import asyncio
import hashlib
import pathlib

import cpuinfo
import torch
import websockets
import json
import psutil
from pathlib import Path
import torch.distributed as dist
import time  # time modülünü ekleyin

import uuid
import logging
import logging.handlers
import os
import sys
import traceback
from datetime import datetime
import platform
from cpuinfo import get_cpu_info  # Değişiklik burada
import GPUtil
from typing import Dict
import requests

from server.src.resource_manager import ResourceManager, ResourceRequirements
from src.training_core import TrainingManager, TrainingConfig
from src.config import init_client_config, setup_logging
# Initialize config and logging
app_config = init_client_config()
logger = setup_logging()


def datetime_handler(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')


class ModelLoader:
    def __init__(self, cache_dir: str = app_config['paths']['model_cache_dir']):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("AIFarmClient")

    def load_model(self, model_url: str, version: str) -> str:
        """Load model code from remote server with caching"""
        try:
            self.logger.info("=" * 50)
            self.logger.info("MODEL LOADING STARTED")
            self.logger.info(f"URL: {model_url}")
            self.logger.info(f"Version: {version}")
            self.logger.info(f"Cache directory: {self.cache_dir}")

            cache_key = hashlib.sha256(f"{model_url}:{version}".encode()).hexdigest()
            cache_path = self.cache_dir / f"{cache_key}.py"
            self.logger.info(f"Cache path: {cache_path}")

            if cache_path.exists():
                self.logger.info("Found model in cache, loading...")
                with open(cache_path, "r") as f:
                    code = f.read()
                self.logger.info(f"Loaded {len(code)} bytes from cache")
                return code

            self.logger.info("Model not in cache, downloading...")

            headers = {
                "Version": version,
                "Accept": "application/json",
                "User-Agent": "AI-Farm-Client/1.0"
            }
            self.logger.info(f"Request headers: {headers}")

            response = requests.get(
                model_url,
                headers=headers,
                timeout=app_config['model']['load_timeout'],
                verify=app_config['model']['verify_ssl']
            )

            self.logger.info(f"Response status: {response.status_code}")
            self.logger.info(f"Response headers: {dict(response.headers)}")

            response.raise_for_status()

            if 'application/json' in response.headers.get('content-type', ''):
                self.logger.info("Parsing JSON response")
                data = response.json()
                self.logger.info(f"JSON keys: {list(data.keys())}")
                code = data.get('code', '')
            else:
                self.logger.info("Using raw response as code")
                code = response.text

            if not code:
                raise ValueError("Empty code received from server")

            self.logger.info(f"Received {len(code)} bytes of code")

            self.logger.info(f"Saving to cache: {cache_path}")
            with open(cache_path, "w", encoding='utf-8') as f:
                f.write(code)
            self.logger.info("Code saved to cache successfully")

            return code

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error: {str(e)}")
            self.logger.error(f"Full error: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to download model: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            self.logger.error(f"Full error: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
        finally:
            self.logger.info("MODEL LOADING ENDED")
            self.logger.info("=" * 50)


class AIFarmClient:
    def __init__(self, server_uri: str = app_config['server']['uri']):
        self.server_uri = server_uri
        self.session_dir = Path(app_config['paths']['session_dir'])
        self.checkpoints_dir = Path(app_config['paths']['checkpoints_dir'])
        self.model_cache_dir = Path(app_config['paths']['model_cache_dir'])

        for directory in [self.session_dir, self.checkpoints_dir, self.model_cache_dir]:
            directory.mkdir(exist_ok=True)

        self.session_file = self.session_dir / "client_session.json"
        self.client_id = self.load_or_create_session()

        self.system_info = self.get_system_info()
        self.running_task = None
        self.reconnect_delay = app_config['server']['reconnect_delay']
        self.max_reconnect_delay = app_config['server']['max_reconnect_delay']

        self.is_connected = asyncio.Event()
        self.resource_manager = ResourceManager()
        self.model_loader = ModelLoader(cache_dir=str(self.model_cache_dir))
        self.training_manager = None
        self.websocket = None
        self.connection_lock = asyncio.Lock()
        self.keepalive_task = None
        self.reconnect_event = asyncio.Event()
        self.last_server_response = time.time()
        self.ping_interval = app_config['server']['ping_interval']
        self.ping_timeout = app_config['server']['ping_timeout']

        self.connection_timeout = app_config['server']['connection_timeout']

        self.is_running = True
        self.retry_interval = app_config['connection']['retry_interval']
        self.max_retries = app_config['connection']['max_retries']
        self.open_connections = 0
        self.connection_limit = app_config['connection']['limit']

        self.reconnection_lock = asyncio.Lock()
        self.receive_lock = asyncio.Lock()
        self.send_lock = asyncio.Lock()
        self.message_queue = asyncio.Queue()
        self.ws_lock = asyncio.Lock()
        self.message_handlers = {}
        self.pending_responses = {}

        self.connection_event = asyncio.Event()
        self.should_stop = False
        self._message_handlers = {}
        self.current_websocket = None

        self.ws_lock = asyncio.Lock()
        self.is_shutting_down = False

    async def ensure_connection(self):
        """Ensure websocket connection is active with improved error handling"""
        if not self.websocket or not self.websocket.open:
            async with self.reconnection_lock:  # Sadece bir reconnect aynı anda
                try:
                    if self.websocket:
                        await self.websocket.close()
                        self.websocket = None

                    # Yeni bağlantı kur
                    self.websocket = await websockets.connect(
                        self.server_uri,
                        ping_interval=15,
                        ping_timeout=self.ping_timeout,
                        close_timeout=10,
                        max_size=None,
                        compression=None,
                        open_timeout=30,
                        extra_headers={
                            'Client-ID': self.client_id
                        }
                    )

                    # Sistem bilgilerini gönder
                    await self.update_system_info()
                    await asyncio.wait_for(
                        self.websocket.send(json.dumps(self.system_info)),
                        timeout=30
                    )
                    logger.info("Reconnected to server and sent system info")

                except Exception as e:
                    logger.error(f"Failed to establish connection: {e}")
                    raise

    async def send_message_with_retry(self, message: Dict, max_retries: int = 3):
        """Send message with improved retry mechanism"""
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                if not self.websocket or not self.websocket.open:
                    await self.ensure_connection()

                async with self.ws_lock:
                    await self.websocket.send(json.dumps(message))
                    return True

            except Exception as e:
                logger.error(f"Send attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                raise

        return False
    async def connect(self):
        """Improved connection management with retry mechanism"""
        recv_lock = asyncio.Lock()  # Recv için özel lock ekleyelim

        while not self.is_shutting_down:
            try:
                if self.open_connections >= self.connection_limit:
                    logger.warning("Connection limit reached, waiting...")
                    await asyncio.sleep(5)
                    continue

                self.open_connections += 1

                async with self.ws_lock:  # Websocket işlemleri için lock kullanımı
                    self.websocket = await asyncio.wait_for(
                        websockets.connect(
                            self.server_uri,
                            ping_interval=15,
                            ping_timeout=self.ping_timeout,
                            close_timeout=10,
                            max_size=None,
                            compression=None,
                            open_timeout=30,
                            extra_headers={
                                'Client-ID': self.client_id
                            }
                        ),
                        timeout=30
                    )

                    # Reset reconnect delay on successful connection
                    self.reconnect_delay = 5

                    # Initial connection setup
                    await self.update_system_info()
                    await self.websocket.send(json.dumps(self.system_info))
                    logger.info("Connected to server and sent system info")

                    # Send initial task request
                    await self.websocket.send(json.dumps({
                        'type': 'task_request',
                        'client_id': self.client_id
                    }))
                    logger.info("Sent initial task request")

                    # Message handling loop
                    while not self.is_shutting_down:
                        if not self.websocket.open:
                            raise websockets.exceptions.ConnectionClosed(1006, "Connection lost")

                        try:
                            # Message handling with timeout and recv_lock
                            async with recv_lock:  # Recv işlemini kilitle
                                message = await asyncio.wait_for(
                                    self.websocket.recv(),
                                    timeout=30
                                )

                                if message:
                                    try:
                                        data = json.loads(message)
                                        await self.handle_message(data)
                                    except json.JSONDecodeError:
                                        logger.error("Invalid message format")
                                    except Exception as e:
                                        logger.error(f"Error handling message: {e}")
                                        continue

                        except asyncio.TimeoutError:
                            # Send heartbeat on timeout
                            try:
                                async with self.ws_lock:  # Heartbeat için ws_lock kullan
                                    heartbeat_success = await asyncio.wait_for(
                                        self.send_heartbeat(),
                                        timeout=5
                                    )
                                    if not heartbeat_success:
                                        raise websockets.exceptions.ConnectionClosed(1001, "Heartbeat failed")
                            except Exception as e:
                                logger.error(f"Heartbeat failed: {e}")
                                break
                            continue

                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("Connection closed by server")
                            break

                        except Exception as e:
                            logger.error(f"Error in message loop: {e}")
                            break

                        # Update last successful message timestamp
                        self.last_server_response = time.time()

                if not self.websocket.open:
                    logger.warning("Connection lost, attempting to reconnect...")
                    break

            except asyncio.TimeoutError:
                logger.error("Connection timeout")
            except Exception as e:
                logger.error(f"Connection error: {e}")
                await self.handle_disconnect()
            finally:
                self.open_connections -= 1

                if self.websocket:
                    try:
                        await self.websocket.close()
                    except:
                        pass
                    self.websocket = None

            # Reconnection delay with exponential backoff
            await asyncio.sleep(self.reconnect_delay)
            self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)

    async def setup_progress_callback(self, task_id: str):
        """Progress callback kurulumu"""

        async def progress_callback(metrics: Dict):
            message = {
                'type': 'task_progress',
                'task_id': task_id,
                'client_id': self.client_id,
                'progress': metrics.get('progress', 0),
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }

            await self.message_queue.put(message)

        await self.training_manager.set_progress_callback(progress_callback)

    async def handle_task_assignment(self, task_data: Dict):
        """
        Task assignment'ı işle:
        1. Task türünü belirle (normal/distributed)
        2. Kaynakları kontrol et ve ayır
        3. Training manager'ı hazırla
        4. İlerlemeyi raporla
        5. Sonucu gönder
        """
        try:
            task_id = task_data['id']
            config = json.loads(task_data['config']) if isinstance(task_data['config'], str) else task_data['config']

            logger.info(f"Starting task {task_id} with config: {config}")

            # Resource requirements kontrolü
            requirements = ResourceRequirements(
                min_gpu_memory=config.get('requirements', {}).get('min_gpu_memory', 0),
                min_cpu_cores=config.get('requirements', {}).get('min_cpu_cores', 1),
                min_ram=config.get('requirements', {}).get('min_ram', 1)
            )

            if not self.resource_manager.check_requirements(requirements):
                raise RuntimeError("Insufficient resources")

            # Kaynakları ayır
            allocated_resources = self.resource_manager.allocate_resources(requirements)

            try:
                # Distributed training için özel ayarlar
                if config.get('distributed'):
                    shard_info = await self.initialize_distributed_training(
                        task_id=task_id,
                        rank=config.get('rank', 0),
                        world_size=config.get('n_clients', 1)
                    )
                    if not shard_info:
                        raise RuntimeError("Failed to initialize distributed training")
                    config['shard_info'] = shard_info

                # Training manager'ı hazırla
                training_config = TrainingConfig(
                    model_name=config['name'],
                    batch_size=config['batch_size'],
                    learning_rate=config['learning_rate'],
                    epochs=config['epochs'],
                    device=f"cuda:{allocated_resources['gpu_id']}" if allocated_resources[
                                                                          'gpu_id'] is not None else "cpu",
                    distributed=config.get('distributed', False),
                    num_workers=allocated_resources['cpu_cores'],
                    checkpoint_freq=config.get('checkpoint_freq', 1)
                )

                self.training_manager = TrainingManager(training_config)

                # Progress callback'i ayarla
                await self.setup_progress_callback(task_id)

                # Model kodunu yükle
                model_code = await self.load_model(
                    url=config['url'],
                    version=config.get('version', 'latest')
                )

                # Training başlat
                training_result = await self.training_manager.train(
                    task_id=task_id,
                    model_code=model_code,
                    train_data={
                        'config': config.get('data_config', {}),
                        'shard_info': config.get('shard_info')
                    }
                )

                # Sonucu işle ve gönder
                await self.handle_training_result(
                    task_id=task_id,
                    result=training_result,
                    config=config
                )

                return training_result

            finally:
                # Kaynakları serbest bırak
                if allocated_resources:
                    self.resource_manager.release_resources(allocated_resources)

        except Exception as e:
            logger.error(f"Task execution failed: {str(e)}\n{traceback.format_exc()}")
            await self.report_task_failure(task_id, str(e))
            return None

    async def handle_training_result(self, task_id: str, result: Dict, config: Dict):
        """Training sonucunu işle ve raporla"""
        try:
            # Model state'i kaydet
            if hasattr(self.training_manager.model, 'state_dict'):
                checkpoint_path = os.path.join(
                    './aggregated_models',
                    f"task_{task_id}_shard_{config.get('rank', 0)}_{datetime.now():%Y%m%d_%H%M%S}.pt"
                )

                torch.save({
                    'model_state': self.training_manager.model.state_dict(),
                    'optimizer_state': self.training_manager.optimizer.state_dict() if self.training_manager.optimizer else None,
                    'config': config,
                    'metrics': result.get('metrics', {}),
                    'shard_info': config.get('shard_info'),
                    'timestamp': datetime.now().isoformat()
                }, checkpoint_path)

                logger.info(f"Saved model checkpoint to {checkpoint_path}")

                # Sonucu server'a bildir
                result_message = {
                    'type': 'task_result',
                    'task_id': task_id,
                    'client_id': self.client_id,
                    'status': result.get('status', 'completed'),
                    'metrics': result.get('metrics', {}),
                    'checkpoint_path': checkpoint_path,
                    'shard_info': config.get('shard_info')
                }

                await self.send_message_with_retry(result_message)

        except Exception as e:
            logger.error(f"Failed to handle training result: {e}")
            raise

    async def report_task_failure(self, task_id: str, error: str):
        """Task hatasını raporla"""
        try:
            message = {
                'type': 'task_result',
                'task_id': task_id,
                'client_id': self.client_id,
                'status': 'failed',
                'error': str(error),
                'timestamp': datetime.now().isoformat()
            }

            await self.send_message_with_retry(message)

        except Exception as e:
            logger.error(f"Failed to report task failure: {e}")
    async def initialize_distributed_training(self, task_id: str, rank: int, world_size: int) -> Dict:
        """Distributed training için initialization"""
        try:
            init_message = {
                'type': 'init_distributed',
                'task_id': task_id,
                'client_id': self.client_id,
                'rank': rank
            }

            # Server'a initialization isteği gönder
            response = await self.send_message_with_retry(init_message)
            if not response or response.get('status') == 'error':
                raise RuntimeError(f"Initialization failed: {response.get('error')}")

            init_data = response.get('data', {})

            # Environment variables
            os.environ.update({
                'MASTER_ADDR': init_data.get('master_addr', 'localhost'),
                'MASTER_PORT': str(init_data.get('master_port', 29500)),
                'WORLD_SIZE': str(world_size),
                'RANK': str(rank)
            })

            # Process group initialization
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(
                    backend='gloo',
                    init_method='env://'
                )

            return init_data.get('shard_info', {
                'id': rank,
                'total_shards': world_size,
                'start_point': 0,
                'end_point': 100
            })

        except Exception as e:
            logger.error(f"Distributed initialization failed: {e}")
            raise
    async def process_message_queue(self):
        """Process queued messages with rate limiting"""
        send_lock = asyncio.Lock()
        rate_limit = 0.1  # saniyeler arası minimum bekleme süresi
        last_send_time = 0

        while True:
            try:
                message = await self.message_queue.get()

                # Rate limiting
                current_time = time.time()
                time_since_last = current_time - last_send_time
                if time_since_last < rate_limit:
                    await asyncio.sleep(rate_limit - time_since_last)

                async with send_lock:
                    try:
                        success = await self.send_message_with_retry(message)
                        if not success:
                            await asyncio.sleep(1)
                            await self.message_queue.put(message)
                    except Exception as e:
                        logger.error(f"Error sending message: {e}")
                        await asyncio.sleep(1)
                        await self.message_queue.put(message)

                last_send_time = time.time()
                self.message_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in message queue processing: {e}")
                await asyncio.sleep(1)
    async def send_heartbeat(self):
        """Send heartbeat with system info"""
        try:
            if self.websocket and self.websocket.open:
                await self.websocket.send(json.dumps({
                    'type': 'heartbeat',
                    'client_id': self.client_id,
                    'timestamp': datetime.now().isoformat(),
                    'system_info': self.system_info
                }))
                return True
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {str(e)}")
            return False
    def generate_machine_id(self) -> str:
        """Benzersiz makine kimliği oluştur"""
        machine_info = {
            'hostname': platform.node(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'mac_addresses': self.get_mac_addresses(),
            'cpu_info': cpuinfo.get_cpu_info()['brand_raw']
        }

        # Makine bilgilerinden hash oluştur
        machine_hash = hashlib.sha256(json.dumps(machine_info, sort_keys=True).encode()).hexdigest()
        return machine_hash[:32]  # İlk 32 karakteri al

    def get_mac_addresses(self) -> list:
        """Sistemdeki tüm MAC adreslerini topla"""
        import uuid
        mac_num = hex(uuid.getnode()).replace('0x', '').upper()
        mac = '-'.join(mac_num[i: i + 2] for i in range(0, 11, 2))
        return [mac]

    def load_or_create_session(self) -> str:
        """Session dosyasını yükle veya yeni oluştur"""
        try:
            if self.session_file.exists():
                with open(self.session_file, 'r') as f:
                    session_data = json.load(f)
                    logger.info(f"Loaded existing session: {session_data['client_id']}")
                    return session_data['client_id']

            # Yeni session oluştur
            client_id = self.generate_machine_id()
            session_data = {
                'client_id': client_id,
                'created_at': datetime.now().isoformat(),
                'last_connected': datetime.now().isoformat(),
                'total_tasks_completed': 0,
                'performance_metrics': {}
            }

            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=4)

            logger.info(f"Created new session: {client_id}")
            return client_id

        except Exception as e:
            logger.error(f"Error managing session: {e}")
            # Fallback olarak UUID kullan
            return str(uuid.uuid4())

    def update_session_metrics(self, metrics: dict):
        """Session metriklerini güncelle"""
        try:
            if self.session_file.exists():
                with open(self.session_file, 'r') as f:
                    session_data = json.load(f)

                session_data['last_connected'] = datetime.now().isoformat()
                session_data['total_tasks_completed'] += metrics.get('completed_tasks', 0)
                session_data['performance_metrics'].update(metrics)

                with open(self.session_file, 'w') as f:
                    json.dump(session_data, f, indent=4)

        except Exception as e:
            logger.error(f"Error updating session metrics: {e}")

    def get_system_info(self) -> Dict:
        """Sistem bilgilerini topla ve session metrikleriyle birleştir"""
        try:
            cpu_info = get_cpu_info()  # cpuinfo.get_cpu_info() yerine get_cpu_info()

            info = {
                'id': self.client_id,
                'hostname': platform.node(),
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor()
                },
                'cpu': {
                    'brand': cpu_info.get('brand_raw', 'Unknown'),
                    'cores_physical': psutil.cpu_count(logical=False),
                    'cores_logical': psutil.cpu_count(logical=True),
                    'frequency': {
                        'current': psutil.cpu_freq().current if psutil.cpu_freq() else None,
                        'min': psutil.cpu_freq().min if psutil.cpu_freq() else None,
                        'max': psutil.cpu_freq().max if psutil.cpu_freq() else None
                    },
                    'usage_percent': psutil.cpu_percent(interval=1)
                },
                'ram': {
                    'total': psutil.virtual_memory().total,
                    'available': psutil.virtual_memory().available,
                    'used': psutil.virtual_memory().used,
                    'percent': psutil.virtual_memory().percent
                }
            }

            # GPU bilgileri
            try:
                gpus = GPUtil.getGPUs()
                info['gpu'] = [{
                    'id': gpu.id,
                    'name': gpu.name,
                    'load': gpu.load,
                    'memory': {
                        'total': gpu.memoryTotal,
                        'used': gpu.memoryUsed,
                        'free': gpu.memoryFree
                    },
                    'temperature': gpu.temperature
                } for gpu in gpus]
            except Exception as e:
                logger.warning(f"Failed to get GPU info: {e}")
                info['gpu'] = None

            # Session metriklerini ekle
            try:
                if self.session_file.exists():
                    with open(self.session_file, 'r') as f:
                        session_data = json.load(f)
                    info['session'] = {
                        'created_at': session_data['created_at'],
                        'total_tasks_completed': session_data['total_tasks_completed'],
                        'performance_metrics': session_data['performance_metrics']
                    }
            except Exception as e:
                logger.error(f"Failed to load session metrics: {e}")

            return info

        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {
                'id': self.client_id,
                'error': str(e)
            }

    async def execute_task(self, task: Dict) -> Dict:
        """Execute task with improved shard handling"""
        try:
            if not task:
                raise ValueError("No task provided")

            task_id = task['id']
            logger.info(f"==================== Task Execution Start ====================")
            logger.info(f"Task ID: {task_id}")

            # Task'ı running_task olarak kaydet
            self.running_task = task

            config = json.loads(task['config']) if isinstance(task['config'], str) else task['config']
            logger.info(f"Task configuration loaded")

            # Resource requirements'ları hazırla
            requirements = ResourceRequirements(
                min_gpu_memory=json.loads(task['requirements'])['min_gpu_memory'] if isinstance(task['requirements'],
                                                                                                str) else
                task['requirements']['min_gpu_memory'],
                min_cpu_cores=json.loads(task['requirements'])['min_cpu_cores'] if isinstance(task['requirements'],
                                                                                              str) else
                task['requirements']['min_cpu_cores'],
                min_ram=json.loads(task['requirements'])['min_ram'] if isinstance(task['requirements'], str) else
                task['requirements']['min_ram']
            )

            if not self.resource_manager.check_requirements(requirements):
                raise RuntimeError("Insufficient resources for task")

            allocated_resources = self.resource_manager.allocate_resources(requirements)
            logger.info("Resources allocated successfully")

            try:
                # Get distributed training assignment if needed
                shard_info = None
                if config.get('distributed'):
                    try:
                        logger.info("Setting up distributed training")
                        shard_info = await self.get_training_assignment(task_id, config)

                        if not shard_info:
                            logger.info("No more shards available for this task")
                            return {
                                'status': 'completed',
                                'metrics': {},
                                'shard_info': config.get('shard_info'),
                                'message': 'No more shards available'
                            }

                        config['shard_info'] = shard_info
                        logger.info(f"Training will use shard: {shard_info}")

                    except Exception as e:
                        logger.error(f"Distributed setup failed: {e}")
                        config['distributed'] = False

                # Initialize training config
                training_config = TrainingConfig(
                    model_name=config['name'],
                    batch_size=config['batch_size'],
                    learning_rate=config['learning_rate'],
                    epochs=config['epochs'],
                    device=f"cuda:{allocated_resources['gpu_id']}" if allocated_resources[
                                                                          'gpu_id'] is not None else "cpu",
                    distributed=config['distributed'],
                    num_workers=allocated_resources['cpu_cores'],
                    checkpoint_freq=config.get('checkpoint_freq', 1)
                )

                # Setup training manager
                self.training_manager = TrainingManager(training_config)
                await self.setup_progress_callback(task_id)

                # Load and verify model
                model_code = await self.load_model(config)
                if not model_code:
                    raise ValueError("Failed to load model code")

                # Start training
                logger.info("Starting model training")
                training_result = await self.training_manager.train(
                    task_id=task_id,
                    model_code=model_code,
                    train_data={
                        'config': config['data_config'],
                        'shard_info': shard_info
                    } if shard_info else config['data_config'],
                    resume_from=None
                )

                result = await self.process_training_result(training_result, allocated_resources)

                # Add shard info to result
                result['shard_info'] = shard_info

                # Cleanup and request next shard
                cleanup_success = await self.cleanup(task_id, config, allocated_resources)

                return result

            except Exception as e:
                raise RuntimeError(f"Training failed: {str(e)}")

        except Exception as e:
            error_msg = f"Task execution failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return {
                'status': 'failed',
                'error': str(e),
                'error_log': error_msg,
                'shard_info': config.get('shard_info') if 'config' in locals() else None
            }
        finally:
            # Task bittikten sonra running_task'ı temizle
            self.running_task = None

            # Resource'ları serbest bırak
            try:
                if 'allocated_resources' in locals():
                    self.resource_manager.release_resources(allocated_resources)
            except Exception as e:
                logger.error(f"Error releasing resources: {e}")

    async def get_training_assignment(self, task_id: str, config: Dict) -> Dict:
        """Get distributed training assignment from server with improved error handling"""
        max_retries = 3
        retry_delay = 1
        timeout = 60  # Timeout'u 60 saniyeye çıkardık

        for attempt in range(max_retries):
            try:
                logger.info("Getting distributed training assignment")

                # Check websocket connection
                if not self.websocket or not self.websocket.open:
                    raise RuntimeError("Websocket connection is not available")

                # Send initialization request
                request_data = {
                    'type': 'init_distributed',
                    'task_id': task_id,
                    'client_id': self.client_id,
                    'rank': config.get('rank', 0)
                }
                logger.info(f"Sending initialization request: {request_data}")

                # Send with timeout
                await asyncio.wait_for(
                    self.websocket.send(json.dumps(request_data)),
                    timeout=timeout
                )

                # Get response with timeout
                response = await asyncio.wait_for(
                    self.websocket.recv(),
                    timeout=timeout
                )

                response_data = json.loads(response)
                logger.info(f"Received initialization response: {response_data}")

                if response_data.get('status') == 'error':
                    raise RuntimeError(f"Server error: {response_data.get('error')}")

                init_data = response_data.get('data', response_data)

                # Set environment variables
                os.environ['MASTER_ADDR'] = init_data.get('master_addr', 'localhost')
                os.environ['MASTER_PORT'] = str(init_data.get('master_port', 29500))
                os.environ['WORLD_SIZE'] = str(init_data.get('world_size', 1))
                os.environ['RANK'] = str(init_data.get('rank', 0))

                # Get shard info
                shard_info = init_data.get('shard_info', {
                    'id': init_data.get('rank', 0),
                    'total_shards': init_data.get('world_size', 1),
                    'start_point': 0,
                    'end_point': 100
                })

                logger.info(f"Initialized with shard info: {shard_info}")
                logger.info(f"Environment variables set: {os.environ}")

                # Notify server that we're ready
                await asyncio.wait_for(
                    self.websocket.send(json.dumps({
                        'type': 'distributed_ready',
                        'task_id': task_id,
                        'client_id': self.client_id,
                        'rank': init_data.get('rank', 0)
                    })),
                    timeout=timeout
                )

                return shard_info

            except Exception as e:
                error_msg = f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")

                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                    retry_delay *= 2  # Exponential backoff
                    continue
                else:
                    raise RuntimeError("Failed to get training assignment")

        raise RuntimeError("Failed to get training assignment")





    async def load_model(self, config: Dict) -> str:
        """Load model code from server"""
        try:
            model_code = self.model_loader.load_model(
                config['url'],
                config.get('version', 'latest')
            )
            logger.info(f"Model loaded successfully ({len(model_code)} bytes)")
            return model_code
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    async def process_training_result(self, training_result: Dict, allocated_resources: Dict) -> Dict:
        """Process and format training result"""
        if isinstance(training_result, dict):
            # Model state'i al
            model_state = None
            if hasattr(self.training_manager.model, 'state_dict'):
                model_state = self.training_manager.model.state_dict()

            return {
                'status': training_result.get('status', 'completed'),
                'metrics': {
                    **training_result.get('metrics', {}),
                    'resources': self.resource_manager.monitor_resources()
                },
                'checkpoint_path': training_result.get('checkpoint_path'),
                'shard_info': training_result.get('shard_info'),
                'model_update': model_state,  # Model state'i ekle
                'rank': training_result.get('rank')
            }
        return {
            'status': 'completed',
            'metrics': {
                'resources': self.resource_manager.monitor_resources()
            }
        }

    async def cleanup(self, task_id: str, config: Dict, allocated_resources: Dict):
        """Enhanced cleanup with task continuation"""
        try:
            # Önce modeli kaydet
            if self.training_manager and hasattr(self.training_manager.model, 'state_dict'):
                # Final model path
                final_model_path = os.path.join(
                    './aggregated_models',
                    f"task_{task_id}_shard_{config.get('shard_info', {}).get('id')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                )

                # Model state ve diğer bilgileri kaydet
                torch.save({
                    'model_state': self.training_manager.model.state_dict(),
                    'optimizer_state': self.training_manager.optimizer.state_dict() if self.training_manager.optimizer else None,
                    'config': config,
                    'metrics': self.training_manager.training_metrics,
                    'shard_info': config.get('shard_info'),
                    'timestamp': datetime.now().isoformat()
                }, final_model_path)

                logger.info(f"Saved final model to {final_model_path}")

            # Calculate completed shard info
            completed_shard = {
                'id': config.get('shard_info', {}).get('id'),
                'start_point': config.get('shard_info', {}).get('start_point'),
                'end_point': config.get('shard_info', {}).get('end_point')
            }

            try:
                if config.get('distributed'):
                    cleanup_message = {
                        'type': 'distributed_cleanup',
                        'task_id': task_id,
                        'client_id': self.client_id,
                        'rank': config.get('rank', 0),
                        'completed_shard': completed_shard,
                        'model_path': final_model_path
                    }

                    # Send with retry
                    for attempt in range(3):
                        try:
                            await self.send_message_with_retry(cleanup_message)
                            break
                        except Exception as e:
                            if attempt == 2:  # Son deneme
                                logger.warning(f"Failed to send distributed cleanup message: {e}")
                            await asyncio.sleep(1)

                # Release resources
                if self.resource_manager and allocated_resources:
                    self.resource_manager.release_resources(allocated_resources)
                    logger.info("Resources released successfully")

                # Request next task/shard
                continuation_message = {
                    'type': 'task_request',
                    'client_id': self.client_id,
                    'previous_task_id': task_id,
                    'completed_shard': completed_shard,
                    'continue_training': True,
                    'final_model_path': final_model_path
                }

                # Send with retry
                for attempt in range(3):
                    try:
                        await self.send_message_with_retry(continuation_message)
                        break
                    except Exception as e:
                        if attempt == 2:  # Son deneme
                            logger.error(f"Failed to request next shard: {e}")
                        await asyncio.sleep(1)

                logger.info("Cleanup completed successfully")
                return True

            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
                return False

        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            return False
        finally:
            # Clear training manager
            self.training_manager = None
    async def run_training(self, task_id: str, model_code: str, config: Dict, resources: Dict) -> Dict:
        """Manage the training process.

        Args:
            task_id (str): Unique identifier for the training task.
            model_code (str): The code or identifier for the model to be trained.
            config (Dict): Configuration dictionary containing training parameters.
            resources (Dict): Dictionary of resource parameters.

        Returns:
            Dict: A dictionary containing the status of the training, metrics, and checkpoint path.
        """
        start_time = datetime.now()

        try:
            # Set environment variables for distributed training
            if config.get('distributed', False):
                os.environ['MASTER_ADDR'] = config.get('master_addr', 'localhost')
                os.environ['MASTER_PORT'] = config.get('master_port', '12355')
                os.environ['WORLD_SIZE'] = str(config.get('world_size', 1))
                os.environ['RANK'] = str(config.get('rank', 0))

                # Backend'i gloo olarak değiştiriyoruz
                dist.init_process_group(
                    backend='gloo',  # nccl yerine gloo kullan
                    init_method='env://',
                    world_size=int(os.environ['WORLD_SIZE']),
                    rank=int(os.environ['RANK'])
                )

            # Progress callback
            async def progress_callback(metrics):
                await self.websocket.send(json.dumps({
                    'type': 'task_progress',
                    'task_id': task_id,
                    'client_id': self.client_id,
                    'progress': metrics.get('progress', 0),
                    'resources': self.resource_manager.monitor_resources(),
                    'metrics': metrics
                }))
                logger.info(f"Progress: {metrics.get('progress', 0):.1f}%, Loss: {metrics.get('loss', 0):.4f}")

            # Start training asynchronously
            training_result = await self.training_manager.train(
                task_id=task_id,
                model_code=model_code,
                train_data=config.get('data_config', {}),
                resume_from=None  # Optional: Add this back if needed for resuming
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Collect metrics
            resource_metrics = self.resource_manager.monitor_resources()

            if isinstance(training_result, dict):
                return {
                    'status': training_result.get('status', 'completed'),
                    'metrics': {
                        **training_result.get('metrics', {}),
                        'duration': duration,
                        'resources': resource_metrics
                    },
                    'checkpoint_path': training_result.get('checkpoint_path')
                }
            else:
                return {
                    'status': 'completed',
                    'metrics': {
                        'duration': duration,
                        'resources': resource_metrics
                    }
                }

        except Exception as e:
            logger.error(f"Training failed for task {task_id}: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}")
    async def update_system_info(self):
        """Sistem bilgilerini güncelle"""
        try:
            self.system_info = self.get_system_info()

            # Session metriklerini güncelle
            metrics = {
                'last_update': datetime.now().isoformat(),
                'system_metrics': {
                    'cpu_usage': psutil.cpu_percent(),
                    'ram_usage': psutil.virtual_memory().percent
                }
            }

            # GPU metrikleri varsa ekle
            if GPUtil.getGPUs():
                metrics['system_metrics']['gpu_usage'] = GPUtil.getGPUs()[0].load * 100
                metrics['system_metrics']['gpu_memory'] = GPUtil.getGPUs()[0].memoryUtil * 100

            self.update_session_metrics(metrics)

        except Exception as e:
            logger.error(f"Failed to update system info: {e}")
            self.system_info = {
                'id': self.client_id,
                'error': str(e)
            }

    async def connect(self):
        """Improved connection management with retry mechanism"""
        recv_lock = asyncio.Lock()  # Websocket recv için lock
        send_lock = asyncio.Lock()  # Websocket send için lock

        while True:
            try:
                if self.open_connections >= self.connection_limit:
                    logger.warning("Connection limit reached, waiting...")
                    await asyncio.sleep(5)
                    continue

                self.open_connections += 1

                # Connection with timeout
                try:
                    self.websocket = await asyncio.wait_for(
                        websockets.connect(
                            self.server_uri,
                            ping_interval=15,
                            ping_timeout=self.ping_timeout,
                            close_timeout=10,
                            max_size=None,
                            compression=None,
                            open_timeout=30,
                            extra_headers={
                                'Client-ID': self.client_id
                            }
                        ),
                        timeout=30
                    )
                except asyncio.TimeoutError:
                    logger.error("Connection attempt timed out")
                    await asyncio.sleep(self.reconnect_delay)
                    continue

                self.reconnect_delay = 5  # Reset delay on successful connection

                # Initial connection setup
                async with send_lock:
                    await self.update_system_info()
                    await asyncio.wait_for(
                        self.websocket.send(json.dumps(self.system_info)),
                        timeout=10
                    )
                    logger.info("Connected to server and sent system info")

                    # İlk task request'ini gönder
                    await self.websocket.send(json.dumps({
                        'type': 'task_request',
                        'client_id': self.client_id
                    }))
                    logger.info("Sent initial task request")

                # Message processing loop
                while True:
                    if not self.websocket.open:
                        raise websockets.exceptions.ConnectionClosed(1006, "Connection lost")

                    try:
                        # Message handling with timeout and lock
                        async with recv_lock:
                            message = await asyncio.wait_for(
                                self.websocket.recv(),
                                timeout=30
                            )
                            if message:
                                try:
                                    data = json.loads(message)
                                    await self.handle_message(json.loads(message))
                                except json.JSONDecodeError:
                                    logger.error("Invalid message format")
                                except Exception as e:
                                    logger.error(f"Error handling message: {e}")
                                    continue

                    except asyncio.TimeoutError:
                        # Send heartbeat on timeout
                        try:
                            async with send_lock:
                                await asyncio.wait_for(
                                    self.send_heartbeat(),
                                    timeout=5
                                )
                        except Exception as e:
                            logger.error(f"Error sending heartbeat: {e}")
                            break
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Connection closed by server")
                        break
                    except Exception as e:
                        logger.error(f"Error in message loop: {e}")
                        break

            except Exception as e:
                logger.error(f"Connection error: {e}")
                await self.handle_disconnect()
            finally:
                self.open_connections -= 1

            # Reconnection delay with exponential backoff
            await asyncio.sleep(self.reconnect_delay)
            self.reconnect_delay = min(self.reconnect_delay * 2, 60)

    async def handle_task_result(self, data: Dict):
        """Handle task completion and request next shard if available"""
        try:
            # Send current task result
            result_message = {
                'type': 'task_result',
                'task_id': data['task_id'],
                'client_id': self.client_id,
                'status': data.get('status', 'completed'),
                'result': data.get('metrics', {}),
                'shard_info': data.get('shard_info'),
                'checkpoint_path': data.get('checkpoint_path'),
                'error_log': data.get('error_log')
            }

            # Send with retry
            for attempt in range(3):
                try:
                    await self.websocket.send(json.dumps(result_message))

                    # Wait for acknowledgment
                    response = await asyncio.wait_for(
                        self.websocket.recv(),
                        timeout=30
                    )
                    response_data = json.loads(response)

                    if response_data.get('type') == 'task_result_ack':
                        logger.info(f"Result acknowledged for task {data['task_id']}")

                        # Request next task/shard
                        next_task_request = {
                            'type': 'task_request',
                            'client_id': self.client_id,
                            'previous_task_id': data['task_id'],
                            'previous_shard': data.get('shard_info', {}).get('id'),
                            'previous_rank': data.get('rank', 0)
                        }

                        await self.websocket.send(json.dumps(next_task_request))
                        logger.info("Sent request for next shard")
                        break

                except Exception as e:
                    logger.error(f"Attempt {attempt + 1}/3 failed to send task result: {e}")
                    if attempt < 2:  # Don't sleep on last attempt
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue

        except Exception as e:
            logger.error(f"Error handling task result: {e}")
            # Try to reconnect if needed
            if not self.websocket or not self.websocket.open:
                await self.handle_disconnect()

    async def handle_message(self, data: Dict):
        """Handle incoming messages with improved synchronization"""
        try:
            if not isinstance(data, dict):
                logger.error(f"Invalid message format: {data}")
                return

            message_type = data.get('type')
            if not message_type:
                logger.error("Message type missing")
                return

            logger.info(f"Received message type: {message_type}")

            if message_type == 'task':
                task_result = await self.execute_task(data['data'])
                try:
                    message = {
                        'type': 'task_result',
                        'task_id': data['data']['id'],
                        'client_id': self.client_id,
                        'status': task_result.get('status', 'completed'),
                        'metrics': task_result.get('metrics', {}),
                        'checkpoint_path': task_result.get('checkpoint_path'),
                        'error_log': task_result.get('error_log'),
                        'shard_info': task_result.get('shard_info'),
                        'rank': data['data'].get('rank', 0)
                    }
                    await self.send_message_with_retry(message)

                except Exception as e:
                    logger.error(f"Error sending task result: {e}")

            elif message_type in ['task_progress_ack', 'task_result_ack', 'heartbeat_ack']:
                logger.info(f"Received acknowledgment for {message_type}")

        except Exception as e:
            logger.error(f"Error handling message: {e}")

        finally:
            self.last_server_response = time.time()

    async def handle_disconnect(self):
        """Improved disconnect handling"""
        try:
            if self.websocket:
                await self.websocket.close()
                self.websocket = None

            if self.running_task:
                logger.warning("Training interrupted due to disconnection")
                if self.training_manager:
                    try:
                        # Save checkpoint
                        checkpoint_path = self.checkpoints_dir / f"interrupt_{self.running_task['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                        self.training_manager.checkpoint_manager.save_checkpoint(
                            self.training_manager.model,
                            self.training_manager.optimizer,
                            self.training_manager.current_epoch,
                            {'status': 'interrupted'},
                            str(checkpoint_path)
                        )
                        logger.info(f"Saved interrupt checkpoint to {checkpoint_path}")
                    except Exception as e:
                        logger.error(f"Error saving checkpoint on disconnect: {e}")

                self.running_task = None

        except Exception as e:
            logger.error(f"Error in handle_disconnect: {e}")

    async def start(self):
        """Start client with improved error handling"""
        try:
            self.should_stop = False

            # Create and start message queue processor
            queue_processor = asyncio.create_task(self.process_message_queue())

            # Start main connection handler
            connection_handler = asyncio.create_task(self.connect())

            # Wait for tasks
            await asyncio.gather(queue_processor, connection_handler)

        except KeyboardInterrupt:
            logger.info("Shutting down client...")
        except Exception as e:
            logger.error(f"Error in client: {e}")
        finally:
            self.should_stop = True
            self.connection_event.set()  # Wake up any waiting coroutines

    async def stop(self):
        """Gracefully stop the client"""
        self.should_stop = True
        self.connection_event.set()
        if self.current_websocket:
            await self.current_websocket.close()
def main():
    try:
        client = AIFarmClient()
        asyncio.run(client.start())
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.critical(f"Client failed: {e}\n{traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main()