import asyncio
import shutil
import signal
import threading
import time
from contextlib import contextmanager

import numpy as np
import torch
import uvicorn
import websockets
import mysql.connector
from mysql.connector import Error, pooling
from fastapi import FastAPI, HTTPException,Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from src.config import init_app_config, DB_CONFIG


from fastapi.middleware.cors import CORSMiddleware
import json
from datetime import datetime
import logging
import logging.handlers
import os
import sys
import traceback
from typing import Dict, List, Optional, Union
from pydantic import BaseModel,Field
from starlette.responses import RedirectResponse, JSONResponse

from src.resource_manager import ResourceManager, LoadBalancer
from src.training_coordinator import TrainingCoordinator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.requests import Request

app_config = init_app_config()
MAX_CONNECTIONS = int(os.getenv('MAX_CONNECTIONS', '100'))

# Logging setup
log_dir = app_config['paths']['log_dir']
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(level=getattr(logging, app_config['logging']['log_level']))
logger = logging.getLogger("AIFarm")

# File handler for all logs
file_handler = logging.handlers.RotatingFileHandler(
    f"{log_dir}/server.log",
    maxBytes=app_config['logging']['max_bytes'],
    backupCount=app_config['logging']['backup_count']
)
file_handler.setFormatter(
    logging.Formatter(app_config['logging']['format'])
)
logger.addHandler(file_handler)

# Error log handler
error_handler = logging.handlers.RotatingFileHandler(
    f"{log_dir}/error.log",
    maxBytes=app_config['logging']['max_bytes'],
    backupCount=app_config['logging']['backup_count']
)
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(
    logging.Formatter(app_config['logging']['error_format'])
)
logger.addHandler(error_handler)

# Create required directories
os.makedirs(app_config['paths']['models_dir'], exist_ok=True)
os.makedirs(app_config['paths']['checkpoints_dir'], exist_ok=True)

# FastAPI Models
class SessionInfo(BaseModel):
    created_at: str
    total_tasks_completed: int = 0
    performance_metrics: Dict = Field(default_factory=dict)
class TaskCreate(BaseModel):
    type: str = "training"
    name: str
    url: str
    version: str
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 10
    distributed: bool
    requirements: Dict = Field(default_factory=lambda: {
        "min_gpu_memory": 0,
        "min_cpu_cores": 1,
        "min_ram": 1
    })
    priority: Optional[int] = 0
    n_clients: Optional[int]  # n_clients alanı burada tanımlanmış
    aggregation_frequency: Optional[int] = 5
    warmup_epochs: Optional[int] = 2
    data_config: Optional[Dict] = Field(default_factory=lambda: {
        "split_method": "random",
        "validation_ratio": 0.2
    })

    # Config sınıfı ile model yapılandırmalarını tanımlıyoruz
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist(),
            torch.Tensor: lambda x: x.cpu().numpy().tolist()
        }



class ClientInfo(BaseModel):
    id: str
    status: str
    cpu_info: Dict
    ram_info: Dict
    gpu_info: Optional[Union[Dict, List[Dict]]] = None
    session_info: Optional[SessionInfo] = None
    last_seen: datetime


class TaskInfo(BaseModel):
    id: int
    type: str
    requirements: Dict
    status: str
    client_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    result: Optional[Dict] = None  # Changed from str to Dict
    priority: int
    error_log: Optional[str] = None





try:
    # Önce DB_CONFIG'den pool parametrelerini çıkaralım
    db_config = app_config['db_config'].copy()
    pool_name = db_config.pop('pool_name', 'ai_farm_pool')
    pool_size = db_config.pop('pool_size', 10)
    if 'max_connections' in db_config:
        db_config.pop('max_connections')
    # Şimdi connection pool'u doğru şekilde oluşturalım
    connection_pool = mysql.connector.pooling.MySQLConnectionPool(
        pool_name=pool_name,
        pool_size=pool_size,
        **db_config
    )
    logger.info("Database connection pool created successfully")
except Error as e:
    logger.critical(f"Failed to create connection pool: {e}")
    sys.exit(1)
def datetime_handler(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

class AIFarmServer:
    def __init__(self):
        self.app_config = init_app_config()
        self.app = FastAPI(title="AI Farm API")
        self.setup_api()
        self.training_coordinators: Dict[int, TrainingCoordinator] = {}

        self.background_tasks = []
        self.shutdown_event = asyncio.Event()
        self.is_shutting_down = False

        self.clients: Dict[str, dict] = {}
        self.setup_database()
        self.setup_admin_panel()

        # Resource manager ve load balancer başlat
        self.resource_manager = ResourceManager()
        self.load_balancer = LoadBalancer(self.resource_manager)
        self.training_coordinators: Dict[int, TrainingCoordinator] = {}  # task_id -> coordinator

        # Config'den değerleri al
        self.inactive_threshold = self.app_config['resource_config']['inactive_threshold']
        self.ping_interval = self.app_config['resource_config']['ping_interval']

        # Database pool configuration from config
        self.db_config = self.app_config['db_config']

        # Initialize connection pool
        try:
            self.connection_pool = mysql.connector.pooling.MySQLConnectionPool(**self.db_config)
            logger.info("Database connection pool created successfully")
        except Error as e:
            logger.critical(f"Failed to create connection pool: {e}")
            sys.exit(1)


        # Cache settings from config
        self._task_cache = {}
        self._last_task_update = 0
        self._task_cache_ttl = self.app_config['cache_config']['ttl']
        self._cache_lock = asyncio.Lock()
        self._clients_cache = {}
        self._clients_cache_ttl = self.app_config['cache_config']['clients_ttl']
        self._last_clients_update = 0
        self._clients_lock = asyncio.Lock()
        self._db_semaphore = asyncio.Semaphore(MAX_CONNECTIONS)

        # Rate limiting
        self.limiter = Limiter(
            key_func=get_remote_address,
            default_limits=[self.app_config['rate_limit']]
        )
        self.app.state.limiter = self.limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        logger.info("Initialized task cache")

    async def shutdown(self):
        """Gracefully shutdown the server"""
        if self.is_shutting_down:
            return

        self.is_shutting_down = True
        logger.info("Initiating server shutdown...")

        # Signal shutdown to all components
        self.shutdown_event.set()

        # Close all websocket connections gracefully
        close_tasks = []
        for client_id, client_data in list(self.clients.items()):
            if 'websocket' in client_data:
                try:
                    close_tasks.append(
                        asyncio.create_task(
                            client_data['websocket'].close(
                                code=1001,
                                reason="Server shutdown"
                            )
                        )
                    )
                except Exception as e:
                    logger.error(f"Error closing websocket for client {client_id}: {e}")

        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Cancel background tasks
        for task in self.background_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error cancelling task: {e}")

        # Close server
        if hasattr(self, 'server'):
            self.server.close()
            await self.server.wait_closed()

        logger.info("Server shutdown completed")

    async def get_tasks_with_cache(self):
        """Get tasks with caching and improved performance"""
        async with self._cache_lock:
            current_time = time.time()

            # Return cached results if still valid
            if current_time - self._last_task_update < self._task_cache_ttl:
                return self._task_cache.get('tasks', [])

            try:
                # Yeni bir bağlantı al
                conn = None
                cursor = None
                try:
                    conn = self.connection_pool.get_connection()
                    cursor = conn.cursor(dictionary=True)

                    # Task'ları tek sorguda al
                    cursor.execute("""
                        SELECT t.*, 
                               GROUP_CONCAT(
                                   JSON_OBJECT(
                                       'client_id', ta.client_id,
                                       'status', ta.status,
                                       'assigned_at', ta.assigned_at
                                   )
                               ) as assignments
                        FROM tasks t
                        LEFT JOIN task_assignments ta ON t.id = ta.task_id
                        GROUP BY t.id
                        ORDER BY t.priority DESC, t.created_at DESC
                        LIMIT 100
                    """)

                    tasks = cursor.fetchall()

                    # JSON alanları parse et
                    for task in tasks:
                        try:
                            if task['requirements']:
                                task['requirements'] = json.loads(task['requirements'])
                            if task['result']:
                                task['result'] = json.loads(task['result'])
                            if task['config']:
                                task['config'] = json.loads(task['config'])
                            if task['assignments']:
                                task['assignments'] = json.loads(f"[{task['assignments']}]")
                        except json.JSONDecodeError:
                            # JSON parse hatası durumunda varsayılan değerler kullan
                            task['requirements'] = {}
                            task['result'] = {}
                            task['config'] = {}
                            task['assignments'] = []

                    # Cache'i güncelle
                    self._task_cache['tasks'] = tasks
                    self._last_task_update = current_time

                    return tasks

                finally:
                    if cursor:
                        cursor.close()
                    if conn and conn.is_connected():
                        conn.close()

            except Exception as e:
                logger.error(f"Database error in get_tasks: {e}")
                # Hata durumunda cache'deki son geçerli veriyi dön
                return self._task_cache.get('tasks', [])
    async def get_db_connection(self):
        """Get database connection with retry mechanism"""
        max_retries = 3
        retry_delay = 1

        for attempt in range(max_retries):
            try:
                conn = self.connection_pool.get_connection()
                return conn
            except Error as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Failed to get connection (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

    @contextmanager
    def get_db_cursor(self):
        """Improved database cursor context manager without timeout argument"""
        conn = None
        cursor = None
        try:
            # Get a connection from the pool (without a timeout argument)
            conn = self.connection_pool.get_connection()

            # Create a cursor, with buffered=True if needed
            cursor = conn.cursor(dictionary=True, buffered=True)

            # Yield the cursor for query execution
            yield cursor

            # Commit the transaction if everything was successful
            conn.commit()

        except Error as e:
            # Rollback in case of error
            if conn:
                conn.rollback()
            raise  # Re-raise the exception to handle it outside

        finally:
            # Ensure the cursor is closed
            if cursor:
                cursor.close()

            # Ensure the connection is returned to the pool
            if conn and conn.is_connected():
                conn.close()
    async def initialize(self):
        """Initialize async components"""
        logger.info("Restoring training coordinators")
        await self.restore_training_coordinators()
        logger.info("Initialization complete")

    def cleanup_old_tasks(self):
        """Cleanup old tasks periodically"""
        try:
            conn = None
            cursor = None
            try:
                conn = self.connection_pool.get_connection()
                cursor = conn.cursor()

                # Delete old completed/failed tasks
                cursor.execute("""
                    DELETE FROM tasks 
                    WHERE status IN ('completed', 'failed') 
                    AND updated_at < DATE_SUB(NOW(), INTERVAL 1 DAY)
                    LIMIT 1000
                """)

                # Reset hanging tasks
                cursor.execute("""
                    UPDATE tasks 
                    SET status = 'pending' 
                    WHERE status = 'running' 
                    AND updated_at < DATE_SUB(NOW(), INTERVAL 1 HOUR)
                """)

                conn.commit()

            finally:
                if cursor:
                    cursor.close()
                if conn and conn.is_connected():
                    conn.close()

        except Error as e:
            logger.error(f"Error in cleanup_old_tasks: {e}")
    async def restore_training_coordinators(self):
        try:
            conn = connection_pool.get_connection()
            cursor = conn.cursor(dictionary=True)

            # Get all running distributed tasks
            cursor.execute("""
                SELECT id, config 
                FROM tasks 
                WHERE status IN ('pending', 'running') 
                AND JSON_EXTRACT(config, '$.distributed') = true
            """)

            tasks = cursor.fetchall()
            for task in tasks:
                try:
                    config = json.loads(task['config'])
                    if config.get('distributed') and config.get('n_clients', 0) > 1:
                        logger.info(f"Restoring coordinator for task {task['id']}")
                        coordinator = TrainingCoordinator(
                            n_clients=config['n_clients'],
                            aggregation_frequency=config.get('aggregation_frequency', 5),
                            warmup_epochs=config.get('warmup_epochs', 2)
                        )
                        self.training_coordinators[task['id']] = coordinator
                        logger.info(f"Restored coordinator for task {task['id']}")
                except Exception as e:
                    logger.error(f"Failed to restore coordinator for task {task['id']}: {e}")

        except Exception as e:
            logger.error(f"Failed to restore training coordinators: {e}")
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()

    def setup_admin_panel(self):
        # CORS middleware'i ekle
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Static dosyaların yolu
        self.static_dir = os.path.join(os.path.dirname(__file__), self.app_config['paths']['static_dir'])
        if not os.path.exists(self.static_dir):
            os.makedirs(self.static_dir)

        # Admin HTML'ini oluştur
        admin_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Farm Admin Panel</title>
            <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
            <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
            <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        </head>
        <body>
            <div id="root"></div>
            <script type="text/babel" data-type="module" data-presets="react" src="/static/admin.js"></script>
        </body>
        </html>
        """

        # Admin HTML'ini dosyaya kaydet
        admin_html_path = os.path.join(self.static_dir, "admin.html")
        with open(admin_html_path, "w", encoding="utf-8") as f:
            f.write(admin_html)

        # Static dosyaları serve et - Düzeltilen kısım
        self.app.mount("/static", StaticFiles(directory=self.static_dir), name="static")

        @self.app.get("/", response_class=RedirectResponse)
        async def root():
            return RedirectResponse(url="/admin")

        @self.app.get("/admin", response_class=HTMLResponse)
        async def admin_panel():
            try:
                with open(os.path.join(self.static_dir, "admin.html"), "r", encoding="utf-8") as f:
                    content = f.read()
                    return HTMLResponse(
                        content=content,
                        headers={
                            "Content-Security-Policy": """
                                default-src 'self';
                                script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com;
                                style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net;
                                img-src 'self' data:;
                                connect-src 'self' http://localhost:8000 ws://localhost:8765;
                                worker-src 'self' blob:;
                            """.replace('\n', ' ').strip()
                        }
                    )
            except Exception as e:
                logger.error(f"Error serving admin panel: {e}")
                return HTMLResponse(content="Error loading admin panel", status_code=500)
    def setup_api(self):
        async def get_cached_clients():
            """Get clients with caching"""
            async with self._clients_lock:
                current_time = time.time()

                if (current_time - self._last_clients_update < self._clients_cache_ttl and
                        self._clients_cache):
                    return list(self._clients_cache.values())

                try:
                    async with self._db_semaphore:
                        with self.get_db_cursor() as cursor:
                            cursor.execute("""
                                       SELECT * FROM clients 
                                       WHERE last_seen > DATE_SUB(NOW(), INTERVAL 5 MINUTE)
                                   """)
                            clients = cursor.fetchall()

                            # Parse JSON fields efficiently
                            parsed_clients = []
                            for client in clients:
                                try:
                                    if client['cpu_info']:
                                        client['cpu_info'] = json.loads(client['cpu_info'])
                                    if client['ram_info']:
                                        client['ram_info'] = json.loads(client['ram_info'])
                                    if client['gpu_info']:
                                        client['gpu_info'] = json.loads(client['gpu_info'])
                                    if client['session_info']:
                                        client['session_info'] = json.loads(client['session_info'])
                                    parsed_clients.append(client)
                                except json.JSONDecodeError:
                                    continue

                            self._clients_cache = {c['id']: c for c in parsed_clients}
                            self._last_clients_update = current_time

                            return parsed_clients

                except Exception as e:
                    logger.error(f"Database error in get_clients: {e}")
                    if self._clients_cache:  # Error durumunda cache'den dön
                        return list(self._clients_cache.values())
                    return []

        @self.app.get("/api/tasks")

        async def get_tasks():
            try:
                tasks = await self.get_tasks_with_cache()
                return tasks
            except Exception as e:
                logger.error(f"Error in get_tasks: {e}")
                raise HTTPException(status_code=500, detail=str(e))



        @self.app.post("/api/tasks")
        async def create_task(task: TaskCreate):
            conn = None
            try:
                conn = connection_pool.get_connection()
                cursor = conn.cursor(dictionary=True)

                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"Creating new task: {task.model_dump()}")

                # Task verilerini hazırla
                task_data = task.model_dump()
                task_data['created_at'] = now
                task_data['updated_at'] = now

                # RETURNING olmadan insert yap
                cursor.execute("""
                    INSERT INTO tasks 
                    (type, requirements, status, created_at, updated_at, priority, config)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    task.type,
                    json.dumps(task.requirements),
                    'pending',
                    now,
                    now,
                    task.priority,
                    json.dumps(task_data)
                ))

                task_id = cursor.lastrowid
                conn.commit()

                # Training coordinator'ı başlat
                if task.distributed and task.n_clients and task.n_clients > 1:
                    logger.info(f"Creating coordinator for distributed task {task_id}")
                    coordinator = TrainingCoordinator(
                        n_clients=task.n_clients,
                        aggregation_frequency=task.aggregation_frequency or 5,
                        warmup_epochs=task.warmup_epochs or 2
                    )
                    self.training_coordinators[task_id] = coordinator
                    logger.info(f"Created coordinator for task {task_id} with {task.n_clients} clients")

                    # Update task config with coordinator info
                    task_data['coordinator_port'] = coordinator.coordinator_port
                    cursor.execute("""
                        UPDATE tasks 
                        SET config = %s 
                        WHERE id = %s
                    """, (json.dumps(task_data), task_id))
                    conn.commit()

                # Ayrı bir sorgu ile yeni task'ı getir
                cursor.execute("SELECT * FROM tasks WHERE id = %s", (task_id,))
                created_task = cursor.fetchone()
                cursor.close()  # Cursor'ı kapat

                if created_task:
                    if created_task['requirements']:
                        created_task['requirements'] = json.loads(created_task['requirements'])
                    if created_task['config']:
                        created_task['config'] = json.loads(created_task['config'])
                    if created_task.get('result'):
                        created_task['result'] = json.loads(created_task['result'])

                logger.info(f"Created task {task_id} successfully")
                return created_task

            except Error as e:
                logger.error(f"Database error in create_task: {e}")
                if conn:
                    try:
                        conn.rollback()
                    except Error:
                        pass
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                if conn:
                    if conn.is_connected():
                        conn.close()

        @self.app.delete("/api/tasks/{task_id}")
        async def delete_task(task_id: int):
            try:
                conn = connection_pool.get_connection()
                cursor = conn.cursor()

                # Önce task assignments'ları sil
                cursor.execute("DELETE FROM task_assignments WHERE task_id = %s", (task_id,))

                # Sonra task'ı sil
                cursor.execute("DELETE FROM tasks WHERE id = %s", (task_id,))

                conn.commit()
                return {"status": "success"}

            except Error as e:
                logger.error(f"Database error in delete_task: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                if 'conn' in locals() and conn.is_connected():
                    cursor.close()
                    conn.close()

        @self.app.put("/api/tasks/{task_id}/status")
        async def update_task_status(task_id: int, status_update: dict):
            try:
                conn = connection_pool.get_connection()
                cursor = conn.cursor()

                cursor.execute("""
                    UPDATE tasks 
                    SET status = %s, updated_at = %s 
                    WHERE id = %s
                """, (status_update['status'], datetime.now(), task_id))

                conn.commit()
                return {"status": "success"}

            except Error as e:
                logger.error(f"Database error in update_task_status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            finally:
                if 'conn' in locals() and conn.is_connected():
                    cursor.close()
                    conn.close()

        @self.app.get("/api/clients")
        async def get_clients():
            try:
                clients = await get_cached_clients()
                return clients
            except Exception as e:
                logger.error(f"Error in get_clients API: {e}")
                raise HTTPException(status_code=500, detail=str(e))



        @self.app.middleware("http")
        async def add_api_timeout(request: Request, call_next):
            try:
                return await asyncio.wait_for(
                    call_next(request),
                    timeout=10.0  # 10 saniye timeout
                )
            except asyncio.TimeoutError:
                return JSONResponse(
                    status_code=504,
                    content={"detail": "Request timeout"}
                )

        # CORS ayarlarını güncelle
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    async def update_task_assignment(self, task_id: int, client_id: str, status: str):
        """Update task assignment status with improved task completion handling"""
        try:
            conn = connection_pool.get_connection()
            cursor = conn.cursor()
            now = datetime.now()

            # Get task info
            cursor.execute("""
                SELECT t.status as task_status, t.config 
                FROM tasks t 
                WHERE t.id = %s
            """, (task_id,))

            task_info = cursor.fetchone()
            if not task_info:
                logger.info(f"Task {task_id} not found")
                return

            config = json.loads(task_info[1]) if isinstance(task_info[1], str) else task_info[1]
            is_distributed = config.get('distributed', False)
            required_clients = config.get('n_clients', 1) if is_distributed else 1

            # Update or insert task assignment
            cursor.execute("""
                SELECT COUNT(*) FROM task_assignments 
                WHERE task_id = %s AND client_id = %s
            """, (task_id, client_id))

            if cursor.fetchone()[0] == 0:
                # Insert new assignment
                cursor.execute("""
                    INSERT INTO task_assignments 
                    (task_id, client_id, assigned_at, status, rank) 
                    VALUES (%s, %s, %s, %s, %s)
                """, (task_id, client_id, now, status, config.get('rank', 0)))
            else:
                # Update existing assignment
                cursor.execute("""
                    UPDATE task_assignments 
                    SET status = %s 
                    WHERE task_id = %s AND client_id = %s
                """, (status, task_id, client_id))

            # Get all assignments status
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_assignments,
                    SUM(CASE WHEN ta.status = 'completed' THEN 1 ELSE 0 END) as completed_assignments,
                    SUM(CASE WHEN ta.status IN ('assigned', 'training') THEN 1 ELSE 0 END) as active_assignments
                FROM task_assignments ta
                WHERE ta.task_id = %s
            """, (task_id,))

            stats = cursor.fetchone()
            total_assignments = stats[0]
            completed_assignments = stats[1] or 0
            active_assignments = stats[2] or 0

            logger.info(f"Task {task_id} assignments - Total: {total_assignments}, "
                        f"Completed: {completed_assignments}, Active: {active_assignments}, "
                        f"Required: {required_clients}")

            # Update task status based on assignments
            if completed_assignments >= required_clients:
                cursor.execute("""
                    UPDATE tasks t 
                    SET t.status = 'completed', 
                        t.updated_at = %s
                    WHERE t.id = %s
                """, (now, task_id))
                logger.info(f"All required shards completed for task {task_id}, marked as completed")
            elif status == 'failed':
                cursor.execute("""
                    UPDATE tasks t 
                    SET t.status = 'failed', 
                        t.updated_at = %s
                    WHERE t.id = %s
                """, (now, task_id))
                logger.info(f"Task {task_id} marked as failed")
            elif active_assignments > 0:
                cursor.execute("""
                    UPDATE tasks t 
                    SET t.status = 'running', 
                        t.updated_at = %s
                    WHERE t.id = %s AND t.status != 'completed'
                """, (now, task_id))
                logger.info(f"Task {task_id} is running with {active_assignments} active assignments")
            else:
                cursor.execute("""
                    UPDATE tasks t 
                    SET t.status = 'pending', 
                        t.updated_at = %s
                    WHERE t.id = %s AND t.status != 'completed'
                """, (now, task_id))
                logger.info(f"No active assignments for task {task_id}, marked as pending")

            conn.commit()

        except Error as e:
            logger.error(f"Failed to update task assignment: {e}")
            if conn:
                try:
                    conn.rollback()
                except Error:
                    pass
            raise
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()

    async def register_client(self, websocket, client_info):
        """Register client with validation"""
        try:
            client_id = client_info.get('id')

            # Validate client_id
            if not client_id or client_id == 'None':
                logger.error("Invalid client ID")
                await websocket.close(1008, "Invalid client ID")
                return False

            # Check if already registered
            if client_id in self.clients:
                logger.warning(f"Client {client_id} already registered, cleaning up old connection")
                await self.handle_client_disconnect(client_id)

            conn = connection_pool.get_connection()
            cursor = conn.cursor()

            try:
                now = datetime.now()
                session_info = client_info.get('session', {
                    'created_at': now.isoformat(),
                    'total_tasks_completed': 0,
                    'performance_metrics': {}
                })

                sql = '''INSERT INTO clients 
                        (id, cpu_info, ram_info, gpu_info, session_info, last_seen, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE 
                        cpu_info=%s, ram_info=%s, gpu_info=%s, session_info=%s, 
                        last_seen=%s, status=%s'''

                values = (
                    client_id,
                    json.dumps(client_info['cpu']),
                    json.dumps(client_info['ram']),
                    json.dumps(client_info.get('gpu', {})),
                    json.dumps(session_info),
                    now,
                    'active',
                    json.dumps(client_info['cpu']),
                    json.dumps(client_info['ram']),
                    json.dumps(client_info.get('gpu', {})),
                    json.dumps(session_info),
                    now,
                    'active'
                )

                cursor.execute(sql, values)
                conn.commit()

                self.clients[client_id] = {
                    'websocket': websocket,
                    'info': client_info,
                    'last_seen': now,
                    'task_count': 0  # Track assigned tasks
                }

                logger.info(f"Client registered: {client_id}")
                await self.log_task_event(None, client_id, 'info',
                                          f"Client registered with session info")
                return True

            except Exception as e:
                logger.error(f"Failed to register client: {e}")
                await websocket.close(1011, "Registration failed")
                return False

            finally:
                if 'conn' in locals() and conn.is_connected():
                    cursor.close()
                    conn.close()

        except Exception as e:
            logger.error(f"Registration error: {e}")
            await websocket.close(1011, "Registration failed")
            return False

    def setup_database(self):

        try:
            conn = connection_pool.get_connection()
            cursor = conn.cursor()

            # 1. Önce clients tablosu (bağımsız tablo)
            cursor.execute('''CREATE TABLE IF NOT EXISTS clients
                           (id VARCHAR(255) PRIMARY KEY,
                           cpu_info JSON NOT NULL,
                           ram_info JSON NOT NULL,
                           gpu_info JSON,
                           session_info JSON,
                           last_seen DATETIME NOT NULL,
                           status ENUM('active', 'inactive') NOT NULL)''')

            # 2. Sonra tasks tablosu (clients'a bağımlı)
            cursor.execute('''CREATE TABLE IF NOT EXISTS tasks
                           (id INT AUTO_INCREMENT PRIMARY KEY,
                           type VARCHAR(255) NOT NULL,
                           requirements JSON NOT NULL,
                           status ENUM('pending', 'running', 'completed', 'failed', 'error') NOT NULL,
                           client_id VARCHAR(255),
                           created_at DATETIME NOT NULL,
                           updated_at DATETIME NOT NULL,
                           result JSON,
                           error_log TEXT,
                           priority INT DEFAULT 0,
                           config JSON,
                           checkpoint_path VARCHAR(255),
                           FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE SET NULL)''')

            # 2.1 Task assignments tablosu (tasks ve clients'a bağımlı)
            cursor.execute('''CREATE TABLE IF NOT EXISTS task_assignments(
                       id INT AUTO_INCREMENT PRIMARY KEY,
                       task_id INT NOT NULL,
                       client_id VARCHAR(255) NOT NULL,
                       assigned_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                       updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                       status ENUM('assigned', 'training', 'completed', 'failed') NOT NULL,
                       rank INT NOT NULL,
                       checkpoint_path VARCHAR(255),  # Yeni eklenen sütun
                       FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE,
                       FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE,
                       UNIQUE KEY unique_assignment (task_id, rank))''')

            # 3. Model versions tablosu (bağımsız tablo)
            cursor.execute('''CREATE TABLE IF NOT EXISTS model_versions
                           (id INT AUTO_INCREMENT PRIMARY KEY,
                           name VARCHAR(255) NOT NULL,
                           version VARCHAR(50) NOT NULL,
                           url TEXT NOT NULL,
                           config JSON,
                           created_at DATETIME NOT NULL,
                           UNIQUE KEY unique_version (name, version))''')

            # 4. Checkpoints tablosu (tasks'a bağımlı)
            cursor.execute('''CREATE TABLE IF NOT EXISTS checkpoints
                           (id INT AUTO_INCREMENT PRIMARY KEY,
                           task_id INT NOT NULL,
                           path VARCHAR(255) NOT NULL,
                           metrics JSON,
                           created_at DATETIME NOT NULL,
                           FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE)''')

            # 5. Task logs tablosu (tasks ve clients'a bağımlı)
            cursor.execute('''CREATE TABLE IF NOT EXISTS task_logs
                           (id INT AUTO_INCREMENT PRIMARY KEY,
                           task_id INT,
                           client_id VARCHAR(255),
                           log_type ENUM('info', 'warning', 'error') NOT NULL,
                           message TEXT NOT NULL,
                           created_at DATETIME NOT NULL,
                           FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE,
                           FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE)''')

            # 6. Client logs tablosu (clients'a bağımlı)
            cursor.execute('''CREATE TABLE IF NOT EXISTS client_logs
                           (id INT AUTO_INCREMENT PRIMARY KEY,
                           client_id VARCHAR(255),
                           log_type ENUM('info', 'warning', 'error') NOT NULL,
                           message TEXT NOT NULL,
                           created_at DATETIME NOT NULL,
                           FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE)''')

            # 7. Performance metrics tablosu (tasks ve clients'a bağımlı)
            cursor.execute('''CREATE TABLE IF NOT EXISTS performance_metrics
                           (id INT AUTO_INCREMENT PRIMARY KEY,
                           client_id VARCHAR(255) NOT NULL,
                           task_id INT,
                           metric_type VARCHAR(50) NOT NULL,
                           value FLOAT NOT NULL,
                           recorded_at DATETIME NOT NULL,
                           FOREIGN KEY (client_id) REFERENCES clients(id) ON DELETE CASCADE,
                           FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE)''')

            conn.commit()

            cursor.execute(''' update tasks set status='pending' ''');
            cursor.execute(''' delete from task_assignments ''');
            conn.commit()
            logger.info("Database tables created successfully")

        except Error as e:
            logger.critical(f"Database setup failed: {e}")
            raise
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()

    async def log_task_event(self, task_id: Optional[int], client_id: str, log_type: str, message: str):
        try:
            conn = connection_pool.get_connection()
            cursor = conn.cursor()

            if task_id is None:
                # Client log
                cursor.execute("""
                    INSERT INTO client_logs (client_id, log_type, message, created_at)
                    VALUES (%s, %s, %s, %s)
                """, (client_id, log_type, message, datetime.now()))
            else:
                # Task log
                cursor.execute("""
                    INSERT INTO task_logs (task_id, client_id, log_type, message, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (task_id, client_id, log_type, message, datetime.now()))

            conn.commit()

        except Error as e:
            logger.error(f"Failed to log event: {e}")
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()

    async def add_performance_metrics(self, client_id: str, task_id: Optional[int], metrics: Dict):
        try:
            conn = connection_pool.get_connection()
            cursor = conn.cursor()

            now = datetime.now()

            for metric_type, value in metrics.items():
                cursor.execute("""
                    INSERT INTO performance_metrics 
                    (client_id, task_id, metric_type, value, recorded_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (client_id, task_id, metric_type, float(value), now))

            conn.commit()

        except Error as e:
            logger.error(f"Failed to add performance metrics: {e}")
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()
    async def register_client(self, websocket, client_info):
        client_id = client_info['id']
        try:
            conn = connection_pool.get_connection()
            cursor = conn.cursor()

            now = datetime.now()
            sql = '''INSERT INTO clients 
                    (id, cpu_info, ram_info, gpu_info, session_info, last_seen, status)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                    cpu_info=%s, ram_info=%s, gpu_info=%s, session_info=%s, 
                    last_seen=%s, status=%s'''

            session_info = json.dumps(client_info.get('session', {}))

            values = (
                client_id,
                json.dumps(client_info['cpu']),
                json.dumps(client_info['ram']),
                json.dumps(client_info.get('gpu', {})),
                session_info,
                now,
                'active',
                json.dumps(client_info['cpu']),
                json.dumps(client_info['ram']),
                json.dumps(client_info.get('gpu', {})),
                session_info,
                now,
                'active'
            )

            cursor.execute(sql, values)
            conn.commit()

            self.clients[client_id] = {
                'websocket': websocket,
                'info': client_info,
                'last_seen': now
            }

            logger.info(f"Client registered: {client_id} with session info")
            await self.log_task_event(None, client_id, 'info',
                                      f"Client registered with session info: {session_info}")

        except Error as e:
            logger.error(f"Failed to register client: {e}")
            raise
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()

    async def assign_task(self, client_id: str):
        """
        Geliştirilmiş task atama mantığı:
        1. Önce pending normal task'ları kontrol et
        2. Distributed task'ları kontrol et (running olsa bile boş slot varsa)
        3. Assignment sayısı n_clients'ı geçemez
        4. Client düşerse task tekrar atanabilir
        """
        try:
            conn = connection_pool.get_connection()
            cursor = conn.cursor(dictionary=True)

            # İki aşamalı task seçimi:
            # 1. Normal tasklar için
            cursor.execute("""
                SELECT t.*,
                       COUNT(ta.id) as current_assignments,
                       JSON_EXTRACT(t.config, '$.distributed') as is_distributed,
                       JSON_EXTRACT(t.config, '$.n_clients') as required_clients
                FROM tasks t
                LEFT JOIN task_assignments ta ON t.id = ta.task_id
                WHERE t.status IN ('pending', 'running')
                AND (
                    -- Normal task için: pending ve hiç assignment yok
                    (JSON_EXTRACT(t.config, '$.distributed') IS NOT TRUE 
                     AND t.status = 'pending'
                     AND NOT EXISTS (
                        SELECT 1 FROM task_assignments 
                        WHERE task_id = t.id
                     )
                    )
                    OR
                    -- Distributed task için: boş slot varsa al
                    (JSON_EXTRACT(t.config, '$.distributed') = TRUE
                     AND (
                        SELECT COUNT(*) 
                        FROM task_assignments
                        WHERE task_id = t.id 
                        AND status IN ('assigned', 'training', 'completed')
                     ) < JSON_EXTRACT(t.config, '$.n_clients')
                    )
                )
                GROUP BY t.id
                ORDER BY t.priority DESC, t.created_at ASC
                LIMIT 1
            """)

            task = cursor.fetchone()
            if not task:
                return None

            config = json.loads(task['config']) if isinstance(task['config'], str) else task['config']
            is_distributed = config.get('distributed', False)
            n_clients = config.get('n_clients', 1) if is_distributed else 1

            # Müsait rank bul
            cursor.execute("""
                SELECT rank FROM task_assignments
                WHERE task_id = %s
                ORDER BY rank ASC
            """, (task['id'],))

            used_ranks = set(row['rank'] for row in cursor.fetchall())
            next_rank = 0
            while next_rank < n_clients:
                if next_rank not in used_ranks:
                    break
                next_rank += 1

            if next_rank >= n_clients:
                logger.warning(f"No available rank for task {task['id']}")
                return None

            # Assignment oluştur/güncelle
            cursor.execute("""
                INSERT INTO task_assignments (
                    task_id, client_id, status, rank, assigned_at
                )
                VALUES (%s, %s, 'assigned', %s, NOW())
                ON DUPLICATE KEY UPDATE
                    client_id = VALUES(client_id),
                    status = 'assigned',
                    assigned_at = NOW()
            """, (task['id'], client_id, next_rank))

            # Task durumunu güncelle
            cursor.execute("""
                UPDATE tasks 
                SET status = 'running',
                    updated_at = NOW()
                WHERE id = %s
            """, (task['id'],))

            # Distributed task için coordinator ayarları
            if is_distributed:
                coordinator = self.training_coordinators.get(task['id'])
                if not coordinator:
                    coordinator = TrainingCoordinator(
                        n_clients=n_clients,
                        aggregation_frequency=config.get('aggregation_frequency', 5),
                        warmup_epochs=config.get('warmup_epochs', 2)
                    )
                    self.training_coordinators[task['id']] = coordinator

                # Config'i güncelle
                config['rank'] = next_rank
                config['world_size'] = n_clients
                cursor.execute("""
                    UPDATE tasks 
                    SET config = %s 
                    WHERE id = %s
                """, (json.dumps(config), task['id']))

            conn.commit()
            logger.info(f"Assigned task {task['id']} to client {client_id} with rank {next_rank}")

            return task

        except Exception as e:
            logger.error(f"Error in assign_task: {e}")
            if 'conn' in locals():
                conn.rollback()
            return None
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals() and conn.is_connected():
                conn.close()

    async def handle_client(self, websocket, path):
        """Handle client websocket connections and messages with improved error handling"""
        client_id = None
        try:
            # Initial registration with timeout
            try:
                register_data = await asyncio.wait_for(
                    websocket.recv(),
                    timeout=30
                )
                client_info = json.loads(register_data)
                client_id = client_info['id']
                await self.register_client(websocket, client_info)
                logger.info(f"New client connected: {client_id}")
            except asyncio.TimeoutError:
                logger.error("Client registration timeout")
                return
            except json.JSONDecodeError:
                logger.error("Invalid registration data")
                return
            except Exception as e:
                logger.error(f"Registration error: {str(e)}")
                return

            last_ping = time.time()
            ping_interval = 15
            ping_timeout = 10

            while True:
                try:
                    if not websocket.open:
                        raise websockets.exceptions.ConnectionClosed(1006, "Connection lost")

                    current_time = time.time()
                    if current_time - last_ping > ping_interval:
                        try:
                            pong_waiter = await asyncio.wait_for(
                                websocket.ping(),
                                timeout=ping_timeout
                            )
                            await asyncio.wait_for(pong_waiter, timeout=ping_timeout)
                            last_ping = current_time

                            # Update client status
                            if client_id in self.clients:
                                self.clients[client_id]['last_seen'] = datetime.now()
                                await self.update_client_status(client_id, 'active')
                        except asyncio.TimeoutError:
                            logger.warning(f"Ping timeout for client {client_id}")
                            raise websockets.exceptions.ConnectionClosed(1002, "Ping timeout")

                    # Message handling with timeout
                    try:
                        message = await asyncio.wait_for(
                            websocket.recv(),
                            timeout=30
                        )

                        if not message:
                            continue

                        data = json.loads(message)
                        msg_type = data.get('type', '')
                        logger.info(f"Received message from client {client_id}: {msg_type}")

                        # Handle different message types
                        if msg_type == 'heartbeat':
                            if client_id in self.clients:
                                self.clients[client_id]['last_seen'] = datetime.now()
                                await websocket.send(json.dumps({'type': 'heartbeat_ack'}))
                                continue

                        elif msg_type == 'init_distributed':
                            try:
                                task_id = int(data['task_id'])
                                coordinator = self.training_coordinators.get(task_id)

                                if not coordinator:
                                    # Create new coordinator if needed
                                    config = await self.get_task_config(task_id)
                                    if config and config.get('distributed'):
                                        coordinator = TrainingCoordinator(
                                            n_clients=config.get('n_clients', 2),
                                            aggregation_frequency=config.get('aggregation_frequency', 5),
                                            warmup_epochs=config.get('warmup_epochs', 2)
                                        )
                                        self.training_coordinators[task_id] = coordinator

                                if coordinator:
                                    init_data = await coordinator.register_for_initialization(
                                        client_id=data['client_id'],
                                        rank=data.get('rank', 0)
                                    )
                                    await websocket.send(json.dumps(init_data))
                                else:
                                    await websocket.send(json.dumps({
                                        'status': 'error',
                                        'error': 'No coordinator available'
                                    }))

                            except Exception as e:
                                logger.error(f"Error in distributed initialization: {e}")
                                await websocket.send(json.dumps({
                                    'status': 'error',
                                    'error': str(e)
                                }))

                        elif msg_type == 'task_request':
                            task = await self.assign_task(data['client_id'])
                            response = {
                                'type': 'task' if task else 'no_task',
                                'data': task
                            } if task else {'type': 'no_task'}
                            await websocket.send(json.dumps(response, default=datetime_handler))

                        elif msg_type in ['task_progress', 'task_result']:
                            await getattr(self, f'handle_{msg_type}')(data)
                            await websocket.send(json.dumps({
                                'type': f'{msg_type}_ack',
                                'task_id': data.get('task_id'),
                                'received': True
                            }))

                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        raise
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error from client {client_id}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")
                        continue

                except websockets.exceptions.ConnectionClosed:
                    logger.warning(f"Client connection closed: {client_id}")
                    break
                except Exception as e:
                    logger.error(f"Connection error: {e}")
                    break

        except Exception as e:
            logger.error(f"Client handling error: {str(e)}\n{traceback.format_exc()}")
        finally:
            if client_id:
                await self.handle_client_disconnect(client_id)
                logger.info(f"Cleaned up after client disconnect: {client_id}")

    async def handle_task_result(self, data):
        try:
            conn = connection_pool.get_connection()
            cursor = conn.cursor(dictionary=True)

            task_id = data['task_id']
            client_id = data['client_id']
            status = data['status']
            checkpoint_path = data.get('checkpoint_path')

            # Assignment güncelle
            cursor.execute("""
                UPDATE task_assignments
                SET status = %s,
                    updated_at = NOW(),
                    checkpoint_path = %s
                WHERE task_id = %s 
                AND client_id = %s
            """, (status, checkpoint_path, task_id, client_id))

            # Task ve assignment durumlarını kontrol et
            cursor.execute("""
                SELECT 
                    t.id,
                    t.status as task_status,
                    t.config,
                    COUNT(ta.id) as total_assignments,
                    SUM(CASE WHEN ta.status = 'completed' THEN 1 ELSE 0 END) as completed_assignments
                FROM tasks t
                LEFT JOIN task_assignments ta ON t.id = ta.task_id
                WHERE t.id = %s
                GROUP BY t.id
            """, (task_id,))

            task_info = cursor.fetchone()
            if not task_info:
                logger.error(f"Task {task_id} not found")
                conn.rollback()
                return

            config = json.loads(task_info['config']) if isinstance(task_info['config'], str) else task_info['config']
            is_distributed = config.get('distributed', False)
            required_clients = config.get('n_clients', 1) if is_distributed else 1
            completed = task_info['completed_assignments'] or 0
            total = task_info['total_assignments'] or 0

            # Task durumunu güncelle
            if completed >= required_clients:
                # Tüm shardlar tamamlandı
                cursor.execute("""
                    UPDATE tasks t
                    SET status = 'completed',
                        updated_at = NOW(),
                        result = JSON_SET(
                            COALESCE(result, '{}'),
                            '$.completion_time', %s,
                            '$.completed_shards', %s,
                            '$.total_shards', %s,
                            '$.status', 'completed'
                        )
                    WHERE id = %s
                """, (datetime.now().isoformat(), completed, required_clients, task_id))

                if is_distributed:
                    # Tüm model dosyalarını birleştir
                    cursor.execute("""
                        SELECT checkpoint_path 
                        FROM task_assignments
                        WHERE task_id = %s 
                        AND status = 'completed'
                    """, (task_id,))

                    model_files = cursor.fetchall()
                    if len(model_files) >= required_clients:
                        aggregated_path = f"aggregated_models/task_{task_id}_final_{datetime.now():%Y%m%d_%H%M%S}.pt"
                        all_models = []

                        for model_info in model_files:
                            if model_info['checkpoint_path'] and os.path.exists(model_info['checkpoint_path']):
                                try:
                                    model_data = torch.load(model_info['checkpoint_path'])
                                    if isinstance(model_data, dict) and 'model_state' in model_data:
                                        all_models.append(model_data)
                                        os.remove(model_info['checkpoint_path'])  # Shard modelini sil
                                except Exception as e:
                                    logger.error(f"Error loading model: {e}")

                        if all_models:
                            try:
                                # Modelleri birleştir
                                aggregated_state = {}
                                first_model = all_models[0]['model_state']

                                for key in first_model.keys():
                                    params = []
                                    for model in all_models:
                                        if key in model['model_state']:
                                            params.append(model['model_state'][key])
                                    if params:
                                        aggregated_state[key] = torch.mean(torch.stack(params), dim=0)

                                # Birleştirilmiş modeli kaydet
                                torch.save({
                                    'model_state': aggregated_state,
                                    'config': config,
                                    'completion_time': datetime.now().isoformat(),
                                    'total_shards': required_clients
                                }, aggregated_path)

                                # Checkpoints tablosuna kaydet
                                cursor.execute("""
                                    INSERT INTO checkpoints 
                                    (task_id, path, metrics, created_at)
                                    VALUES (%s, %s, %s, NOW())
                                """, (
                                    task_id,
                                    aggregated_path,
                                    json.dumps({'status': 'completed', 'total_shards': required_clients})
                                ))

                            except Exception as e:
                                logger.error(f"Error aggregating models: {e}")

                # Coordinator'ı temizle
                if task_id in self.training_coordinators:
                    del self.training_coordinators[task_id]

            else:
                # Task devam ediyor
                cursor.execute("""
                    UPDATE tasks
                    SET status = 'running',
                        updated_at = NOW(),
                        result = JSON_SET(
                            COALESCE(result, '{}'),
                            '$.status', 'running',
                            '$.completed_shards', %s,
                            '$.total_shards', %s
                        )
                    WHERE id = %s
                """, (completed, required_clients, task_id))

            conn.commit()
            logger.info(f"Task {task_id} - {completed}/{required_clients} shards completed")

            # Yeni task ata
            if status == 'completed' and (not is_distributed or completed < required_clients):
                await self.assign_task(client_id)

        except Exception as e:
            logger.error(f"Error handling task result: {e}")
            if 'conn' in locals():
                try:
                    conn.rollback()
                except:
                    pass
        finally:
            if 'cursor' in locals(): cursor.close()
            if 'conn' in locals() and conn.is_connected(): conn.close()
    async def get_task_config(self, task_id: int) -> Optional[Dict]:
        """Get task configuration from database"""
        try:
            with self.get_db_cursor() as cursor:
                cursor.execute("SELECT config FROM tasks WHERE id = %s", (task_id,))
                result = cursor.fetchone()
                if result and result['config']:
                    return json.loads(result['config'])
        except Exception as e:
            logger.error(f"Error getting task config: {e}")
        return None
    async def update_client_status(self, client_id: str, status: str):
        """Update client status in database"""
        try:
            conn = connection_pool.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE clients 
                SET status = %s, last_seen = %s 
                WHERE id = %s
            """, (status, datetime.now(), client_id))

            conn.commit()

        except Error as e:
            logger.error(f"Failed to update client status: {e}")
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()

    async def handle_client_disconnect(self, client_id: str):
        """Client bağlantısı koptuğunda yapılacak işlemler"""
        try:
            conn = connection_pool.get_connection()
            cursor = conn.cursor()

            # Client'ın aktif assignment'larını bul
            cursor.execute("""
                SELECT task_id, rank
                FROM task_assignments
                WHERE client_id = %s 
                AND status IN ('assigned', 'training')
            """, (client_id,))

            active_assignments = cursor.fetchall()

            # Her assignment için:
            for (task_id, rank) in active_assignments:  # Tuple unpacking kullanıyoruz
                # Assignment'ı failed olarak işaretle
                cursor.execute("""
                    UPDATE task_assignments
                    SET status = 'failed',
                        updated_at = NOW() 
                    WHERE task_id = %s AND rank = %s
                """, (task_id, rank))

                # Task'ın tüm assignment'larını kontrol et
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed
                    FROM task_assignments 
                    WHERE task_id = %s
                """, (task_id,))

                stats = cursor.fetchone()
                total = stats[0]
                failed = stats[1] or 0
                completed = stats[2] or 0

                # Tüm assignment'lar failed ise task'ı pending yap
                if failed == total:
                    cursor.execute("""
                        UPDATE tasks
                        SET status = 'pending',
                            updated_at = NOW()
                        WHERE id = %s
                    """, (task_id,))

            # Client'ı inactive olarak işaretle
            cursor.execute("""
                UPDATE clients
                SET status = 'inactive',
                    last_seen = NOW()
                WHERE id = %s
            """, (client_id,))

            conn.commit()

            # Client'ı in-memory listeden çıkar
            if client_id in self.clients:
                del self.clients[client_id]

            logger.info(f"Client disconnected and cleaned up: {client_id}")

        except Exception as e:
            logger.error(f"Error in handle_client_disconnect: {e}")
            if 'conn' in locals() and conn:
                conn.rollback()
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()
            if 'conn' in locals() and conn and conn.is_connected():
                conn.close()

    async def handle_task_progress(self, data):
        try:
            task_id = data['task_id']
            client_id = data['client_id']
            progress = data.get('progress', 0)
            metrics = data.get('metrics', {})

            await self.log_task_event(
                task_id,
                client_id,
                'info',
                f"Progress update: {progress}%"
            )

            # İlerleme ve metrikleri veritabanına kaydet
            conn = connection_pool.get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE tasks 
                SET result = JSON_MERGE(
                    JSON_SET(
                        COALESCE(result, '{}'),
                        '$.progress', %s
                    ),
                    JSON_SET(
                        COALESCE(result, '{}'),
                        '$.metrics', %s
                    )
                )
                WHERE id = %s
            """, (progress, json.dumps(metrics), task_id))

            conn.commit()

        except Exception as e:
            logger.error(f"Failed to handle task progress: {e}")
        finally:
            if 'conn' in locals() and conn.is_connected():
                cursor.close()
                conn.close()

    async def monitor_clients(self):
        """Monitor client connections"""
        try:
            while True:
                try:
                    now = datetime.now()
                    inactive_threshold = 60  # seconds

                    for client_id, client_data in list(self.clients.items()):
                        last_seen = client_data['last_seen']
                        if (now - last_seen).total_seconds() > inactive_threshold:
                            await self.handle_client_disconnect(client_id)

                    await asyncio.sleep(30)  # Check every 30 seconds

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error in client monitor: {e}")
                    await asyncio.sleep(5)

        except asyncio.CancelledError:
            logger.info("Monitor task cancelled")
            raise

    async def start(self):
        """Start server with improved configuration"""
        try:
            # Initialize server components
            await self.initialize()

            # Port kullanımda mı kontrol et
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            port_available = sock.connect_ex(('localhost', 8000)) != 0
            sock.close()

            if not port_available:
                new_port = 8001  # Alternatif port
                logger.warning(f"Port 8000 in use, trying port {new_port}")
            else:
                new_port = 8000

            # Configure uvicorn for better performance
            config = uvicorn.Config(
                app=self.app,
                host="0.0.0.0",
                port=new_port,  # Dinamik port kullan
                workers=4,
                loop="uvloop" if sys.platform != "win32" else "asyncio",
                limit_concurrency=100,
                timeout_keep_alive=30,
                access_log=False,
                log_level="warning"
            )
            api_server = uvicorn.Server(config)

            # Start API server in separate thread
            api_thread = threading.Thread(
                target=api_server.run,
                daemon=True
            )
            api_thread.start()

            # WebSocket server
            self.server = await websockets.serve(
                self.handle_client,
                "0.0.0.0",
                8765,
                ping_interval=30,
                ping_timeout=120,
                max_size=None,
                compression=None
            )

            # Background tasks
            background_tasks = [
                asyncio.create_task(self.monitor_clients()),
                asyncio.create_task(self._periodic_cleanup())
            ]

            # Wait for any task to complete
            try:
                done, pending = await asyncio.wait(
                    background_tasks,
                    return_when=asyncio.FIRST_COMPLETED
                )
                for task in pending:
                    task.cancel()
            except asyncio.CancelledError:
                pass

        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            if hasattr(self, 'server'):
                self.server.close()
            await self.cleanup()

    async def _periodic_cleanup(self):
        """Run cleanup tasks periodically"""
        try:
            while True:
                try:
                    # Cleanup old tasks
                    self.cleanup_old_tasks()

                    # Cleanup resources
                    await self.cleanup_resources()

                    # Wait for next cleanup
                    await asyncio.sleep(3600)  # Run every hour

                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error in periodic cleanup: {e}")
                    await asyncio.sleep(60)  # Retry after 1 minute

        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            raise

    async def cleanup_resources(self):
        """Cleanup server resources"""
        try:
            # Cleanup inactive clients
            now = datetime.now()
            for client_id, client_data in list(self.clients.items()):
                if (now - client_data['last_seen']).total_seconds() > 300:  # 5 minutes
                    await self.handle_client_disconnect(client_id)

            conn = None
            cursor = None
            try:
                # Get connection from pool
                conn = self.connection_pool.get_connection()
                cursor = conn.cursor()

                # Reset hanging tasks
                cursor.execute("""
                    UPDATE tasks 
                    SET status = 'pending' 
                    WHERE status = 'running' 
                    AND updated_at < DATE_SUB(NOW(), INTERVAL 1 HOUR)
                """)

                conn.commit()

                # Cleanup old records
                self.cleanup_old_tasks()

            finally:
                if cursor:
                    cursor.close()
                if conn and conn.is_connected():
                    conn.close()

        except Exception as e:
            logger.error(f"Error in cleanup_resources: {e}")
            if 'conn' in locals() and conn and conn.is_connected():
                conn.close()

def main():
    """Main entry point with improved error handling and shutdown"""
    server = None
    loop = None

    try:
        # Windows için özel event loop politikası ayarla
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        server = AIFarmServer()

        # API server için config
        api_config = {
            "app": server.app,
            "host": "0.0.0.0",
            "port": 8000,
            "log_level": "info",
            "loop": "auto",
            "workers": 1,
            "reload": False,  # Reload'u devre dışı bırak
            "timeout_keep_alive": 30,  # Keep-alive timeout'unu artır
            "access_log": False  # Access log'u devre dışı bırak
        }

        def run_api_server():
            import uvicorn
            uvicorn.run(**api_config)

        # API server'ı ayrı bir thread'de başlat
        api_thread = threading.Thread(target=run_api_server, daemon=True)
        api_thread.start()

        # WebSocket server için ana event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Shutdown handler
        async def shutdown_handler():
            if server:
                await server.shutdown()
            loop.stop()

        def signal_handler():
            logger.info("Received shutdown signal")
            if not loop.is_running():
                return
            loop.create_task(shutdown_handler())

        # Signal handler'ları ayarla
        if sys.platform != 'win32':
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, signal_handler)
        else:
            signal.signal(signal.SIGINT, lambda s, f: loop.call_soon_threadsafe(signal_handler))
            signal.signal(signal.SIGTERM, lambda s, f: loop.call_soon_threadsafe(signal_handler))

        try:
            # WebSocket server'ı başlat
            loop.run_until_complete(server.start())
            loop.run_forever()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Server error: {e}\n{traceback.format_exc()}")
        finally:
            # Cleanup işlemlerini yürüt
            if server:
                loop.run_until_complete(server.shutdown())

            # Event loop'u kapat
            if loop and not loop.is_closed():
                loop.close()

    except Exception as e:
        logger.critical(f"Fatal server error: {e}\n{traceback.format_exc()}")
        sys.exit(1)



if __name__ == "__main__":
    main()