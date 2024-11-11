from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import asyncio
import logging
import json
import time

logger = logging.getLogger("TrainingCoordinator")


@dataclass
class ModelUpdate:
    client_id: str
    epoch: int
    parameters: Dict[str, torch.Tensor]
    metrics: Dict
    timestamp: float


class FederatedAveraging:
    @staticmethod
    def aggregate_parameters(updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Farklı clientlerden gelen model parametrelerini ağırlıklı ortalama ile birleştir"""
        aggregated_params = {}
        total_samples = sum(update.metrics.get('samples_seen', 1) for update in updates)

        for param_name in updates[0].parameters.keys():
            weighted_sum = torch.zeros_like(updates[0].parameters[param_name])

            for update in updates:
                weight = update.metrics.get('samples_seen', 1) / total_samples
                weighted_sum += update.parameters[param_name] * weight

            aggregated_params[param_name] = weighted_sum

        return aggregated_params


class TrainingCoordinator:
    def __init__(self, n_clients: int, aggregation_frequency: int = 5, warmup_epochs: int = 2):
        self.n_clients = n_clients
        self.aggregation_frequency = aggregation_frequency
        self.warmup_epochs = warmup_epochs
        self.ready_clients = set()

        self.client_states = {}
        self.coordinator_port = 29500

        # Shard yönetimi için data structure'ı düzelt
        self.data_shards = []  # Liste olarak başlat
        self.initialize_shards()  # Shardları başlangıçta oluştur

        self.global_model_version = 0
        self.client_updates = {}
        self.global_parameters = None
        self.best_metrics = {'loss': float('inf')}

        self.lock = asyncio.Lock()
        self.aggregator = FederatedAveraging()
        logger.info(f"Training coordinator initialized for {n_clients} clients")

    def initialize_shards(self):
        """Initialize shards at startup"""
        shard_size = 100.0 / self.n_clients
        for i in range(self.n_clients):
            self.data_shards.append({
                'id': i,
                'status': 'pending',
                'assigned_to': None,
                'start_time': None,
                'progress': 0,
                'start_point': i * shard_size,
                'end_point': (i + 1) * shard_size
            })

    def get_next_shard(self, client_id: str) -> Optional[dict]:
        """Client için uygun bir iş parçası bul"""
        # Pending shardları kontrol et
        for shard in self.data_shards:
            if shard['status'] == 'pending':
                shard['status'] = 'in_progress'
                shard['assigned_to'] = client_id
                shard['start_time'] = time.time()
                return shard

        # En yavaş ilerleyen shardı bul
        slowest_shard = None
        min_progress = float('inf')
        for shard in self.data_shards:
            if shard['status'] == 'in_progress' and shard['progress'] < min_progress:
                min_progress = shard['progress']
                slowest_shard = shard

        if slowest_shard and slowest_shard['progress'] < 50:
            # İşi böl
            current_endpoint = slowest_shard['end_point']
            split_point = slowest_shard['start_point'] + (slowest_shard['progress'] / 100.0) * (
                        current_endpoint - slowest_shard['start_point'])

            new_shard = {
                'id': len(self.data_shards),
                'status': 'in_progress',
                'assigned_to': client_id,
                'start_time': time.time(),
                'progress': 0,
                'parent_shard': slowest_shard['id'],
                'start_point': split_point,
                'end_point': current_endpoint
            }

            # Orijinal shardı güncelle
            slowest_shard['end_point'] = split_point

            self.data_shards.append(new_shard)
            return new_shard

        return None

    async def register_client(self, client_id: str):
        """Register a new client"""
        logger.info(f"Registering client {client_id}")
        self.client_states[client_id] = {
            'status': 'registered',
            'start_time': datetime.now(),
            'last_update': datetime.now(),
            'current_epoch': 0,
            'shard': None,
            'metrics': {},
            'progress': 0
        }
        logger.info(f"Client {client_id} registered successfully")

    async def register_for_initialization(self, client_id: str, rank: int):
        """Client'ı distributed training için kaydet ve initialization bilgilerini dön"""
        async with self.lock:
            if client_id not in self.client_states:
                await self.register_client(client_id)

            # Mevcut shard durumlarını kontrol et
            available_shards = [s for s in self.data_shards if s['status'] == 'pending' or
                                (s['status'] == 'in_progress' and s['assigned_to'] == client_id)]

            if not available_shards:
                # Tüm shardlar tamamlanmışsa yeni task'a geç
                if all(s['status'] == 'completed' for s in self.data_shards):
                    return {
                        'status': 'completed',
                        'message': 'All shards completed'
                    }

                # Mevcut shardlardan birini yeniden kullan
                available_shards = [s for s in self.data_shards if s['status'] != 'completed']

            # Shard seç
            shard = available_shards[0]
            shard['status'] = 'in_progress'
            shard['assigned_to'] = client_id
            shard['start_time'] = time.time()

            self.client_states[client_id].update({
                'rank': rank,
                'status': 'training',
                'current_shard': shard['id']
            })

            logger.info(f"Client {client_id} assigned to shard {shard['id']}")

            return {
                'master_addr': 'localhost',
                'master_port': self.coordinator_port,
                'world_size': self.n_clients,
                'rank': shard['id'],
                'model_version': self.global_model_version,
                'parameters': self.global_parameters,
                'aggregation_frequency': self.aggregation_frequency,
                'warmup_epochs': self.warmup_epochs,
                'shard_info': {
                    'id': shard['id'],
                    'total_shards': len(self.data_shards),
                    'start_point': shard.get('start_point', 0),
                    'end_point': shard.get('end_point', 100)
                }
            }

    def calculate_shard(self, client_id: str) -> dict:
        """Calculate shard for given client"""
        # Mevcut işleri kontrol et
        assigned_ranges = [s['range'] for s in self.data_shards['assignments'].values()]
        total_processed = sum(r[1] - r[0] for r in assigned_ranges) if assigned_ranges else 0

        # Yeni range hesapla
        start = total_processed
        end = min(100, start + (100 - total_processed))

        shard = {
            'id': len(self.data_shards['assignments']),
            'range': (start, end),
            'total_shards': self.n_clients
        }

        # Kaydet
        self.data_shards['assignments'][client_id] = shard
        return shard
    async def mark_client_ready(self, client_id: str):
        """Mark client as ready for training"""
        async with self.lock:
            if client_id in self.client_states:
                self.client_states[client_id]['status'] = 'ready'
                logger.info(f"Client {client_id} marked as ready")
                return {'status': 'success'}
            return {'status': 'error', 'message': 'Client not found'}
    def get_next_shard(self, client_id: str) -> Optional[dict]:
        """Client için uygun bir iş parçası bul"""
        # Pending shardları kontrol et
        for shard in self.data_shards:
            if shard['status'] == 'pending':
                shard['status'] = 'in_progress'
                shard['assigned_to'] = client_id
                shard['start_time'] = time.time()
                return shard

        # En yavaş ilerleyen shardı bul
        slowest_shard = None
        min_progress = float('inf')
        for shard in self.data_shards:
            if shard['status'] == 'in_progress' and shard['progress'] < min_progress:
                min_progress = shard['progress']
                slowest_shard = shard

        if slowest_shard and slowest_shard['progress'] < 50:
            # İşi böl
            new_shard = {
                'id': len(self.data_shards),
                'status': 'in_progress',
                'assigned_to': client_id,
                'start_time': time.time(),
                'progress': 0,
                'parent_shard': slowest_shard['id'],
                'start_point': slowest_shard['progress'],
                'end_point': 100
            }
            # Orijinal shardı güncelle
            slowest_shard['end_point'] = slowest_shard['progress']
            self.data_shards.append(new_shard)
            return new_shard

        return None

    async def register_for_initialization(self, client_id: str, rank: int):
        """Client'ı distributed training için kaydet ve initialization bilgilerini dön"""
        async with self.lock:
            if client_id not in self.client_states:
                await self.register_client(client_id)

            # İş parçası ata
            shard = self.get_next_shard(client_id)
            if not shard:
                # Yeni shard oluştur
                shard = {
                    'id': len(self.data_shards),
                    'status': 'in_progress',
                    'assigned_to': client_id,
                    'start_time': time.time(),
                    'progress': 0,
                    'start_point': 0,
                    'end_point': 100,
                }
                self.data_shards.append(shard)

            self.client_states[client_id].update({
                'rank': rank,
                'status': 'training',
                'current_shard': shard['id']
            })

            logger.info(f"Client {client_id} assigned to shard {shard['id']}")

            # Hemen initialization data dön
            return {
                'master_addr': 'localhost',
                'master_port': self.coordinator_port,
                'world_size': self.n_clients,
                'rank': shard['id'],
                'model_version': self.global_model_version,
                'parameters': self.global_parameters,
                'aggregation_frequency': self.aggregation_frequency,
                'warmup_epochs': self.warmup_epochs,
                'shard_info': {
                    'id': shard['id'],
                    'total_shards': len(self.data_shards),
                    'start_point': shard.get('start_point', 0),
                    'end_point': shard.get('end_point', 100)
                }
            }

    async def update_progress(self, client_id: str, progress: float, metrics: Dict = None):
        """Update client progress"""
        async with self.lock:
            if client_id in self.client_states:
                self.client_states[client_id].update({
                    'progress': progress,
                    'last_update': datetime.now(),
                    'metrics': metrics or {}
                })
                return {'status': 'success'}
            return {'status': 'error', 'message': 'Client not found'}

    async def submit_update(self, client_id: str, epoch: int, parameters: Dict, metrics: Dict) -> Dict:
        """Model güncellemelerini topla ve gerekirse birleştir"""
        async with self.lock:
            try:
                # İlk update ise parametreleri direkt kaydet
                if self.global_parameters is None:
                    self.global_parameters = parameters
                    self.global_model_version = 0
                    return {'status': 'ok', 'action': 'continue'}

                # Client update'ini kaydet
                if client_id not in self.client_updates:
                    self.client_updates[client_id] = []

                # Model update'i oluştur
                model_update = ModelUpdate(
                    client_id=client_id,
                    epoch=epoch,
                    parameters=parameters,  # Model parametrelerini ekle
                    metrics=metrics,
                    timestamp=time.time()
                )
                self.client_updates[client_id].append(model_update)

                # Update şartlarını kontrol et
                should_update = (epoch > self.warmup_epochs and
                                 epoch % self.aggregation_frequency == 0 and
                                 len(self.client_updates) >= 1)

                if should_update:
                    # En son updateler
                    recent_updates = [updates[-1] for updates in self.client_updates.values()]

                    try:
                        # Modelleri birleştir
                        self.global_parameters = self.aggregator.aggregate_parameters(recent_updates)
                        self.global_model_version += 1

                        # Final modeli kaydet
                        save_path = os.path.join(
                            'aggregated_models',
                            f'task_aggregated_model_v{self.global_model_version}.pt'
                        )

                        # State dict'i kaydet
                        torch.save({
                            'model_state': self.global_parameters,
                            'metrics': metrics,
                            'version': self.global_model_version,
                            'timestamp': datetime.now().isoformat(),
                            'updates': [u.__dict__ for u in recent_updates]
                        }, save_path)

                        logger.info(f"Saved aggregated model to: {save_path}")

                        # Metrikleri güncelle
                        if metrics.get('loss', float('inf')) < self.best_metrics['loss']:
                            self.best_metrics = metrics

                        self.client_updates.clear()

                        return {
                            'status': 'ok',
                            'action': 'sync',
                            'model_version': self.global_model_version,
                            'parameters': self.global_parameters,
                            'model_path': save_path
                        }
                    except Exception as e:
                        logger.error(f"Error aggregating models: {str(e)}")
                        return {'status': 'error', 'message': str(e)}

                return {'status': 'ok', 'action': 'continue'}

            except Exception as e:
                logger.error(f"Error in submit_update: {str(e)}")
                return {'status': 'error', 'message': str(e)}

    async def unregister_client(self, client_id: str):
        """Client'ı coordinator'dan kaldır"""
        async with self.lock:
            if client_id in self.ready_clients:
                self.ready_clients.remove(client_id)
                self.ready_clients.remove(client_id)
            if client_id in self.client_states:
                del self.client_states[client_id]
            if client_id in self.client_updates:
                del self.client_updates[client_id]
            logger.info(f"Client {client_id} unregistered from coordinator")

    async def get_training_status(self) -> Dict:
        """Eğitim durumunu getir"""
        async with self.lock:
            return {
                'model_version': self.global_model_version,
                'n_clients': len(self.client_updates),
                'best_metrics': self.best_metrics,
                'client_status': {
                    client_id: {
                        'last_update': self.client_states[client_id]['last_update'],
                        'current_epoch': self.client_states[client_id]['current_epoch'],
                        'status': self.client_states[client_id]['status'],
                        'metrics': self.client_states[client_id]['metrics'],
                        'n_updates': self.client_states[client_id]['n_updates']
                    }
                    for client_id in self.client_states
                }
            }

    async def monitor_clients(self):
        """Client durumlarını monitör et"""
        while True:
            try:
                async with self.lock:
                    current_time = time.time()
                    for client_id, state in list(self.client_states.items()):
                        last_update = state.get('last_update', 0)
                        if current_time - last_update > 300:  # 5 dakika timeout
                            await self.unregister_client(client_id)
                            logger.warning(f"Client {client_id} timed out and was unregistered")
            except Exception as e:
                logger.error(f"Error in client monitor: {str(e)}")

            await asyncio.sleep(60)  # Her dakika kontrol et