import psutil
import GPUtil
from typing import Dict, List
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger("AIFarmResource")


@dataclass
class ResourceRequirements:
    min_gpu_memory: float  # GB
    min_cpu_cores: int
    min_ram: float  # GB


class ResourceManager:
    def __init__(self):
        self.resources = self.get_system_resources()
        self.allocated_resources = {}  # Track allocated resources

    def get_system_resources(self) -> Dict:
        resources = {
            'cpu': {
                'cores': psutil.cpu_count(),
                'usage': psutil.cpu_percent(interval=1),
                'frequency': psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            'ram': {
                'total': psutil.virtual_memory().total / (1024 ** 3),  # GB
                'available': psutil.virtual_memory().available / (1024 ** 3),  # GB
                'percent': psutil.virtual_memory().percent
            },
            'gpus': []
        }

        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                resources['gpus'].append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal / 1024,  # GB
                    'memory_free': gpu.memoryFree / 1024,  # GB
                    'utilization': gpu.load * 100
                })
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")

        return resources

    def check_requirements(self, requirements: ResourceRequirements) -> bool:
        """Check if system meets the requirements"""
        # Update current resources
        self.resources = self.get_system_resources()

        # CPU check
        if self.resources['cpu']['cores'] < requirements.min_cpu_cores:
            return False

        # RAM check
        if self.resources['ram']['available'] < requirements.min_ram:
            return False

        # GPU check
        if requirements.min_gpu_memory > 0:
            available_gpu = False
            for gpu in self.resources['gpus']:
                if gpu['memory_free'] >= requirements.min_gpu_memory:
                    available_gpu = True
                    break
            if not available_gpu:
                return False

        return True

    def allocate_resources(self, requirements: ResourceRequirements) -> Dict:
        """Allocate resources for a task"""
        if not self.check_requirements(requirements):
            raise RuntimeError("Insufficient resources")

        # Find best GPU if needed
        selected_gpu = None
        if requirements.min_gpu_memory > 0:
            gpus = [(i, gpu) for i, gpu in enumerate(self.resources['gpus'])
                    if gpu['memory_free'] >= requirements.min_gpu_memory]
            if gpus:
                # Select GPU with most free memory
                selected_gpu = max(gpus, key=lambda x: x[1]['memory_free'])[0]

        # Create allocation
        allocation = {
            'gpu_id': selected_gpu,
            'cpu_cores': requirements.min_cpu_cores,
            'ram': requirements.min_ram,
            'allocation_id': id(requirements)  # Unique identifier for this allocation
        }

        # Track allocation
        self.allocated_resources[allocation['allocation_id']] = allocation
        logger.info(
            f"Resources allocated: GPU={selected_gpu}, CPU cores={requirements.min_cpu_cores}, RAM={requirements.min_ram}GB")

        return allocation

    def release_resources(self, allocation: Dict) -> None:
        """Release previously allocated resources"""
        try:
            allocation_id = allocation.get('allocation_id')
            if allocation_id and allocation_id in self.allocated_resources:
                released = self.allocated_resources.pop(allocation_id)
                logger.info(
                    f"Released resources: GPU={released['gpu_id']}, CPU cores={released['cpu_cores']}, RAM={released['ram']}GB")
            else:
                # Try to find allocation by values if no ID match
                for alloc_id, alloc in list(self.allocated_resources.items()):
                    if alloc['gpu_id'] == allocation['gpu_id'] and \
                            alloc['cpu_cores'] == allocation['cpu_cores'] and \
                            alloc['ram'] == allocation['ram']:
                        released = self.allocated_resources.pop(alloc_id)
                        logger.info(
                            f"Released resources found by values: GPU={released['gpu_id']}, CPU cores={released['cpu_cores']}, RAM={released['ram']}GB")
                        break
        except Exception as e:
            logger.error(f"Error releasing resources: {str(e)}")
            # Continue without raising to prevent disrupting cleanup

    def monitor_resources(self) -> Dict:
        """Monitor current resource usage"""
        current = self.get_system_resources()

        # Calculate resource utilization trends
        cpu_trend = np.mean([current['cpu']['usage']])
        ram_trend = current['ram']['percent']

        gpu_trends = []
        for gpu in current['gpus']:
            gpu_trends.append({
                'id': gpu['id'],
                'utilization': gpu['utilization'],
                'memory_usage': (gpu['memory_total'] - gpu['memory_free']) / gpu['memory_total'] * 100
            })

        # Include allocation info in monitoring
        active_allocations = len(self.allocated_resources)

        return {
            'cpu_utilization': cpu_trend,
            'ram_utilization': ram_trend,
            'gpu_utilization': gpu_trends,
            'active_allocations': active_allocations,
            'status': 'healthy' if cpu_trend < 90 and ram_trend < 90 else 'overloaded'
        }


class LoadBalancer:
    def __init__(self, resource_manager: ResourceManager):
        self.resource_manager = resource_manager

    def calculate_task_score(self, task: Dict, resources: Dict) -> float:
        """Calculate priority score for task based on resources"""
        score = task.get('priority', 0) * 10  # Base priority

        # Resource availability factor
        if resources['status'] == 'healthy':
            score += 5

        # GPU utilization factor
        if task.get('requires_gpu', False) and resources['gpu_utilization']:
            avg_gpu_util = np.mean([gpu['utilization'] for gpu in resources['gpu_utilization']])
            score -= avg_gpu_util / 10  # Lower score for high GPU usage

        return score

    def select_next_task(self, tasks: List[Dict]) -> Dict:
        """Select next task based on resources and priorities"""
        if not tasks:
            return None

        resources = self.resource_manager.monitor_resources()

        # Calculate scores for all tasks
        task_scores = [(task, self.calculate_task_score(task, resources))
                       for task in tasks]

        # Select task with highest score
        selected_task = max(task_scores, key=lambda x: x[1])[0]

        return selected_task