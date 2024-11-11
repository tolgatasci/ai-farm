# AI Farm Client Documentation

The client component is responsible for executing machine learning training tasks, managing local resources, and coordinating with the server for distributed training.

## Technical Stack

- **Python 3.8+**
- **PyTorch** - ML Framework
- **WebSocket** - Server Communication
- **CUDA** (optional) - GPU Support
- **cpuinfo** - Resource Monitoring

## Installation

### System Requirements

- Python 3.8+
- CUDA Toolkit (optional for GPU support)
- 4GB+ RAM
- Storage for model checkpoints and cache

### Dependencies Installation

```bash
pip install -r requirements.txt
```

### Configuration

Create `config.py` with the following settings:

```python
# Server Connection
SERVER_URI = "ws://localhost:8765"
MODEL_SERVER_URL = "http://localhost:5000"

# Client Settings
SESSION_DIR = "sessions"
CHECKPOINTS_DIR = "checkpoints"
MODEL_CACHE_DIR = "model_cache"

# Resource Management
MAX_TASKS = 3
PING_INTERVAL = 15
CONNECTION_TIMEOUT = 30

# Training Settings
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4
```

## Client Components

### 1. Resource Manager

Monitors and manages system resources:

```python
class ResourceManager:
    def __init__(self):
        self.resources = self.get_system_resources()
        self.allocated_resources = {}

    def get_system_resources(self) -> Dict:
        return {
            'cpu': {
                'cores': psutil.cpu_count(),
                'usage': psutil.cpu_percent(),
                'frequency': psutil.cpu_freq().current
            },
            'ram': {
                'total': psutil.virtual_memory().total,
                'available': psutil.virtual_memory().available
            },
            'gpus': self.get_gpu_info()
        }
```

### 2. Training Manager

Handles model training operations:

```python
class TrainingManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.checkpoint_manager = CheckpointManager()

    async def train(
        self, 
        task_id: str, 
        model_code: str, 
        train_data: Dict,
        resume_from: Optional[str] = None
    ) -> Dict:
        # Training implementation
```

### 3. Model Loader

Manages model code caching and loading:

```python
class ModelLoader:
    def __init__(self, cache_dir: str = ".model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def load_model(self, model_url: str, version: str) -> str:
        # Model loading implementation
```

## Client Operation Flow

### 1. Startup and Registration

```python
async def start(self):
    """Start client operations"""
    try:
        # Initialize components
        self.system_info = self.get_system_info()
        await self.connect()
        
        # Start background tasks
        self.background_tasks.extend([
            asyncio.create_task(self.process_message_queue()),
            asyncio.create_task(self.monitor_connection())
        ])
        
        # Main operation loop
        while not self.should_stop:
            await self.handle_tasks()
            
    except Exception as e:
        logger.error(f"Client operation error: {e}")
    finally:
        await self.cleanup()
```

### 2. Task Handling

```python
async def handle_task(self, task: Dict):
    """Process received task"""
    try:
        # Resource check
        requirements = ResourceRequirements(**task['requirements'])
        if not self.resource_manager.check_requirements(requirements):
            raise RuntimeError("Insufficient resources")

        # Resource allocation
        resources = self.resource_manager.allocate_resources(requirements)

        try:
            # Training setup
            config = task['config']
            training_config = TrainingConfig(
                device=f"cuda:{resources['gpu_id']}" if resources['gpu_id'] is not None else "cpu",
                **config
            )

            # Execute training
            result = await self.execute_training(task['id'], training_config)
            return result

        finally:
            # Release resources
            self.resource_manager.release_resources(resources)

    except Exception as e:
        logger.error(f"Task handling error: {e}")
        raise
```

### 3. Progress Reporting

```python
async def report_progress(self, task_id: str, progress: float, metrics: Dict):
    """Send progress update to server"""
    message = {
        'type': 'task_progress',
        'task_id': task_id,
        'progress': progress,
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    await self.send_message(message)
```

## Error Handling

### Connection Errors

```python
async def handle_connection_error(self):
    """Handle connection failures"""
    try:
        logger.warning("Connection lost, attempting to reconnect...")
        await asyncio.sleep(self.reconnect_delay)
        self.reconnect_delay = min(
            self.reconnect_delay * 2,
            self.max_reconnect_delay
        )
        await self.connect()
    except Exception as e:
        logger.error(f"Reconnection failed: {e}")
```

### Training Errors

```python
async def handle_training_error(self, task_id: str, error: Exception):
    """Handle training failures"""
    try:
        # Save checkpoint if possible
        if self.training_manager and self.training_manager.current_state:
            await self.save_emergency_checkpoint(task_id)

        # Report error
        await self.report_task_failure(task_id, str(error))
    except Exception as e:
        logger.error(f"Error handling failed: {e}")
```

## Resource Management

### GPU Management

```python
def get_gpu_info(self) -> List[Dict]:
    """Get GPU information"""
    try:
        gpus = GPUtil.getGPUs()
        return [{
            'id': gpu.id,
            'name': gpu.name,
            'memory': {
                'total': gpu.memoryTotal,
                'used': gpu.memoryUsed,
                'free': gpu.memoryFree
            },
            'utilization': gpu.load * 100,
            'temperature': gpu.temperature
        } for gpu in gpus]
    except Exception:
        return []
```

### Resource Allocation

```python
def allocate_resources(self, requirements: ResourceRequirements) -> Dict:
    """Allocate resources for task"""
    allocation = {
        'cpu_cores': requirements.min_cpu_cores,
        'ram': requirements.min_ram,
        'gpu_id': None
    }

    if requirements.min_gpu_memory > 0:
        allocation['gpu_id'] = self._find_suitable_gpu(
            requirements.min_gpu_memory
        )

    return allocation
```

## Checkpointing

### Checkpoint Management

```python
class CheckpointManager:
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
        task_id: str
    ):
        """Save training checkpoint"""
        path = os.path.join(
            self.save_dir,
            f"task_{task_id}_epoch_{epoch}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, path)
        
        return path
```

## Monitoring and Logging

### Log Configuration

```python
logging.config.dictConfig({
    'version': 1,
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'client.log',
            'maxBytes': 10485760,
            'backupCount': 5
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file']
    }
})
```

### Performance Monitoring

```python
def monitor_resources(self) -> Dict:
    """Monitor resource utilization"""
    return {
        'cpu_usage': psutil.cpu_percent(interval=1),
        'ram_usage': psutil.virtual_memory().percent,
        'gpu_usage': self.get_gpu_usage(),
        'disk_usage': psutil.disk_usage('/').percent
    }
```

## Development and Testing

### Running Tests

```bash
pytest tests/
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows

# Install development dependencies
pip install -r requirements-dev.txt

# Run client in debug mode
python client.py --debug
```

## Troubleshooting

Common issues and solutions:

1. Connection Issues
```
Issue: "WebSocket connection failed"
Solution: Check server address and network connection
```

2. Resource Issues
```
Issue: "CUDA out of memory"
Solution: Reduce batch size or free GPU memory
```

3. Training Issues
```
Issue: "Model code execution failed"
Solution: Check model code and dependencies
```

## Support and Feedback

For support:
1. Check logs in `logs/` directory
2. Review error messages
3. Contact system administrator

For feature requests and bug reports:
- Open an issue in the repository
- Include logs and error messages
- Describe steps to reproduce
