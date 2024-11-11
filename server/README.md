# AI Farm Server Documentation

The server component is the central coordination unit of the AI Farm platform. It handles task distribution, client coordination, and model aggregation.

## Technical Stack

- **Python 3.8+**
- **FastAPI** - Web framework
- **WebSocket** - Real-time communication
- **MySQL** - State management
- **Redis** (optional) - Caching
- **TorchServe** (optional) - Model serving

## Installation

### System Requirements

- Python 3.8+
- MySQL 8.0+
- 4GB+ RAM
- Sufficient storage for model checkpoints

### Dependencies Installation

```bash
pip install -r requirements.txt
```

### MySQL Setup

1. Create the database:
```sql
CREATE DATABASE ai_farm CHARACTER SET utf8mb4 COLLATE utf8mb4_general_ci;
```

2. Configure database connection in `config.py`:
```python
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'ai_farm',
    'charset': 'utf8mb4',
    'collation': 'utf8mb4_general_ci'
}
```

## Configuration

The server configuration is managed through `config.py`:

```python
# Server Configuration
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000
WEBSOCKET_PORT = 8765
MODEL_SERVER_PORT = 5000

# Resource Limits
MAX_CLIENTS = 100
MAX_TASKS_PER_CLIENT = 3
CONNECTION_TIMEOUT = 60

# Training Configuration
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
CHECKPOINT_FREQUENCY = 5

# Paths
LOG_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"
MODEL_CACHE_DIR = "model_cache"
```

## Database Schema

### Tasks Table
```sql
CREATE TABLE tasks (
    id INT AUTO_INCREMENT PRIMARY KEY,
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
    checkpoint_path VARCHAR(255)
);
```

### Task Assignments Table
```sql
CREATE TABLE task_assignments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    task_id INT NOT NULL,
    client_id VARCHAR(255) NOT NULL,
    assigned_at DATETIME NOT NULL,
    status ENUM('assigned', 'training', 'completed', 'failed') NOT NULL,
    rank INT NOT NULL,
    checkpoint_path VARCHAR(255),
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);
```

### Clients Table
```sql
CREATE TABLE clients (
    id VARCHAR(255) PRIMARY KEY,
    cpu_info JSON NOT NULL,
    ram_info JSON NOT NULL,
    gpu_info JSON,
    session_info JSON,
    last_seen DATETIME NOT NULL,
    status ENUM('active', 'inactive') NOT NULL
);
```

## Server Components

### 1. WebSocket Server
Handles real-time communication with clients:
- Client registration
- Task distribution
- Progress updates
- Resource monitoring

### 2. HTTP API Server
Provides RESTful endpoints for:
- Task management
- Client management
- Resource monitoring
- Admin interface

### 3. Training Coordinator
Manages distributed training:
- Shard distribution
- Model aggregation
- Progress tracking
- Fault tolerance

## API Endpoints

### WebSocket Events

#### Client -> Server
```python
# Task Request
{
    'type': 'task_request',
    'client_id': 'client_uuid'
}

# Task Progress
{
    'type': 'task_progress',
    'task_id': 'task_id',
    'progress': 50.0,
    'metrics': {...}
}

# Task Result
{
    'type': 'task_result',
    'task_id': 'task_id',
    'status': 'completed',
    'metrics': {...}
}
```

#### Server -> Client
```python
# Task Assignment
{
    'type': 'task',
    'data': {
        'id': 'task_id',
        'config': {...},
        'requirements': {...}
    }
}

# Control Messages
{
    'type': 'pause/resume/stop',
    'task_id': 'task_id'
}
```

### HTTP API

#### Tasks
- `GET /api/tasks` - List all tasks
- `POST /api/tasks` - Create new task
- `GET /api/tasks/{id}` - Get task details
- `PUT /api/tasks/{id}/status` - Update task status
- `DELETE /api/tasks/{id}` - Delete task

#### Clients
- `GET /api/clients` - List all clients
- `GET /api/clients/{id}` - Get client details
- `PUT /api/clients/{id}/status` - Update client status

## Error Handling

### Database Errors
```python
try:
    conn = connection_pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    # ... database operations
except Error as e:
    logger.error(f"Database error: {e}")
    if 'conn' in locals(): conn.rollback()
finally:
    if 'cursor' in locals(): cursor.close()
    if 'conn' in locals() and conn.is_connected(): conn.close()
```

### WebSocket Errors
```python
try:
    await websocket.send(json.dumps(message))
except websockets.exceptions.ConnectionClosed:
    await self.handle_client_disconnect(client_id)
except Exception as e:
    logger.error(f"WebSocket error: {e}")
```

## Monitoring and Logging

### Log Files
- `server.log` - General server logs
- `error.log` - Error logs
- `access.log` - HTTP access logs

### Metrics
```python
{
    'cpu_utilization': 45.2,
    'ram_utilization': 60.5,
    'gpu_utilization': [{
        'id': 0,
        'utilization': 80.0,
        'memory_used': 4096
    }],
    'active_tasks': 5,
    'connected_clients': 3
}
```

## Security

### Authentication
- WebSocket: Client ID based
- HTTP API: Token based
- Admin Panel: Username/Password

### Rate Limiting
```python
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"]
)
```

## Troubleshooting

### Common Issues

1. Database Connection Issues
```python
"Error: Lost connection to MySQL server"
Solution: Check MySQL service status and connection settings
```

2. WebSocket Connection Issues
```python
"Error: Connection closed abnormally"
Solution: Check network connectivity and client status
```

3. Resource Issues
```python
"Error: Insufficient resources for task"
Solution: Adjust task requirements or add more resources
```

## Development and Testing

### Setup Development Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate  # Windows
pip install -r requirements-dev.txt
```

### Running Tests
```bash
pytest tests/
```

### Docker Deployment
```bash
docker build -t ai-farm-server .
docker run -p 8000:8000 -p 8765:8765 ai-farm-server
```
