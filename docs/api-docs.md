# AI Farm API Documentation

## Overview

AI Farm provides three main API interfaces:
1. WebSocket API for client-server communication
2. HTTP REST API for management and monitoring
3. Internal API for component interaction

## WebSocket API

### Connection

```python
WebSocket URL: ws://server:8765
```

### Authentication

```python
# Connection Headers
headers = {
    'Client-ID': 'unique_client_id',
    'Authorization': 'Bearer <token>'
}
```

### Message Format

All messages follow this basic structure:
```json
{
    "type": "message_type",
    "timestamp": "2024-11-04T12:00:00Z",
    "data": {}
}
```

### Client Messages

#### 1. Task Request
```json
{
    "type": "task_request",
    "client_id": "client_uuid",
    "timestamp": "2024-11-04T12:00:00Z"
}
```

#### 2. Task Progress
```json
{
    "type": "task_progress",
    "task_id": "task_uuid",
    "client_id": "client_uuid",
    "progress": 45.5,
    "metrics": {
        "loss": 0.0023,
        "accuracy": 0.982,
        "learning_rate": 0.001
    },
    "timestamp": "2024-11-04T12:00:00Z"
}
```

#### 3. Task Result
```json
{
    "type": "task_result",
    "task_id": "task_uuid",
    "client_id": "client_uuid",
    "status": "completed",
    "metrics": {
        "final_loss": 0.0018,
        "final_accuracy": 0.988,
        "training_time": 3600
    },
    "checkpoint_path": "/path/to/model.pt",
    "timestamp": "2024-11-04T12:00:00Z"
}
```

#### 4. Heartbeat
```json
{
    "type": "heartbeat",
    "client_id": "client_uuid",
    "system_info": {
        "cpu_usage": 45.2,
        "ram_usage": 60.5,
        "gpu_usage": 80.0
    },
    "timestamp": "2024-11-04T12:00:00Z"
}
```

### Server Messages

#### 1. Task Assignment
```json
{
    "type": "task",
    "data": {
        "id": "task_uuid",
        "type": "training",
        "config": {
            "name": "model_name",
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10,
            "distributed": true,
            "rank": 0,
            "world_size": 3
        },
        "requirements": {
            "min_gpu_memory": 4096,
            "min_cpu_cores": 4,
            "min_ram": 8192
        }
    },
    "timestamp": "2024-11-04T12:00:00Z"
}
```

#### 2. Control Messages
```json
{
    "type": "control",
    "action": "pause|resume|stop",
    "task_id": "task_uuid",
    "timestamp": "2024-11-04T12:00:00Z"
}
```

#### 3. Acknowledgments
```json
{
    "type": "task_result_ack",
    "task_id": "task_uuid",
    "received": true,
    "timestamp": "2024-11-04T12:00:00Z"
}
```

## HTTP REST API

### Authentication

```http
Authorization: Bearer <token>
```

### Tasks Endpoints

#### List Tasks
```http
GET /api/tasks

Response 200:
{
    "tasks": [
        {
            "id": "task_uuid",
            "type": "training",
            "status": "running",
            "progress": 45.5,
            "created_at": "2024-11-04T12:00:00Z",
            "updated_at": "2024-11-04T12:00:00Z"
        }
    ],
    "total": 10,
    "page": 1
}
```

#### Create Task
```http
POST /api/tasks

Request:
{
    "type": "training",
    "name": "model_name",
    "url": "http://model-server/models/model/1.0",
    "version": "1.0",
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 10,
    "distributed": true,
    "requirements": {
        "min_gpu_memory": 0,
        "min_cpu_cores": 1,
        "min_ram": 1
    },
    "n_clients": 3
}

Response 201:
{
    "id": "task_uuid",
    "status": "pending",
    "created_at": "2024-11-04T12:00:00Z"
}
```

#### Get Task Details
```http
GET /api/tasks/{task_id}

Response 200:
{
    "id": "task_uuid",
    "type": "training",
    "status": "running",
    "progress": 45.5,
    "metrics": {
        "loss": 0.0023,
        "accuracy": 0.982
    },
    "config": {},
    "requirements": {},
    "created_at": "2024-11-04T12:00:00Z",
    "updated_at": "2024-11-04T12:00:00Z"
}
```

#### Update Task Status
```http
PUT /api/tasks/{task_id}/status

Request:
{
    "status": "paused",
    "reason": "Resource constraints"
}

Response 200:
{
    "id": "task_uuid",
    "status": "paused",
    "updated_at": "2024-11-04T12:00:00Z"
}
```

#### Delete Task
```http
DELETE /api/tasks/{task_id}

Response 204
```

### Clients Endpoints

#### List Clients
```http
GET /api/clients

Response 200:
{
    "clients": [
        {
            "id": "client_uuid",
            "status": "active",
            "resources": {
                "cpu": {},
                "ram": {},
                "gpu": {}
            },
            "last_seen": "2024-11-04T12:00:00Z"
        }
    ]
}
```

#### Get Client Details
```http
GET /api/clients/{client_id}

Response 200:
{
    "id": "client_uuid",
    "status": "active",
    "resources": {
        "cpu": {
            "cores": 8,
            "usage": 45.2
        },
        "ram": {
            "total": 16384,
            "used": 8192
        },
        "gpu": [
            {
                "id": 0,
                "name": "NVIDIA RTX 3080",
                "memory": {
                    "total": 10240,
                    "used": 3072
                }
            }
        ]
    },
    "tasks": [
        {
            "id": "task_uuid",
            "status": "running",
            "progress": 45.5
        }
    ],
    "performance": {
        "completed_tasks": 10,
        "average_time": 3600,
        "success_rate": 0.95
    },
    "last_seen": "2024-11-04T12:00:00Z"
}
```

### Metrics Endpoints

#### Get System Metrics
```http
GET /api/metrics

Response 200:
{
    "clients": {
        "total": 10,
        "active": 8
    },
    "tasks": {
        "total": 100,
        "running": 5,
        "completed": 90,
        "failed": 5
    },
    "resources": {
        "cpu_usage": 45.2,
        "ram_usage": 60.5,
        "gpu_usage": 80.0
    },
    "timestamp": "2024-11-04T12:00:00Z"
}
```

#### Get Task Metrics
```http
GET /api/metrics/tasks/{task_id}

Response 200:
{
    "progress": 45.5,
    "metrics": {
        "loss": 0.0023,
        "accuracy": 0.982,
        "learning_rate": 0.001
    },
    "resources": {
        "cpu_usage": 45.2,
        "ram_usage": 60.5,
        "gpu_usage": 80.0
    },
    "timestamp": "2024-11-04T12:00:00Z"
}
```

## Error Handling

### Error Response Format
```json
{
    "error": {
        "code": "ERROR_CODE",
        "message": "Error description",
        "details": {}
    }
}
```

### Common Error Codes

```python
ERROR_CODES = {
    'INVALID_REQUEST': 400,
    'UNAUTHORIZED': 401,
    'FORBIDDEN': 403,
    'NOT_FOUND': 404,
    'CONFLICT': 409,
    'INTERNAL_ERROR': 500,
    'SERVICE_UNAVAILABLE': 503
}
```

### WebSocket Close Codes

```python
WS_CLOSE_CODES = {
    1000: "Normal Closure",
    1001: "Going Away",
    1002: "Protocol Error",
    1003: "Unsupported Data",
    1006: "Abnormal Closure",
    1008: "Policy Violation",
    1011: "Internal Error"
}
```

## Rate Limiting

```http
HTTP Headers:
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1635945600
```

## Pagination

```http
Query Parameters:
?page=1&limit=10

Response Headers:
X-Total-Count: 100
X-Page-Count: 10
Link: <https://api/resources?page=2>; rel="next"
```

## Versioning

```http
API Version Header:
X-API-Version: 1.0

Base URLs:
- v1: https://api/v1
- v2: https://api/v2
```
