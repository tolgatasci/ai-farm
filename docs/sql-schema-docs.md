# AI Farm Database Schema Documentation

## Overview

AI Farm uses MySQL 8.0+ with UTF-8 encoding for all database operations. The schema is designed to support distributed training operations, client management, and task tracking.

## Database Configuration

```sql
CREATE DATABASE ai_farm
  CHARACTER SET = utf8mb4
  COLLATE = utf8mb4_general_ci;
```

## Core Tables

### 1. Tasks Table
Stores all training tasks and their configurations.

```sql
CREATE TABLE tasks (
    id INT AUTO_INCREMENT PRIMARY KEY,
    type VARCHAR(255) NOT NULL,
    requirements JSON NOT NULL,
    status ENUM(
        'pending',
        'running',
        'completed',
        'failed',
        'error'
    ) NOT NULL,
    client_id VARCHAR(255),
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    result JSON,
    error_log TEXT,
    priority INT DEFAULT 0,
    config JSON,
    checkpoint_path VARCHAR(255),
    
    INDEX idx_status (status),
    INDEX idx_priority (priority),
    INDEX idx_created_at (created_at),
    
    FOREIGN KEY (client_id) 
        REFERENCES clients(id) 
        ON DELETE SET NULL,
        
    CHECK (JSON_VALID(requirements)),
    CHECK (JSON_VALID(config)),
    CHECK (JSON_VALID(result))
) ENGINE=InnoDB;
```

Example Task Record:
```json
{
    "id": 1,
    "type": "training",
    "requirements": {
        "min_gpu_memory": 4096,
        "min_cpu_cores": 4,
        "min_ram": 8192
    },
    "status": "running",
    "config": {
        "name": "mnist_model",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "distributed": true,
        "n_clients": 3
    },
    "result": {
        "progress": 45.5,
        "metrics": {
            "loss": 0.0023,
            "accuracy": 0.982
        }
    }
}
```

### 2. Task Assignments Table
Manages the relationship between tasks and clients in distributed training.

```sql
CREATE TABLE task_assignments (
    id INT AUTO_INCREMENT PRIMARY KEY,
    task_id INT NOT NULL,
    client_id VARCHAR(255) NOT NULL,
    assigned_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP 
        ON UPDATE CURRENT_TIMESTAMP,
    status ENUM(
        'assigned',
        'training',
        'completed',
        'failed'
    ) NOT NULL,
    rank INT NOT NULL,
    checkpoint_path VARCHAR(255),
    metrics JSON,
    
    INDEX idx_task_status (task_id, status),
    INDEX idx_client_status (client_id, status),
    UNIQUE KEY unique_assignment (task_id, rank),
    
    FOREIGN KEY (task_id) 
        REFERENCES tasks(id) 
        ON DELETE CASCADE,
    FOREIGN KEY (client_id) 
        REFERENCES clients(id) 
        ON DELETE CASCADE,
        
    CHECK (JSON_VALID(metrics))
) ENGINE=InnoDB;
```

### 3. Clients Table
Stores client information and resource capabilities.

```sql
CREATE TABLE clients (
    id VARCHAR(255) PRIMARY KEY,
    cpu_info JSON NOT NULL,
    ram_info JSON NOT NULL,
    gpu_info JSON,
    session_info JSON,
    last_seen DATETIME NOT NULL,
    status ENUM('active', 'inactive') NOT NULL,
    version VARCHAR(50),
    hostname VARCHAR(255),
    
    INDEX idx_status (status),
    INDEX idx_last_seen (last_seen),
    
    CHECK (JSON_VALID(cpu_info)),
    CHECK (JSON_VALID(ram_info)),
    CHECK (JSON_VALID(gpu_info)),
    CHECK (JSON_VALID(session_info))
) ENGINE=InnoDB;
```

### 4. Checkpoints Table
Manages model checkpoints and training states.

```sql
CREATE TABLE checkpoints (
    id INT AUTO_INCREMENT PRIMARY KEY,
    task_id INT NOT NULL,
    client_id VARCHAR(255),
    path VARCHAR(255) NOT NULL,
    type ENUM(
        'regular',
        'distributed',
        'emergency',
        'final'
    ) NOT NULL,
    metrics JSON,
    created_at DATETIME NOT NULL,
    
    INDEX idx_task_type (task_id, type),
    
    FOREIGN KEY (task_id) 
        REFERENCES tasks(id) 
        ON DELETE CASCADE,
    FOREIGN KEY (client_id) 
        REFERENCES clients(id) 
        ON DELETE SET NULL,
        
    CHECK (JSON_VALID(metrics))
) ENGINE=InnoDB;
```

### 5. Model Versions Table
Tracks model versions and their configurations.

```sql
CREATE TABLE model_versions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    url TEXT NOT NULL,
    config JSON,
    hash VARCHAR(64),
    created_at DATETIME NOT NULL,
    
    UNIQUE KEY unique_version (name, version),
    INDEX idx_name_version (name, version),
    
    CHECK (JSON_VALID(config))
) ENGINE=InnoDB;
```

## Logging Tables

### 1. Task Logs
```sql
CREATE TABLE task_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    task_id INT,
    client_id VARCHAR(255),
    log_type ENUM('info', 'warning', 'error') NOT NULL,
    message TEXT NOT NULL,
    created_at DATETIME NOT NULL,
    
    INDEX idx_task_time (task_id, created_at),
    INDEX idx_client_time (client_id, created_at),
    
    FOREIGN KEY (task_id) 
        REFERENCES tasks(id) 
        ON DELETE CASCADE,
    FOREIGN KEY (client_id) 
        REFERENCES clients(id) 
        ON DELETE CASCADE
) ENGINE=InnoDB;
```

### 2. Client Logs
```sql
CREATE TABLE client_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    client_id VARCHAR(255),
    log_type ENUM('info', 'warning', 'error') NOT NULL,
    message TEXT NOT NULL,
    created_at DATETIME NOT NULL,
    
    INDEX idx_client_time (client_id, created_at),
    
    FOREIGN KEY (client_id) 
        REFERENCES clients(id) 
        ON DELETE CASCADE
) ENGINE=InnoDB;
```

### 3. Performance Metrics
```sql
CREATE TABLE performance_metrics (
    id INT AUTO_INCREMENT PRIMARY KEY,
    client_id VARCHAR(255) NOT NULL,
    task_id INT,
    metric_type VARCHAR(50) NOT NULL,
    value FLOAT NOT NULL,
    recorded_at DATETIME NOT NULL,
    
    INDEX idx_client_time (client_id, recorded_at),
    INDEX idx_task_time (task_id, recorded_at),
    
    FOREIGN KEY (client_id) 
        REFERENCES clients(id) 
        ON DELETE CASCADE,
    FOREIGN KEY (task_id) 
        REFERENCES tasks(id) 
        ON DELETE CASCADE
) ENGINE=InnoDB;
```

## Common Queries

### Task Management
```sql
-- Get active tasks with client info
SELECT t.*, c.status as client_status, c.last_seen
FROM tasks t
LEFT JOIN clients c ON t.client_id = c.id
WHERE t.status = 'running'
ORDER BY t.priority DESC, t.created_at ASC;

-- Get task completion statistics
SELECT 
    type,
    COUNT(*) as total,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
    AVG(CASE 
        WHEN status = 'completed' 
        THEN TIMESTAMPDIFF(SECOND, created_at, updated_at)
        ELSE NULL 
    END) as avg_completion_time
FROM tasks
GROUP BY type;
```

### Client Management
```sql
-- Get active clients with task counts
SELECT 
    c.*,
    COUNT(DISTINCT ta.task_id) as active_tasks,
    MAX(ta.updated_at) as last_task_update
FROM clients c
LEFT JOIN task_assignments ta 
    ON c.id = ta.client_id 
    AND ta.status IN ('assigned', 'training')
WHERE c.status = 'active'
GROUP BY c.id;

-- Get client performance metrics
SELECT 
    client_id,
    metric_type,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value
FROM performance_metrics
WHERE recorded_at > DATE_SUB(NOW(), INTERVAL 1 HOUR)
GROUP BY client_id, metric_type;
```

### Distributed Training
```sql
-- Get distributed task progress
SELECT 
    t.id,
    t.status,
    COUNT(DISTINCT ta.client_id) as active_clients,
    SUM(CASE WHEN ta.status = 'completed' THEN 1 ELSE 0 END) as completed_shards,
    JSON_EXTRACT(t.config, '$.n_clients') as required_clients
FROM tasks t
LEFT JOIN task_assignments ta ON t.id = ta.task_id
WHERE JSON_EXTRACT(t.config, '$.distributed') = true
GROUP BY t.id;

-- Get available ranks for task
SELECT 
    t.id,
    t.status,
    ranks.rank
FROM tasks t
CROSS JOIN (
    SELECT 0 as rank UNION SELECT 1 UNION SELECT 2
) ranks
LEFT JOIN task_assignments ta 
    ON t.id = ta.task_id 
    AND ranks.rank = ta.rank
WHERE t.id = ? 
    AND ta.id IS NULL
    AND ranks.rank < JSON_EXTRACT(t.config, '$.n_clients');
```

## Maintenance Queries

### Cleanup
```sql
-- Delete old completed tasks
DELETE FROM tasks 
WHERE status IN ('completed', 'failed')
AND updated_at < DATE_SUB(NOW(), INTERVAL 30 DAY);

-- Remove inactive clients
UPDATE clients 
SET status = 'inactive'
WHERE last_seen < DATE_SUB(NOW(), INTERVAL 5 MINUTE);

-- Clean orphaned assignments
DELETE FROM task_assignments
WHERE updated_at < DATE_SUB(NOW(), INTERVAL 1 HOUR)
AND status = 'assigned';
```

### Monitoring
```sql
-- Check table sizes
SELECT 
    table_name,
    table_rows,
    data_length/1024/1024 as data_size_mb,
    index_length/1024/1024 as index_size_mb
FROM information_schema.tables
WHERE table_schema = 'ai_farm'
ORDER BY data_length DESC;

-- Find slow queries
SELECT *
FROM performance_schema.events_statements_summary_by_digest
WHERE schema_name = 'ai_farm'
ORDER BY avg_timer_wait DESC
LIMIT 10;
```

## Backup and Recovery
```sql
-- Create backup
mysqldump -u root -p ai_farm > backup.sql

-- Restore from backup
mysql -u root -p ai_farm < backup.sql

-- Point-in-time recovery
SET GLOBAL binlog_format = 'ROW';
SET GLOBAL expire_logs_days = 30;
```
