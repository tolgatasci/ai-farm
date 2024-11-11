# AI Farm Deployment Guide

## Table of Contents
1. [Development Environment](#development-environment)
2. [Staging Environment](#staging-environment)
3. [Production Environment](#production-environment)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Monitoring & Maintenance](#monitoring--maintenance)

## Development Environment

### Local Setup

#### 1. Prerequisites
```bash
# System requirements
python >= 3.8
mysql >= 8.0
redis >= 6.0
nvidia-docker2 (optional for GPU support)

# Python virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

#### 2. Server Installation
```bash
# Clone repository
git clone https://github.com/tolgatasci/ai-farm.git
cd ai-farm/server

# Install dependencies
pip install -r requirements-dev.txt

# Setup database
mysql -u root -p < scripts/init_db.sql

# Environment variables
cat > .env << EOL
DB_HOST=localhost
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=ai_farm
REDIS_HOST=localhost
REDIS_PORT=6379
DEBUG=True
EOL
```

#### 3. Client Installation
```bash
cd ../client

# Install dependencies
pip install -r requirements-dev.txt

# Environment variables
cat > .env << EOL
SERVER_URL=ws://localhost:8765
MODEL_SERVER_URL=http://localhost:5000
DEBUG=True
EOL
```

#### 4. Development Server
```bash
# Start server in development mode
python server.py --debug

# Start model server
python model_server.py --debug

# Start client
python client.py --debug
```

### Development Tools

#### Code Quality
```bash
# Install development tools
pip install black isort mypy pylint pytest

# Format code
black .
isort .

# Run type checking
mypy .

# Run linting
pylint src/

# Run tests
pytest tests/
```

#### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
-   repo: https://github.com/psf/black
    rev: 21.5b2
    hooks:
    -   id: black
-   repo: https://github.com/PyCQA/isort
    rev: 5.9.1
    hooks:
    -   id: isort
-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.910
    hooks:
    -   id: mypy
```

## Staging Environment

### Docker Compose Setup

#### 1. Docker Compose File
```yaml
# docker-compose.yml
version: '3.8'

services:
  server:
    build: 
      context: ./server
      dockerfile: Dockerfile.staging
    ports:
      - "8000:8000"
      - "8765:8765"
    environment:
      - DB_HOST=db
      - REDIS_HOST=redis
      - ENV=staging
    depends_on:
      - db
      - redis

  model-server:
    build: ./model-server
    ports:
      - "5000:5000"
    volumes:
      - model-data:/data

  db:
    image: mysql:8.0
    environment:
      - MYSQL_ROOT_PASSWORD=staging_password
      - MYSQL_DATABASE=ai_farm
    volumes:
      - db-data:/var/lib/mysql

  redis:
    image: redis:6.2
    volumes:
      - redis-data:/data

volumes:
  db-data:
  redis-data:
  model-data:
```

#### 2. Staging Configuration
```python
# config/staging.py
WEBSOCKET_PORT = 8765
HTTP_PORT = 8000
MODEL_SERVER_PORT = 5000

DB_CONFIG = {
    'host': 'db',
    'user': 'root',
    'password': 'staging_password',
    'database': 'ai_farm',
    'pool_size': 20
}

REDIS_CONFIG = {
    'host': 'redis',
    'port': 6379,
    'db': 0
}

LOGGING = {
    'version': 1,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/ai-farm/server.log',
            'maxBytes': 10485760,
            'backupCount': 5
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}
```

#### 3. Deployment Script
```bash
#!/bin/bash
# deploy-staging.sh

# Build and deploy
docker-compose build
docker-compose up -d

# Run migrations
docker-compose exec server python manage.py migrate

# Health check
./scripts/health_check.sh

# Load test data
docker-compose exec server python manage.py load_test_data
```

## Production Environment

### Kubernetes Deployment

#### 1. Kubernetes Manifests

```yaml
# kubernetes/server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-farm-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-farm-server
  template:
    metadata:
      labels:
        app: ai-farm-server
    spec:
      containers:
      - name: server
        image: ai-farm-server:latest
        ports:
        - containerPort: 8000
        - containerPort: 8765
        env:
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: ai-farm-secrets
              key: db-host
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
```

```yaml
# kubernetes/client-daemonset.yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: ai-farm-client
spec:
  selector:
    matchLabels:
      app: ai-farm-client
  template:
    metadata:
      labels:
        app: ai-farm-client
    spec:
      containers:
      - name: client
        image: ai-farm-client:latest
        env:
        - name: SERVER_URL
          value: "ws://ai-farm-server:8765"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: nvidia-gpu
          mountPath: /usr/local/nvidia
      volumes:
      - name: nvidia-gpu
        hostPath:
          path: /usr/local/nvidia
```

#### 2. Production Configuration

```yaml
# kubernetes/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ai-farm-config
data:
  config.yaml: |
    server:
      websocket_port: 8765
      http_port: 8000
      max_connections: 1000
      connection_timeout: 30
    
    monitoring:
      prometheus_port: 9090
      grafana_port: 3000
      
    resources:
      max_tasks_per_client: 3
      task_timeout: 3600
      checkpoint_frequency: 300
```

#### 3. Secrets Management

```yaml
# kubernetes/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ai-farm-secrets
type: Opaque
data:
  db-password: base64_encoded_password
  redis-password: base64_encoded_password
  jwt-secret: base64_encoded_secret
```

#### 4. Production Deployment Script

```bash
#!/bin/bash
# deploy-prod.sh

# Check prerequisites
if ! command -v kubectl &> /dev/null; then
    echo "kubectl not found"
    exit 1
fi

# Apply configurations
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/secrets.yaml

# Deploy infrastructure
kubectl apply -f kubernetes/mysql-statefulset.yaml
kubectl apply -f kubernetes/redis-statefulset.yaml

# Wait for infrastructure
kubectl wait --for=condition=ready pod -l app=mysql --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis --timeout=300s

# Deploy application
kubectl apply -f kubernetes/server-deployment.yaml
kubectl apply -f kubernetes/client-daemonset.yaml
kubectl apply -f kubernetes/model-server-deployment.yaml

# Deploy monitoring
kubectl apply -f kubernetes/prometheus/
kubectl apply -f kubernetes/grafana/

# Verify deployment
kubectl get pods -n ai-farm
```

## Monitoring & Maintenance

### Prometheus Monitoring

```yaml
# prometheus/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ai-farm-server'
    static_configs:
      - targets: ['ai-farm-server:9090']

  - job_name: 'ai-farm-client'
    kubernetes_sd_configs:
      - role: pod
        selectors:
          - role: ai-farm-client
```

### Grafana Dashboards

```json
{
  "dashboard": {
    "id": null,
    "title": "AI Farm Overview",
    "panels": [
      {
        "title": "Active Clients",
        "type": "stat",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "sum(ai_farm_active_clients)"
          }
        ]
      },
      {
        "title": "Running Tasks",
        "type": "graph",
        "datasource": "Prometheus",
        "targets": [
          {
            "expr": "sum(ai_farm_running_tasks) by (type)"
          }
        ]
      }
    ]
  }
}
```

### Backup Procedures

```bash
#!/bin/bash
# backup.sh

# Database backup
mysqldump -h $DB_HOST -u $DB_USER -p$DB_PASSWORD ai_farm > \
    "/backups/db/ai_farm_$(date +%Y%m%d).sql"

# Model checkpoints backup
rsync -av /data/checkpoints/ /backups/checkpoints/

# Compress backups
tar -czf "/backups/ai_farm_full_$(date +%Y%m%d).tar.gz" \
    /backups/db/ /backups/checkpoints/

# Upload to cloud storage
aws s3 cp "/backups/ai_farm_full_$(date +%Y%m%d).tar.gz" \
    s3://ai-farm-backups/

# Cleanup old backups
find /backups -type f -mtime +30 -delete
```

### Scaling Procedures

```bash
# Horizontal scaling
kubectl scale deployment ai-farm-server --replicas=5

# Vertical scaling
kubectl set resources deployment ai-farm-server \
    --requests=cpu=1000m,memory=2Gi \
    --limits=cpu=2000m,memory=4Gi
```

### Maintenance Scripts

```python
# scripts/maintenance.py
import asyncio
from datetime import datetime, timedelta

async def cleanup_old_data():
    """Cleanup old data periodically"""
    while True:
        try:
            # Clean old tasks
            await db.execute("""
                DELETE FROM tasks 
                WHERE status in ('completed', 'failed')
                AND updated_at < NOW() - INTERVAL 30 DAY
            """)
            
            # Clean old logs
            await db.execute("""
                DELETE FROM task_logs
                WHERE created_at < NOW() - INTERVAL 90 DAY
            """)
            
            # Clean old metrics
            await db.execute("""
                DELETE FROM performance_metrics
                WHERE recorded_at < NOW() - INTERVAL 7 DAY
            """)
            
            await asyncio.sleep(86400)  # Run daily
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            await asyncio.sleep(300)  # Retry after 5 minutes
```

## Troubleshooting

### Common Issues

1. Database Connection Issues
```bash
# Check MySQL connection
mysqladmin ping -h $DB_HOST -u $DB_USER -p$DB_PASSWORD

# Check connection pool
kubectl exec -it ai-farm-server-0 -- python -c "
import mysql.connector
pool = mysql.connector.pooling.MySQLConnectionPool(
    pool_name='ai_farm',
    pool_size=5,
    host='$DB_HOST',
    user='$DB_USER',
    password='$DB_PASSWORD'
)
print('Connection successful')
"
```

2. WebSocket Connection Issues
```python
# Test WebSocket connection
import websockets
import asyncio

async def test_connection():
    uri = "ws://localhost:8765"
    async with websockets.connect(uri) as websocket:
        await websocket.send('{"type": "ping"}')
        response = await websocket.recv()
        print(f"Received: {response}")

asyncio.run(test_connection())
```

3. Resource Issues
```bash
# Check node resources
kubectl describe node

# Check pod resources
kubectl top pods

# Check logs
kubectl logs -f deployment/ai-farm-server
```

### Recovery Procedures

```bash
# Database recovery
kubectl exec -it mysql-0 -- mysql -u root -p$MYSQL_ROOT_PASSWORD ai_farm < backup.sql

# Emergency cleanup
kubectl delete pods -l app=ai-farm-server
kubectl delete pvc -l app=ai-farm-server
kubectl apply -f kubernetes/server-deployment.yaml
```

Would you like me to continue with any specific part in more detail or move on to another documentation section?