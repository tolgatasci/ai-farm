import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Logging configuration
LOG_DIR = os.getenv('LOG_DIR', 'logs')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Database configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME', 'ai_farm'),
    'charset': os.getenv('DB_CHARSET', 'utf8mb4'),
    'collation': os.getenv('DB_COLLATION', 'utf8mb4_general_ci'),
    'connection_timeout': int(os.getenv('DB_CONNECTION_TIMEOUT', '10')),
    'buffered': True,
    'consume_results': True,
    'autocommit': True,
    'get_warnings': True,
    'raise_on_warnings': False,
    'pool_reset_session': True
}

# System configuration
MAX_CONNECTIONS = int(os.getenv('MAX_CONNECTIONS', '100'))
# Pool configuration separately
DB_POOL_NAME = os.getenv('DB_POOL_NAME', 'ai_farm_pool')
DB_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '10'))
# Server configuration
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('SERVER_PORT', '8000'))
WEBSOCKET_PORT = int(os.getenv('WEBSOCKET_PORT', '8765'))
API_WORKERS = int(os.getenv('API_WORKERS', '4'))
KEEP_ALIVE_TIMEOUT = int(os.getenv('KEEP_ALIVE_TIMEOUT', '30'))

# Resource management
INACTIVE_THRESHOLD = int(os.getenv('INACTIVE_THRESHOLD', '300'))  # 5 minutes
PING_INTERVAL = int(os.getenv('PING_INTERVAL', '30'))
MAX_CONNECTIONS = int(os.getenv('MAX_CONNECTIONS', '100'))

# Path configurations
STATIC_DIR = os.getenv('STATIC_DIR', 'static')
MODELS_DIR = os.getenv('MODELS_DIR', 'aggregated_models')
CHECKPOINTS_DIR = os.getenv('CHECKPOINTS_DIR', 'checkpoints')

# Cache settings
CACHE_TTL = int(os.getenv('CACHE_TTL', '2'))
CLIENTS_CACHE_TTL = int(os.getenv('CLIENTS_CACHE_TTL', '2'))

# Rate limiting
RATE_LIMIT = os.getenv('RATE_LIMIT', '60/minute')


def setup_logging():
    """Configure logging based on environment variables"""
    os.makedirs(LOG_DIR, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper()),
        format=LOG_FORMAT,
        handlers=[
            logging.handlers.RotatingFileHandler(
                f"{LOG_DIR}/server.log",
                maxBytes=10485760,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler()
        ]
    )


def create_required_directories():
    """Create necessary directories if they don't exist"""
    directories = [LOG_DIR, STATIC_DIR, MODELS_DIR, CHECKPOINTS_DIR]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def init_app_config():
    """Initialize application configuration"""
    setup_logging()
    create_required_directories()
    return {
        'db_config': {**DB_CONFIG,
            'pool_name': DB_POOL_NAME,
            'pool_size': DB_POOL_SIZE},
        'pool_name': os.getenv('DB_POOL_NAME', 'ai_farm_pool'),
        'server_config': {
            'host': SERVER_HOST,
            'port': SERVER_PORT,
            'websocket_port': WEBSOCKET_PORT,
            'workers': API_WORKERS,
            'keep_alive_timeout': KEEP_ALIVE_TIMEOUT,
            'max_connections': MAX_CONNECTIONS
        },
        'websocket_config': {
            'host': SERVER_HOST,
            'port': WEBSOCKET_PORT,
            'ping_interval': PING_INTERVAL,
            'ping_timeout': 30,
            'max_size': None,
        },
        'cache_config': {
            'ttl': CACHE_TTL,
            'clients_ttl': CLIENTS_CACHE_TTL
        },
        'paths': {
            'log_dir': LOG_DIR,
            'static_dir': STATIC_DIR,
            'models_dir': MODELS_DIR,
            'checkpoints_dir': CHECKPOINTS_DIR
        },
        'logging': {
            'log_level': LOG_LEVEL,
            'format': LOG_FORMAT,
            'error_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n%(message)s\n',
            'max_bytes': 10485760,  # 10MB
            'backup_count': 5
        },
        'resource_config': {
            'inactive_threshold': INACTIVE_THRESHOLD,
            'ping_interval': PING_INTERVAL,
            'max_connections': MAX_CONNECTIONS
        },
        'rate_limit': RATE_LIMIT
    }