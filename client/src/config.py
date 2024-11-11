import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Base directories
BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / 'src'

# Server Connection
SERVER_URI = os.getenv('SERVER_URI', 'ws://localhost:8765')
RECONNECT_DELAY = int(os.getenv('RECONNECT_DELAY', '5'))
MAX_RECONNECT_DELAY = int(os.getenv('MAX_RECONNECT_DELAY', '60'))
CONNECTION_TIMEOUT = int(os.getenv('CONNECTION_TIMEOUT', '30'))
PING_INTERVAL = int(os.getenv('PING_INTERVAL', '15'))
PING_TIMEOUT = int(os.getenv('PING_TIMEOUT', '10'))

# Resource Management
CONNECTION_LIMIT = int(os.getenv('CONNECTION_LIMIT', '5'))
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
RETRY_INTERVAL = int(os.getenv('RETRY_INTERVAL', '5'))

# Paths Configuration (project yapısına göre düzenlendi)
LOG_DIR = BASE_DIR / 'logs'
SESSION_DIR = BASE_DIR / 'sessions'
CHECKPOINTS_DIR = BASE_DIR / 'checkpoints'
MODEL_CACHE_DIR = BASE_DIR / 'model_cache'
AGGREGATED_MODELS_DIR = BASE_DIR / 'aggregated_models'

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOG_MAX_BYTES = int(os.getenv('LOG_MAX_BYTES', '10485760'))  # 10MB
LOG_BACKUP_COUNT = int(os.getenv('LOG_BACKUP_COUNT', '5'))

# Model Loading
MODEL_LOAD_TIMEOUT = int(os.getenv('MODEL_LOAD_TIMEOUT', '30'))
VERIFY_SSL = os.getenv('VERIFY_SSL', 'False').lower() == 'true'


def setup_logging():
    """Configure logging based on environment variables"""
    os.makedirs(LOG_DIR, exist_ok=True)

    logging.basicConfig(level=getattr(logging, LOG_LEVEL.upper()))
    logger = logging.getLogger("AIFarmClient")

    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "client.log",
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    # Error log handler
    error_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "error.log",
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n%(message)s\n')
    )
    logger.addHandler(error_handler)

    return logger


def init_client_config():
    """Initialize client configuration"""
    # Create required directories
    for directory in [LOG_DIR, SESSION_DIR, CHECKPOINTS_DIR, MODEL_CACHE_DIR, AGGREGATED_MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    return {
        'server': {
            'uri': SERVER_URI,
            'reconnect_delay': RECONNECT_DELAY,
            'max_reconnect_delay': MAX_RECONNECT_DELAY,
            'connection_timeout': CONNECTION_TIMEOUT,
            'ping_interval': PING_INTERVAL,
            'ping_timeout': PING_TIMEOUT,
        },
        'connection': {
            'limit': CONNECTION_LIMIT,
            'max_retries': MAX_RETRIES,
            'retry_interval': RETRY_INTERVAL,
        },
        'paths': {
            'base_dir': str(BASE_DIR),
            'src_dir': str(SRC_DIR),
            'log_dir': str(LOG_DIR),
            'session_dir': str(SESSION_DIR),
            'checkpoints_dir': str(CHECKPOINTS_DIR),
            'model_cache_dir': str(MODEL_CACHE_DIR),
            'aggregated_models_dir': str(AGGREGATED_MODELS_DIR),
        },
        'logging': {
            'level': LOG_LEVEL,
            'format': LOG_FORMAT,
            'max_bytes': LOG_MAX_BYTES,
            'backup_count': LOG_BACKUP_COUNT,
        },
        'model': {
            'load_timeout': MODEL_LOAD_TIMEOUT,
            'verify_ssl': VERIFY_SSL,
        }
    }