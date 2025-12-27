import logging.config


LOG_FILE = "aafm.log"


def setup_logging(level: str="DEBUG", log_to_file: bool=True, filename: str=LOG_FILE) -> None:
    """
    Configure logging dynamically.
    :param level: Log level (DEBUG, INFO, WARNING, ERROR)
    :param log_to_file: Boolean to enable/disable file logging
    :param filename: Log file name
    """
    # Base config (console only)
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'},
            'detailed': {'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'},
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout',
            },
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console'],
                'level': level,
                'propagate': True
            },
            'dd.bdd': {
                'level': 'WARNING',
                'propagate': False
            },
        }
    }

    # 2. Only if log_to_file is True, add the file handler and link it
    if log_to_file:
        config['handlers']['file'] = {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': level,
            'formatter': 'detailed',
            'filename': filename,
            'maxBytes': 10485760,
            'backupCount': 5,
            'encoding': 'utf8',
        }
        # Añadimos 'file' a la lista de handlers del root logger
        config['loggers']['']['handlers'].append('file')

    logging.config.dictConfig(config)