import sys
import logging
import logging.handlers

logging.basicConfig(
    level='DEBUG',
    format="%(asctime)s [pid:%(process)d - tid:%(thread)d] %(filename)s:%(lineno)s %(levelname)s  %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler('/var/log/eapl_ws/nlp_pipeline.log', mode='a', maxBytes=1024*1024*25, backupCount=2)
    ])

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': "%(asctime)s.%(msecs)03d [pid:%(process)d - tid:%(thread)d] %(levelname)s %(filename)s:%(lineno)s %(message)s",
            'datefmt': "%d/%b/%Y %H:%M:%S"
        },
        'simple': {
            'format': '%(message)s'
        },
    },
    'handlers': {
        'eapl_ws_req': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/eapl_ws/request.log',
            'formatter': 'verbose',
            'maxBytes': 1024 * 1024 * 25,
            'backupCount': 4,
        },
        'eapl_ws_err': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/eapl_ws/error.log',
            'formatter': 'verbose',
            'maxBytes': 1024 * 1024 * 25,
            'backupCount': 4,
        },

        'console': {
            'class': 'logging.StreamHandler',
            'stream': sys.stdout,
        }
    },
    'loggers': {
        'eapl_ws_req': {
            'handlers': ['eapl_ws_req', 'console'],
            'level': 'DEBUG',
        },
        'eapl_ws_err': {
            'handlers': ['eapl_ws_err', 'console'],
            'level': 'DEBUG',
        },
    }
}

