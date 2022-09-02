import os
import environ
import ast

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
env_path = f"{BASE_DIR}/.env"
env = environ.Env()
env.read_env(env_path)

redis_init_params = ast.literal_eval(env.get_value('REDIS_INIT_PARAMS'))

hosts = {
    'hosts_list': ["localhost", "127.0.0.1","*"]
}

DEBUG = False

RQ_QUEUES = {
    'default': {
        'HOST': redis_init_params.get('host'),
        'PORT': redis_init_params.get('port'),
        'DB': 0,
        # 'PASSWORD': 'some-password',
        'DEFAULT_TIMEOUT': 18000,
    }
}

