import configparser
import random


class ParamConfigurator:
    """Parameter configurator class."""

    def __init__(self):
        config = configparser.ConfigParser()
        config.read('config.ini')

        self.id = str(random.randint(0, 10000))
        self.learning_rate = config['training'].getfloat('learning_rate')
        self.gamma = config['training'].getfloat('gamma')
        self.gae_lambda = config['training'].getfloat('gae_lambda')
        self.batch_size = config['training'].getint('batch_size')
        self.step_count = config['training'].getint('step_count')
        self.n_steps = config['training'].getint('n_steps')
        self.path = None
        self.n_envs = config['training'].getint('n_envs')
        self.map_emb_dim = config['models'].getint('map_emb_dim')
