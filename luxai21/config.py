import configparser
import random
import yaml
import pathlib

default_config = pathlib.Path(__file__).parent.joinpath('config.yaml')


class ParamConfigurator:
    """Parameter configurator class."""

    def __init__(self):
        with default_config.open("r") as f:
            self.config = yaml.safe_load(f)

        # training
        self.id = str(random.randint(0, 10000))
        self.learning_rate = self.config['training']['learning_rate']
        self.gamma = self.config['training']['gamma']
        self.gae_lambda = self.config['training']['gae_lambda']
        self.batch_size = self.config['training']['batch_size']
        self.step_count = self.config['training']['step_count']
        self.n_steps = self.config['training']['n_steps']
        self.path = None
        self.n_envs = self.config['training']['n_envs']

        # model
        self.map_emb_dim = self.config['models']['map_emb_dim']
        self.net_arch_shared_layers = self.config["models"]["net_arch_shared_layers"]
        self.net_arch_pi = self.config["models"]["net_arch_pi"]
        self.net_arch_vf = self.config["models"]["net_arch_vf"]


if __name__ == '__main__':
    config = ParamConfigurator()
    print(config.net_arch_shared_layers)
