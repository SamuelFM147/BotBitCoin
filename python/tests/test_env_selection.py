from config.config import Config
import os

def test_config_fast_values_for_dev_test():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "config_fast.yaml")
    c = Config(path)
    assert c.training.total_episodes == 5
    assert c.dqn.batch_size == 16

def test_config_full_values_for_prod():
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "config_full.yaml")
    c = Config(path)
    assert c.training.total_episodes == 2000
    assert c.dqn.batch_size == 64