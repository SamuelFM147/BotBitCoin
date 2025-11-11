"""
Configuration management for Bitcoin RL Trading System
"""
import os
from dataclasses import dataclass
from typing import Dict, Any
import yaml


@dataclass
class EnvironmentConfig:
    """Trading environment configuration"""
    initial_balance: float = 10000.0
    max_position_size: float = 0.3  # 30% of balance
    transaction_cost: float = 0.001  # 0.1% transaction fee
    reward_scaling: float = 1.0
    lookback_window: int = 50  # Number of past observations
    

@dataclass
class DQNConfig:
    """Deep Q-Network configuration"""
    learning_rate: float = 0.0001
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    buffer_size: int = 100000
    batch_size: int = 64
    target_update_freq: int = 1000
    hidden_layers: list = None
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 256, 128]


@dataclass
class PPOConfig:
    """Proximal Policy Optimization configuration"""
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    ent_coef: float = 0.01
    vf_coef: float = 0.5


@dataclass
class TrainingConfig:
    """Training process configuration"""
    total_episodes: int = 5000
    eval_frequency: int = 100
    checkpoint_frequency: int = 500
    early_stopping_patience: int = 50
    min_episodes_before_stopping: int = 1000
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    # Optional cap to speed up training by limiting steps per episode
    max_steps_per_episode: int | None = None
    

@dataclass
class DataConfig:
    """Data processing configuration"""
    data_source: str = "binance"  # or "file", "ccxt"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    validation_split: float = 0.2
    test_split: float = 0.1
    

class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: str = None):
        self.environment = EnvironmentConfig()
        self.dqn = DQNConfig()
        self.ppo = PPOConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_yaml(config_path)
    
    def load_from_yaml(self, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        if 'environment' in config_dict:
            self.environment = EnvironmentConfig(**config_dict['environment'])
        if 'dqn' in config_dict:
            self.dqn = DQNConfig(**config_dict['dqn'])
        if 'ppo' in config_dict:
            self.ppo = PPOConfig(**config_dict['ppo'])
        if 'training' in config_dict:
            self.training = TrainingConfig(**config_dict['training'])
        if 'data' in config_dict:
            self.data = DataConfig(**config_dict['data'])
    
    def save_to_yaml(self, path: str):
        """Save configuration to YAML file"""
        config_dict = {
            'environment': vars(self.environment),
            'dqn': vars(self.dqn),
            'ppo': vars(self.ppo),
            'training': vars(self.training),
            'data': vars(self.data)
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'environment': vars(self.environment),
            'dqn': vars(self.dqn),
            'ppo': vars(self.ppo),
            'training': vars(self.training),
            'data': vars(self.data)
        }


# Default configuration instance
default_config = Config()
