"""
Training loop for RL agents
"""
import numpy as np
import torch
from typing import Dict, Any, Optional
import logging
import os
from tqdm import tqdm
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles training loop for RL agents
    """
    
    def __init__(self,
                 agent,
                 env,
                 eval_env=None,
                 total_episodes: int = 5000,
                 eval_frequency: int = 100,
                 checkpoint_frequency: int = 500,
                 early_stopping_patience: int = 50,
                 min_episodes_before_stopping: int = 1000,
                 log_dir: str = "logs",
                 checkpoint_dir: str = "checkpoints"):
        """
        Args:
            agent: RL agent to train
            env: Training environment
            eval_env: Evaluation environment (optional)
            total_episodes: Total number of episodes to train
            eval_frequency: Frequency of evaluation
            checkpoint_frequency: Frequency of checkpoints
            early_stopping_patience: Patience for early stopping
            min_episodes_before_stopping: Minimum episodes before early stopping
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
        """
        self.agent = agent
        self.env = env
        self.eval_env = eval_env
        self.total_episodes = total_episodes
        self.eval_frequency = eval_frequency
        self.checkpoint_frequency = checkpoint_frequency
        self.early_stopping_patience = early_stopping_patience
        self.min_episodes_before_stopping = min_episodes_before_stopping
        
        # Create directories
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training history
        self.history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'epsilon': [],
            'eval_rewards': [],
            'portfolio_values': []
        }
        
        # Best model tracking
        self.best_reward = -np.inf
        self.episodes_without_improvement = 0
        
        logger.info("Trainer initialized")
    
    def train_episode(self) -> Dict[str, float]:
        """
        Train for one episode
        
        Returns:
            Dictionary with episode statistics
        """
        state, _ = self.env.reset()
        episode_reward = 0
        episode_loss = 0
        steps = 0
        done = False
        
        while not done:
            # Select and perform action
            action = self.agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            loss = self.agent.train()
            
            episode_reward += reward
            episode_loss += loss
            steps += 1
            state = next_state
        
        avg_loss = episode_loss / steps if steps > 0 else 0
        
        return {
            'reward': episode_reward,
            'loss': avg_loss,
            'steps': steps,
            'portfolio_value': info.get('total_value', 0),
            'profit': info.get('profit', 0),
            'n_trades': info.get('n_trades', 0)
        }
    
    def evaluate(self, n_episodes: int = 5) -> Dict[str, float]:
        """
        Evaluate agent
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        eval_env = self.eval_env if self.eval_env is not None else self.env
        
        eval_rewards = []
        eval_profits = []
        eval_trades = []
        
        for _ in range(n_episodes):
            state, _ = eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
            
            eval_rewards.append(episode_reward)
            eval_profits.append(info.get('profit', 0))
            eval_trades.append(info.get('n_trades', 0))
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_profit': np.mean(eval_profits),
            'mean_trades': np.mean(eval_trades)
        }
    
    def train(self) -> Dict[str, list]:
        """
        Main training loop
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.total_episodes} episodes")
        
        pbar = tqdm(range(self.total_episodes), desc="Training")
        
        for episode in pbar:
            # Train one episode
            stats = self.train_episode()
            
            # Log statistics
            self.history['episodes'].append(episode)
            self.history['rewards'].append(stats['reward'])
            self.history['losses'].append(stats['loss'])
            self.history['epsilon'].append(self.agent.epsilon)
            self.history['portfolio_values'].append(stats['portfolio_value'])
            
            # Update progress bar
            pbar.set_postfix({
                'reward': f"{stats['reward']:.2f}",
                'loss': f"{stats['loss']:.4f}",
                'profit': f"${stats['profit']:.2f}",
                'epsilon': f"{self.agent.epsilon:.3f}"
            })
            
            # Evaluation
            if (episode + 1) % self.eval_frequency == 0:
                eval_stats = self.evaluate()
                self.history['eval_rewards'].append(eval_stats['mean_reward'])
                
                logger.info(f"\nEpisode {episode + 1}/{self.total_episodes}")
                logger.info(f"  Eval Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
                logger.info(f"  Eval Profit: ${eval_stats['mean_profit']:.2f}")
                logger.info(f"  Eval Trades: {eval_stats['mean_trades']:.1f}")
                
                # Check for improvement
                if eval_stats['mean_reward'] > self.best_reward:
                    self.best_reward = eval_stats['mean_reward']
                    self.episodes_without_improvement = 0
                    
                    # Save best model
                    best_model_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
                    self.agent.save(best_model_path)
                    logger.info(f"  New best model saved! Reward: {self.best_reward:.2f}")
                else:
                    self.episodes_without_improvement += self.eval_frequency
                
                # Early stopping
                if (episode >= self.min_episodes_before_stopping and 
                    self.episodes_without_improvement >= self.early_stopping_patience):
                    logger.info(f"\nEarly stopping at episode {episode}")
                    logger.info(f"No improvement for {self.episodes_without_improvement} episodes")
                    break
            
            # Checkpoint
            if (episode + 1) % self.checkpoint_frequency == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir, 
                    f'checkpoint_episode_{episode + 1}.pth'
                )
                self.agent.save(checkpoint_path)
                self.save_history()
        
        # Final save
        final_model_path = os.path.join(self.checkpoint_dir, 'final_model.pth')
        self.agent.save(final_model_path)
        self.save_history()
        
        logger.info("\nTraining completed!")
        logger.info(f"Best reward: {self.best_reward:.2f}")
        
        return self.history
    
    def save_history(self):
        """Save training history to file"""
        history_path = os.path.join(self.log_dir, 'training_history.json')
        
        # Convert numpy types to Python types for JSON serialization
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                history_serializable[key] = [
                    float(v) if isinstance(v, (np.floating, np.integer)) else v 
                    for v in value
                ]
            else:
                history_serializable[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=2)
        
        logger.info(f"History saved to {history_path}")
    
    def load_history(self, filepath: str):
        """Load training history from file"""
        with open(filepath, 'r') as f:
            self.history = json.load(f)
        logger.info(f"History loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    # from environment.bitcoin_env import BitcoinTradingEnv
    # from models.dqn_agent import DQNAgent
    # import pandas as pd
    
    # # Load data
    # df = pd.read_csv('data/bitcoin_features.csv')
    
    # # Create environments
    # train_env = BitcoinTradingEnv(df)
    # eval_env = BitcoinTradingEnv(df)
    
    # # Create agent
    # state_dim = train_env.observation_space.shape[0]
    # action_dim = train_env.action_space.n
    # agent = DQNAgent(state_dim, action_dim)
    
    # # Create trainer
    # trainer = Trainer(agent, train_env, eval_env)
    
    # # Train
    # history = trainer.train()
    pass
