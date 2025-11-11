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
                 checkpoint_dir: str = "checkpoints",
                 supabase_client=None,
                 agent_id: str = "DQN-v2.1",
                 max_steps_per_episode: int | None = None):
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
        self.max_steps_per_episode = max_steps_per_episode
        
        # Create directories
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Frontend public dir (para consumo direto pelo app web)
        self.public_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "public")
        try:
            os.makedirs(self.public_dir, exist_ok=True)
        except Exception:
            # Se não conseguir criar, mantém apenas logs/checkpoints
            pass
        
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

        # Supabase integration (optional)
        self.supabase = supabase_client
        self.agent_id = agent_id
        
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
        start_time = datetime.utcnow()
        
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

            # Optional cap to speed up episodes
            if self.max_steps_per_episode is not None and steps >= int(self.max_steps_per_episode):
                truncated = True
                done = True
        
        avg_loss = episode_loss / steps if steps > 0 else 0
        duration_seconds = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'reward': episode_reward,
            'loss': avg_loss,
            'steps': steps,
            'portfolio_value': info.get('total_value', 0),
            'profit': info.get('profit', 0),
            'n_trades': info.get('n_trades', 0),
            'duration_seconds': duration_seconds
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

            # Persist episode and trades to Supabase (best-effort)
            if self.supabase is not None:
                try:
                    resp = self.supabase.save_episode(
                        agent_id=self.agent_id,
                        episode_number=episode + 1,
                        total_reward=float(stats['reward']),
                        avg_loss=float(stats['loss']),
                        epsilon=float(self.agent.epsilon),
                        actions_taken=int(stats['steps']),
                        duration_seconds=float(stats.get('duration_seconds', 0.0)),
                    )
                    episode_row = resp.get('episode') or {}
                    episode_id = episode_row.get('id')
                    # Persist trades if the environment exposes them
                    trades = getattr(self.env, 'trades', [])
                    if episode_id is not None and trades:
                        self.supabase.save_trades_batch(
                            agent_id=self.agent_id,
                            episode_id=episode_id,
                            trades=trades,
                            default_confidence=None,
                        )
                except Exception as e:
                    logger.warning(f"Supabase persistence failed for episode {episode + 1}: {e}")

            # Atualiza arquivos consumidos pelo frontend (status e trades) a cada episódio
            try:
                self._save_frontend_status(stats, episode)
                self._save_frontend_trades()
            except Exception as e:
                logger.warning(f"Falha ao salvar arquivos de status/trades no public: {e}")
            
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
        # Espelha também em /public para o frontend consumir como fallback
        try:
            public_history_path = os.path.join(self.public_dir, 'training_history.json')
            with open(public_history_path, 'w') as f:
                json.dump(history_serializable, f, indent=2)
            logger.info(f"History mirrored to {public_history_path}")
        except Exception as e:
            logger.warning(f"Não foi possível espelhar histórico no public: {e}")
    
    def load_history(self, filepath: str):
        """Load training history from file"""
        with open(filepath, 'r') as f:
            self.history = json.load(f)
        logger.info(f"History loaded from {filepath}")

    def _save_frontend_status(self, stats: Dict[str, Any], episode: int):
        """Salva um arquivo simples com o status em tempo real para o frontend.

        Conteúdo: episódio, epsilon, device, recompensa, perda média, passos, valor de portfólio,
        lucro, trades no episódio, timestamp e flags de execução.
        """
        try:
            status = {
                'episode_number': int(episode + 1),
                'epsilon': float(self.agent.epsilon),
                'device': str(getattr(self.agent, 'device', 'cpu')),
                'gpu_available': bool(torch.cuda.is_available()),
                'reward': float(stats.get('reward', 0.0)),
                'loss': float(stats.get('loss', 0.0)),
                'steps': int(stats.get('steps', 0)),
                'portfolio_value': float(stats.get('portfolio_value', 0.0)),
                'profit': float(stats.get('profit', 0.0)),
                'n_trades': int(stats.get('n_trades', 0)),
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'running': True,
                'agent_id': self.agent_id,
            }
            status_path = os.path.join(self.public_dir, 'training_status.json')
            with open(status_path, 'w') as f:
                json.dump(status, f, indent=2)
            logger.info(f"Training status written to {status_path}")
        except Exception as e:
            logger.warning(f"Falha ao escrever training_status.json: {e}")

    def _save_frontend_trades(self):
        """Salva as trades do episódio corrente em /public/trades.json para o frontend.

        O ambiente expõe self.env.trades como lista de dicts; convertemos para um
        formato simples e serializável.
        """
        try:
            trades = getattr(self.env, 'trades', []) or []
            # Normaliza tipos para JSON e adiciona timestamp se não existir
            normalized = []
            now_iso = datetime.utcnow().isoformat() + 'Z'
            for t in trades:
                nt = {
                    'trade_type': str(t.get('action', 'hold')),
                    'price': float(t.get('price', 0.0)),
                    'executed_price': float(t.get('executed_price', t.get('price', 0.0))),
                    'amount': float(t.get('amount', 0.0)),
                    'fee': float(t.get('fee', 0.0)),
                    'revenue': float(t.get('revenue', 0.0)) if t.get('revenue') is not None else None,
                    'profit_loss': float(t.get('profit_loss', 0.0)) if t.get('profit_loss') is not None else None,
                    'step': int(t.get('step', 0)),
                    'timestamp': t.get('timestamp') or now_iso,
                }
                normalized.append(nt)

            trades_path = os.path.join(self.public_dir, 'trades.json')
            with open(trades_path, 'w') as f:
                json.dump(normalized, f, indent=2)
            logger.info(f"Trades written to {trades_path}")
        except Exception as e:
            logger.warning(f"Falha ao escrever trades.json: {e}")


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
