"""
Main script for Bitcoin RL Trading System
"""
import argparse
import pandas as pd
import numpy as np
import logging
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from data.preprocessor import DataPreprocessor
from data.feature_engineer import FeatureEngineer
from environment.bitcoin_env import BitcoinTradingEnv
from models.dqn_agent import DQNAgent
from training.trainer import Trainer
from integrations.supabase_client import SupabaseEdgeClient
from evaluation.backtester import Backtester
from utils.risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(config: Config, data_path: str, agent_id: str = "DQN-v2.1"):
    """Train RL model"""
    logger.info("="*50)
    logger.info("STARTING TRAINING PIPELINE")
    logger.info("="*50)
    
    # 1. Load and preprocess data
    logger.info("\n[1/6] Loading and preprocessing data...")
    preprocessor = DataPreprocessor(scaling_method='standard')
    df = preprocessor.load_data(data_path)
    df = preprocessor.clean_data(df)
    
    # 2. Engineer features
    logger.info("\n[2/6] Engineering features...")
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)
    
    logger.info(f"Dataset shape: {df_features.shape}")
    logger.info(f"Features: {len(engineer.feature_names)}")
    
    # 3. Normalize data
    logger.info("\n[3/6] Normalizing data...")
    df_normalized = preprocessor.normalize_data(df_features, fit=True)
    
    # 4. Split data for training and evaluation
    logger.info("\n[4/6] Creating environments...")
    total_rows = len(df_normalized)
    lookback = config.environment.lookback_window
    min_window = lookback + 1

    if total_rows < min_window:
        raise ValueError(
            f"Dataset após feature engineering/normalização é muito curto: {total_rows} linhas. "
            f"Necessário pelo menos {min_window} para lookback_window={lookback}."
        )

    # Para evitar episódios com apenas 1 passo, o conjunto de treino precisa ser
    # substancialmente maior que o lookback: len(train) >= 2*lookback + 2
    required_train_rows = 2 * lookback + 2

    # Split padrão 80/20
    train_size = int(total_rows * 0.8)

    # Ajuste para garantir avaliação mínima
    if (total_rows - train_size) < min_window:
        train_size = max(min_window, total_rows - min_window)

    # Garantir tamanho mínimo do treino para episodios com vários passos
    if train_size < required_train_rows:
        if total_rows >= required_train_rows:
            # Concede mais linhas ao treino garantindo avaliação mínima
            train_size = max(required_train_rows, total_rows - min_window)
        else:
            # Dataset muito curto para split; usa todo o dataset para treino e desabilita eval
            logger.warning(
                (
                    f"Dataset ({total_rows} linhas) é insuficiente para split respeitando lookback={lookback}. "
                    f"Usando todo o dataset para treino e desabilitando avaliação. "
                    f"Considere reduzir lookback_window ou usar um dataset maior."
                )
            )
            train_size = total_rows

    df_train = df_normalized.iloc[:train_size]
    df_eval = df_normalized.iloc[train_size:]
    
    # Create environments
    train_env = BitcoinTradingEnv(
        df_train,
        initial_balance=config.environment.initial_balance,
        lookback_window=config.environment.lookback_window,
        transaction_cost=config.environment.transaction_cost,
        max_position_size=config.environment.max_position_size
    )
    
    # Avaliação: criar somente se houver dados suficientes
    eval_env = None
    if len(df_eval) > lookback:
        eval_env = BitcoinTradingEnv(
            df_eval,
            initial_balance=config.environment.initial_balance,
            lookback_window=lookback,
            transaction_cost=config.environment.transaction_cost,
            max_position_size=config.environment.max_position_size
        )
    else:
        logger.warning(
            f"Conjunto de avaliação insuficiente (linhas={len(df_eval)} < lookback_window+1). "
            "A avaliação usará o conjunto de treino."
        )
    
    # 5. Create agent
    logger.info("\n[5/6] Creating DQN agent...")
    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=config.dqn.learning_rate,
        gamma=config.dqn.gamma,
        epsilon_start=config.dqn.epsilon_start,
        epsilon_end=config.dqn.epsilon_end,
        epsilon_decay=config.dqn.epsilon_decay,
        buffer_size=config.dqn.buffer_size,
        batch_size=config.dqn.batch_size,
        target_update_freq=config.dqn.target_update_freq,
        hidden_layers=config.dqn.hidden_layers
    )
    
    # 6. Train agent
    # Supabase client (optional, best-effort). If env vars are missing, training proceeds without persistence.
    supabase_client = None
    try:
        supabase_client = SupabaseEdgeClient()
        logger.info("Supabase client initialized; episodes and trades will be persisted.")
    except Exception as e:
        logger.warning(f"Supabase client not configured: {e}. Proceeding without Supabase persistence.")

    logger.info("\n[6/6] Training agent...")
    trainer = Trainer(
        agent=agent,
        env=train_env,
        eval_env=eval_env,
        total_episodes=config.training.total_episodes,
        eval_frequency=config.training.eval_frequency,
        checkpoint_frequency=config.training.checkpoint_frequency,
        early_stopping_patience=config.training.early_stopping_patience,
        min_episodes_before_stopping=config.training.min_episodes_before_stopping,
        log_dir=config.training.log_dir,
        checkpoint_dir=config.training.checkpoint_dir,
        supabase_client=supabase_client,
        agent_id=agent_id,
        max_steps_per_episode=config.training.max_steps_per_episode,
    )
    
    history = trainer.train()
    
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETED!")
    logger.info("="*50)
    
    return agent, history


def backtest_model(config: Config, data_path: str, model_path: str):
    """Backtest trained model"""
    logger.info("="*50)
    logger.info("STARTING BACKTESTING")
    logger.info("="*50)
    
    # Load and preprocess data
    logger.info("\n[1/4] Loading test data...")
    preprocessor = DataPreprocessor(scaling_method='standard')
    df = preprocessor.load_data(data_path)
    df = preprocessor.clean_data(df)
    
    # Engineer features
    logger.info("\n[2/4] Engineering features...")
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)
    df_normalized = preprocessor.normalize_data(df_features, fit=False)
    
    # Create environment
    logger.info("\n[3/4] Creating test environment...")
    test_env = BitcoinTradingEnv(
        df_normalized,
        initial_balance=config.environment.initial_balance,
        lookback_window=config.environment.lookback_window,
        transaction_cost=config.environment.transaction_cost,
        max_position_size=config.environment.max_position_size
    )
    
    # Load agent
    state_dim = test_env.observation_space.shape[0]
    action_dim = test_env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    agent.load(model_path)
    
    # Run backtest
    logger.info("\n[4/4] Running backtest...")
    backtester = Backtester(
        initial_balance=config.environment.initial_balance,
        transaction_cost=config.environment.transaction_cost
    )
    
    results = backtester.run_backtest(agent, test_env)
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("BACKTEST RESULTS")
    logger.info("="*50)
    
    metrics = results['metrics']
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.4f}")
    
    # Plot results
    try:
        backtester.plot_results('results/backtest_results.png')
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Bitcoin RL Trading System')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'backtest', 'both'],
                       help='Mode: train, backtest, or both')
    parser.add_argument('--data', type=str, default='data/bitcoin_historical.csv',
                       help='Path to data file')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to model file (for backtest)')
    parser.add_argument('--agent-id', type=str, default='DQN-v2.1',
                       help='Identificador do agente para métricas no Supabase')
    
    args = parser.parse_args()
    
    # Load configuration
    config = Config(args.config) if args.config else Config()
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Execute based on mode
    if args.mode in ['train', 'both']:
        agent, history = train_model(config, args.data, agent_id=args.agent_id)
        
        if args.mode == 'both':
            logger.info("\n" + "="*50)
            logger.info("Proceeding to backtesting...")
            logger.info("="*50 + "\n")
    
    if args.mode in ['backtest', 'both']:
        results = backtest_model(config, args.data, args.model)
    
    logger.info("\nAll tasks completed successfully!")


if __name__ == "__main__":
    # Example: python main.py --mode train --data data/bitcoin_historical.csv
    # Example: python main.py --mode backtest --data data/bitcoin_test.csv --model checkpoints/best_model.pth
    # Example: python main.py --mode both --data data/bitcoin_historical.csv
    
    main()
