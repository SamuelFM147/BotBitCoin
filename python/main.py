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
from environment.env_factory import create_env
from models.agent_factory import create_agent, load_agent, SB3PolicyAdapter
from training.trainer import Trainer
from integrations.supabase_client import SupabaseEdgeClient
from evaluation.backtester import Backtester
from utils.risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(config: Config,
                data_path: str,
                agent_id: str = "DQN-v2.1",
                algo: str = "dqn",
                exploration_episodes: int = 50,
                drawdown_max: float | None = 0.06,
                grad_accum_steps: int | None = None,
                updates_per_step: int | None = None,
                vec_envs: int | None = None):
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

    min_eval_rows = lookback + 1
    train_size = int(total_rows * 0.8)
    if (total_rows - train_size) < min_eval_rows:
        train_size = max(required_train_rows, total_rows - min_eval_rows)
    if train_size < required_train_rows:
        if total_rows >= required_train_rows + min_eval_rows:
            train_size = total_rows - min_eval_rows
        else:
            logger.warning(
                (
                    f"Dataset ({total_rows} linhas) insuficiente para treino+avaliação com lookback={lookback}. "
                    f"Usando todo o dataset para treino e desabilitando avaliação."
                )
            )
            train_size = total_rows

    df_train = df_normalized.iloc[:train_size]
    df_eval = df_normalized.iloc[train_size:]
    
    # Create environments
    train_env = create_env(df_train, config.environment) if not vec_envs or int(vec_envs) <= 1 else __import__('environment.env_factory', fromlist=['create_vec_env']).create_vec_env(df_train, config.environment, int(vec_envs))
    
    # Avaliação: criar somente se houver dados suficientes
    eval_env = None
    if len(df_eval) > lookback:
        cfg_eval = config.environment
        cfg_eval.lookback_window = lookback
        eval_env = create_env(df_eval, cfg_eval)
    else:
        logger.warning(
            f"Conjunto de avaliação insuficiente (linhas={len(df_eval)} < lookback_window+1). "
            "A avaliação usará o conjunto de treino."
        )
    
    logger.info("\n[5/6] Creating agent...")
    agent, agent_type = create_agent(algo, train_env, config)
    if grad_accum_steps is not None and agent_type == "custom" and hasattr(agent, 'set_accumulate_steps'):
        try:
            agent.set_accumulate_steps(int(grad_accum_steps))
        except Exception:
            pass
    
    
    # 6. Train agent
    # Supabase client (optional, best-effort). If env vars are missing, training proceeds without persistence.
    supabase_client = None
    try:
        supabase_client = SupabaseEdgeClient()
        logger.info("Supabase client initialized; episodes and trades will be persisted.")
    except Exception as e:
        logger.warning(f"Supabase client not configured: {e}. Proceeding without Supabase persistence.")

    logger.info("\n[6/6] Training agent...")
    history = {}
    if agent_type == "sb3":
        steps_per_episode = config.training.max_steps_per_episode or max(1, len(df_train) - lookback - 1)
        total_timesteps = int(config.training.total_episodes) * int(steps_per_episode)
        agent.learn(total_timesteps=total_timesteps, progress_bar=True)
        os.makedirs(config.training.checkpoint_dir, exist_ok=True)
        agent.save(os.path.join(config.training.checkpoint_dir, 'best_model'))
    else:
        # Auto-redução da exploração quando episódios são muito baixos (<50)
        expl_eps = int(exploration_episodes)
        try:
            total_eps = int(config.training.total_episodes)
            if total_eps < 50:
                expl_eps = max(1, min(expl_eps, total_eps // 2))
        except Exception:
            pass
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
            exploration_episodes=int(expl_eps),
            drawdown_max=float(drawdown_max) if drawdown_max is not None else None,
            updates_per_step=int(updates_per_step) if updates_per_step is not None else 1,
            num_envs=int(vec_envs) if vec_envs is not None else None,
        )
        history = trainer.train()
    
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETED!")
    logger.info("="*50)
    
    return agent, history


def backtest_model(config: Config, data_path: str, model_path: str, algo: str = "dqn"):
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
    df_normalized = preprocessor.normalize_data(df_features, fit=True)
    
    # Create environment
    logger.info("\n[3/4] Creating test environment...")
    test_env = create_env(df_normalized, config.environment)
    
    agent = load_agent(algo, test_env, config, model_path)
    
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
    try:
        import json
        os.makedirs('results', exist_ok=True)
        with open('results/backtest_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info("Backtest metrics written to results/backtest_metrics.json")
    except Exception as e:
        logger.warning(f"Could not write backtest metrics: {e}")
    
    # Plot results
    try:
        backtester.plot_results('results/backtest_results.png')
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")
    
    return results


def backtest_trained_agent(agent, config: Config, data_path: str):
    logger.info("="*50)
    logger.info("STARTING BACKTESTING (IN-MEMORY AGENT)")
    logger.info("="*50)
    preprocessor = DataPreprocessor(scaling_method='standard')
    df = preprocessor.load_data(data_path)
    df = preprocessor.clean_data(df)
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df)
    df_normalized = preprocessor.normalize_data(df_features, fit=True)
    test_env = create_env(df_normalized, config.environment)
    if hasattr(agent, 'predict'):
        agent = SB3PolicyAdapter(agent)
    backtester = Backtester(
        initial_balance=config.environment.initial_balance,
        transaction_cost=config.environment.transaction_cost
    )
    results = backtester.run_backtest(agent, test_env)
    metrics = results['metrics']
    try:
        import json
        os.makedirs('results', exist_ok=True)
        with open('results/backtest_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Bitcoin RL Trading System')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'backtest', 'both', 'validate_xauusd'],
                       help='Mode: train, backtest, or both')
    parser.add_argument('--data', type=str, default='data/bitcoin_historical.csv',
                       help='Path to data file')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to separate test data file for backtesting')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to model file (custom: .pth; SB3: .zip; base name accepted)')
    parser.add_argument('--agent-id', type=str, default='DQN-v2.1',
                       help='Identificador do agente para métricas no Supabase')
    parser.add_argument('--algo', type=str, default='dqn', choices=['dqn', 'qrdqn', 'ppo', 'sac', 'td3', 'sb3_dqn'],
                       help='Algorithm: dqn or qrdqn')
    parser.add_argument('--env', type=str, default=None,
                       help='Environment id: ohlcv_discrete or orderbook_discrete')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Override total training episodes')
    parser.add_argument('--max-steps', type=int, default=None,
                       help='Cap steps per episode')
    parser.add_argument('--eval-freq', type=int, default=None,
                       help='Override eval frequency')
    parser.add_argument('--lookback', type=int, default=None,
                       help='Override environment lookback window')
    parser.add_argument('--exploration-episodes', type=int, default=50,
                       help='Número de episódios de exploração (freeze do epsilon decay)')
    parser.add_argument('--drawdown-max', type=float, default=0.06,
                       help='Regra de risco: drawdown máximo (ex.: 0.06 para 6%) após exploração')
    parser.add_argument('--device', type=str, default=None,
                       help='Dispositivo para modelos SB3 (ex.: cpu, cuda)')
    parser.add_argument('--policy', type=str, default=None,
                       choices=['mlp','ts_cnn','ts_transformer'],
                       help='Policy para PPO/SB3: mlp ou ts_cnn')
    parser.add_argument('--dqn-batch', type=int, default=None,
                       help='Override DQN batch size')
    parser.add_argument('--dqn-hidden', type=str, default=None,
                       help='Override DQN hidden layers, ex.: 512,512,256')
    parser.add_argument('--updates-per-step', type=int, default=None,
                       help='Quantas atualizações de gradiente por passo de ambiente')
    parser.add_argument('--grad-accum-steps', type=int, default=None,
                       help='Acumular N minibatches antes do optimizer (DQN)')
    parser.add_argument('--vec-envs', type=int, default=None,
                       help='Número de ambientes paralelos (Gymnasium SyncVectorEnv)')
    
    args = parser.parse_args()
    
    env_name = (os.getenv('APP_ENV') or '').strip().lower()
    default_config_map = {
        'development': 'config_fast.yaml',
        'dev': 'config_fast.yaml',
        'test': 'config_fast.yaml',
        'testing': 'config_fast.yaml',
        'production': 'config_full.yaml',
        'prod': 'config_full.yaml',
    }
    selected_config = None
    if not args.config:
        selected_config = default_config_map.get(env_name)
        if selected_config and not os.path.isabs(selected_config):
            selected_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), selected_config)
        if not selected_config or not os.path.exists(selected_config):
            selected_config = None
    config = Config(args.config or selected_config) if (args.config or selected_config) else Config()
    if args.episodes is not None:
        config.training.total_episodes = int(args.episodes)
    if args.max_steps is not None:
        config.training.max_steps_per_episode = int(args.max_steps)
    if args.eval_freq is not None:
        config.training.eval_frequency = int(args.eval_freq)
    if args.lookback is not None:
        config.environment.lookback_window = int(args.lookback)
    if args.env is not None:
        config.environment.env_id = str(args.env)
    if args.device is not None:
        config.ppo.device = str(args.device)
    if args.policy is not None:
        config.ppo.policy_name = str(args.policy)
    if args.dqn_batch is not None:
        config.dqn.batch_size = int(args.dqn_batch)
    if args.dqn_hidden is not None:
        try:
            config.dqn.hidden_layers = [int(x) for x in str(args.dqn_hidden).split(',') if x]
        except Exception:
            pass
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Execute based on mode
    if args.mode in ['train', 'both']:
        agent, history = train_model(
            config,
            args.data,
            agent_id=args.agent_id,
            algo=args.algo,
            exploration_episodes=int(args.exploration_episodes),
            drawdown_max=float(args.drawdown_max) if args.drawdown_max is not None else None,
            grad_accum_steps=int(args.grad_accum_steps) if args.grad_accum_steps is not None else None,
            updates_per_step=int(args.updates_per_step) if args.updates_per_step is not None else None,
            vec_envs=int(args.vec_envs) if args.vec_envs is not None else None,
        )
        # updates_per_step já é passado ao Trainer acima
        
        if args.mode == 'both':
            logger.info("\n" + "="*50)
            logger.info("Proceeding to backtesting...")
            logger.info("="*50 + "\n")
            _ = backtest_trained_agent(agent, config, args.data)
    
    if args.mode in ['backtest', 'both']:
        model_arg = args.model
        if args.algo in ['ppo', 'sb3_dqn', 'sac', 'td3'] and model_arg.endswith('.pth'):
            model_arg = os.path.splitext(model_arg)[0]
        data_arg = args.test_data if args.test_data else args.data
        results = backtest_model(config, data_arg, model_arg, algo=args.algo)

    if args.mode == 'validate_xauusd':
        try:
            from evaluation.validation_xauusd import validate_xauusd
            daily_csv = args.data if args.data else os.path.join(os.path.dirname(__file__), 'data', 'xauusd_test.csv')
            hourly_csv = args.test_data
            rep_paths = validate_xauusd(daily_csv=daily_csv, hourly_csv=hourly_csv, initial_cash=config.environment.initial_balance, commission=config.environment.transaction_cost)
            logger.info(f"HTML: {rep_paths['html_report']}")
            logger.info(f"PDF: {rep_paths['pdf_report']}")
        except Exception as e:
            logger.error(f"Validação XAUUSD falhou: {e}")
    
    logger.info("\nAll tasks completed successfully!")


if __name__ == "__main__":
    # Example: python main.py --mode train --data data/bitcoin_historical.csv
    # Example: python main.py --mode backtest --data data/bitcoin_test.csv --model checkpoints/best_model.pth
    # Example: python main.py --mode both --data data/bitcoin_historical.csv
    
    main()
