import os
import numpy as np
from gymnasium import spaces
from models.dqn_agent import DQNAgent
from models.qrdqn_agent import QRDQNAgent
from stable_baselines3 import PPO, DQN as SB3DQN, SAC, TD3
TimeSeriesCNNExtractor = None
TimeSeriesTransformerExtractor = None


def create_agent(algo: str, env, config):
    if algo == "dqn":
        obs_dim = env.single_observation_space.shape[0] if hasattr(env, "single_observation_space") else env.observation_space.shape[0]
        act_dim = env.single_action_space.n if hasattr(env, "single_action_space") else env.action_space.n
        return DQNAgent(
            state_dim=obs_dim,
            action_dim=act_dim,
            learning_rate=config.dqn.learning_rate,
            gamma=config.dqn.gamma,
            epsilon_start=config.dqn.epsilon_start,
            epsilon_end=config.dqn.epsilon_end,
            epsilon_decay=config.dqn.epsilon_decay,
            buffer_size=config.dqn.buffer_size,
            batch_size=config.dqn.batch_size,
            target_update_freq=config.dqn.target_update_freq,
            hidden_layers=config.dqn.hidden_layers,
        ), "custom"
    if algo == "qrdqn":
        obs_dim = env.single_observation_space.shape[0] if hasattr(env, "single_observation_space") else env.observation_space.shape[0]
        act_dim = env.single_action_space.n if hasattr(env, "single_action_space") else env.action_space.n
        return QRDQNAgent(
            state_dim=obs_dim,
            action_dim=act_dim,
            learning_rate=config.dqn.learning_rate,
            gamma=config.dqn.gamma,
            epsilon_start=config.dqn.epsilon_start,
            epsilon_end=config.dqn.epsilon_end,
            epsilon_decay=config.dqn.epsilon_decay,
            buffer_size=config.dqn.buffer_size,
            batch_size=config.dqn.batch_size,
            target_update_freq=config.dqn.target_update_freq,
            hidden_layers=config.dqn.hidden_layers,
            n_quantiles=51,
        ), "custom"
    if algo == "ppo":
        obs_dim = env.single_observation_space.shape[0] if hasattr(env, "single_observation_space") else env.observation_space.shape[0]
        lookback = int(config.ppo and getattr(config.environment, "lookback_window", 50))
        n_features = max(1, (obs_dim - 3) // lookback)
        policy_kwargs = None
        if getattr(config.ppo, "policy_name", "mlp") in {"ts_cnn", "ts_transformer"}:
            try:
                from models.feature_extractors import TimeSeriesCNNExtractor as _CNN, TimeSeriesTransformerExtractor as _TR
                globals()["TimeSeriesCNNExtractor"] = _CNN
                globals()["TimeSeriesTransformerExtractor"] = _TR
            except Exception:
                pass
            policy_kwargs = {
                "features_extractor_class": (TimeSeriesCNNExtractor if getattr(config.ppo, "policy_name", "mlp") == "ts_cnn" else TimeSeriesTransformerExtractor),
                "features_extractor_kwargs": {
                    "lookback": lookback,
                    "n_features": n_features,
                    "features_dim": 128,
                },
            }
        return PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.ppo.learning_rate,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
            clip_range=config.ppo.clip_range,
            n_steps=config.ppo.n_steps,
            batch_size=config.ppo.batch_size,
            n_epochs=config.ppo.n_epochs,
            ent_coef=config.ppo.ent_coef,
            vf_coef=config.ppo.vf_coef,
            verbose=0,
            device=getattr(config.ppo, "device", "cpu"),
            policy_kwargs=policy_kwargs,
        ), "sb3"
    if algo == "sb3_dqn":
        return SB3DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.dqn.learning_rate,
            gamma=config.dqn.gamma,
            batch_size=config.dqn.batch_size,
            verbose=0,
            device=getattr(config.ppo, "device", "cpu"),
        ), "sb3"
    if algo in {"sac", "td3"}:
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("Algoritmo contínuo requer action space contínuo (Box).")
        if algo == "sac":
            return SAC(
                policy="MlpPolicy",
                env=env,
                learning_rate=config.ppo.learning_rate,
                gamma=config.ppo.gamma,
                verbose=0,
                device=getattr(config.ppo, "device", "cpu"),
            ), "sb3"
        return TD3(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.ppo.learning_rate,
            gamma=config.ppo.gamma,
            verbose=0,
            device=getattr(config.ppo, "device", "cpu"),
        ), "sb3"
    raise ValueError(f"Algoritmo desconhecido: {algo}")


class SB3PolicyAdapter:
    def __init__(self, model):
        self.model = model
        self.epsilon = 0.0
        self.device = str(getattr(model, "device", "cpu"))
    def select_action(self, state, training: bool = False):
        action, _ = self.model.predict(state, deterministic=False)
        if isinstance(action, np.ndarray):
            if action.shape == ():
                return int(action.item())
            return action
        return int(action)
    def save(self, filepath: str):
        self.model.save(filepath)


def load_agent(algo: str, env, config, model_path: str):
    if algo == "dqn":
        obs_dim = env.single_observation_space.shape[0] if hasattr(env, "single_observation_space") else env.observation_space.shape[0]
        act_dim = env.single_action_space.n if hasattr(env, "single_action_space") else env.action_space.n
        agent = DQNAgent(
            state_dim=obs_dim,
            action_dim=act_dim,
            learning_rate=config.dqn.learning_rate,
            gamma=config.dqn.gamma,
            epsilon_start=config.dqn.epsilon_start,
            epsilon_end=config.dqn.epsilon_end,
            epsilon_decay=config.dqn.epsilon_decay,
            buffer_size=config.dqn.buffer_size,
            batch_size=config.dqn.batch_size,
            target_update_freq=config.dqn.target_update_freq,
            hidden_layers=config.dqn.hidden_layers,
        )
        agent.load(model_path)
        return agent
    if algo == "qrdqn":
        obs_dim = env.single_observation_space.shape[0] if hasattr(env, "single_observation_space") else env.observation_space.shape[0]
        act_dim = env.single_action_space.n if hasattr(env, "single_action_space") else env.action_space.n
        agent = QRDQNAgent(
            state_dim=obs_dim,
            action_dim=act_dim,
            learning_rate=config.dqn.learning_rate,
            gamma=config.dqn.gamma,
            epsilon_start=config.dqn.epsilon_start,
            epsilon_end=config.dqn.epsilon_end,
            epsilon_decay=config.dqn.epsilon_decay,
            buffer_size=config.dqn.buffer_size,
            batch_size=config.dqn.batch_size,
            target_update_freq=config.dqn.target_update_freq,
            hidden_layers=config.dqn.hidden_layers,
            n_quantiles=51,
        )
        agent.load(model_path)
        return agent
    if algo == "ppo":
        base = model_path
        if base.endswith(".pth"):
            base = os.path.splitext(base)[0]
        path = base if os.path.exists(base) else (base + ".zip")
        if not os.path.exists(path) and os.path.exists(base + ".zip"):
            path = base + ".zip"
        model = PPO.load(path, env=env)
        return SB3PolicyAdapter(model)
    if algo == "sb3_dqn":
        base = model_path
        if base.endswith(".pth"):
            base = os.path.splitext(base)[0]
        path = base if os.path.exists(base) else (base + ".zip")
        if not os.path.exists(path) and os.path.exists(base + ".zip"):
            path = base + ".zip"
        model = SB3DQN.load(path, env=env)
        return SB3PolicyAdapter(model)
    if algo in {"sac", "td3"}:
        if not isinstance(env.action_space, spaces.Box):
            raise ValueError("Algoritmo contínuo requer action space contínuo (Box).")
        if algo == "sac":
            base = model_path
            if base.endswith(".pth"):
                base = os.path.splitext(base)[0]
            path = base if os.path.exists(base) else (base + ".zip")
            model = SAC.load(path, env=env)
        else:
            base = model_path
            if base.endswith(".pth"):
                base = os.path.splitext(base)[0]
            path = base if os.path.exists(base) else (base + ".zip")
            model = TD3.load(path, env=env)
        return SB3PolicyAdapter(model)
    raise ValueError(f"Algoritmo desconhecido: {algo}")
