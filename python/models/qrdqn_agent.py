import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8),
        )
    def __len__(self):
        return len(self.buffer)


class QuantileNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list, n_quantiles: int):
        super().__init__()
        layers = []
        input_dim = state_dim
        for h in hidden_layers:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        self.feature = nn.Sequential(*layers)
        self.out = nn.Linear(input_dim, action_dim * n_quantiles)
        self.action_dim = action_dim
        self.n_quantiles = n_quantiles
    def forward(self, x):
        z = self.feature(x)
        q = self.out(z)
        q = q.view(-1, self.action_dim, self.n_quantiles)
        return q


class QRDQNAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000,
                 hidden_layers: list = [256, 256, 128],
                 n_quantiles: int = 51,
                 device: str | None = None,
                 use_amp: bool = True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_start = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_quantiles = n_quantiles
        self.taus = torch.linspace(0.0, 1.0, steps=n_quantiles + 1)[1:] - 0.5 / n_quantiles
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        try:
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
            if torch.cuda.is_available():
                torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        self.policy_net = QuantileNetwork(state_dim, action_dim, hidden_layers, n_quantiles).to(self.device)
        self.target_net = QuantileNetwork(state_dim, action_dim, hidden_layers, n_quantiles).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.scaler = amp.GradScaler('cuda', enabled=(self.device.type == 'cuda' and use_amp))
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.train_step = 0
        self.decay_enabled = True
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                q = self.policy_net(s).mean(dim=2)
            return int(q.argmax().item())
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    def _quantile_huber_loss(self, td: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
        abs_td = torch.abs(td)
        huber = torch.where(abs_td < 1.0, 0.5 * td.pow(2), abs_td - 0.5)
        tau = taus.view(1, -1, 1)
        weight = torch.abs(tau - (td.detach() < 0.0).float())
        return (weight * huber).mean()
    def train(self) -> float:
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        with amp.autocast('cuda', enabled=self.scaler.is_enabled()):
            q_dist = self.policy_net(states)
        action_mask = actions.view(-1, 1, 1).expand(-1, 1, self.n_quantiles)
        current = q_dist.gather(1, action_mask).squeeze(1)
        with torch.no_grad():
            next_q = self.policy_net(next_states).mean(dim=2)
            next_actions = next_q.argmax(dim=1)
            next_mask = next_actions.view(-1, 1, 1).expand(-1, 1, self.n_quantiles)
            next_dist = self.target_net(next_states).gather(1, next_mask).squeeze(1)
            target = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * self.gamma * next_dist
        td = target.unsqueeze(2) - current.unsqueeze(1)
        taus = self.taus.to(self.device)
        with amp.autocast('cuda', enabled=self.scaler.is_enabled()):
            loss = self._quantile_huber_loss(td, taus)
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        if self.decay_enabled:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())
    def set_decay_enabled(self, enabled: bool):
        self.decay_enabled = bool(enabled)
    def save(self, filepath: str):
        torch.save({
            'policy': self.policy_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
        }, filepath)
    def load(self, filepath: str):
        ck = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(ck['policy'])
        self.target_net.load_state_dict(ck['target'])
        self.optimizer.load_state_dict(ck['optimizer'])
        self.epsilon = ck['epsilon']
        self.train_step = ck['train_step']
