"""
Deep Q-Network (DQN) Agent for Bitcoin Trading

Enhancements:
- Target network (already present) kept and configurable
- Double DQN update rule to reduce overestimation
- Optional Dueling architecture (Value and Advantage streams)
- Optional Prioritized Experience Replay (PER) with beta annealing
- Flexible epsilon decay (exponential or linear)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience Replay Buffer"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )
    
    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Rank-based Prioritized Experience Replay (simplified).

    Uses priorities to sample important transitions more frequently.
    Maintains compatibility with the standard buffer API: push, sample, len.

    Sampling probabilities: P(i) = p_i^alpha / sum_j p_j^alpha
    Importance sampling weights: w_i = (N * P(i))^-beta / max_w
    """

    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta_start: float = 0.4, beta_frames: int = 100000, eps: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.eps = eps

        self.buffer: List[Tuple] = []
        self.priorities: List[float] = []
        self.pos = 0
        self.frame = 1

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        max_prio = max(self.priorities) if self.priorities else 1.0
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_prio)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        # Beta annealing
        self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.frame / self.beta_frames))
        self.frame += 1

        # Importance sampling weights
        N = len(self.buffer)
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max() + self.eps
        weights = np.array(weights, dtype=np.float32)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8),
            indices,
            weights,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        td_errors = np.abs(td_errors) + self.eps
        for idx, err in zip(indices, td_errors):
            self.priorities[int(idx)] = float(err)


class DQNetwork(nn.Module):
    """Deep Q-Network architecture"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list = [256, 256, 128]):
        super(DQNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class DuelingDQNetwork(nn.Module):
    """Dueling DQN architecture with Value and Advantage streams."""

    def __init__(self, state_dim: int, action_dim: int, hidden_layers: list = [256, 256, 128]):
        super().__init__()
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim

        self.feature_layer = nn.Sequential(*layers)
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_vals


class DQNAgent:
    """DQN Agent for trading"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 epsilon_linear_frames: Optional[int] = None,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000,
                 hidden_layers: list = [256, 256, 128],
                 dueling: bool = True,
                 double_dqn: bool = True,
                 use_per: bool = True,
                 per_alpha: float = 0.6,
                 per_beta_start: float = 0.4,
                 per_beta_frames: int = 100000,
                 device: str = None):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting epsilon for exploration
            epsilon_end: Minimum epsilon
            epsilon_decay: Epsilon decay rate
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
            hidden_layers: List of hidden layer sizes
            device: Device to use ('cuda' or 'cpu')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_linear_frames = epsilon_linear_frames
        self.epsilon_start = epsilon_start
        self.double_dqn = double_dqn
        self.use_per = use_per
        self.per_alpha = per_alpha
        self.per_beta_start = per_beta_start
        self.per_beta_frames = per_beta_frames
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Networks
        NetCls = DuelingDQNetwork if dueling else DQNetwork
        self.policy_net = NetCls(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net = NetCls(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()
        
        # Replay buffer
        if self.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size,
                alpha=self.per_alpha,
                beta_start=self.per_beta_start,
                beta_frames=self.per_beta_frames,
            )
        else:
            self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training step counter
        self.train_step = 0
        
        logger.info(f"DQN Agent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: If True, use epsilon-greedy; otherwise greedy
            
        Returns:
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self) -> float:
        """
        Train the agent on a batch from replay buffer
        
        Returns:
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        if self.use_per:
            states, actions, rewards, next_states, dones, indices, weights = self.replay_buffer.sample(self.batch_size)
            weights_t = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            indices, weights_t = None, None
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values with Double DQN option
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.policy_net(next_states).argmax(dim=1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        td_errors = target_q_values - current_q_values
        if self.use_per:
            # Weighted Huber loss
            loss = (weights_t * torch.nn.functional.smooth_l1_loss(current_q_values, target_q_values, reduction='none')).mean()
        else:
            loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update priorities if PER
        if self.use_per and indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())

        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.info(f"Target network updated at step {self.train_step}")
        
        # Decay epsilon (linear optional)
        if self.epsilon_linear_frames is not None and self.train_step < self.epsilon_linear_frames:
            decay_per_step = (self.epsilon_start - self.epsilon_end) / max(1, self.epsilon_linear_frames)
            self.epsilon = max(self.epsilon_end, self.epsilon - decay_per_step)
        else:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, filepath: str):
        """Save agent to file"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    state_dim = 100
    action_dim = 3
    
    agent = DQNAgent(state_dim, action_dim)
    
    # Simulate training
    # for episode in range(100):
    #     state = env.reset()
    #     done = False
    #     while not done:
    #         action = agent.select_action(state)
    #         next_state, reward, done, _ = env.step(action)
    #         agent.store_transition(state, action, reward, next_state, done)
    #         loss = agent.train()
    #         state = next_state
