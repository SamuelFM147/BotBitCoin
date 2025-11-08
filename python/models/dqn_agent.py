"""
Deep Q-Network (DQN) Agent for Bitcoin Trading
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple
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
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 1000,
                 hidden_layers: list = [256, 256, 128],
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
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Networks
        self.policy_net = DQNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()
        
        # Replay buffer
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
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.info(f"Target network updated at step {self.train_step}")
        
        # Decay epsilon
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
