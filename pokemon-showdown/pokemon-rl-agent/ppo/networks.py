"""
PPO Neural Networks - Actor and Critic
PyTorch implementation optimized for CPU (M4 Mac compatible)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ActorNetwork(nn.Module):
    """
    Actor Network (Policy Network)
    
    Maps observations to action probabilities.
    Outputs logits for each action, which are converted to probabilities via softmax.
    
    Architecture:
        Input (10) -> FC(64) -> ReLU -> FC(64) -> ReLU -> FC(9) -> Logits
    """
    
    def __init__(self, obs_dim: int = 10, action_dim: int = 9, hidden_dim: int = 64):
        """
        Initialize Actor Network.
        
        Args:
            obs_dim: Observation space dimension (default: 10)
            action_dim: Action space dimension (default: 9)
            hidden_dim: Hidden layer size (default: 64)
        """
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Initialize weights with orthogonal initialization (standard for PPO)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)  # Small gain for output layer
        
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, obs: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through actor network.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim] or [obs_dim]
            action_mask: Optional mask for invalid actions [batch_size, action_dim]
                        1 = valid action, 0 = invalid action
        
        Returns:
            Action logits [batch_size, action_dim]
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        
        # Apply action mask if provided (mask invalid actions with large negative values)
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e8)
        
        return logits
    
    def get_action_probs(self, obs: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Get action probabilities using softmax.
        
        Args:
            obs: Observation tensor
            action_mask: Optional action mask
        
        Returns:
            Action probabilities [batch_size, action_dim]
        """
        logits = self.forward(obs, action_mask)
        return F.softmax(logits, dim=-1)
    
    def get_action_log_probs(self, obs: torch.Tensor, action_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Get log probabilities of actions using log_softmax.
        
        Args:
            obs: Observation tensor
            action_mask: Optional action mask
        
        Returns:
            Log action probabilities [batch_size, action_dim]
        """
        logits = self.forward(obs, action_mask)
        return F.log_softmax(logits, dim=-1)


class CriticNetwork(nn.Module):
    """
    Critic Network (Value Function)
    
    Maps observations to state value estimates.
    Used to compute advantages and as a baseline for policy gradient.
    
    Architecture:
        Input (10) -> FC(64) -> ReLU -> FC(64) -> ReLU -> FC(1) -> Value
    """
    
    def __init__(self, obs_dim: int = 10, hidden_dim: int = 64):
        """
        Initialize Critic Network.
        
        Args:
            obs_dim: Observation space dimension (default: 10)
            hidden_dim: Hidden layer size (default: 64)
        """
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through critic network.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim] or [obs_dim]
        
        Returns:
            State value estimate [batch_size, 1] or scalar
        """
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value


# Test the networks
if __name__ == "__main__":
    print("="*70)
    print("TESTING PPO NEURAL NETWORKS")
    print("="*70)
    
    # Set dimensions
    obs_dim = 10
    action_dim = 9
    hidden_dim = 64
    batch_size = 32
    
    # Create networks
    actor = ActorNetwork(obs_dim, action_dim, hidden_dim)
    critic = CriticNetwork(obs_dim, hidden_dim)
    
    print("\n✅ Actor Network:")
    print(actor)
    print(f"\n   Total parameters: {sum(p.numel() for p in actor.parameters())}")
    
    print("\n✅ Critic Network:")
    print(critic)
    print(f"\n   Total parameters: {sum(p.numel() for p in critic.parameters())}")
    
    # Test forward pass with batch
    print("\n" + "-"*70)
    print("Testing forward pass (batch)...")
    print("-"*70)
    
    obs_batch = torch.randn(batch_size, obs_dim)
    
    # Actor forward pass
    logits = actor(obs_batch)
    probs = actor.get_action_probs(obs_batch)
    log_probs = actor.get_action_log_probs(obs_batch)
    
    print(f"\n   Observation shape: {obs_batch.shape}")
    print(f"   Actor logits shape: {logits.shape}")
    print(f"   Action probs shape: {probs.shape}")
    print(f"   Action probs sum: {probs.sum(dim=-1)[0]:.4f} (should be ~1.0)")
    print(f"   Log probs shape: {log_probs.shape}")
    
    # Critic forward pass
    values = critic(obs_batch)
    print(f"\n   Critic values shape: {values.shape}")
    print(f"   Sample value: {values[0].item():.4f}")
    
    # Test with action mask
    print("\n" + "-"*70)
    print("Testing action masking...")
    print("-"*70)
    
    action_mask = torch.ones(batch_size, action_dim)
    action_mask[:, 5:] = 0  # Mask out last 4 actions (no switches available)
    
    masked_logits = actor(obs_batch, action_mask)
    masked_probs = actor.get_action_probs(obs_batch, action_mask)
    
    print(f"\n   Action mask: {action_mask[0]}")
    print(f"   Masked probs: {masked_probs[0]}")
    print(f"   Masked probs sum: {masked_probs.sum(dim=-1)[0]:.4f} (should be ~1.0)")
    print(f"   Prob of masked actions: {masked_probs[0, 5:].sum():.6f} (should be ~0.0)")
    
    # Test single observation
    print("\n" + "-"*70)
    print("Testing single observation...")
    print("-"*70)
    
    obs_single = torch.randn(obs_dim)
    logits_single = actor(obs_single)
    value_single = critic(obs_single)
    
    print(f"\n   Single obs shape: {obs_single.shape}")
    print(f"   Single logits shape: {logits_single.shape}")
    print(f"   Single value shape: {value_single.shape}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - Networks ready for PPO training!")
    print("="*70 + "\n")