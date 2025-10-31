"""
PPO Rollout Memory Buffer
Stores experiences and computes Generalized Advantage Estimation (GAE)
"""

import torch
import numpy as np
from typing import List, Tuple, Generator


class RolloutBuffer:
    """
    Buffer for storing trajectories experienced by the PPO agent.
    
    Stores:
        - observations, actions, rewards, log_probs, values, dones
    
    Computes:
        - returns (discounted rewards)
        - advantages (using GAE - Generalized Advantage Estimation)
    
    Provides mini-batches for PPO training.
    """
    
    def __init__(
        self,
        buffer_size: int,
        obs_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cpu"
    ):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Maximum number of timesteps to store
            obs_dim: Observation space dimension
            gamma: Discount factor for rewards (default: 0.99)
            gae_lambda: GAE lambda parameter (default: 0.95)
            device: Device to store tensors on (default: "cpu")
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        # Storage arrays
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        # Computed during finalize()
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        
        # Tracking
        self.ptr = 0  # Current position in buffer
        self.path_start_idx = 0  # Start of current trajectory
        self.is_full = False
    
    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ):
        """
        Add a single timestep to the buffer.
        
        Args:
            obs: Observation at timestep t
            action: Action taken at timestep t
            reward: Reward received at timestep t
            log_prob: Log probability of action
            value: Value estimate V(s_t)
            done: Whether episode ended
        """
        assert self.ptr < self.buffer_size, "Buffer overflow! Call reset() first."
        
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        
        self.ptr += 1
    
    def finish_path(self, last_value: float = 0.0):
        """
        Call this at the end of a trajectory to compute returns and advantages.
        
        Uses GAE-Lambda for advantage estimation:
            δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
            A_t = δ_t + (γ*λ)*δ_{t+1} + (γ*λ)^2*δ_{t+2} + ...
        
        Args:
            last_value: Value estimate for the final state (0 if terminal, V(s) if truncated)
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        
        # Get trajectory data
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # Compute deltas: δ_t = r_t + γ*V(s_{t+1}) - V(s_t)
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        
        # Compute GAE advantages
        advantages = self._discount_cumsum(deltas, self.gamma * self.gae_lambda)
        
        # Compute returns: R_t = A_t + V(s_t)
        returns = advantages + self.values[path_slice]
        
        # Store computed values
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns
        
        # Update path start for next trajectory
        self.path_start_idx = self.ptr
    
    def _discount_cumsum(self, x: np.ndarray, discount: float) -> np.ndarray:
        """
        Compute discounted cumulative sum.
        
        output[t] = x[t] + discount * x[t+1] + discount^2 * x[t+2] + ...
        
        Args:
            x: Input array
            discount: Discount factor
        
        Returns:
            Discounted cumulative sum
        """
        cumsum = np.zeros_like(x)
        cumsum[-1] = x[-1]
        for t in reversed(range(len(x) - 1)):
            cumsum[t] = x[t] + discount * cumsum[t + 1]
        return cumsum
    
    def get(self) -> Tuple[torch.Tensor, ...]:
        """
        Get all data from buffer and convert to PyTorch tensors.
        
        Also normalizes advantages to have mean=0 and std=1.
        
        Returns:
            Tuple of (observations, actions, log_probs, returns, advantages, values)
        """
        assert self.ptr == self.buffer_size, "Buffer not full! Fill buffer before getting data."
        
        # Normalize advantages (mean=0, std=1)
        adv_mean = np.mean(self.advantages)
        adv_std = np.std(self.advantages)
        self.advantages = (self.advantages - adv_mean) / (adv_std + 1e-8)
        
        # Convert to tensors
        data = (
            torch.tensor(self.observations, dtype=torch.float32, device=self.device),
            torch.tensor(self.actions, dtype=torch.long, device=self.device),
            torch.tensor(self.log_probs, dtype=torch.float32, device=self.device),
            torch.tensor(self.returns, dtype=torch.float32, device=self.device),
            torch.tensor(self.advantages, dtype=torch.float32, device=self.device),
            torch.tensor(self.values, dtype=torch.float32, device=self.device),
        )
        
        return data
    
    def get_batches(self, batch_size: int) -> Generator[Tuple[torch.Tensor, ...], None, None]:
        """
        Generate random mini-batches for training.
        
        Args:
            batch_size: Size of each mini-batch
        
        Yields:
            Mini-batches of (observations, actions, old_log_probs, returns, advantages, old_values)
        """
        # Get all data
        obs, actions, log_probs, returns, advantages, values = self.get()
        
        # Generate random indices
        indices = np.arange(self.buffer_size)
        np.random.shuffle(indices)
        
        # Yield mini-batches
        for start_idx in range(0, self.buffer_size, batch_size):
            end_idx = start_idx + batch_size
            batch_indices = indices[start_idx:end_idx]
            
            yield (
                obs[batch_indices],
                actions[batch_indices],
                log_probs[batch_indices],
                returns[batch_indices],
                advantages[batch_indices],
                values[batch_indices],
            )
    
    def reset(self):
        """Clear the buffer for new data collection."""
        self.ptr = 0
        self.path_start_idx = 0
        self.is_full = False
    
    def size(self) -> int:
        """Return the current number of stored timesteps."""
        return self.ptr
    
    def is_ready(self) -> bool:
        """Check if buffer is full and ready for training."""
        return self.ptr == self.buffer_size


# Test the buffer
if __name__ == "__main__":
    print("="*70)
    print("TESTING PPO ROLLOUT BUFFER")
    print("="*70)
    
    # Parameters
    buffer_size = 100
    obs_dim = 10
    batch_size = 32
    
    # Create buffer
    buffer = RolloutBuffer(
        buffer_size=buffer_size,
        obs_dim=obs_dim,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    print(f"\n✅ Buffer created:")
    print(f"   Buffer size: {buffer_size}")
    print(f"   Obs dimension: {obs_dim}")
    print(f"   Gamma: {buffer.gamma}")
    print(f"   GAE Lambda: {buffer.gae_lambda}")
    
    # Simulate data collection
    print("\n" + "-"*70)
    print("Simulating data collection (2 episodes)...")
    print("-"*70)
    
    episode_count = 0
    
    while not buffer.is_ready():
        # Simulate an episode
        episode_length = np.random.randint(10, 30)
        episode_count += 1
        
        print(f"\n   Episode {episode_count}: {episode_length} steps")
        
        for step in range(episode_length):
            if buffer.is_ready():
                break
            
            # Fake data
            obs = np.random.randn(obs_dim).astype(np.float32)
            action = np.random.randint(0, 9)
            reward = np.random.randn()
            log_prob = np.random.randn()
            value = np.random.randn()
            done = (step == episode_length - 1)
            
            buffer.add(obs, action, reward, log_prob, value, done)
        
        # Finish trajectory
        last_value = 0.0 if done else np.random.randn()
        buffer.finish_path(last_value)
    
    print(f"\n✅ Buffer filled: {buffer.size()}/{buffer.buffer_size} steps")
    
    # Test get() method
    print("\n" + "-"*70)
    print("Testing get() method...")
    print("-"*70)
    
    obs, actions, log_probs, returns, advantages, values = buffer.get()
    
    print(f"\n   Observations shape: {obs.shape}")
    print(f"   Actions shape: {actions.shape}")
    print(f"   Log probs shape: {log_probs.shape}")
    print(f"   Returns shape: {returns.shape}")
    print(f"   Advantages shape: {advantages.shape}")
    print(f"   Values shape: {values.shape}")
    
    print(f"\n   Advantages mean: {advantages.mean().item():.6f} (should be ~0)")
    print(f"   Advantages std: {advantages.std().item():.6f} (should be ~1)")
    
    # Test mini-batch generation
    print("\n" + "-"*70)
    print("Testing mini-batch generation...")
    print("-"*70)
    
    batch_count = 0
    for batch in buffer.get_batches(batch_size):
        batch_count += 1
        obs_batch, actions_batch, log_probs_batch, returns_batch, adv_batch, values_batch = batch
        
        if batch_count == 1:  # Print first batch info
            print(f"\n   Batch {batch_count}:")
            print(f"     Observations: {obs_batch.shape}")
            print(f"     Actions: {actions_batch.shape}")
            print(f"     Returns: {returns_batch.shape}")
    
    print(f"\n   Total batches generated: {batch_count}")
    
    # Test reset
    print("\n" + "-"*70)
    print("Testing reset...")
    print("-"*70)
    
    buffer.reset()
    print(f"\n   Buffer size after reset: {buffer.size()}")
    print(f"   Buffer ready: {buffer.is_ready()}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - Buffer ready for PPO training!")
    print("="*70 + "\n")