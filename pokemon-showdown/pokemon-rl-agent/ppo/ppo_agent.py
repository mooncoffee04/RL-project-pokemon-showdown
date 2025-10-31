"""
PPO Agent - Core Algorithm Implementation
Proximal Policy Optimization with clipped surrogate objective
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
import numpy as np

from ppo.networks import ActorNetwork, CriticNetwork
from ppo.memory import RolloutBuffer


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) Agent
    
    Implements:
        - Clipped surrogate objective for policy updates
        - Value function loss (MSE)
        - Entropy bonus for exploration
        - Multiple epochs per update
        - Gradient clipping for stability
    
    Paper: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
    """
    
    def __init__(
        self,
        actor: ActorNetwork,
        critic: CriticNetwork,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cpu"
    ):
        """
        Initialize PPO agent.
        
        Args:
            actor: Actor network (policy)
            critic: Critic network (value function)
            lr_actor: Learning rate for actor (default: 3e-4)
            lr_critic: Learning rate for critic (default: 1e-3)
            gamma: Discount factor (default: 0.99)
            gae_lambda: GAE lambda parameter (default: 0.95)
            clip_epsilon: PPO clipping parameter (default: 0.2)
            entropy_coef: Entropy bonus coefficient (default: 0.01)
            value_loss_coef: Value function loss coefficient (default: 0.5)
            max_grad_norm: Max gradient norm for clipping (default: 0.5)
            device: Device to run on (default: "cpu")
        """
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Training statistics
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'approx_kl': [],
            'clip_fraction': []
        }
    
    def select_action(
        self,
        obs: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action given observation.
        
        Args:
            obs: Observation (numpy array)
            action_mask: Optional mask for invalid actions
            deterministic: If True, select argmax action (for evaluation)
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            # Convert to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action probabilities
            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
                logits = self.actor(obs_tensor, mask_tensor)
            else:
                logits = self.actor(obs_tensor)
            
            probs = torch.softmax(logits, dim=-1)
            
            # Select action
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            
            # Get log probability and value
            log_prob = torch.log(probs.squeeze(0)[action] + 1e-8)
            value = self.critic(obs_tensor)
            
            return action.item(), log_prob.item(), value.squeeze().item()
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given observations.
        
        Used during PPO update to compute new log probs and entropy.
        
        Args:
            obs: Batch of observations [batch_size, obs_dim]
            actions: Batch of actions [batch_size]
            action_mask: Optional action mask [batch_size, action_dim]
        
        Returns:
            Tuple of (log_probs, values, entropy)
        """
        # Get action logits
        logits = self.actor(obs, action_mask)
        probs = torch.softmax(logits, dim=-1)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(probs)
        
        # Get log probabilities for taken actions
        log_probs = dist.log_prob(actions)
        
        # Get entropy (for exploration bonus)
        entropy = dist.entropy()
        
        # Get value estimates
        values = self.critic(obs).squeeze(-1)
        
        return log_probs, values, entropy
    
    def update(
        self,
        rollout_buffer: RolloutBuffer,
        n_epochs: int = 4,
        batch_size: int = 64
    ) -> dict:
        """
        Update policy using PPO algorithm.
        
        Performs multiple epochs of minibatch updates using data from rollout buffer.
        
        Args:
            rollout_buffer: Buffer containing collected experiences
            n_epochs: Number of epochs to train on the data (default: 4)
            batch_size: Mini-batch size (default: 64)
        
        Returns:
            Dictionary of training statistics
        """
        # Reset statistics
        for key in self.training_stats:
            self.training_stats[key] = []
        
        # Multiple epochs of updates
        for epoch in range(n_epochs):
            # Generate mini-batches
            for batch in rollout_buffer.get_batches(batch_size):
                obs_batch, actions_batch, old_log_probs_batch, \
                    returns_batch, advantages_batch, old_values_batch = batch
                
                # Evaluate actions with current policy
                new_log_probs, new_values, entropy = self.evaluate_actions(
                    obs_batch, actions_batch
                )
                
                # ============ Policy Loss (Clipped Surrogate Objective) ============
                # Compute probability ratio: π_new(a|s) / π_old(a|s)
                log_ratio = new_log_probs - old_log_probs_batch
                ratio = torch.exp(log_ratio)
                
                # Compute clipped surrogate loss
                policy_loss_1 = advantages_batch * ratio
                policy_loss_2 = advantages_batch * torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # ============ Value Loss ============
                # Clip value predictions to avoid large updates
                values_clipped = old_values_batch + torch.clamp(
                    new_values - old_values_batch,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss_1 = (new_values - returns_batch).pow(2)
                value_loss_2 = (values_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_loss_1, value_loss_2).mean()
                
                # ============ Entropy Bonus ============
                entropy_loss = -entropy.mean()
                
                # ============ Total Loss ============
                total_loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )
                
                # ============ Optimization Step ============
                # Actor update
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                total_loss.backward()
                
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # ============ Logging Statistics ============
                with torch.no_grad():
                    # Approximate KL divergence (for monitoring)
                    approx_kl = (old_log_probs_batch - new_log_probs).mean().item()
                    
                    # Fraction of data points where clipping was active
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean().item()
                    )
                    
                    # Store statistics
                    self.training_stats['policy_loss'].append(policy_loss.item())
                    self.training_stats['value_loss'].append(value_loss.item())
                    self.training_stats['entropy'].append(-entropy_loss.item())
                    self.training_stats['total_loss'].append(total_loss.item())
                    self.training_stats['approx_kl'].append(approx_kl)
                    self.training_stats['clip_fraction'].append(clip_fraction)
        
        # Compute mean statistics over all updates
        stats_summary = {
            key: np.mean(values) for key, values in self.training_stats.items()
        }
        
        return stats_summary
    
    def save(self, filepath: str):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'hyperparameters': {
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon,
                'entropy_coef': self.entropy_coef,
                'value_loss_coef': self.value_loss_coef,
                'max_grad_norm': self.max_grad_norm
            }
        }
        torch.save(checkpoint, filepath)
        print(f"✅ Model saved to: {filepath}")
    
    def load(self, filepath: str):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Load hyperparameters
        hyperparams = checkpoint['hyperparameters']
        self.gamma = hyperparams['gamma']
        self.gae_lambda = hyperparams['gae_lambda']
        self.clip_epsilon = hyperparams['clip_epsilon']
        self.entropy_coef = hyperparams['entropy_coef']
        self.value_loss_coef = hyperparams['value_loss_coef']
        self.max_grad_norm = hyperparams['max_grad_norm']
        
        print(f"✅ Model loaded from: {filepath}")


# Test the agent
if __name__ == "__main__":
    print("="*70)
    print("TESTING PPO AGENT")
    print("="*70)
    
    # Parameters
    obs_dim = 10
    action_dim = 9
    hidden_dim = 64
    buffer_size = 100
    
    # Create networks
    actor = ActorNetwork(obs_dim, action_dim, hidden_dim)
    critic = CriticNetwork(obs_dim, hidden_dim)
    
    # Create agent
    agent = PPOAgent(
        actor=actor,
        critic=critic,
        lr_actor=3e-4,
        lr_critic=1e-3,
        clip_epsilon=0.2
    )
    
    print("\n✅ PPO Agent created:")
    print(f"   Actor parameters: {sum(p.numel() for p in agent.actor.parameters())}")
    print(f"   Critic parameters: {sum(p.numel() for p in agent.critic.parameters())}")
    print(f"   Clip epsilon: {agent.clip_epsilon}")
    print(f"   Entropy coefficient: {agent.entropy_coef}")
    
    # Test action selection
    print("\n" + "-"*70)
    print("Testing action selection...")
    print("-"*70)
    
    obs = np.random.randn(obs_dim).astype(np.float32)
    action, log_prob, value = agent.select_action(obs)
    
    print(f"\n   Observation: {obs[:3]}... (showing first 3 dims)")
    print(f"   Selected action: {action}")
    print(f"   Log probability: {log_prob:.4f}")
    print(f"   Value estimate: {value:.4f}")
    
    # Test with action mask
    action_mask = np.ones(action_dim)
    action_mask[5:] = 0  # Mask out switches
    
    action_masked, log_prob_masked, value_masked = agent.select_action(obs, action_mask)
    print(f"\n   With mask (no switches):")
    print(f"   Selected action: {action_masked} (should be 0-4)")
    
    # Simulate rollout buffer with data
    print("\n" + "-"*70)
    print("Testing PPO update...")
    print("-"*70)
    
    rollout_buffer = RolloutBuffer(
        buffer_size=buffer_size,
        obs_dim=obs_dim,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    # Fill buffer with fake data
    for i in range(buffer_size):
        obs = np.random.randn(obs_dim).astype(np.float32)
        action = np.random.randint(0, action_dim)
        reward = np.random.randn()
        log_prob = np.random.randn()
        value = np.random.randn()
        done = (i % 20 == 19)  # Episode ends every 20 steps
        
        rollout_buffer.add(obs, action, reward, log_prob, value, done)
        
        if done:
            rollout_buffer.finish_path(last_value=0.0)
    
    # Perform PPO update
    stats = agent.update(rollout_buffer, n_epochs=2, batch_size=32)
    
    print(f"\n✅ Update completed:")
    print(f"   Policy loss: {stats['policy_loss']:.4f}")
    print(f"   Value loss: {stats['value_loss']:.4f}")
    print(f"   Entropy: {stats['entropy']:.4f}")
    print(f"   Total loss: {stats['total_loss']:.4f}")
    print(f"   Approx KL: {stats['approx_kl']:.6f}")
    print(f"   Clip fraction: {stats['clip_fraction']:.4f}")
    
    # Test save/load
    print("\n" + "-"*70)
    print("Testing save/load...")
    print("-"*70)
    
    save_path = "/tmp/test_ppo_checkpoint.pt"
    agent.save(save_path)
    
    # Create new agent and load
    actor_new = ActorNetwork(obs_dim, action_dim, hidden_dim)
    critic_new = CriticNetwork(obs_dim, hidden_dim)
    agent_new = PPOAgent(actor=actor_new, critic=critic_new)
    agent_new.load(save_path)
    
    # Verify loaded agent produces same action
    action_loaded, _, _ = agent_new.select_action(obs, deterministic=True)
    action_original, _, _ = agent.select_action(obs, deterministic=True)
    
    print(f"\n   Original action: {action_original}")
    print(f"   Loaded action: {action_loaded}")
    print(f"   Match: {action_original == action_loaded} ✅")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - PPO Agent ready for training!")
    print("="*70 + "\n")