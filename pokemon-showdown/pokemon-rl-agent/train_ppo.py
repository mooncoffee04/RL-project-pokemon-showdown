"""
PPO Training Script for Pokemon Showdown - FIXED VERSION
Properly tracks wins and collects rewards after battle completion
"""

import asyncio
import os
import json
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

from poke_env.player import RandomPlayer
from rl_env.simple_rl_player import SimpleRLPlayer
from ppo.networks import ActorNetwork, CriticNetwork
from ppo.ppo_agent import PPOAgent
from ppo.memory import RolloutBuffer


class TrainableRLPlayer(SimpleRLPlayer):
    """
    RL player that uses PPO agent and collects data properly.
    """
    
    def __init__(self, agent: PPOAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.episode_transitions = []  # Store (obs, action, log_prob, value) during episode
        
    def choose_move(self, battle):
        """Choose action using PPO agent."""
        # Get observation
        obs = self.embed_battle(battle)
        
        # Get action mask
        action_mask = self._get_action_mask(battle)
        
        # Select action
        action, log_prob, value = self.agent.select_action(obs, action_mask)
        
        # Store transition (reward will be computed after battle)
        self.episode_transitions.append({
            'obs': obs,
            'action': action,
            'log_prob': log_prob,
            'value': value,
            'battle_tag': battle.battle_tag
        })
        
        # Convert to battle order
        return self._action_to_move(action, battle)
    
    def _get_action_mask(self, battle):
        """Create action mask for valid actions."""
        mask = np.zeros(9, dtype=np.float32)
        
        # Available moves (actions 0-3)
        for i in range(min(len(battle.available_moves), 4)):
            mask[i] = 1.0
        
        # Available switches (actions 4-8)
        for i in range(min(len(battle.available_switches), 5)):
            mask[4 + i] = 1.0
        
        # Ensure at least one action is valid
        if mask.sum() == 0:
            mask[0] = 1.0
        
        return mask
    
    def _action_to_move(self, action, battle):
        """Convert action index to battle order."""
        # Actions 0-3: Use moves
        if action < 4 and action < len(battle.available_moves):
            return self.create_order(battle.available_moves[action])
        
        # Actions 4-8: Switch Pokemon
        switch_idx = action - 4
        if switch_idx >= 0 and switch_idx < len(battle.available_switches):
            return self.create_order(battle.available_switches[switch_idx])
        
        # Fallback
        return self.choose_random_move(battle)
    
    def get_episode_data_with_rewards(self):
        """
        After battle completes, compute rewards for all transitions.
        Returns list of (obs, action, reward, log_prob, value, done) tuples.
        """
        if not self.episode_transitions:
            return []
        
        # Get the completed battle
        if not self._battles:
            return []
        
        battle = list(self._battles.values())[-1]
        
        if not battle.finished:
            return []
        
        # Compute final reward based on battle outcome
        if battle.won:
            final_reward = 30.0
        else:
            final_reward = -30.0
        
        # Create episode data with rewards
        episode_data = []
        num_transitions = len(self.episode_transitions)
        
        for i, trans in enumerate(self.episode_transitions):
            # Give final reward only on last step, small intermediate rewards otherwise
            if i == num_transitions - 1:
                reward = final_reward
                done = True
            else:
                reward = 0.0  # Sparse rewards - only terminal reward matters
                done = False
            
            episode_data.append({
                'obs': trans['obs'],
                'action': trans['action'],
                'reward': reward,
                'log_prob': trans['log_prob'],
                'value': trans['value'],
                'done': done
            })
        
        return episode_data
    
    def reset_episode(self):
        """Reset for new episode."""
        self.episode_transitions = []
        self._prev_battle_state = None


async def play_episode(rl_player, opponent):
    """
    Play one complete battle and return episode data.
    """
    # Reset for new episode
    rl_player.reset_episode()
    
    # Play battle
    await rl_player.battle_against(opponent, n_battles=1)
    
    # Get episode data with computed rewards
    episode_data = rl_player.get_episode_data_with_rewards()
    
    return episode_data


async def collect_rollout(rl_player, opponent, buffer):
    """
    Collect experiences until buffer is full.
    
    Returns:
        Number of episodes, number of wins
    """
    episodes = 0
    wins = 0
    
    while not buffer.is_ready():
        # Play one episode
        episode_data = await play_episode(rl_player, opponent)
        
        if not episode_data:
            continue
        
        episodes += 1
        
        # Check if won (final reward = +30)
        if episode_data[-1]['reward'] > 0:
            wins += 1
        
        # Add to buffer
        for step in episode_data:
            if buffer.is_ready():
                break
            
            buffer.add(
                obs=step['obs'],
                action=step['action'],
                reward=step['reward'],
                log_prob=step['log_prob'],
                value=step['value'],
                done=step['done']
            )
        
        # Finish path for GAE
        last_value = 0.0 if episode_data[-1]['done'] else episode_data[-1]['value']
        buffer.finish_path(last_value)
    
    return episodes, wins


async def main():
    """Main training loop."""
    
    print("="*70)
    print("POKEMON SHOWDOWN PPO TRAINING - FIXED VERSION")
    print("="*70)
    
    # Hyperparameters
    n_total_episodes = 5000
    buffer_size = 2048
    batch_size = 64
    n_epochs = 4
    save_interval = 500
    device = "cpu"
    
    print(f"\nTraining Configuration:")
    print(f"  Total episodes: {n_total_episodes}")
    print(f"  Buffer size: {buffer_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Update epochs: {n_epochs}")
    print(f"  Save interval: {save_interval}")
    
    # Create directories
    os.makedirs("training/checkpoints", exist_ok=True)
    os.makedirs("training/logs", exist_ok=True)
    
    # Create networks
    print("\n📊 Creating neural networks...")
    actor = ActorNetwork(obs_dim=10, action_dim=9, hidden_dim=64)
    critic = CriticNetwork(obs_dim=10, hidden_dim=64)
    
    # Create agent
    print("🤖 Creating PPO agent...")
    agent = PPOAgent(
        actor=actor,
        critic=critic,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        device=device
    )
    
    # Create buffer
    buffer = RolloutBuffer(
        buffer_size=buffer_size,
        obs_dim=10,
        gamma=0.99,
        gae_lambda=0.95,
        device=device
    )
    
    # Create RL player
    print("🎮 Creating RL player...")
    rl_player = TrainableRLPlayer(agent=agent, battle_format="gen9randombattle")
    
    # Create opponents (reuse throughout training)
    opponent_phase1 = RandomPlayer(battle_format="gen9randombattle")
    opponent_phase2 = RandomPlayer(battle_format="gen9randombattle")
    
    # Training stats
    stats = {
        'episodes': [],
        'win_rates': [],
        'policy_losses': [],
        'value_losses': []
    }
    
    total_episodes = 0
    total_wins = 0
    
    print("\n📚 Curriculum Learning:")
    print("  Phase 1 (Episodes 1-200): vs RandomPlayer")
    print("  Phase 2 (Episodes 201+): vs RandomPlayer (harder)")
    
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70 + "\n")
    
    with tqdm(total=n_total_episodes, desc="Training Progress") as pbar:
        while total_episodes < n_total_episodes:
            # Reset buffer
            buffer.reset()
            
            # Select opponent
            if total_episodes < 200:
                opponent = opponent_phase1
                phase = "Phase1"
            else:
                opponent = opponent_phase2
                phase = "Phase2"
            
            # Collect rollout
            episodes, wins = await collect_rollout(rl_player, opponent, buffer)
            
            # Update totals
            total_episodes += episodes
            total_wins += wins
            
            # PPO update
            update_stats = agent.update(buffer, n_epochs=n_epochs, batch_size=batch_size)
            
            # Calculate metrics
            win_rate = wins / episodes if episodes > 0 else 0.0
            overall_win_rate = total_wins / total_episodes if total_episodes > 0 else 0.0
            
            # Store stats
            stats['episodes'].append(total_episodes)
            stats['win_rates'].append(win_rate)
            stats['policy_losses'].append(update_stats['policy_loss'])
            stats['value_losses'].append(update_stats['value_loss'])
            
            # Update progress
            pbar.update(episodes)
            pbar.set_postfix({
                'Phase': phase,
                'WinRate': f"{win_rate:.2%}",
                'OverallWR': f"{overall_win_rate:.2%}",
                'Wins': f"{total_wins}/{total_episodes}"
            })
            
            # Save checkpoint
            if total_episodes % save_interval == 0 and total_episodes > 0:
                checkpoint_path = f"training/checkpoints/ppo_checkpoint_ep{total_episodes}.pt"
                agent.save(checkpoint_path)
    
    # Save final model
    final_path = "training/checkpoints/ppo_checkpoint_final.pt"
    agent.save(final_path)
    
    # Save stats
    stats_path = "training/logs/training_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📈 Final Statistics:")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total wins: {total_wins}")
    print(f"  Overall win rate: {total_wins/total_episodes:.2%}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()