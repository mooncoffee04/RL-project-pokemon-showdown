"""
Trained vs Untrained Agent Comparison
Compare replays from trained and untrained agents
"""

import asyncio
import os
import json
from datetime import datetime
import numpy as np
import torch

from poke_env.player import RandomPlayer
from rl_env.simple_rl_player import SimpleRLPlayer
from ppo.networks import ActorNetwork, CriticNetwork
from ppo.ppo_agent import PPOAgent


class ReplayRLPlayer(SimpleRLPlayer):
    """RL player that saves battle information."""
    
    def __init__(self, agent: PPOAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.battle_log = []
        
    def choose_move(self, battle):
        """Choose action and log it."""
        obs = self.embed_battle(battle)
        action_mask = self._get_action_mask(battle)
        action, log_prob, value = self.agent.select_action(obs, action_mask, deterministic=True)
        
        # Log decision
        move_info = {
            'turn': battle.turn,
            'action': int(action),
            'value_estimate': float(value),
            'active_pokemon': str(battle.active_pokemon) if battle.active_pokemon else None,
            'opponent_active': str(battle.opponent_active_pokemon) if battle.opponent_active_pokemon else None,
        }
        self.battle_log.append(move_info)
        
        return self._action_to_move(action, battle)
    
    def _get_action_mask(self, battle):
        """Create action mask."""
        mask = np.zeros(9, dtype=np.float32)
        for i in range(min(len(battle.available_moves), 4)):
            mask[i] = 1.0
        for i in range(min(len(battle.available_switches), 5)):
            mask[4 + i] = 1.0
        if mask.sum() == 0:
            mask[0] = 1.0
        return mask
    
    def _action_to_move(self, action, battle):
        """Convert action to move."""
        if action < 4 and action < len(battle.available_moves):
            return self.create_order(battle.available_moves[action])
        switch_idx = action - 4
        if switch_idx >= 0 and switch_idx < len(battle.available_switches):
            return self.create_order(battle.available_switches[switch_idx])
        return self.choose_random_move(battle)
    
    def reset_log(self):
        """Reset battle log."""
        self.battle_log = []


async def save_battle_replay(player, battle, replay_dir, battle_num, outcome):
    """Save battle replay."""
    try:
        battle_id = battle.battle_tag
        
        metadata = {
            'battle_num': battle_num,
            'outcome': outcome,
            'turns': battle.turn,
            'timestamp': datetime.now().isoformat(),
            'battle_id': battle_id,
            'replay_url': f"http://localhost:8000/{battle_id}",
            'decision_log': player.battle_log
        }
        
        filename = f"battle_{battle_num:03d}_{outcome}.json"
        filepath = os.path.join(replay_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  💾 {filename} - {metadata['replay_url']}")
        
    except Exception as e:
        print(f"  ⚠️  Failed to save: {e}")


async def test_agent(agent, agent_type, n_battles=20):
    """
    Test an agent and save replays.
    
    Args:
        agent: PPO agent (trained or untrained)
        agent_type: "trained" or "untrained"
        n_battles: Number of battles
    
    Returns:
        Win rate and replay directory
    """
    print(f"\n{'='*70}")
    print(f"TESTING {agent_type.upper()} AGENT")
    print(f"{'='*70}")
    
    # Create directory
    replay_dir = f"replays/{agent_type}"
    os.makedirs(replay_dir, exist_ok=True)
    
    # Create players
    rl_player = ReplayRLPlayer(agent=agent, battle_format="gen9randombattle")
    opponent = RandomPlayer(battle_format="gen9randombattle")
    
    print(f"\nPlaying {n_battles} battles...")
    
    wins = 0
    losses = 0
    
    for i in range(n_battles):
        rl_player.reset_log()
        await rl_player.battle_against(opponent, n_battles=1)
        
        battle = list(rl_player._battles.values())[-1]
        
        if battle.won:
            outcome = "win"
            wins += 1
        else:
            outcome = "loss"
            losses += 1
        
        print(f"Battle {i+1}/{n_battles}: {outcome.upper()}")
        await save_battle_replay(rl_player, battle, replay_dir, i+1, outcome)
    
    win_rate = wins / n_battles
    
    print(f"\n{'='*70}")
    print(f"{agent_type.upper()} RESULTS")
    print(f"{'='*70}")
    print(f"Wins: {wins}/{n_battles} ({win_rate:.1%})")
    print(f"Losses: {losses}/{n_battles}")
    print(f"Replays: {replay_dir}/")
    
    return win_rate, replay_dir


async def main():
    """Main comparison function."""
    
    print("="*70)
    print("TRAINED VS UNTRAINED AGENT COMPARISON")
    print("="*70)
    
    # ========== TRAINED AGENT ==========
    print("\n📂 Loading TRAINED agent...")
    
    actor_trained = ActorNetwork(obs_dim=10, action_dim=9, hidden_dim=64)
    critic_trained = CriticNetwork(obs_dim=10, hidden_dim=64)
    agent_trained = PPOAgent(actor=actor_trained, critic=critic_trained)
    
    checkpoint_path = "training/checkpoints/ppo_checkpoint_final.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    agent_trained.load(checkpoint_path)
    print("✅ Trained agent loaded (91% win rate model)")
    
    # ========== UNTRAINED AGENT ==========
    print("\n📂 Creating UNTRAINED agent...")
    
    actor_untrained = ActorNetwork(obs_dim=10, action_dim=9, hidden_dim=64)
    critic_untrained = CriticNetwork(obs_dim=10, hidden_dim=64)
    agent_untrained = PPOAgent(actor=actor_untrained, critic=critic_untrained)
    
    print("✅ Untrained agent created (random weights)")
    
    # ========== TEST BOTH AGENTS ==========
    n_battles = 20
    
    trained_wr, trained_dir = await test_agent(agent_trained, "trained", n_battles)
    untrained_wr, untrained_dir = await test_agent(agent_untrained, "untrained", n_battles)
    
    # ========== COMPARISON SUMMARY ==========
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    print(f"\n📊 Win Rates:")
    print(f"  Trained:   {trained_wr:.1%} ⭐")
    print(f"  Untrained: {untrained_wr:.1%}")
    print(f"  Improvement: +{(trained_wr - untrained_wr)*100:.1f}%")
    
    print(f"\n📁 Replay Locations:")
    print(f"  Trained:   {trained_dir}/")
    print(f"  Untrained: {untrained_dir}/")
    
    print("\n💡 What to look at:")
    print("  - Open loss replays from TRAINED agent to see rare mistakes")
    print("  - Open loss replays from UNTRAINED agent to see random play")
    print("  - Compare decision logs to see how training improved strategy")
    
    print("\n🌐 View replays at: http://localhost:8000/<battle-id>")
    print("="*70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()