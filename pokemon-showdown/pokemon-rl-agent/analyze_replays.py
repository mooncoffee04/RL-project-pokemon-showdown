"""
Replay Analysis Script
Saves replays of wins and losses for detailed analysis
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
    """
    RL player that saves battle information for replay analysis.
    """
    
    def __init__(self, agent: PPOAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        self.battle_log = []
        
    def choose_move(self, battle):
        """Choose action and log it."""
        # Get observation
        obs = self.embed_battle(battle)
        
        # Get action mask
        action_mask = self._get_action_mask(battle)
        
        # Select action (deterministic for evaluation)
        action, log_prob, value = self.agent.select_action(obs, action_mask, deterministic=True)
        
        # Log the decision
        move_info = {
            'turn': battle.turn,
            'action': int(action),
            'value_estimate': float(value),
            'active_pokemon': str(battle.active_pokemon) if battle.active_pokemon else None,
            'opponent_active': str(battle.opponent_active_pokemon) if battle.opponent_active_pokemon else None,
            'available_moves': [str(m) for m in battle.available_moves],
            'available_switches': [str(p) for p in battle.available_switches],
            'team_alive': sum(1 for mon in battle.team.values() if not mon.fainted),
            'opponent_alive': sum(1 for mon in battle.opponent_team.values() if not mon.fainted)
        }
        
        self.battle_log.append(move_info)
        
        # Convert to battle order
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
        """Reset battle log for new battle."""
        self.battle_log = []


async def save_battle_replay(player, battle, replay_dir, battle_num, outcome):
    """
    Save battle replay with metadata and decision log.
    
    Args:
        player: RL player with battle log
        battle: Completed battle object
        replay_dir: Directory to save replays
        battle_num: Battle number
        outcome: 'win' or 'loss'
    """
    try:
        battle_id = battle.battle_tag
        
        # Battle metadata
        metadata = {
            'battle_num': battle_num,
            'outcome': outcome,
            'turns': battle.turn,
            'timestamp': datetime.now().isoformat(),
            'battle_id': battle_id,
            'replay_url': f"http://localhost:8000/{battle_id}",
            'team': [
                {
                    'species': str(mon.species),
                    'fainted': mon.fainted,
                    'hp_fraction': mon.current_hp_fraction
                }
                for mon in battle.team.values()
            ],
            'opponent_team': [
                {
                    'species': str(mon.species),
                    'fainted': mon.fainted,
                    'hp_fraction': mon.current_hp_fraction
                }
                for mon in battle.opponent_team.values()
            ],
            'decision_log': player.battle_log
        }
        
        # Save to file
        filename = f"battle_{battle_num:03d}_{outcome}.json"
        filepath = os.path.join(replay_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  💾 Saved: {filename}")
        print(f"     URL: {metadata['replay_url']}")
        
    except Exception as e:
        print(f"  ⚠️  Failed to save replay: {e}")


async def collect_replays(agent, n_battles=20):
    """
    Play battles and save replays of all outcomes.
    
    Args:
        agent: Trained PPO agent
        n_battles: Number of battles to play
    """
    print("="*70)
    print("COLLECTING BATTLE REPLAYS")
    print("="*70)
    
    # Create directories
    replay_dir = "replays/analysis"
    os.makedirs(replay_dir, exist_ok=True)
    
    # Create players
    rl_player = ReplayRLPlayer(agent=agent, battle_format="gen9randombattle")
    opponent = RandomPlayer(battle_format="gen9randombattle")
    
    print(f"\nPlaying {n_battles} battles...")
    print(f"Saving replays to: {replay_dir}\n")
    
    wins = 0
    losses = 0
    
    # Play battles one at a time
    for i in range(n_battles):
        # Reset log for new battle
        rl_player.reset_log()
        
        # Play one battle
        await rl_player.battle_against(opponent, n_battles=1)
        
        # Get the battle that just finished
        battle = list(rl_player._battles.values())[-1]
        
        # Determine outcome
        if battle.won:
            outcome = "win"
            wins += 1
        else:
            outcome = "loss"
            losses += 1
        
        # Save replay
        print(f"\nBattle {i+1}/{n_battles}: {outcome.upper()}")
        await save_battle_replay(rl_player, battle, replay_dir, i+1, outcome)
    
    # Summary
    print("\n" + "="*70)
    print("REPLAY COLLECTION COMPLETE")
    print("="*70)
    print(f"\nResults:")
    print(f"  Wins: {wins}/{n_battles} ({wins/n_battles:.1%})")
    print(f"  Losses: {losses}/{n_battles} ({losses/n_battles:.1%})")
    print(f"\nReplays saved to: {replay_dir}/")
    print(f"\nLoss replays:")
    
    # List loss files
    for i in range(1, n_battles + 1):
        loss_file = f"battle_{i:03d}_loss.json"
        if os.path.exists(os.path.join(replay_dir, loss_file)):
            print(f"  - {loss_file}")


async def analyze_loss(replay_file):
    """
    Analyze a specific loss replay.
    
    Args:
        replay_file: Path to replay JSON file
    """
    print("="*70)
    print("ANALYZING LOSS REPLAY")
    print("="*70)
    
    with open(replay_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nBattle #{data['battle_num']}")
    print(f"Outcome: {data['outcome']}")
    print(f"Turns: {data['turns']}")
    print(f"Replay URL: {data['replay_url']}")
    
    print("\n" + "-"*70)
    print("FINAL TEAM STATE")
    print("-"*70)
    print("\nYour Team:")
    for mon in data['team']:
        status = "💀 FAINTED" if mon['fainted'] else f"❤️  {mon['hp_fraction']:.1%} HP"
        print(f"  {mon['species']}: {status}")
    
    print("\nOpponent Team:")
    for mon in data['opponent_team']:
        status = "💀 FAINTED" if mon['fainted'] else f"❤️  {mon['hp_fraction']:.1%} HP"
        print(f"  {mon['species']}: {status}")
    
    print("\n" + "-"*70)
    print("DECISION LOG (Last 10 turns)")
    print("-"*70)
    
    # Show last 10 decisions
    log = data['decision_log'][-10:]
    for decision in log:
        print(f"\nTurn {decision['turn']}:")
        print(f"  Active: {decision['active_pokemon']} vs {decision['opponent_active']}")
        print(f"  Action: {decision['action']} (Value estimate: {decision['value_estimate']:.2f})")
        print(f"  Team alive: {decision['team_alive']}/6 | Opponent alive: {decision['opponent_alive']}/6")
        
        if decision['action'] < 4:
            print(f"  → Used move (action {decision['action']})")
        else:
            print(f"  → Switched Pokemon (action {decision['action']})")
    
    print("\n" + "="*70)


async def main():
    """Main function."""
    
    # Load trained agent
    print("📂 Loading trained agent...")
    
    actor = ActorNetwork(obs_dim=10, action_dim=9, hidden_dim=64)
    critic = CriticNetwork(obs_dim=10, hidden_dim=64)
    agent = PPOAgent(actor=actor, critic=critic)
    
    checkpoint_path = "training/checkpoints/ppo_checkpoint_final.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    agent.load(checkpoint_path)
    print(f"✅ Model loaded\n")
    
    # Collect replays
    await collect_replays(agent, n_battles=20)
    
    # Find and analyze first loss
    replay_dir = "replays/analysis"
    for i in range(1, 21):
        loss_file = os.path.join(replay_dir, f"battle_{i:03d}_loss.json")
        if os.path.exists(loss_file):
            print("\n")
            await analyze_loss(loss_file)
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()