"""
PPO Agent Evaluation Script - FIXED VERSION
Evaluates trained agent against baseline opponents
"""

import asyncio
import os
import json
from datetime import datetime
import numpy as np
import torch
from tabulate import tabulate

from poke_env.player import RandomPlayer
from rl_env.simple_rl_player import SimpleRLPlayer
from ppo.networks import ActorNetwork, CriticNetwork
from ppo.ppo_agent import PPOAgent


class EvalRLPlayer(SimpleRLPlayer):
    """
    RL player for evaluation (deterministic policy).
    """
    
    def __init__(self, agent: PPOAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent
        
    def choose_move(self, battle):
        """Choose action using PPO agent (deterministic)."""
        # Get observation
        obs = self.embed_battle(battle)
        
        # Get action mask
        action_mask = self._get_action_mask(battle)
        
        # Select action (deterministic)
        action, _, _ = self.agent.select_action(obs, action_mask, deterministic=True)
        
        # Convert to battle order
        return self._action_to_move(action, battle)
    
    def _get_action_mask(self, battle):
        """Create action mask for valid actions."""
        mask = np.zeros(9, dtype=np.float32)
        
        # Available moves
        for i in range(min(len(battle.available_moves), 4)):
            mask[i] = 1.0
        
        # Available switches
        for i in range(min(len(battle.available_switches), 5)):
            mask[4 + i] = 1.0
        
        # Ensure at least one valid action
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


class PPOEvaluator:
    """
    Evaluator for trained PPO agent.
    """
    
    def __init__(self, agent: PPOAgent, eval_player: EvalRLPlayer):
        """
        Initialize evaluator.
        
        Args:
            agent: Trained PPO agent
            eval_player: RL player for evaluation
        """
        self.agent = agent
        self.eval_player = eval_player
    
    async def evaluate_against(self, opponent, opponent_name: str, n_battles: int = 100):
        """
        Evaluate agent against a specific opponent.
        
        Args:
            opponent: Opponent player
            opponent_name: Name of opponent
            n_battles: Number of battles
        
        Returns:
            Dictionary with evaluation results
        """
        print(f"\n{'='*70}")
        print(f"Evaluating against {opponent_name}")
        print(f"{'='*70}")
        
        # Play battles
        await self.eval_player.battle_against(opponent, n_battles=n_battles)
        
        # Get win statistics
        wins = 0
        for battle in self.eval_player.battles.values():
            if battle.won:
                wins += 1
        
        win_rate = wins / n_battles
        
        print(f"\n  Results: {wins}/{n_battles} wins ({win_rate:.2%})")
        
        results = {
            'opponent': opponent_name,
            'n_battles': n_battles,
            'wins': wins,
            'losses': n_battles - wins,
            'win_rate': win_rate
        }
        
        return results
    
    async def cross_evaluate(self, opponents: list, opponent_names: list, n_battles: int = 100):
        """
        Evaluate agent against multiple opponents.
        
        Args:
            opponents: List of opponent players
            opponent_names: Names of opponents
            n_battles: Battles per opponent
        
        Returns:
            Dictionary with all results
        """
        print("\n" + "="*70)
        print("CROSS-EVALUATION")
        print("="*70)
        print(f"\nEvaluating against {len(opponents)} opponents")
        print(f"Battles per opponent: {n_battles}")
        
        all_results = {}
        
        for opponent, name in zip(opponents, opponent_names):
            results = await self.evaluate_against(opponent, name, n_battles)
            all_results[name] = results
            
            # Reset battles for next opponent
            self.eval_player._battles = {}
        
        # Print summary
        print("\n" + "="*70)
        print("EVALUATION SUMMARY")
        print("="*70)
        
        table_data = []
        for name, res in all_results.items():
            table_data.append([
                res['opponent'],
                f"{res['win_rate']*100:.1f}%",
                f"{res['wins']}/{res['n_battles']}"
            ])
        
        headers = ["Opponent", "Win Rate", "W/L"]
        print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
        print()
        
        return all_results


async def main():
    """Main evaluation function."""
    
    print("="*70)
    print("PPO AGENT EVALUATION")
    print("="*70)
    
    # Load trained agent
    print("\n📂 Loading trained agent...")
    
    # Create networks
    actor = ActorNetwork(obs_dim=10, action_dim=9, hidden_dim=64)
    critic = CriticNetwork(obs_dim=10, hidden_dim=64)
    
    # Create agent
    agent = PPOAgent(actor=actor, critic=critic)
    
    # Load checkpoint
    checkpoint_path = "training/checkpoints/ppo_checkpoint_final.pt"
    
    if not os.path.exists(checkpoint_path):
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print("   Please train the agent first")
        return
    
    agent.load(checkpoint_path)
    print(f"✅ Model loaded from: {checkpoint_path}")
    
    # Create RL player
    print("\n🎮 Creating RL player...")
    eval_player = EvalRLPlayer(agent=agent, battle_format="gen9randombattle")
    
    # Create evaluator
    evaluator = PPOEvaluator(agent=agent, eval_player=eval_player)
    
    # Create opponents
    print("\n🤖 Creating opponents...")
    opponents = [
        RandomPlayer(battle_format="gen9randombattle"),
    ]
    opponent_names = ["RandomPlayer"]
    
    # Run cross-evaluation
    results = await evaluator.cross_evaluate(
        opponents=opponents,
        opponent_names=opponent_names,
        n_battles=100
    )
    
    # Save results
    results_dir = "evaluation/results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, "evaluation_results.json")
    
    # Convert to serializable format
    results_serializable = {}
    for name, res in results.items():
        results_serializable[name] = {
            'opponent': res['opponent'],
            'n_battles': res['n_battles'],
            'wins': res['wins'],
            'losses': res['losses'],
            'win_rate': res['win_rate']
        }
    
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()