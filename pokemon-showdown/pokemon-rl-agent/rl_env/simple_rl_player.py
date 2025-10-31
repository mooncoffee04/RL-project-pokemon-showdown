"""
RL Environment Wrapper for Pokemon Showdown Gen 9 Random Battles
Custom Gymnasium wrapper built on top of poke_env.player.Player
"""

import numpy as np
from gymnasium.spaces import Box, Discrete
from poke_env.player import Player
from poke_env.data import GenData


class SimpleRLPlayer(Player):
    """
    Custom RL player for Gen 9 Random Battles with Gymnasium interface.
    
    Observation Space (10 dims):
        - moves_base_power[4]: Base power of available moves (normalized, -1 if unavailable)
        - moves_dmg_multiplier[4]: Type effectiveness multipliers
        - remaining_mon_team: Fraction of remaining Pokemon on our team
        - remaining_mon_opponent: Fraction of remaining Pokemon on opponent's team
    
    Action Space:
        - Discrete(9): 4 moves + 5 switches (max 6 Pokemon, -1 for active)
    
    Reward Function:
        - Win: +30.0
        - Faint opponent Pokemon: +2.0
        - Lose own Pokemon: -2.0
        - HP advantage: +1.0 (scaled by HP difference)
    """
    
    def __init__(self, **kwargs):
        """Initialize the RL player with observation/action spaces."""
        super().__init__(**kwargs)
        
        # Observation space: 10-dimensional vector
        self._observation_space = Box(
            low=np.array([-1, -1, -1, -1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([3, 3, 3, 3, 4, 4, 4, 4, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: 4 moves + 5 switches = 9 discrete actions
        self._action_space = Discrete(9)
        
        # Track current battle and previous state for rewards
        self.current_battle = None
        self._prev_battle_state = None
        
        # Get type chart for Gen 9 (load once during initialization)
        self.gen_data = GenData.from_gen(9)
        self.type_chart = self.gen_data.type_chart
    
    def choose_move(self, battle):
        """
        Override to prevent automatic move selection.
        This will be called by poke-env but we control actions externally.
        """
        # This should not be called during RL training
        # If it is, return a random move
        return self.choose_random_move(battle)
    
    def calc_reward(self, battle) -> float:
        """
        Calculate reward based on battle state changes.
        
        Returns:
            float: Calculated reward
        """
        # Win/loss rewards
        if battle.finished:
            return 30.0 if battle.won else -30.0
        
        # Count fainted Pokemon
        own_fainted = sum(1 for mon in battle.team.values() if mon.fainted)
        opp_fainted = sum(1 for mon in battle.opponent_team.values() if mon.fainted)
        
        # HP-based rewards
        own_hp = sum(mon.current_hp_fraction for mon in battle.team.values() if not mon.fainted)
        opp_hp = sum(mon.current_hp_fraction for mon in battle.opponent_team.values() if not mon.fainted)
        
        # Compare with previous state if available
        if self._prev_battle_state:
            prev_own_fainted, prev_opp_fainted, prev_own_hp, prev_opp_hp = self._prev_battle_state
            
            # Reward for fainting opponent's Pokemon
            opp_fainted_reward = 2.0 * (opp_fainted - prev_opp_fainted)
            
            # Penalty for losing own Pokemon
            own_fainted_penalty = -2.0 * (own_fainted - prev_own_fainted)
            
            # HP advantage reward
            hp_reward = 1.0 * ((own_hp - opp_hp) - (prev_own_hp - prev_opp_hp))
            
            total_reward = opp_fainted_reward + own_fainted_penalty + hp_reward
        else:
            total_reward = 0.0
        
        # Update previous state
        self._prev_battle_state = (own_fainted, opp_fainted, own_hp, opp_hp)
        
        return total_reward
    
    def embed_battle(self, battle) -> np.ndarray:
        """
        Convert battle state into observation vector.
        
        Returns:
            np.ndarray: 10-dimensional observation vector (dtype=float32)
        """
        # Initialize move features with default values
        moves_base_power = -np.ones(4, dtype=np.float32)
        moves_dmg_multiplier = np.ones(4, dtype=np.float32)
        
        # Extract features for each available move
        for i, move in enumerate(battle.available_moves[:4]):  # Max 4 moves
            # Normalize base power to [0, 3] range
            moves_base_power[i] = move.base_power / 100 if move.base_power else 0
            
            # Calculate type effectiveness if move has a type
            if move.type and battle.opponent_active_pokemon:
                target_types = battle.opponent_active_pokemon.types
                if target_types:
                    # Use PokemonType.damage_multiplier method (correct API)
                    multiplier = move.type.damage_multiplier(
                        *target_types,
                        type_chart=self.type_chart
                    )
                    moves_dmg_multiplier[i] = multiplier
        
        # Calculate remaining Pokemon ratios
        remaining_mon_team = (
            len([mon for mon in battle.team.values() if mon.fainted is False]) / 6
        )
        remaining_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted is False]) / 6
        )
        
        # Concatenate all features
        observation = np.concatenate([
            moves_base_power,
            moves_dmg_multiplier,
            [remaining_mon_team, remaining_mon_opponent]
        ]).astype(np.float32)
        
        return observation
    
    @property
    def observation_space(self):
        """Return the observation space."""
        return self._observation_space
    
    @property
    def action_space(self):
        """Return the action space."""
        return self._action_space
    
    def describe_embedding(self) -> None:
        """Print observation space description."""
        print("\n" + "="*70)
        print("OBSERVATION SPACE DESCRIPTION (10 dimensions)")
        print("="*70)
        print("\n[0:4] moves_base_power:")
        print("  - Normalized base power of available moves (0-3 range)")
        print("  - -1 if move slot is empty")
        print("\n[4:8] moves_dmg_multiplier:")
        print("  - Type effectiveness vs opponent (0.25, 0.5, 1, 2, 4)")
        print("\n[8] remaining_mon_team:")
        print("  - Fraction of alive Pokemon on our team (0-1)")
        print("\n[9] remaining_mon_opponent:")
        print("  - Fraction of alive Pokemon on opponent's team (0-1)")
        print("\n" + "="*70)
        print("ACTION SPACE: Discrete(9)")
        print("="*70)
        print("  - Actions 0-3: Use moves 1-4")
        print("  - Actions 4-8: Switch to Pokemon 2-6")
        print("="*70 + "\n")


# Test the environment
if __name__ == "__main__":
    print("Testing SimpleRLPlayer environment...")
    
    env = SimpleRLPlayer(battle_format="gen9randombattle")
    env.describe_embedding()
    
    print("✅ Observation space:", env.observation_space)
    print("✅ Action space:", env.action_space)
    print("\n✅ SimpleRLPlayer environment created successfully!\n")