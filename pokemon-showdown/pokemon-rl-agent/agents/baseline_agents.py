"""
Baseline Agents for Pokemon Showdown RL Training

This module contains simple baseline agents that serve as benchmarks
for our RL agent's performance:
1. RandomPlayer - Chooses actions uniformly at random
2. MaxDamagePlayer - Greedy agent that always picks highest damage move

Author: Pokemon RL Project
Date: October 31, 2025
"""

import numpy as np
from poke_env.player import Player
from poke_env.data import GenData


class RandomPlayer(Player):
    """
    Random Baseline Agent
    
    Selects moves and switches uniformly at random from available actions.
    Expected win rate: ~20-30% in Gen 9 Random Battles.
    
    This agent serves as the lower bound for performance - any intelligent
    agent should easily outperform random play.
    """
    
    def choose_move(self, battle):
        """
        Choose a random legal move from available actions.
        
        Args:
            battle: Current battle state from poke-env
            
        Returns:
            Random move order string
        """
        # Get all available moves and switches
        available_orders = battle.available_moves + battle.available_switches
        
        # If no moves available (forced switch), choose random switch
        if not available_orders:
            available_orders = battle.available_switches
            
        # Select uniformly at random
        return self.choose_random_move(battle)


class MaxDamagePlayer(Player):
    """
    Greedy Max Damage Baseline Agent
    
    Always selects the move that deals maximum expected damage to the
    opponent's active Pokemon. Does not consider:
    - Type advantages for switching
    - Status moves
    - Long-term strategy
    
    Expected win rate: ~40-50% in Gen 9 Random Battles.
    
    This represents a simple but effective strategy that our RL agent
    should surpass by learning defensive switching and status play.
    """
    
    def choose_move(self, battle):
        """
        Choose the move with highest base power against opponent.
        
        Args:
            battle: Current battle state from poke-env
            
        Returns:
            Move order string for highest damage move
        """
        # If we can attack, find the best damaging move
        if battle.available_moves:
            # Calculate expected damage for each move
            best_move = max(
                battle.available_moves,
                key=lambda move: self._calculate_damage(move, battle)
            )
            return self.create_order(best_move)
        
        # If forced to switch, pick the Pokemon with best type matchup
        elif battle.available_switches:
            best_switch = self._choose_best_switch(battle)
            return self.create_order(best_switch)
        
        # Fallback to random if no actions available
        return self.choose_random_move(battle)
    
    def _calculate_damage(self, move, battle):
        """
        Calculate expected damage for a move.
        
        Simplified calculation considering:
        - Base power
        - Type effectiveness
        - STAB bonus
        
        Args:
            move: Move object from poke-env
            battle: Current battle state
            
        Returns:
            float: Expected damage value
        """
        # Base power (moves without damage have power 0)
        if move.base_power == 0:
            return 0
        
        damage = move.base_power
        
        # Get opponent's active Pokemon
        opponent_pokemon = battle.opponent_active_pokemon
        
        if opponent_pokemon:
            # Type effectiveness multiplier
            # Get Gen 9 type chart for damage calculations
            type_chart = GenData.from_gen(9).type_chart
            effectiveness = move.type.damage_multiplier(
                opponent_pokemon.type_1,
                opponent_pokemon.type_2,
                type_chart=type_chart
            )
            damage *= effectiveness
            
            # STAB (Same Type Attack Bonus) - 1.5x if move type matches user
            if move.type in battle.active_pokemon.types:
                damage *= 1.5
        
        return damage
    
    def _choose_best_switch(self, battle):
        """
        Choose the best Pokemon to switch in based on type matchup.
        
        Args:
            battle: Current battle state
            
        Returns:
            Pokemon: Best switch option
        """
        opponent = battle.opponent_active_pokemon
        
        if not opponent:
            # If no opponent info, pick first available switch
            return battle.available_switches[0]
        
        # Evaluate each switch option
        best_switch = None
        best_score = -float('inf')
        
        for pokemon in battle.available_switches:
            # Calculate defensive score (resistance to opponent's types)
            score = 0
            
            # Check resistance to opponent's types
            if opponent.type_1:
                score += self._get_defensive_multiplier(
                    opponent.type_1, pokemon.type_1, pokemon.type_2
                )
            if opponent.type_2:
                score += self._get_defensive_multiplier(
                    opponent.type_2, pokemon.type_1, pokemon.type_2
                )
            
            # Prefer Pokemon with higher HP
            score += pokemon.current_hp_fraction * 0.5
            
            if score > best_score:
                best_score = score
                best_switch = pokemon
        
        return best_switch if best_switch else battle.available_switches[0]
    
    def _get_defensive_multiplier(self, attack_type, def_type1, def_type2):
        """
        Calculate defensive type effectiveness.
        
        Args:
            attack_type: Attacking type
            def_type1: Defender's primary type
            def_type2: Defender's secondary type (can be None)
            
        Returns:
            float: Defensive multiplier (lower is better for defender)
        """
        # Get Gen 9 type chart for damage calculations
        type_chart = GenData.from_gen(9).type_chart
        
        # Get type effectiveness
        multiplier = attack_type.damage_multiplier(def_type1, def_type2, type_chart=type_chart)
        
        # Return inverted score (resistance is good, weakness is bad)
        return 2.0 - multiplier  # 0.5x resist -> 1.5 score, 2x weak -> 0 score


# For easy importing
__all__ = ['RandomPlayer', 'MaxDamagePlayer']