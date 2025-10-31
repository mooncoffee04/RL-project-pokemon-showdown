"""
Agents Package for Pokemon Showdown RL Training

Contains baseline agents and RL agents for Pokemon battles.
"""

from .baseline_agents import RandomPlayer, MaxDamagePlayer

__all__ = ['RandomPlayer', 'MaxDamagePlayer']