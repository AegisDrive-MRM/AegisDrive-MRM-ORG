"""Lightweight paper-reproduction module for world-model-assisted MRM.

The module reproduces the core idea of driving-aware world-model planning:
roll out candidate actions, predict short-horizon risk, and select the action
with the lowest risk-aware cost. It is deliberately lightweight and does not
try to reproduce the full CARLA/RL/MPC training stack.
"""

from .evaluate_candidates import run_paper_world_model

__all__ = ["run_paper_world_model"]
