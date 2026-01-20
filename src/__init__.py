"""
Resonant Sparse Geometry Networks (RSGN)
A Brain-Inspired Architecture with Self-Organizing Sparse Hierarchical Connectivity
"""

from .model import RSGNetwork, RSGClassifier
from .hyperbolic import (
    HyperbolicDistance, poincare_distance, mobius_add,
    project_to_poincare, hyperbolic_log_map, hyperbolic_exp_map
)
from .hebbian import HebbianLearner
from .utils import generate_hierarchical_data

__version__ = "0.1.0"
__all__ = [
    "RSGNetwork",
    "RSGClassifier",
    "HyperbolicDistance",
    "poincare_distance",
    "mobius_add",
    "HebbianLearner",
    "generate_hierarchical_data"
]
