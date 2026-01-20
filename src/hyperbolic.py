"""
Hyperbolic geometry operations for RSGN.
Implements the Poincare ball model of hyperbolic space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def poincare_distance(x, y, eps=1e-5):
    """
    Compute the hyperbolic distance in the Poincare ball model.

    d_H(x, y) = arcosh(1 + 2 * ||x - y||^2 / ((1 - ||x||^2)(1 - ||y||^2)))

    Args:
        x: Tensor of shape (..., d) - points in Poincare ball
        y: Tensor of shape (..., d) - points in Poincare ball
        eps: Small constant for numerical stability

    Returns:
        Tensor of hyperbolic distances
    """
    diff_norm_sq = torch.sum((x - y) ** 2, dim=-1)
    x_norm_sq = torch.sum(x ** 2, dim=-1)
    y_norm_sq = torch.sum(y ** 2, dim=-1)

    numerator = 2 * diff_norm_sq
    denominator = (1 - x_norm_sq) * (1 - y_norm_sq) + eps

    arg = 1 + numerator / denominator
    return torch.acosh(torch.clamp(arg, min=1 + eps))


def mobius_add(x, y, eps=1e-5):
    """
    Mobius addition in the Poincare ball.

    x ⊕ y = ((1 + 2<x,y> + ||y||^2)x + (1 - ||x||^2)y) /
            (1 + 2<x,y> + ||x||^2||y||^2)

    Args:
        x, y: Tensors of shape (..., d)
        eps: Numerical stability constant

    Returns:
        Tensor of shape (..., d) - result of Mobius addition
    """
    x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
    y_norm_sq = torch.sum(y ** 2, dim=-1, keepdim=True)
    xy_dot = torch.sum(x * y, dim=-1, keepdim=True)

    numerator = (1 + 2 * xy_dot + y_norm_sq) * x + (1 - x_norm_sq) * y
    denominator = 1 + 2 * xy_dot + x_norm_sq * y_norm_sq + eps

    return numerator / denominator


def project_to_poincare(x, max_norm=0.99, eps=1e-5):
    """
    Project points to the interior of the Poincare ball.

    Args:
        x: Tensor of shape (..., d)
        max_norm: Maximum allowed norm (< 1)
        eps: Numerical stability

    Returns:
        Projected tensor
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    scale = torch.where(
        norm >= 1,
        max_norm / (norm + eps),
        torch.ones_like(norm)
    )
    return x * scale


def hyperbolic_log_map(x, y, eps=1e-5):
    """
    Logarithmic map in the Poincare ball at point x.
    Maps y to the tangent space at x.

    Args:
        x: Base point, shape (..., d)
        y: Target point, shape (..., d)
        eps: Numerical stability

    Returns:
        Tangent vector at x pointing toward y
    """
    diff = mobius_add(-x, y, eps)
    diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=eps)

    x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
    lambda_x = 2 / (1 - x_norm_sq + eps)

    return (2 / lambda_x) * torch.atanh(diff_norm.clamp(max=1-eps)) * (diff / diff_norm)


def hyperbolic_exp_map(x, v, eps=1e-5):
    """
    Exponential map in the Poincare ball at point x.
    Maps tangent vector v to a point in the ball.

    Args:
        x: Base point, shape (..., d)
        v: Tangent vector, shape (..., d)
        eps: Numerical stability

    Returns:
        Point in Poincare ball
    """
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)
    x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
    lambda_x = 2 / (1 - x_norm_sq + eps)

    direction = v / v_norm
    magnitude = torch.tanh(lambda_x * v_norm / 2) * direction

    return mobius_add(x, magnitude, eps)


class HyperbolicDistance(nn.Module):
    """
    Module for computing pairwise hyperbolic distances in the Poincare ball.
    """

    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        """
        Compute pairwise distances between all points in x and y.

        Args:
            x: Tensor of shape (batch, n, d)
            y: Tensor of shape (batch, m, d)

        Returns:
            Distance matrix of shape (batch, n, m)
        """
        # Expand for broadcasting
        x_exp = x.unsqueeze(2)  # (batch, n, 1, d)
        y_exp = y.unsqueeze(1)  # (batch, 1, m, d)

        diff_norm_sq = torch.sum((x_exp - y_exp) ** 2, dim=-1)
        x_norm_sq = torch.sum(x_exp ** 2, dim=-1)
        y_norm_sq = torch.sum(y_exp ** 2, dim=-1)

        numerator = 2 * diff_norm_sq
        denominator = (1 - x_norm_sq) * (1 - y_norm_sq) + self.eps

        arg = 1 + numerator / denominator
        return torch.acosh(torch.clamp(arg, min=1 + self.eps))


class HyperbolicMLR(nn.Module):
    """
    Hyperbolic Multinomial Logistic Regression.
    For classification in hyperbolic space.
    """

    def __init__(self, dim, num_classes, curvature=1.0):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.curvature = curvature

        # Class prototypes in Poincare ball
        self.prototypes = nn.Parameter(torch.randn(num_classes, dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        """
        Args:
            x: Points in Poincare ball, shape (batch, dim)

        Returns:
            Logits, shape (batch, num_classes)
        """
        prototypes = project_to_poincare(self.prototypes)

        # Compute hyperbolic distances to each prototype
        distances = []
        for i in range(self.num_classes):
            d = poincare_distance(x, prototypes[i].unsqueeze(0))
            distances.append(d)

        distances = torch.stack(distances, dim=-1)  # (batch, num_classes)

        # Convert distances to logits (negative distance)
        logits = -distances + self.bias
        return logits
