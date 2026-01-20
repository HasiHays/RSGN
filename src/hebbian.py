"""
Hebbian Structural Learning for RSGN.

Implements slow learning rules for:
- Affinity updates (co-activation strengthening)
- Position drift (clustering of co-active nodes)
- Threshold adaptation (maintaining target sparsity)
- Synaptic pruning and sprouting
"""

import torch
import torch.nn as nn
import numpy as np
from collections import deque


class HebbianLearner:
    """
    Manages slow Hebbian-style structural learning for RSG networks.

    Two-timescale learning:
    - FAST: Gradient descent on task loss (handled by optimizer)
    - SLOW: Hebbian structural updates (handled by this class)

    Args:
        model: RSGNetwork instance
        lr_affinity: Learning rate for affinity updates
        lr_position: Learning rate for position drift
        lr_threshold: Learning rate for threshold adaptation
        decay: Weight decay for affinities
        prune_threshold: Threshold below which connections are pruned
        sprout_threshold: Correlation threshold for sprouting new connections
        sparsity_target: Target activation sparsity
        history_size: Size of activation history buffer
    """

    def __init__(
        self,
        model,
        lr_affinity=0.001,
        lr_position=0.0001,
        lr_threshold=0.001,
        decay=0.995,
        prune_threshold=0.01,
        sprout_threshold=0.7,
        sparsity_target=0.1,
        history_size=100
    ):
        self.model = model
        self.lr_affinity = lr_affinity
        self.lr_position = lr_position
        self.lr_threshold = lr_threshold
        self.decay = decay
        self.prune_threshold = prune_threshold
        self.sprout_threshold = sprout_threshold
        self.sparsity_target = sparsity_target

        # Running activation history for correlation computation
        self.activation_history = deque(maxlen=history_size)

        # Statistics tracking
        self.stats = {
            'hebbian_updates': 0,
            'prune_events': 0,
            'sprout_events': 0,
            'avg_correlation': []
        }

    def update(self, alpha, reward=1.0):
        """
        Perform Hebbian structural updates based on activations.

        Updates:
        1. Affinity factors (Hebbian: co-activation strengthens)
        2. Thresholds (adapt to maintain target sparsity)
        3. Positions (drift toward co-active neighbors)

        Args:
            alpha: Activation levels, shape (batch, num_nodes)
            reward: Global reward/modulation signal (e.g., -loss)
        """
        with torch.no_grad():
            device = alpha.device

            # Average activation over batch
            alpha_mean = alpha.mean(dim=0).cpu()

            # Store for correlation computation
            self.activation_history.append(alpha_mean.clone())

            # Compute correlation matrix if enough history
            if len(self.activation_history) >= 10:
                act_stack = torch.stack(list(self.activation_history))
                corr = self._compute_correlation(act_stack)
            else:
                corr = torch.zeros(self.model.num_nodes, self.model.num_nodes)

            # === 1. Affinity Update (Hebbian) ===
            # Co-activation strengthens connections: delta_a = eta * alpha_i * alpha_j * R
            outer = alpha_mean.unsqueeze(1) * alpha_mean.unsqueeze(0)
            delta_affinity = self.lr_affinity * outer * reward

            # Update factorized affinity parameters
            self._update_affinity_factors(delta_affinity, device)

            # === 2. Threshold Adaptation ===
            # Adjust thresholds to maintain target sparsity
            sparsity_error = alpha_mean.to(device) - self.sparsity_target
            self.model.thresholds.data += self.lr_threshold * sparsity_error
            self.model.thresholds.data.clamp_(min=0.01, max=0.99)

            # === 3. Position Drift ===
            # Move nodes toward highly correlated neighbors
            if len(self.activation_history) >= 10:
                self._update_positions(corr, device)

            # Invalidate weight cache after structural changes
            self.model.invalidate_cache()

            self.stats['hebbian_updates'] += 1

    def _compute_correlation(self, act_stack):
        """Compute correlation matrix from activation history."""
        # Standardize
        mean = act_stack.mean(dim=0, keepdim=True)
        std = act_stack.std(dim=0, keepdim=True) + 1e-8
        standardized = (act_stack - mean) / std

        # Correlation matrix
        n = act_stack.shape[0]
        corr = (standardized.T @ standardized) / n
        corr = torch.nan_to_num(corr, 0)

        # Track statistics
        self.stats['avg_correlation'].append(corr.abs().mean().item())

        return corr

    def _update_affinity_factors(self, delta_affinity, device):
        """Update factorized affinity parameters u, v."""
        u = self.model.affinity_u.data
        v = self.model.affinity_v.data

        # Current affinity matrix (before sigmoid)
        current_pre_sigmoid = u @ v.T

        # Target: current + delta, with decay
        target = torch.sigmoid(current_pre_sigmoid) + delta_affinity.to(device)
        target = target * self.decay + torch.sigmoid(current_pre_sigmoid) * (1 - self.decay)

        # Gradient step to move toward target (approximate)
        # Using simple gradient: d(||UV^T - target||^2)/dU = 2(UV^T - target)V
        error = torch.sigmoid(current_pre_sigmoid) - target
        sigmoid_deriv = torch.sigmoid(current_pre_sigmoid) * (1 - torch.sigmoid(current_pre_sigmoid))
        grad_u = (error * sigmoid_deriv) @ v

        self.model.affinity_u.data -= self.lr_affinity * grad_u

    def _update_positions(self, corr, device):
        """Update node positions based on co-activation patterns."""
        pos = self.model.get_positions()
        corr = corr.to(device)

        # Move each node toward positively correlated nodes
        for i in range(self.model.num_nodes):
            # Weight by positive correlations
            weights = corr[i] * (corr[i] > 0.1).float()
            if weights.sum() > 0:
                # Compute weighted centroid direction
                direction = ((pos - pos[i]) * weights.unsqueeze(1)).sum(0)
                direction = direction / (direction.norm() + 1e-6)

                # Update position
                self.model.positions.data[i] += self.lr_position * direction

        # Re-project positions to Poincare ball
        norms = self.model.positions.data.norm(dim=-1, keepdim=True)
        mask = norms >= 1
        self.model.positions.data = torch.where(
            mask,
            self.model.positions.data / (norms + 1e-5) * 0.95,
            self.model.positions.data
        )

    def prune_and_sprout(self):
        """
        Periodic structural plasticity:
        - Prune: Weaken connections with consistently low affinity
        - Sprout: Initialize connections between highly correlated but
                  weakly connected nodes
        """
        with torch.no_grad():
            u = self.model.affinity_u.data
            v = self.model.affinity_v.data
            affinity = torch.sigmoid(u @ v.T)

            # === Pruning ===
            # Push weak affinities further toward zero
            weak_mask = affinity < self.prune_threshold
            if weak_mask.any():
                # Reduce affinity factors for weak connections
                weak_indices = weak_mask.nonzero()
                n_pruned = min(len(weak_indices), 50)  # Limit per step

                for idx in weak_indices[:n_pruned]:
                    i, j = idx[0].item(), idx[1].item()
                    # Reduce correlation between u[i] and v[j]
                    self.model.affinity_u.data[i] *= 0.99
                    self.model.affinity_v.data[j] *= 0.99

                self.stats['prune_events'] += n_pruned

            # === Sprouting ===
            if len(self.activation_history) >= 10:
                act_stack = torch.stack(list(self.activation_history))
                corr = self._compute_correlation(act_stack).to(u.device)

                # Find highly correlated but weakly connected pairs
                sprout_mask = (corr > self.sprout_threshold) & (affinity < 0.1)

                if sprout_mask.any():
                    sprout_indices = sprout_mask.nonzero()
                    n_sprouted = min(len(sprout_indices), 20)

                    for idx in sprout_indices[:n_sprouted]:
                        i, j = idx[0].item(), idx[1].item()
                        # Increase alignment between u[i] and v[j]
                        self.model.affinity_u.data[i] += 0.01 * v[j]
                        self.model.affinity_v.data[j] += 0.01 * u[i]

                    self.stats['sprout_events'] += n_sprouted

            # Invalidate cache
            self.model.invalidate_cache()

    def get_statistics(self):
        """Return learning statistics."""
        return {
            'total_hebbian_updates': self.stats['hebbian_updates'],
            'total_prune_events': self.stats['prune_events'],
            'total_sprout_events': self.stats['sprout_events'],
            'avg_correlation': np.mean(self.stats['avg_correlation'][-100:])
                if self.stats['avg_correlation'] else 0.0
        }

    def reset_statistics(self):
        """Reset learning statistics."""
        self.stats = {
            'hebbian_updates': 0,
            'prune_events': 0,
            'sprout_events': 0,
            'avg_correlation': []
        }


class RewardModulator:
    """
    Implements reward-modulated plasticity (similar to dopamine signaling).

    Recent active connections get eligibility traces that are modulated
    by reward signals.
    """

    def __init__(self, model, trace_decay=0.95):
        self.model = model
        self.trace_decay = trace_decay

        # Eligibility traces for connections
        self.eligibility = torch.zeros(model.num_nodes, model.num_nodes)

    def update_eligibility(self, alpha):
        """
        Update eligibility traces based on co-activation.

        Args:
            alpha: Activation levels, shape (batch, num_nodes)
        """
        with torch.no_grad():
            alpha_mean = alpha.mean(dim=0).cpu()

            # Co-activation creates eligibility
            co_activation = alpha_mean.unsqueeze(1) * alpha_mean.unsqueeze(0)

            # Decay existing traces and add new
            self.eligibility = self.eligibility * self.trace_decay + co_activation

    def apply_reward(self, reward, learning_rate=0.001):
        """
        Apply reward signal to modulate recent connections.

        Args:
            reward: Scalar reward signal
            learning_rate: Learning rate for reward-modulated updates
        """
        with torch.no_grad():
            device = self.model.affinity_u.device

            # Compute update based on eligibility and reward
            delta = learning_rate * reward * self.eligibility.to(device)

            # Apply to affinity factors (approximate)
            u = self.model.affinity_u.data
            v = self.model.affinity_v.data

            # Simple update: adjust affinities in direction of eligibility
            self.model.affinity_u.data += delta.mean(dim=1, keepdim=True).expand_as(u) * 0.1
            self.model.affinity_v.data += delta.mean(dim=0, keepdim=True).T.expand_as(v) * 0.1

            # Invalidate cache
            self.model.invalidate_cache()
