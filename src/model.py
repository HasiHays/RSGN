"""
Resonant Sparse Geometry Network (RSGN) - Core Model Implementation

This module implements the main RSGN architecture with:
- Nodes embedded in hyperbolic space (Poincare ball model)
- Distance-based connectivity
- Input-dependent ignition
- Iterative propagation with soft thresholds
- Local inhibition for winner-take-more dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .hyperbolic import (
    HyperbolicDistance,
    poincare_distance,
    project_to_poincare,
    hyperbolic_log_map
)


class RSGNetwork(nn.Module):
    """
    Resonant Sparse Geometry Network.

    Nodes exist in learned hyperbolic space. Connectivity is determined by
    distance in this space. Activation propagates through the network via
    iterative dynamics with soft thresholds and local inhibition.

    Args:
        num_nodes: Number of computational nodes
        dim_input: Input dimension
        dim_hidden: Hidden state dimension
        dim_space: Dimension of hyperbolic embedding space
        num_steps: Number of propagation steps
        temperature: Soft threshold temperature (anneals during training)
        tau: Distance decay temperature
        sparsity_target: Target activation sparsity
        inhibition_radius: Radius for local inhibition
    """

    def __init__(
        self,
        num_nodes=256,
        dim_input=64,
        dim_hidden=128,
        dim_space=3,
        num_steps=5,
        temperature=1.0,
        tau=1.0,
        sparsity_target=0.1,
        inhibition_radius=0.3
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_space = dim_space
        self.num_steps = num_steps
        self.temperature = temperature
        self.tau = tau
        self.sparsity_target = sparsity_target
        self.inhibition_radius = inhibition_radius

        # Node positions in Poincare ball (slow-learned)
        # Initialize with small norm to stay inside ball
        self.positions = nn.Parameter(
            torch.randn(num_nodes, dim_space) * 0.3
        )

        # Node thresholds (slow-learned)
        self.thresholds = nn.Parameter(
            torch.ones(num_nodes) * 0.5
        )

        # Hierarchical levels (slow-learned)
        self.levels = nn.Parameter(
            torch.linspace(0, 1, num_nodes)
        )

        # Factorized affinity: a_ij = sigmoid(u_i^T v_j)
        affinity_rank = 32
        self.affinity_u = nn.Parameter(
            torch.randn(num_nodes, affinity_rank) * 0.01
        )
        self.affinity_v = nn.Parameter(
            torch.randn(num_nodes, affinity_rank) * 0.01
        )

        # Input embedding to hyperbolic space
        self.input_embed = nn.Sequential(
            nn.Linear(dim_input, dim_space * 4),
            nn.GELU(),
            nn.Linear(dim_space * 4, dim_space),
            nn.Tanh()  # Outputs in (-1, 1)^d
        )
        self.input_scale = nn.Parameter(torch.tensor(0.7))

        # State transformation
        self.W_h = nn.Linear(dim_hidden, dim_hidden)
        self.layer_norm = nn.LayerNorm(dim_hidden)

        # Initial state projection from input
        self.init_state = nn.Linear(dim_input, dim_hidden)

        # Hyperbolic distance module
        self.hyp_dist = HyperbolicDistance()

        # Cache for connection weights
        self._cached_weights = None
        self._cache_valid = False

    def get_positions(self):
        """Return positions projected to Poincare ball interior."""
        return project_to_poincare(self.positions, max_norm=0.95)

    def invalidate_cache(self):
        """Invalidate cached connection weights."""
        self._cache_valid = False
        self._cached_weights = None

    def compute_connection_weights(self, use_cache=True):
        """
        Compute N x N connection weight matrix.

        w_ij = sigma(a_ij) * exp(-d_H(p_i, p_j) / tau) * phi(l_i - l_j)

        Returns:
            Weight matrix of shape (num_nodes, num_nodes)
        """
        if use_cache and self._cache_valid and self._cached_weights is not None:
            return self._cached_weights

        pos = self.get_positions()  # (N, D)

        # Compute pairwise hyperbolic distances
        pos_batch = pos.unsqueeze(0)  # (1, N, D)
        dist = self.hyp_dist(pos_batch, pos_batch)[0]  # (N, N)

        # Distance-based weights: exp(-d / tau)
        dist_weight = torch.exp(-dist / self.tau)

        # Affinity weights: sigma(u^T v)
        affinity = torch.sigmoid(self.affinity_u @ self.affinity_v.T)

        # Level factor: softplus(l_i - l_j + 1) favors upward flow
        level_diff = self.levels.unsqueeze(1) - self.levels.unsqueeze(0)
        level_factor = F.softplus(level_diff + 1)

        # Combined weight
        W = affinity * dist_weight * level_factor

        # Zero out self-connections
        W = W * (1 - torch.eye(self.num_nodes, device=W.device))

        # Cache the result
        self._cached_weights = W
        self._cache_valid = True

        return W

    def soft_threshold(self, x, theta):
        """
        Differentiable soft threshold activation.

        SoftThresh(x, theta, T) = sigma((x - theta) / T)

        Args:
            x: Input values
            theta: Threshold values

        Returns:
            Soft-thresholded activations in [0, 1]
        """
        return torch.sigmoid((x - theta) / self.temperature)

    def ignite(self, x):
        """
        Create initial activation from input via input-dependent ignition.

        Input tokens are embedded as "spark points" in hyperbolic space.
        Nodes near these sparks become initially active.

        Args:
            x: Input tensor of shape (batch, seq_len, dim_input)

        Returns:
            Initial activations of shape (batch, num_nodes)
        """
        batch_size, seq_len, _ = x.shape

        # Embed input tokens to hyperbolic space
        sparks = self.input_embed(x) * self.input_scale  # (B, T, D_space)
        sparks = project_to_poincare(sparks, max_norm=0.9)

        # Get node positions
        pos = self.get_positions()  # (N, D_space)

        # Compute distances from each node to nearest spark
        sparks_exp = sparks.unsqueeze(2)  # (B, T, 1, D)
        pos_exp = pos.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)

        # Use Euclidean distance approximation for efficiency
        # (Hyperbolic can be used but is slower)
        dist_sq = ((sparks_exp - pos_exp) ** 2).sum(-1)  # (B, T, N)
        min_dist_sq = dist_sq.min(dim=1)[0]  # (B, N)

        # Gaussian activation field
        sigma_ign = 0.4
        activation = torch.exp(-min_dist_sq / (2 * sigma_ign ** 2))

        return activation

    def propagate(self, h, alpha, W):
        """
        One step of activation propagation through the network.

        Active nodes send signals to neighbors weighted by connection strength.
        Receiving nodes update their activation via soft thresholding.

        Args:
            h: Hidden states, shape (batch, num_nodes, dim_hidden)
            alpha: Activation levels, shape (batch, num_nodes)
            W: Connection weights, shape (num_nodes, num_nodes)

        Returns:
            Updated (h, alpha)
        """
        batch_size = h.shape[0]

        # Soft activity mask
        active_mask = (alpha > 0.01).float()

        # Weighted states: scale hidden states by activation level
        h_weighted = h * alpha.unsqueeze(-1) * active_mask.unsqueeze(-1)

        # Message passing: aggregate from neighbors
        # h_msg[i] = sum_j W[j,i] * h_weighted[j]
        h_msg = torch.einsum('ij,bjd->bid', W.T, h_weighted)

        # Transform messages
        h_new = self.W_h(h_msg)

        # Update activation based on incoming signal strength
        signal_strength = h_new.norm(dim=-1)
        alpha_new = self.soft_threshold(
            alpha + signal_strength * 0.1,
            self.thresholds
        )

        # Residual connection and normalization
        h_out = self.layer_norm(h_new + h)
        h_out = h_out * alpha_new.unsqueeze(-1)

        return h_out, alpha_new

    def local_inhibition(self, alpha):
        """
        Apply local winner-take-more inhibition.

        Within spatial neighborhoods, activations compete via normalization.

        Args:
            alpha: Activations, shape (batch, num_nodes)

        Returns:
            Inhibited activations
        """
        # Use detached positions for inhibition (doesn't need gradients)
        pos = self.get_positions().detach()

        # Compute pairwise Euclidean distances (faster than hyperbolic)
        dist = torch.cdist(pos, pos)

        # Neighborhood mask
        neighborhood = (dist < self.inhibition_radius).float()

        # Normalize within neighborhoods
        neighbor_sum = torch.einsum('ij,bj->bi', neighborhood, alpha) + 1e-6
        alpha_normalized = alpha * neighborhood.sum(dim=1) / neighbor_sum

        return alpha_normalized.clamp(0, 1)

    def forward(self, x):
        """
        Forward pass through the RSG network.

        1. Ignite: Create initial activation from input
        2. Propagate: Iteratively spread activation
        3. Inhibit: Apply local competition
        4. Readout: Collect output from active nodes

        Args:
            x: Input tensor of shape (batch, seq_len, dim_input)

        Returns:
            output: Network output, shape (batch, dim_hidden)
            alpha: Final activations, shape (batch, num_nodes)
        """
        batch_size = x.shape[0]

        # Compute connection weights (no caching during training to avoid autograd issues)
        W = self.compute_connection_weights(use_cache=not self.training)

        # Initialize activations via ignition
        alpha = self.ignite(x)

        # Initialize hidden states from pooled input
        x_pooled = x.mean(dim=1)  # (B, dim_input)
        h = self.init_state(x_pooled).unsqueeze(1)  # (B, 1, D_h)
        h = h.expand(-1, self.num_nodes, -1).clone()  # (B, N, D_h)
        h = h * alpha.unsqueeze(-1)  # Mask by initial activation

        # Store activation trajectory for analysis
        alpha_trajectory = [alpha.clone()]

        # Propagation dynamics
        for step in range(self.num_steps):
            h, alpha = self.propagate(h, alpha, W)
            alpha = self.local_inhibition(alpha)
            alpha_trajectory.append(alpha.clone())

        # Readout: weighted sum of active states
        output = (h * alpha.unsqueeze(-1)).sum(dim=1)

        return output, alpha

    def get_sparsity(self, alpha, threshold=0.01):
        """Compute fraction of active nodes."""
        return (alpha > threshold).float().mean()

    def get_effective_connectivity(self, alpha, threshold=0.01):
        """Compute fraction of active connections given activations."""
        W = self.compute_connection_weights()
        active = (alpha > threshold).float()

        # Active connections: both endpoints active
        active_conn = torch.einsum('bi,ij,bj->b', active, W, active)
        total_conn = W.sum()

        return (active_conn / (total_conn + 1e-6)).mean()


class RSGClassifier(nn.Module):
    """
    RSG network with classification head.

    Args:
        num_classes: Number of output classes
        num_nodes: Number of RSG nodes
        dim_input: Input feature dimension
        dim_hidden: Hidden state dimension
        **kwargs: Additional arguments for RSGNetwork
    """

    def __init__(
        self,
        num_classes,
        num_nodes=256,
        dim_input=64,
        dim_hidden=128,
        **kwargs
    ):
        super().__init__()

        self.rsg = RSGNetwork(
            num_nodes=num_nodes,
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            **kwargs
        )
        self.classifier = nn.Linear(dim_hidden, num_classes)

    def forward(self, x):
        """
        Forward pass for classification.

        Args:
            x: Input tensor, shape (batch, seq_len, dim_input)

        Returns:
            logits: Class logits, shape (batch, num_classes)
            alpha: Node activations, shape (batch, num_nodes)
        """
        features, alpha = self.rsg(x)
        logits = self.classifier(features)
        return logits, alpha


class RSGSequenceModel(nn.Module):
    """
    RSG for sequence-to-sequence tasks.

    Processes each position with a shared RSG network.
    """

    def __init__(
        self,
        dim_input,
        dim_output,
        num_nodes=256,
        dim_hidden=128,
        **kwargs
    ):
        super().__init__()

        self.rsg = RSGNetwork(
            num_nodes=num_nodes,
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            **kwargs
        )
        self.output_proj = nn.Linear(dim_hidden, dim_output)

    def forward(self, x):
        """
        Args:
            x: Input sequence, shape (batch, seq_len, dim_input)

        Returns:
            output: Output sequence, shape (batch, seq_len, dim_output)
        """
        batch_size, seq_len, _ = x.shape
        outputs = []

        for t in range(seq_len):
            # Context window around position t
            start = max(0, t - 4)
            end = min(seq_len, t + 5)
            context = x[:, start:end, :]

            out, _ = self.rsg(context)
            outputs.append(out)

        output = torch.stack(outputs, dim=1)
        return self.output_proj(output)
