"""
Utility functions for RSGN experiments.

Includes:
- Synthetic data generation
- Training utilities
- Visualization helpers
- Baseline models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


def generate_hierarchical_data(
    num_samples=10000,
    seq_len=32,
    dim=64,
    num_classes=10,
    noise=0.1,
    seed=42
):
    """
    Generate hierarchical classification data.

    Structure:
    - Level 1: Local patterns (3-grams)
    - Level 2: Compositions of local patterns (spread across sequence)
    - Level 3: Global class signature

    Args:
        num_samples: Number of samples to generate
        seq_len: Sequence length
        dim: Feature dimension
        num_classes: Number of classes
        noise: Noise level
        seed: Random seed

    Returns:
        X: Tensor of shape (num_samples, seq_len, dim)
        y: Tensor of shape (num_samples,)
    """
    np.random.seed(seed)

    X = np.random.randn(num_samples, seq_len, dim) * noise
    y = np.random.randint(0, num_classes, num_samples)

    # Generate class-specific hierarchical patterns
    level1_patterns = np.random.randn(num_classes, 5, 3, dim) * 0.5
    level2_patterns = np.random.randn(num_classes, 3, dim) * 0.3
    level3_patterns = np.random.randn(num_classes, dim) * 0.2

    for i in range(num_samples):
        c = y[i]

        # Insert level-1 patterns (local 3-grams)
        for p in range(3):
            pos = np.random.randint(0, max(1, seq_len - 3))
            pattern_idx = np.random.randint(0, 5)
            end_pos = min(pos + 3, seq_len)
            X[i, pos:end_pos] += level1_patterns[c, pattern_idx, :end_pos-pos]

        # Add level-2 patterns (spread across sequence)
        for p in range(3):
            pos = (seq_len // 3) * p + np.random.randint(0, max(1, seq_len // 6))
            pos = min(pos, seq_len - 1)
            X[i, pos] += level2_patterns[c, p]

        # Add level-3 global signature
        X[i] += level3_patterns[c] * 0.1

    return torch.FloatTensor(X), torch.LongTensor(y)


def generate_copy_task_data(
    num_samples=5000,
    seq_len=20,
    vocab_size=10,
    seed=42
):
    """
    Generate data for copy task.

    Input: [sequence] [delimiter] [zeros]
    Target: [zeros] [delimiter] [sequence]

    Args:
        num_samples: Number of samples
        seq_len: Length of sequence to copy
        vocab_size: Size of vocabulary
        seed: Random seed

    Returns:
        X: Input sequences
        y: Target sequences
    """
    np.random.seed(seed)

    total_len = 2 * seq_len + 1
    X = np.zeros((num_samples, total_len), dtype=np.int64)
    y = np.zeros((num_samples, total_len), dtype=np.int64)

    for i in range(num_samples):
        # Generate random sequence
        seq = np.random.randint(1, vocab_size, seq_len)

        # Input: sequence + delimiter + zeros
        X[i, :seq_len] = seq
        X[i, seq_len] = vocab_size  # delimiter

        # Target: zeros + delimiter + sequence
        y[i, seq_len] = vocab_size
        y[i, seq_len+1:] = seq

    return torch.LongTensor(X), torch.LongTensor(y)


class HierarchicalDataset(Dataset):
    """Dataset wrapper for hierarchical classification data."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(X, y, train_ratio=0.8, batch_size=64, seed=42):
    """Create train and validation dataloaders."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    n = len(y)
    indices = np.random.permutation(n)
    train_size = int(n * train_ratio)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    train_dataset = HierarchicalDataset(X[train_idx], y[train_idx])
    val_dataset = HierarchicalDataset(X[val_idx], y[val_idx])

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader


# ============== Baseline Models ==============

class MLPClassifier(nn.Module):
    """MLP baseline for classification."""

    def __init__(self, seq_len, dim_input, dim_hidden, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(seq_len * dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_classes)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x), None


class TransformerClassifier(nn.Module):
    """Transformer baseline for classification."""

    def __init__(self, dim_input, dim_hidden, num_classes, num_heads=4, num_layers=2):
        super().__init__()
        self.embed = nn.Linear(dim_input, dim_hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden,
            nhead=num_heads,
            dim_feedforward=dim_hidden * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim_hidden, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x), None


class SparseTransformerClassifier(nn.Module):
    """Sparse Transformer baseline with fixed sparsity pattern."""

    def __init__(self, dim_input, dim_hidden, num_classes, num_heads=4, num_layers=2, sparsity=0.75):
        super().__init__()
        self.embed = nn.Linear(dim_input, dim_hidden)
        self.sparsity = sparsity
        self.num_heads = num_heads
        self.dim_hidden = dim_hidden

        # Attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(dim_hidden, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden * 4),
                nn.GELU(),
                nn.Linear(dim_hidden * 4, dim_hidden)
            )
            for _ in range(num_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(dim_hidden) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(dim_hidden) for _ in range(num_layers)])
        self.classifier = nn.Linear(dim_hidden, num_classes)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x = self.embed(x)

        # Create sparse attention mask (local + strided global)
        mask = torch.ones(seq_len, seq_len, device=x.device)
        for i in range(seq_len):
            # Local attention (window of 5)
            start = max(0, i - 2)
            end = min(seq_len, i + 3)
            mask[i, start:end] = 0
            # Strided global attention
            for j in range(0, seq_len, 4):
                mask[i, j] = 0

        mask = mask.bool()

        for attn, ffn, norm1, norm2 in zip(
            self.attention_layers, self.ffn_layers, self.norms1, self.norms2
        ):
            # Self-attention with sparse mask
            attn_out, _ = attn(x, x, x, attn_mask=mask)
            x = norm1(x + attn_out)
            x = norm2(x + ffn(x))

        x = x.mean(dim=1)
        return self.classifier(x), None


class LSTMClassifier(nn.Module):
    """LSTM baseline for classification."""

    def __init__(self, dim_input, dim_hidden, num_classes, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            dim_input, dim_hidden, num_layers,
            batch_first=True, bidirectional=True
        )
        self.classifier = nn.Linear(dim_hidden * 2, num_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        # Concatenate forward and backward
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.classifier(h), None


# ============== Training Utilities ==============

def train_epoch(model, loader, optimizer, criterion, device, hebbian=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_sparsity = 0
    n_batches = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits, alpha = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        # Hebbian update if provided
        if hebbian is not None and alpha is not None:
            reward = -loss.item()
            hebbian.update(alpha.detach(), reward=reward)

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        total_correct += (preds == batch_y).sum().item()
        total_samples += batch_y.size(0)

        if alpha is not None:
            total_sparsity += (alpha > 0.01).float().mean().item()
        n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'accuracy': total_correct / total_samples,
        'sparsity': total_sparsity / n_batches if n_batches > 0 else 0
    }


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_sparsity = 0
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits, alpha = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += batch_y.size(0)

            if alpha is not None:
                total_sparsity += (alpha > 0.01).float().mean().item()
            n_batches += 1

    return {
        'loss': total_loss / n_batches,
        'accuracy': total_correct / total_samples,
        'sparsity': total_sparsity / n_batches if n_batches > 0 else 0
    }


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops_per_sample(model, sample_input):
    """
    Estimate FLOPs per forward pass.
    This is a rough estimate based on linear layer counts.
    """
    total_flops = 0

    def count_linear(module, input, output):
        nonlocal total_flops
        if isinstance(module, nn.Linear):
            total_flops += module.in_features * module.out_features

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(count_linear))

    model.eval()
    with torch.no_grad():
        model(sample_input)

    for hook in hooks:
        hook.remove()

    return total_flops
