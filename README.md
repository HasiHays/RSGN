# Resonant Sparse Geometry Networks (RSGN)

A brain-inspired neural architecture with self-organizing sparse hierarchical connectivity.

## Overview

RSGN implements a novel neural network architecture that:

- Embeds computational nodes in learned hyperbolic space
- Uses distance-based connectivity that naturally enforces sparsity
- Employs input-dependent ignition for dynamic routing
- Combines fast gradient descent with slow Hebbian structural learning

## Installation

```bash
pip install torch numpy matplotlib seaborn jupyter
```

## Project Structure

```
RSGN/
├── src/
│   ├── __init__.py      # Package exports
│   ├── model.py         # Core RSGN architecture
│   ├── hyperbolic.py    # Hyperbolic geometry operations
│   ├── hebbian.py       # Hebbian structural learning
│   └── utils.py         # Data generation and baselines
├── analysis/
│   └── rsgn_experiments.ipynb  # Main experiments notebook
├── tests/               # Unit tests
└── examples/            # Usage examples
```

## Quick Start

```python
import torch
from src.model import RSGClassifier
from src.hebbian import HebbianLearner
from src.utils import generate_hierarchical_data, create_dataloaders

# Generate data
X, y = generate_hierarchical_data(num_samples=5000, seq_len=32, dim=64, num_classes=10)
train_loader, val_loader = create_dataloaders(X, y)

# Create model
model = RSGClassifier(
    num_classes=10,
    num_nodes=256,
    dim_input=64,
    dim_hidden=128,
    num_steps=5
)

# Training with Hebbian learning
hebbian = HebbianLearner(model.rsg)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        logits, alpha = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        # Hebbian structural update
        hebbian.update(alpha.detach(), reward=-loss.item())
```

## Key Components

### RSGNetwork

The core network with:

- Node positions in Poincare ball (hyperbolic space)
- Distance-based connection weights
- Soft-threshold activation dynamics
- Local inhibition for competition

### HebbianLearner

Manages slow structural learning:

- Co-activation strengthens affinities
- Position drift toward correlated nodes
- Threshold adaptation for sparsity control
- Periodic pruning and sprouting

## Running Experiments

Open and run the Jupyter notebook:

```bash
cd analysis
jupyter notebook rsgn_experiments.ipynb
```

This generates:

- Performance comparison tables
- Training curves
- Architecture visualizations
- Ablation study results

## Citation

```bibtex
@article{hays2026rsgn,
  title={Resonant Sparse Geometry Networks: A Brain-Inspired Architecture
         with Self-Organizing Sparse Hierarchical Connectivity},
  author={Hays, Hasi},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT License
