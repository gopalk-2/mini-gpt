"""
Layer Normalization — implemented from scratch.

Normalizes activations across the feature dimension (last axis)
for each individual sample, independent of the batch. This stabilizes
training by ensuring each layer's inputs have consistent mean and
variance.

Unlike Batch Normalization:
    - Operates on a single sample (no batch statistics)
    - Works identically during training and inference
    - No running mean/variance to track

The formula:
    LayerNorm(x) = gamma * (x - mean) / sqrt(variance + eps) + beta

Where gamma (scale) and beta (shift) are learnable parameters that
allow the network to undo the normalization if needed.

Reference:
    Ba et al., "Layer Normalization" (2016).
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization with learnable affine parameters.

    Args:
        embedding_dim: Size of the feature dimension to normalize over.
        eps: Small constant for numerical stability in division.
    """

    def __init__(self, embedding_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embedding_dim))
        self.beta = nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x):
        """Apply layer normalization.

        Args:
            x: Input tensor of shape (..., embedding_dim).

        Returns:
            Normalized tensor of the same shape.
        """
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(variance + self.eps)
        return self.gamma * normalized + self.beta