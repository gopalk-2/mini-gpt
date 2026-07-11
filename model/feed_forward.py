"""
Position-wise Feed-Forward Network.

Applied independently to each position after multi-head attention.
This is a simple two-layer MLP that expands the representation to
a higher dimensionality, applies a non-linearity, then projects
back down:

    FFN(x) = Dropout(Linear_2(GELU(Linear_1(x))))

The expansion factor (default 4x) gives the network more capacity
to learn complex transformations at each position.

GPT-2 uses GELU activation instead of ReLU (original Transformer).
GELU is smoother and empirically performs better for language models.

Reference:
    - Section 3.3 of "Attention Is All You Need" (Vaswani et al., 2017)
    - Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (2016)
"""

import torch.nn as nn


class FeedForward(nn.Module):
    """Two-layer feed-forward network with GELU activation and dropout.

    Args:
        embedding_dim: Input and output dimensionality.
        expansion_factor: Hidden layer size = embedding_dim * expansion_factor.
        dropout: Dropout probability after activation.
    """

    def __init__(
        self,
        embedding_dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_dim = embedding_dim * expansion_factor

        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Apply feed-forward transformation.

        Args:
            x: Input tensor of shape (batch, seq_len, embedding_dim).

        Returns:
            Output tensor of the same shape.
        """
        return self.net(x)