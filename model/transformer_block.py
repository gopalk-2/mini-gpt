"""
Transformer Block — one layer of the GPT decoder stack.

Each block applies two sub-layers with residual connections:
    1. Multi-Head Self-Attention
    2. Position-wise Feed-Forward Network

This implementation uses **Pre-LN** (Pre-Layer Normalization),
where LayerNorm is applied BEFORE each sub-layer. This is the
GPT-2 convention and differs from the original Transformer paper
which applies LayerNorm AFTER (Post-LN).

    Pre-LN:  x = x + Attention(LayerNorm(x))     ← Used here (GPT-2)
    Post-LN: x = LayerNorm(x + Attention(x))      ← Original paper

Pre-LN provides more stable gradient flow during training because
the residual path is an unobstructed identity mapping.

Reference:
    - Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2)
    - Xiong et al., "On Layer Normalization in the Transformer Architecture" (2020)
"""

from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.layer_norm import LayerNorm


import torch.nn as nn


class TransformerBlock(nn.Module):
    """Single Transformer decoder block with Pre-LN architecture.

    Args:
        embedding_dim: Dimensionality of the token representations.
        num_heads: Number of attention heads.
        expansion_factor: Feed-forward hidden layer expansion ratio.
        dropout: Dropout probability for regularization.
        block_size: Maximum sequence length (for causal mask).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        expansion_factor: int = 4,
        dropout: float = 0.0,
        block_size: int = 256,
    ):
        super().__init__()

        # Layer Norms (applied BEFORE each sub-layer in Pre-LN)
        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

        # Sub-layers
        self.attention = MultiHeadAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            block_size=block_size,
        )

        self.ffn = FeedForward(
            embedding_dim=embedding_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
        )

    def forward(self, x):
        """Apply one Transformer block.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            Output tensor of the same shape.

        Pre-LN residual connections:
            x = x + Attention(LayerNorm(x))
            x = x + FFN(LayerNorm(x))
        """
        # Multi-Head Attention with Pre-LN residual
        x = x + self.attention(self.norm1(x))

        # Feed-Forward with Pre-LN residual
        x = x + self.ffn(self.norm2(x))

        return x