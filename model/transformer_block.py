from model.multi_head_attention import MultiHeadAttention
from model.feed_forward import FeedForward
from model.layer_norm import LayerNorm


import torch.nn as nn


class TransformerBlock(nn.Module):
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
        # Multi-Head Attention with Pre-LN residual
        x = x + self.attention(self.norm1(x))

        # Feed-Forward with Pre-LN residual
        x = x + self.ffn(self.norm2(x))

        return x