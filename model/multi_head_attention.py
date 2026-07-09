import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int):
        super().__init__()

        assert (
            embedding_dim % num_heads == 0
        ), "embedding_dim must be divisible by num_heads"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Single projection for Q, K and V
        self.qkv = nn.Linear(
            embedding_dim,
            embedding_dim * 3,
            bias=False,
        )

        # Output projection
        self.output = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

    def forward(self, x):
        """
        x shape:
            (batch_size, sequence_length, embedding_dim)
        """

        batch_size, sequence_length, _ = x.shape

        # --------------------------------------------------
        # Generate Q, K, V
        # --------------------------------------------------
        qkv = self.qkv(x)

        # (B, S, 3*E) -> 3 x (B, S, E)
        Q, K, V = qkv.chunk(3, dim=-1)

        # --------------------------------------------------
        # Split into heads
        # (B,S,E) -> (B,H,S,D)
        # --------------------------------------------------
        Q = (
            Q.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        K = (
            K.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        V = (
            V.view(
                batch_size,
                sequence_length,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

        # --------------------------------------------------
        # Attention Scores
        # (B,H,S,D) x (B,H,D,S)
        # -> (B,H,S,S)
        # --------------------------------------------------
        scores = Q @ K.transpose(-2, -1)

        # --------------------------------------------------
        # Scale
        # --------------------------------------------------
        scores = scores / math.sqrt(self.head_dim)

        # --------------------------------------------------
        # Causal Mask
        # --------------------------------------------------
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        scores = scores.masked_fill(mask, float("-inf"))

        # --------------------------------------------------
        # Softmax
        # --------------------------------------------------
        attention = torch.softmax(scores, dim=-1)

        # Save for visualization/debugging
        self.attention_weights = attention

        # --------------------------------------------------
        # Weighted Sum
        # (B,H,S,S) x (B,H,S,D)
        # -> (B,H,S,D)
        # --------------------------------------------------
        output = attention @ V

        # --------------------------------------------------
        # Merge Heads
        # (B,H,S,D) -> (B,S,H,D)
        # --------------------------------------------------
        output = output.transpose(1, 2)

        # --------------------------------------------------
        # Concatenate Heads
        # (B,S,H,D) -> (B,S,E)
        # --------------------------------------------------
        output = output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        # --------------------------------------------------
        # Final Linear Projection
        # --------------------------------------------------
        output = self.output(output)

        return output