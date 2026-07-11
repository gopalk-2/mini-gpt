"""
Multi-Head Self-Attention — the core mechanism of the Transformer.

Instead of computing a single attention function, multi-head attention
runs `num_heads` parallel attention computations, each operating on
a `head_dim`-sized slice of the embedding. The outputs are concatenated
and linearly projected.

This allows the model to jointly attend to information from different
representation subspaces at different positions.

Implementation details:
    - Fused QKV projection: A single linear layer produces Q, K, V
      in one matrix multiply, then splits them — more efficient than
      three separate projections.
    - Causal mask: Registered as a buffer (not a parameter) so it
      moves with the model to GPU/CPU but isn't trained.
    - Dropout: Applied on attention weights to prevent overfitting.

The math:
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

Reference:
    Section 3.2 of "Attention Is All You Need" (Vaswani et al., 2017).
"""

import math

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head causal self-attention.

    Args:
        embedding_dim: Total embedding dimensionality.
        num_heads: Number of parallel attention heads.
        dropout: Dropout probability on attention weights and output.
        block_size: Maximum sequence length (for causal mask buffer).
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        block_size: int = 256,
    ):
        super().__init__()

        assert (
            embedding_dim % num_heads == 0
        ), f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # --- Projections ---
        # Fused QKV: one big linear layer, then split into Q, K, V
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3, bias=False)

        # Output projection after concatenating heads
        self.output_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # --- Regularization ---
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # --- Causal Mask ---
        # Registered as a buffer so it's part of the model state but not
        # a learnable parameter. Prevents attending to future positions.
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(block_size, block_size, dtype=torch.bool),
                diagonal=1,
            ),
        )

    def forward(self, x):
        """Apply multi-head causal self-attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim).

        Returns:
            Output tensor of shape (batch_size, seq_len, embedding_dim).

        The computation flow:
            1. Project input to Q, K, V via fused linear layer
            2. Reshape to (batch, heads, seq_len, head_dim)
            3. Compute scaled dot-product attention with causal mask
            4. Concatenate heads and apply output projection
        """
        batch_size, seq_len, _ = x.shape

        # ---- Step 1: Fused QKV Projection ----
        # (B, S, E) → (B, S, 3*E) → 3 × (B, S, E)
        qkv = self.qkv(x)
        Q, K, V = qkv.chunk(3, dim=-1)

        # ---- Step 2: Reshape into heads ----
        # (B, S, E) → (B, S, H, D) → (B, H, S, D)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # ---- Step 3: Scaled Dot-Product Attention ----
        # (B, H, S, D) @ (B, H, D, S) → (B, H, S, S)
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        # Apply causal mask (prevent attending to future tokens)
        scores = scores.masked_fill(
            self.causal_mask[:seq_len, :seq_len],
            float("-inf"),
        )

        # Softmax over the key dimension (turns -inf into 0 probability)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        # Weighted sum of values
        # (B, H, S, S) @ (B, H, S, D) → (B, H, S, D)
        output = attention_weights @ V

        # ---- Step 4: Concatenate heads and project ----
        # (B, H, S, D) → (B, S, H, D) → (B, S, E)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embedding_dim
        )

        # Final linear projection + dropout
        output = self.resid_dropout(self.output_proj(output))

        return output