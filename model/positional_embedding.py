"""
Learnable Positional Embedding layer.

Adds position information to token embeddings so the Transformer
knows the order of tokens in a sequence. Without this, the
self-attention mechanism is permutation-invariant — it treats
"the cat sat" the same as "sat the cat".

GPT-2 uses learned positional embeddings (as opposed to the
sinusoidal encodings from the original Transformer paper).
Each position 0..block_size-1 gets its own learnable vector.

Reference:
    - Section 3.5 of "Attention Is All You Need" (Vaswani et al., 2017)
    - Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
"""

import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Learned positional embedding table.

    Args:
        max_seq_length: Maximum number of positions (block_size).
        embedding_dim: Dimensionality of each position vector.
    """

    def __init__(self, max_seq_length: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_length, embedding_dim)

    def forward(self, sequence_length: int):
        """Retrieve positional embeddings for positions 0..sequence_length-1.

        Args:
            sequence_length: Number of positions to retrieve.

        Returns:
            Tensor of shape (sequence_length, embedding_dim).
        """
        import torch

        positions = torch.arange(sequence_length, device=self.embedding.weight.device)
        return self.embedding(positions)