"""
Token Embedding layer.

Converts integer token IDs into dense vector representations.
Each token in the vocabulary gets its own learnable embedding vector
of size `embedding_dim`.

This is the first step in the Transformer pipeline:
    token_ids → Embedding → dense vectors (batch, seq_len, embedding_dim)

Reference:
    Section 3.4 of "Attention Is All You Need" (Vaswani et al., 2017).
"""

import torch.nn as nn


class Embedding(nn.Module):
    """Learnable token embedding lookup table.

    Args:
        vocab_size: Number of unique tokens in the vocabulary.
        embedding_dim: Dimensionality of each embedding vector.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_ids):
        """Look up embeddings for the given token IDs.

        Args:
            token_ids: Integer tensor of shape (batch_size, sequence_length).

        Returns:
            Tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        return self.embedding(token_ids)
