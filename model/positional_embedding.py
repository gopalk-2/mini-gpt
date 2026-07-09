import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):

    def __init__(self, max_seq_length, embedding_dim):

        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(max_seq_length, embedding_dim)
        )

    def forward(self, sequence_length):

        return self.weight[:sequence_length]