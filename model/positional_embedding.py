import torch.nn as nn


class PositionalEmbedding(nn.Module):

    def __init__(self, max_seq_length: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_length, embedding_dim)

    def forward(self, sequence_length: int):
        import torch

        positions = torch.arange(sequence_length, device=self.embedding.weight.device)
        return self.embedding(positions)