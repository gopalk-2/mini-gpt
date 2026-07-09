import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, embedding_dim):

        super().__init__()

        self.Wq = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.Wk = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

        self.Wv = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )

    def forward(self, x):

        Q = self.Wq(x)

        K = self.Wk(x)

        V = self.Wv(x)

        return Q, K, V