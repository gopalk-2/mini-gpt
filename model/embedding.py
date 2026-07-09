import torch
import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim):

        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(vocab_size, embedding_dim)
        )

    def forward(self, token_ids):

        return self.weight[token_ids]
    
