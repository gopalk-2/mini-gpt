import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.query = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.key = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.value = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 1. Calculate raw dot-product scores
        scores = Q @ K.transpose(-2, -1)

        # 2. Scale the scores
        scores = scores / math.sqrt(self.embedding_dim)
        sequence_length = scores.size(-1)
        mask = torch.triu(
            torch.ones(sequence_length, sequence_length, device=x.device), # Good practice to keep on same device (CPU/GPU)
            diagonal=1
        ).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        # 3. Softmax turns -inf into 0 probability 
        attention = torch.softmax(scores, dim=-1)
        self.attention_weights = attention

        # 4. Multiply by values
        output = attention @ V

        return output