import torch

from model.self_attention import SelfAttention


attention = SelfAttention(embedding_dim=8)

x = torch.randn(4,8)
output = attention(x)

print(scores.shape)