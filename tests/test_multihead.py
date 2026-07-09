import torch

from model.multi_head_attention import MultiHeadAttention

model = MultiHeadAttention(
    embedding_dim=768,
    num_heads=12
)

x = torch.randn(2,16,768)

with torch.no_grad():

    batch_size, sequence_length, _ = x.shape

    qkv = model.qkv(x)

    Q, K, V = qkv.chunk(3, dim=-1)

    Q = Q.view(batch_size, sequence_length, model.num_heads, model.head_dim).transpose(1,2)
    K = K.view(batch_size, sequence_length, model.num_heads, model.head_dim).transpose(1,2)
    V = V.view(batch_size, sequence_length, model.num_heads, model.head_dim).transpose(1,2)

    print("Q:", Q.shape)
    print("K:", K.shape)
    print("V:", V.shape)