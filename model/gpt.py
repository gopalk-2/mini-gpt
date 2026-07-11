import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig
from model.embedding import Embedding
from model.positional_embedding import PositionalEmbedding
from model.transformer_block import TransformerBlock
from model.layer_norm import LayerNorm


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = PositionalEmbedding(config.block_size, config.n_embd)
        self.embedding_dropout = nn.Dropout(config.dropout)

        # --- Transformer Blocks ---
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=config.n_embd,
                    num_heads=config.n_head,
                    dropout=config.dropout,
                    block_size=config.block_size,
                )
                for _ in range(config.n_layer)
            ]
        )

        # --- Output Head ---
        self.final_norm = LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None):

        batch_size, seq_len = token_ids.shape

        assert seq_len <= self.config.block_size, (
            f"Sequence length {seq_len} exceeds block_size {self.config.block_size}"
        )

        # Token embeddings: (B, S) → (B, S, E)
        tok_emb = self.token_embedding(token_ids)

        # Positional embeddings: (S,) → (S, E) → (1, S, E) via broadcast
        pos_emb = self.position_embedding(seq_len)

        # Combine and apply dropout
        x = self.embedding_dropout(tok_emb + pos_emb)

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm (Pre-LN requires a final norm after all blocks)
        x = self.final_norm(x)

        # Project to vocabulary: (B, S, E) → (B, S, V)
        logits = self.lm_head(x)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Reshape for cross-entropy: (B*S, V) and (B*S,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            # Crop to block_size if sequence is too long
            idx_cond = idx[:, -self.config.block_size :]

            # Forward pass (no targets → no loss)
            logits, _ = self(idx_cond)

            # Take logits at the last position: (B, V)
            logits = logits[:, -1, :]

            # Apply temperature scaling
            logits = logits / temperature

            # Optional top-k filtering
            if top_k is not None:
                # Zero out all logits below the top-k threshold
                top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                threshold = top_k_values[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float("-inf")

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)