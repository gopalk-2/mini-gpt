import pytest
import torch

from config import GPTConfig
from model.gpt import GPT
from model.multi_head_attention import MultiHeadAttention
from model.transformer_block import TransformerBlock
from model.feed_forward import FeedForward
from model.embedding import Embedding
from model.positional_embedding import PositionalEmbedding
from model.layer_norm import LayerNorm


@pytest.fixture
def config():
    """Small model config for fast tests."""
    return GPTConfig(
        vocab_size=32,
        block_size=16,
        n_layer=2,
        n_head=4,
        n_embd=64,
        dropout=0.0,  # Disable dropout for deterministic tests
    )


@pytest.fixture
def model(config):
    """Create a small GPT model for testing."""
    return GPT(config)



class TestEmbedding:
    """Tests for the token embedding layer."""

    def test_output_shape(self):
        emb = Embedding(vocab_size=32, embedding_dim=64)
        ids = torch.randint(0, 32, (2, 8))
        output = emb(ids)
        assert output.shape == (2, 8, 64)

    def test_different_ids_different_vectors(self):
        emb = Embedding(vocab_size=32, embedding_dim=64)
        ids = torch.tensor([[0, 1]])
        output = emb(ids)
        assert not torch.allclose(output[0, 0], output[0, 1])


class TestPositionalEmbedding:
    """Tests for the positional embedding layer."""

    def test_output_shape(self):
        pos = PositionalEmbedding(max_seq_length=16, embedding_dim=64)
        output = pos(10)
        assert output.shape == (10, 64)

    def test_full_length(self):
        pos = PositionalEmbedding(max_seq_length=16, embedding_dim=64)
        output = pos(16)
        assert output.shape == (16, 64)



class TestLayerNorm:
    """Tests for the custom LayerNorm implementation."""

    def test_output_shape(self):
        norm = LayerNorm(64)
        x = torch.randn(2, 8, 64)
        output = norm(x)
        assert output.shape == x.shape

    def test_normalized_statistics(self):
        """After normalization, mean ≈ 0 and var ≈ 1 (before affine)."""
        norm = LayerNorm(64)
        # Reset affine params to identity
        norm.gamma.data.fill_(1.0)
        norm.beta.data.fill_(0.0)

        x = torch.randn(4, 8, 64) * 5 + 3  # Non-zero mean and variance
        output = norm(x)

        mean = output.mean(dim=-1)
        var = output.var(dim=-1, unbiased=False)

        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4)



class TestFeedForward:
    """Tests for the position-wise feed-forward network."""

    def test_output_shape(self):
        ffn = FeedForward(embedding_dim=64, expansion_factor=4)
        x = torch.randn(2, 8, 64)
        output = ffn(x)
        assert output.shape == (2, 8, 64)

    def test_expansion_factor(self):
        """Hidden dim should be embedding_dim * expansion_factor."""
        ffn = FeedForward(embedding_dim=64, expansion_factor=4)
        # First layer: 64 → 256
        assert ffn.net[0].in_features == 64
        assert ffn.net[0].out_features == 256



class TestMultiHeadAttention:
    """Tests for multi-head causal self-attention."""

    def test_output_shape(self):
        mha = MultiHeadAttention(embedding_dim=64, num_heads=4, block_size=16)
        x = torch.randn(2, 8, 64)
        output = mha(x)
        assert output.shape == (2, 8, 64)

    def test_causal_mask_exists(self):
        """The causal mask buffer should be registered."""
        mha = MultiHeadAttention(embedding_dim=64, num_heads=4, block_size=16)
        assert hasattr(mha, "causal_mask")
        assert mha.causal_mask.shape == (16, 16)

    def test_causal_mask_is_upper_triangular(self):
        """Causal mask should be True above diagonal (future positions)."""
        mha = MultiHeadAttention(embedding_dim=64, num_heads=4, block_size=8)
        # Position (i, j) should be masked when j > i (future token)
        assert mha.causal_mask[0, 1] == True   # Can't attend to future
        assert mha.causal_mask[0, 0] == False   # Can attend to self
        assert mha.causal_mask[2, 1] == False   # Can attend to past

    def test_embedding_dim_must_divide_by_heads(self):
        """Should raise assertion if embedding_dim not divisible by num_heads."""
        with pytest.raises(AssertionError):
            MultiHeadAttention(embedding_dim=65, num_heads=4, block_size=16)

    def test_fused_qkv_projection(self):
        """QKV projection should output 3x embedding_dim."""
        mha = MultiHeadAttention(embedding_dim=64, num_heads=4, block_size=16)
        assert mha.qkv.out_features == 64 * 3


class TestTransformerBlock:
    """Tests for the Transformer decoder block."""

    def test_output_shape(self):
        block = TransformerBlock(embedding_dim=64, num_heads=4, block_size=16)
        x = torch.randn(2, 8, 64)
        output = block(x)
        assert output.shape == (2, 8, 64)

    def test_pre_ln_architecture(self):
        block = TransformerBlock(
            embedding_dim=64,
            num_heads=4,
            dropout=0.0,
            block_size=16,
        )

        # Zero out all sublayer weights so sublayer output ≈ 0
        for param in block.attention.parameters():
            param.data.zero_()
        for param in block.ffn.parameters():
            param.data.zero_()

        x = torch.randn(2, 8, 64)
        output = block(x)

        # With zeroed sublayers, Pre-LN gives: x + 0 = x
        assert torch.allclose(output, x, atol=1e-5)



class TestGPT:
    """Tests for the complete GPT model."""

    def test_forward_logits_shape(self, model, config):
        """Logits should have shape (batch, seq_len, vocab_size)."""
        tokens = torch.randint(0, config.vocab_size, (2, 8))
        logits, loss = model(tokens)
        assert logits.shape == (2, 8, config.vocab_size)
        assert loss is None  # No targets provided

    def test_forward_with_targets_returns_loss(self, model, config):
        """When targets are provided, forward should return a loss."""
        tokens = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))
        logits, loss = model(tokens, targets)
        assert loss is not None
        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0  # Loss should be positive

    def test_initial_loss_near_random(self, config):
        """Initial loss should be ≈ -ln(1/vocab_size) = ln(vocab_size)."""
        import math
        model = GPT(config)
        tokens = torch.randint(0, config.vocab_size, (32, config.block_size))
        targets = torch.randint(0, config.vocab_size, (32, config.block_size))
        _, loss = model(tokens, targets)

        expected = math.log(config.vocab_size)  # ~3.47 for vocab_size=32
        assert abs(loss.item() - expected) < 0.5  # Within 0.5 of expected

    def test_generate_extends_sequence(self, model, config):
        """Generate should extend the input sequence."""
        start = torch.randint(0, config.vocab_size, (1, 4))
        generated = model.generate(start, max_new_tokens=10)
        assert generated.shape == (1, 14)  # 4 + 10

    def test_generate_respects_max_tokens(self, model, config):
        """Generate should produce exactly max_new_tokens new tokens."""
        start = torch.zeros((1, 1), dtype=torch.long)
        for n in [1, 5, 20]:
            generated = model.generate(start, max_new_tokens=n)
            assert generated.shape[1] == 1 + n

    def test_weight_tying(self, model):
        """Token embedding and LM head should share weights."""
        assert model.lm_head.weight is model.token_embedding.embedding.weight

    def test_count_parameters(self, model):
        """Should return a positive integer."""
        count = model.count_parameters()
        assert count > 0
        assert isinstance(count, int)

    def test_sequence_too_long_raises(self, model, config):
        """Sequence exceeding block_size should raise an error."""
        tokens = torch.randint(0, config.vocab_size, (1, config.block_size + 1))
        with pytest.raises(AssertionError):
            model(tokens)
