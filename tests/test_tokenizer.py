"""
Tests for the character-level tokenizer.

Verifies vocabulary construction, encoding/decoding roundtrips,
and edge cases.
"""

import pytest

from tokenizer.tokenizer import CharTokenizer


class TestCharTokenizer:
    """Tests for CharTokenizer."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer with a simple vocabulary."""
        t = CharTokenizer()
        t.build_vocab("hello world")
        return t

    @pytest.fixture
    def shakespeare_tokenizer(self):
        """Create a tokenizer from Tiny Shakespeare data."""
        t = CharTokenizer()
        with open("data/input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        t.build_vocab(text)
        return t

    def test_build_vocab_creates_mappings(self, tokenizer):
        """Vocabulary should have bidirectional char↔int mappings."""
        assert len(tokenizer.char_to_idx) > 0
        assert len(tokenizer.idx_to_char) > 0
        assert len(tokenizer.char_to_idx) == len(tokenizer.idx_to_char)

    def test_vocab_size(self, tokenizer):
        """'hello world' has 8 unique characters: ' ', 'd', 'e', 'h', 'l', 'o', 'r', 'w'."""
        assert tokenizer.vocab_size == 8

    def test_shakespeare_vocab_size(self, shakespeare_tokenizer):
        """Tiny Shakespeare should have 65 unique characters."""
        assert shakespeare_tokenizer.vocab_size == 65

    def test_vocab_is_sorted(self, tokenizer):
        """Characters should be assigned IDs in sorted order."""
        chars = list(tokenizer.char_to_idx.keys())
        assert chars == sorted(chars)

    def test_encode_returns_list_of_ints(self, tokenizer):
        """Encoding should return a list of integers."""
        encoded = tokenizer.encode("hello")
        assert isinstance(encoded, list)
        assert all(isinstance(x, int) for x in encoded)

    def test_decode_returns_string(self, tokenizer):
        """Decoding should return a string."""
        encoded = tokenizer.encode("hello")
        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str)

    def test_encode_decode_roundtrip(self, tokenizer):
        """Encoding then decoding should recover the original string."""
        original = "hello world"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        assert decoded == original

    def test_encode_decode_roundtrip_shakespeare(self, shakespeare_tokenizer):
        """Roundtrip should work for Shakespeare-style text."""
        text = "ROMEO:\nWherefore art thou Romeo?\n"
        encoded = shakespeare_tokenizer.encode(text)
        decoded = shakespeare_tokenizer.decode(encoded)
        assert decoded == text

    def test_encode_length(self, tokenizer):
        """Encoded length should equal input string length (char-level)."""
        text = "hello"
        encoded = tokenizer.encode(text)
        assert len(encoded) == len(text)

    def test_encode_unknown_char_raises(self, tokenizer):
        """Encoding a character not in vocabulary should raise KeyError."""
        with pytest.raises(KeyError):
            tokenizer.encode("hello world!")  # '!' not in vocab

    def test_empty_string(self, tokenizer):
        """Empty string should encode to empty list and vice versa."""
        assert tokenizer.encode("") == []
        assert tokenizer.decode([]) == ""

    def test_repr(self, tokenizer):
        """Repr should show vocab size."""
        assert "8" in repr(tokenizer)
