"""
Character-level tokenizer for Mini-GPT.

Maps each unique character in the training corpus to an integer ID.
This is the same approach Karpathy uses for Tiny Shakespeare — simple,
transparent, and keeps the focus on the Transformer architecture.

The vocabulary is built from the training text: every unique character
gets a unique integer, sorted alphabetically for deterministic ordering.

Example:
    >>> tokenizer = CharTokenizer()
    >>> tokenizer.build_vocab("hello world")
    >>> tokenizer.encode("hello")
    [3, 2, 5, 5, 6]
    >>> tokenizer.decode([3, 2, 5, 5, 6])
    'hello'
"""


class CharTokenizer:
    """Character-level tokenizer.

    Builds a bidirectional mapping between characters and integer IDs.
    Vocabulary is sorted alphabetically so the mapping is deterministic
    across runs.

    Attributes:
        char_to_idx: Mapping from character to token ID.
        idx_to_char: Mapping from token ID to character.
    """

    def __init__(self):
        self.char_to_idx: dict[str, int] = {}
        self.idx_to_char: dict[int, str] = {}

    def build_vocab(self, text: str) -> None:
        """Build vocabulary from a text corpus.

        Extracts all unique characters, sorts them, and assigns
        sequential integer IDs starting from 0.

        Args:
            text: The full training text to extract characters from.
        """
        chars = sorted(set(text))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of token IDs.

        Args:
            text: Input string to encode.

        Returns:
            List of integer token IDs.

        Raises:
            KeyError: If text contains characters not in vocabulary.
        """
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        """Convert a list of token IDs back into a string.

        Args:
            token_ids: List of integer token IDs.

        Returns:
            Decoded string.

        Raises:
            KeyError: If any token ID is not in vocabulary.
        """
        return "".join(self.idx_to_char[idx] for idx in token_ids)

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary."""
        return len(self.char_to_idx)

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"