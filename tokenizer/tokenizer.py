
class CharTokenizer:
    def __init__(self):
        self.char_to_idx: dict[str, int] = {}
        self.idx_to_char: dict[int, str] = {}

    def build_vocab(self, text: str) -> None:
        chars = sorted(set(text))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.idx_to_char[idx] for idx in token_ids)

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the vocabulary."""
        return len(self.char_to_idx)

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"