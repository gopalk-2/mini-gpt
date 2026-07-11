"""
Dataset utilities for Mini-GPT training.

Follows Karpathy's simple and elegant approach:
    1. Load the entire text file into memory
    2. Tokenize it into a single long integer tensor
    3. Split into train/val tensors (90/10 by default)
    4. Sample random batches of (input, target) pairs

No PyTorch Dataset/DataLoader overhead — for a single-file corpus
like Tiny Shakespeare (~1MB), this approach is simpler and faster.

Each training example is a chunk of `block_size` consecutive tokens.
The target is the same chunk shifted right by one position:

    Input:  "First Citizen"  → [F, i, r, s, t,  , C, i, t, i, z, e]
    Target: "irst Citizen:"  → [i, r, s, t,  , C, i, t, i, z, e, n]

This means every position simultaneously learns to predict the
next token, giving us `block_size` training examples per sequence.
"""

import torch

from tokenizer.tokenizer import CharTokenizer


def load_dataset(
    data_path: str,
    tokenizer: CharTokenizer,
    train_split: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load text data, tokenize, and split into train/val tensors.

    Args:
        data_path: Path to the text file (e.g., "data/input.txt").
        tokenizer: A CharTokenizer with vocabulary already built,
                   OR an uninitialized one (vocab will be built here).
        train_split: Fraction of data for training (default 0.9 = 90%).

    Returns:
        Tuple of (train_data, val_data) as LongTensors.
    """
    # Read raw text
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Dataset: {len(text):,} characters")

    # Build vocabulary if not already built
    if tokenizer.vocab_size == 0:
        tokenizer.build_vocab(text)

    print(f"Vocabulary: {tokenizer.vocab_size} unique characters")

    # Tokenize the entire text into a single tensor
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    print(f"Tokens: {len(data):,}")

    # Split into train and validation
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Train: {len(train_data):,} tokens | Val: {len(val_data):,} tokens")

    return train_data, val_data


def get_batch(
    split: str,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a random batch of (input, target) pairs.

    Randomly samples `batch_size` starting positions from the data,
    then extracts chunks of length `block_size`.

    Args:
        split: "train" or "val" to select which data to sample from.
        train_data: Training data tensor.
        val_data: Validation data tensor.
        block_size: Context window length.
        batch_size: Number of sequences per batch.
        device: Device to place tensors on ("cpu", "cuda", "mps").

    Returns:
        Tuple of (x, y) where:
            x: Input tokens of shape (batch_size, block_size).
            y: Target tokens of shape (batch_size, block_size),
               shifted right by 1 position.
    """
    data = train_data if split == "train" else val_data

    # Random starting indices (ensuring we don't overflow)
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Stack into batch tensors
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    # Move to device
    x = x.to(device)
    y = y.to(device)

    return x, y