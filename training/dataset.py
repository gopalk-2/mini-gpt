import torch

from tokenizer.tokenizer import CharTokenizer


def load_dataset(
    data_path: str,
    tokenizer: CharTokenizer,
    train_split: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:

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