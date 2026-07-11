from dataclasses import dataclass


@dataclass
class GPTConfig:
    """Model architecture configuration.

    These defaults produce a ~10.7M parameter model suitable for
    character-level language modeling on Tiny Shakespeare.

    Attributes:
        vocab_size: Number of unique tokens (65 chars in Tiny Shakespeare).
        block_size: Maximum context length (sequence length).
        n_layer: Number of stacked Transformer blocks.
        n_head: Number of attention heads per block.
        n_embd: Embedding dimensionality (must be divisible by n_head).
        dropout: Dropout probability applied after attention, FFN, and embeddings.
    """

    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2


@dataclass
class TrainConfig:
    """Training hyperparameters.

    Attributes:
        max_iters: Total number of training iterations.
        eval_interval: Evaluate train/val loss every N iterations.
        eval_iters: Number of batches to average over when estimating loss.
        learning_rate: Peak learning rate for AdamW optimizer.
        batch_size: Number of independent sequences per training step.
        weight_decay: L2 regularization coefficient for AdamW.
        warmup_iters: Linear LR warmup steps before cosine decay begins.
        min_lr: Minimum learning rate at the end of cosine decay.
        seed: Random seed for reproducibility.
        device: Compute device — "auto" detects cuda > mps > cpu.
        checkpoint_dir: Directory to save model checkpoints.
        data_path: Path to the training text file.
        train_split: Fraction of data used for training (rest is validation).
        log_interval: Print training loss every N iterations.
        sample_interval: Generate sample text every N iterations.
        sample_tokens: Number of tokens to generate in each sample.
    """

    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 100
    learning_rate: float = 3e-4
    batch_size: int = 32
    weight_decay: float = 1e-1
    warmup_iters: int = 100
    min_lr: float = 3e-5
    seed: int = 1337
    device: str = "auto"
    checkpoint_dir: str = "checkpoints"
    data_path: str = "data/input.txt"
    train_split: float = 0.9
    log_interval: int = 100
    sample_interval: int = 1000
    sample_tokens: int = 200
