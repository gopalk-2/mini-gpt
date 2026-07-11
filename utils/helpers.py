"""
Utility functions for Mini-GPT.

Helper functions used across training and inference scripts
for reproducibility, device management, and model inspection.
"""

import os
import random

import torch
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> str:
    """Detect the best available compute device.

    Priority: CUDA GPU → Apple MPS → CPU.

    Args:
        preference: "auto" to auto-detect, or force a specific device
                    ("cuda", "mps", "cpu").

    Returns:
        Device string suitable for torch.device().
    """
    if preference != "auto":
        return preference

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def count_parameters(model: torch.nn.Module) -> str:
    """Count trainable parameters and return human-readable string.

    Args:
        model: PyTorch model.

    Returns:
        Formatted string like "10.65M parameters".
    """
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total >= 1_000_000:
        return f"{total / 1_000_000:.2f}M parameters"
    elif total >= 1_000:
        return f"{total / 1_000:.2f}K parameters"
    else:
        return f"{total} parameters"


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, block_size, batch_size, device):
    """Estimate average loss over multiple batches for train and val.

    Switches model to eval mode, averages loss over `eval_iters`
    random batches, then switches back to train mode.

    Args:
        model: The GPT model.
        train_data: Training data tensor.
        val_data: Validation data tensor.
        eval_iters: Number of batches to average over.
        block_size: Context window length.
        batch_size: Batch size for evaluation.
        device: Compute device string.

    Returns:
        Dict with "train" and "val" average losses.
    """
    from training.dataset import get_batch

    model.eval()
    losses = {}

    for split in ["train", "val"]:
        total_loss = 0.0

        for _ in range(eval_iters):
            x, y = get_batch(split, train_data, val_data, block_size, batch_size, device)
            _, loss = model(x, y)
            total_loss += loss.item()

        losses[split] = total_loss / eval_iters

    model.train()
    return losses


def save_checkpoint(model, optimizer, iteration, loss, path):
    """Save a training checkpoint.

    Args:
        model: The GPT model.
        optimizer: The optimizer.
        iteration: Current training iteration.
        loss: Current loss value.
        path: File path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
            "loss": loss,
            "config": model.config,
        },
        path,
    )
    print(f"  💾 Checkpoint saved: {path}")


def load_checkpoint(path, device="cpu"):
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        device: Device to load tensors to.

    Returns:
        Dict containing model_state_dict, optimizer_state_dict,
        iteration, loss, and config.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint
