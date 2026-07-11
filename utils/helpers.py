import os
import random

import torch
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def count_parameters(model: torch.nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if total >= 1_000_000:
        return f"{total / 1_000_000:.2f}M parameters"
    elif total >= 1_000:
        return f"{total / 1_000:.2f}K parameters"
    else:
        return f"{total} parameters"


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, block_size, batch_size, device):
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
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint
