"""
Training script for Mini-GPT.

End-to-end training pipeline that:
    1. Loads the Tiny Shakespeare dataset
    2. Builds a character-level tokenizer
    3. Creates the GPT model
    4. Trains with AdamW optimizer and cosine LR scheduling
    5. Periodically evaluates on train/val sets
    6. Generates sample text at intervals
    7. Saves the best checkpoint

Usage:
    python train.py

The default config trains a ~10.7M parameter model for 5000 iterations.
On a Mac M-series GPU (MPS): ~10-15 minutes
On CPU: ~30-45 minutes
On NVIDIA GPU: ~3-5 minutes

Expected results:
    - Starting loss: ~4.17 (random = ln(65))
    - Final loss: ~1.4-1.5
    - Generated text: recognizable Shakespeare-like output
"""

import math
import os

import torch
from tqdm import tqdm

from config import GPTConfig, TrainConfig
from model.gpt import GPT
from tokenizer.tokenizer import CharTokenizer
from training.dataset import load_dataset, get_batch
from utils.helpers import (
    set_seed,
    get_device,
    count_parameters,
    estimate_loss,
    save_checkpoint,
)


def get_lr(iteration: int, train_config: TrainConfig) -> float:
    """Compute learning rate with linear warmup + cosine decay.

    This schedule:
        1. Linearly increases LR from 0 to peak during warmup
        2. Cosine-decays LR from peak to min_lr over remaining iterations

    Args:
        iteration: Current training iteration.
        train_config: Training configuration.

    Returns:
        Learning rate for this iteration.
    """
    # Linear warmup phase
    if iteration < train_config.warmup_iters:
        return train_config.learning_rate * (iteration + 1) / train_config.warmup_iters

    # After all iterations, return minimum LR
    if iteration >= train_config.max_iters:
        return train_config.min_lr

    # Cosine decay phase
    progress = (iteration - train_config.warmup_iters) / (
        train_config.max_iters - train_config.warmup_iters
    )
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return train_config.min_lr + (train_config.learning_rate - train_config.min_lr) * cosine_decay


def train():
    """Run the full training pipeline."""

    # =========================================================
    # Configuration
    # =========================================================
    gpt_config = GPTConfig()
    train_config = TrainConfig()

    # =========================================================
    # Setup
    # =========================================================
    set_seed(train_config.seed)
    device = get_device(train_config.device)

    print("=" * 60)
    print("  Mini-GPT Training")
    print("=" * 60)
    print(f"  Device:       {device}")
    print(f"  Seed:         {train_config.seed}")
    print()

    # =========================================================
    # Data
    # =========================================================
    print("--- Loading Data ---")
    tokenizer = CharTokenizer()
    train_data, val_data = load_dataset(
        data_path=train_config.data_path,
        tokenizer=tokenizer,
        train_split=train_config.train_split,
    )
    print()

    # Update vocab_size from actual data
    gpt_config.vocab_size = tokenizer.vocab_size

    # =========================================================
    # Model
    # =========================================================
    print("--- Creating Model ---")
    model = GPT(gpt_config)
    model = model.to(device)
    print(f"  Architecture: {gpt_config.n_layer} layers, {gpt_config.n_head} heads, {gpt_config.n_embd} dim")
    print(f"  Block size:   {gpt_config.block_size}")
    print(f"  Vocab size:   {gpt_config.vocab_size}")
    print(f"  Parameters:   {count_parameters(model)}")
    print(f"  Dropout:      {gpt_config.dropout}")
    print()

    # =========================================================
    # Optimizer
    # =========================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )

    # =========================================================
    # Training Loop
    # =========================================================
    print("--- Training ---")
    print(f"  Max iterations:  {train_config.max_iters}")
    print(f"  Batch size:      {train_config.batch_size}")
    print(f"  Learning rate:   {train_config.learning_rate}")
    print(f"  Weight decay:    {train_config.weight_decay}")
    print()

    best_val_loss = float("inf")
    checkpoint_path = os.path.join(train_config.checkpoint_dir, "best_model.pt")

    progress_bar = tqdm(range(train_config.max_iters), desc="Training", unit="iter")

    for iteration in progress_bar:

        # --- Learning Rate Schedule ---
        lr = get_lr(iteration, train_config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # --- Evaluate periodically ---
        if iteration % train_config.eval_interval == 0 or iteration == train_config.max_iters - 1:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                eval_iters=train_config.eval_iters,
                block_size=gpt_config.block_size,
                batch_size=train_config.batch_size,
                device=device,
            )

            progress_bar.write(
                f"\n  Step {iteration:5d} | "
                f"Train Loss: {losses['train']:.4f} | "
                f"Val Loss: {losses['val']:.4f} | "
                f"LR: {lr:.2e}"
            )

            # Save best model
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=iteration,
                    loss=best_val_loss,
                    path=checkpoint_path,
                )

        # --- Generate sample text periodically ---
        if iteration > 0 and iteration % train_config.sample_interval == 0:
            model.eval()
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated = model.generate(
                context,
                max_new_tokens=train_config.sample_tokens,
                temperature=0.8,
                top_k=40,
            )
            text = tokenizer.decode(generated[0].tolist())
            progress_bar.write(f"\n  --- Sample (step {iteration}) ---")
            progress_bar.write(f"  {text[:200]}")
            progress_bar.write(f"  ---")
            model.train()

        # --- Training Step ---
        x, y = get_batch(
            "train",
            train_data,
            val_data,
            gpt_config.block_size,
            train_config.batch_size,
            device,
        )

        # Forward pass
        _, loss = model(x, y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Update progress bar
        progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

    # =========================================================
    # Final Evaluation & Generation
    # =========================================================
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)

    final_losses = estimate_loss(
        model=model,
        train_data=train_data,
        val_data=val_data,
        eval_iters=train_config.eval_iters,
        block_size=gpt_config.block_size,
        batch_size=train_config.batch_size,
        device=device,
    )
    print(f"  Final Train Loss: {final_losses['train']:.4f}")
    print(f"  Final Val Loss:   {final_losses['val']:.4f}")
    print(f"  Best Val Loss:    {best_val_loss:.4f}")
    print()

    # Generate a final sample
    print("--- Generated Shakespeare ---")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(
        context,
        max_new_tokens=500,
        temperature=0.8,
        top_k=40,
    )
    print(tokenizer.decode(generated[0].tolist()))
    print()
    print(f"  Checkpoint saved to: {checkpoint_path}")
    print("=" * 60)


if __name__ == "__main__":
    train()