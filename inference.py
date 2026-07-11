"""
Inference script for Mini-GPT.

Loads a trained checkpoint and generates text from a given prompt.
Supports both single-shot and interactive generation modes.

Usage:
    # Generate from a prompt
    python inference.py --prompt "ROMEO:" --max_tokens 500

    # Interactive mode (keep generating from different prompts)
    python inference.py --interactive

    # Adjust generation parameters
    python inference.py --prompt "To be" --temperature 0.5 --top_k 20

    # Use a specific checkpoint
    python inference.py --checkpoint checkpoints/best_model.pt --prompt "JULIET:"
"""

import argparse
import sys

import torch

from model.gpt import GPT
from tokenizer.tokenizer import CharTokenizer
from utils.helpers import get_device, load_checkpoint


def load_model(checkpoint_path: str, device: str) -> tuple[GPT, CharTokenizer]:
    """Load a trained model and tokenizer from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to load the model on.

    Returns:
        Tuple of (model, tokenizer).
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path, device=device)
    config = checkpoint["config"]

    # Rebuild tokenizer from the training data
    # (The vocab is deterministic from the text, so we rebuild it)
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer()
    tokenizer.build_vocab(text)

    # Rebuild model and load weights
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    iteration = checkpoint.get("iteration", "?")
    loss = checkpoint.get("loss", "?")
    print(f"  Loaded from iteration {iteration} (val loss: {loss})")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {device}")

    return model, tokenizer


def generate_text(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_k: int | None,
    device: str,
    stream: bool = True,
) -> str:
    """Generate text from a prompt.

    Args:
        model: Trained GPT model in eval mode.
        tokenizer: Character tokenizer.
        prompt: Starting text (or empty for unconditional generation).
        max_tokens: Number of tokens to generate.
        temperature: Sampling temperature (lower = more deterministic).
        top_k: Top-k filtering (None = no filtering).
        device: Compute device.
        stream: If True, print tokens as they're generated.

    Returns:
        The complete generated text (prompt + generated).
    """
    # Encode prompt
    if prompt:
        token_ids = tokenizer.encode(prompt)
        idx = torch.tensor([token_ids], dtype=torch.long, device=device)
    else:
        # Start with a newline for unconditional generation
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    if stream:
        # Stream tokens one at a time for a nice effect
        sys.stdout.write(prompt)
        sys.stdout.flush()

        with torch.no_grad():
            for _ in range(max_tokens):
                # Crop to block_size
                idx_cond = idx[:, -model.config.block_size :]

                # Get prediction for next token
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    threshold = top_k_vals[:, -1].unsqueeze(-1)
                    logits[logits < threshold] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, next_token], dim=1)

                # Print the new character
                char = tokenizer.decode([next_token.item()])
                sys.stdout.write(char)
                sys.stdout.flush()

        sys.stdout.write("\n")
        return tokenizer.decode(idx[0].tolist())

    else:
        # Generate all at once
        generated = model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        return tokenizer.decode(generated[0].tolist())


def interactive_mode(model, tokenizer, args, device):
    """Interactive REPL for text generation.

    Type a prompt and press Enter to generate. Type 'quit' to exit.
    """
    print("\n" + "=" * 60)
    print("  Mini-GPT Interactive Mode")
    print("=" * 60)
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-k:       {args.top_k or 'disabled'}")
    print(f"  Max tokens:  {args.max_tokens}")
    print("  Type a prompt and press Enter. Type 'quit' to exit.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if prompt.strip().lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if not prompt.strip():
            prompt = ""

        print()
        generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print()


def main():
    parser = argparse.ArgumentParser(description="Generate text with Mini-GPT")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint (default: checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Starting text for generation (default: unconditional)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=500,
        help="Number of tokens to generate (default: 500)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Top-k sampling (default: 40, 0 = disabled)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enter interactive generation mode",
    )

    args = parser.parse_args()

    # Treat top_k=0 as disabled
    if args.top_k == 0:
        args.top_k = None

    device = get_device()
    model, tokenizer = load_model(args.checkpoint, device)

    if args.interactive:
        interactive_mode(model, tokenizer, args, device)
    else:
        print()
        generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )


if __name__ == "__main__":
    main()
