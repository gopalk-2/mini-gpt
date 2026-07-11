# Mini-GPT: Transformer Language Model from Scratch

A decoder-only Transformer language model built **entirely from scratch** in PyTorch — no `nn.TransformerDecoder`, no Hugging Face, no shortcuts. Every component is implemented from first principles to demonstrate deep understanding of the architecture behind GPT-2.

Trained on the [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) dataset to generate Shakespeare-like text.

Inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper.

---

## Architecture

```
Input Token IDs
      │
      ▼
┌─────────────┐     ┌─────────────────────┐
│   Token      │     │   Positional         │
│   Embedding  │  +  │   Embedding          │
└──────┬──────┘     └──────────┬──────────┘
       │                       │
       └───────────┬───────────┘
                   │
              ┌────▼────┐
              │ Dropout  │
              └────┬────┘
                   │
          ┌────────▼─────────┐
          │  Transformer      │
          │  Block × N        │
          │                   │
          │  ┌─────────────┐  │
          │  │ LayerNorm    │  │
          │  │ Multi-Head   │  │  ← Pre-LN
          │  │ Attention    │  │    Residual
          │  │ + Residual   │  │
          │  ├─────────────┤  │
          │  │ LayerNorm    │  │
          │  │ Feed-Forward │  │  ← Pre-LN
          │  │ + Residual   │  │    Residual
          │  └─────────────┘  │
          └────────┬─────────┘
                   │
            ┌──────▼──────┐
            │  LayerNorm   │
            └──────┬──────┘
                   │
          ┌────────▼─────────┐
          │   Linear Head     │  ← Weight-tied with
          │   (→ vocab logits)│    Token Embedding
          └────────┬─────────┘
                   │
                   ▼
          Next Token Prediction
```

### Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Norm placement | **Pre-LN** (GPT-2 style) | More stable gradients than Post-LN |
| Activation | **GELU** | Smoother than ReLU, standard for GPT |
| Positional encoding | **Learned** (not sinusoidal) | GPT-2 convention, simpler |
| Weight tying | **Embedding ↔ LM Head** | Reduces params, improves coherence |
| QKV projection | **Fused** (single linear) | More efficient than 3 separate |
| Tokenization | **Character-level** | Perfect for this dataset size |

---

## Components Built from Scratch

Every module lives in `model/` and is implemented without using PyTorch's built-in Transformer layers:

- **`embedding.py`** — Token embedding lookup table
- **`positional_embedding.py`** — Learned positional embeddings
- **`layer_norm.py`** — Layer Normalization (custom, not `nn.LayerNorm`)
- **`multi_head_attention.py`** — Multi-head causal self-attention with fused QKV
- **`feed_forward.py`** — Position-wise FFN with GELU activation
- **`transformer_block.py`** — Pre-LN Transformer decoder block
- **`gpt.py`** — Full GPT model with generation capabilities

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/YOUR_USERNAME/mini-gpt.git
cd mini-gpt
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Download Data

```bash
mkdir -p data
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### 3. Train

```bash
python train.py
```

Training outputs:
- Progress bar with live loss
- Periodic train/val loss evaluation
- Sample text generation every 1000 steps
- Best model checkpoint saved to `checkpoints/best_model.pt`

### 4. Generate Text

```bash
# Single prompt
python inference.py --prompt "ROMEO:" --max_tokens 500

# Interactive mode
python inference.py --interactive

# Adjust creativity
python inference.py --prompt "To be" --temperature 0.5 --top_k 20
```

### 5. Run Tests

```bash
python -m pytest tests/ -v
```

---

## Default Configuration

```python
# Model (~10.7M parameters)
vocab_size  = 65     # Tiny Shakespeare unique characters
block_size  = 256    # Context window
n_layer     = 6      # Transformer blocks
n_head      = 6      # Attention heads
n_embd      = 384    # Embedding dimension
dropout     = 0.2    # Regularization

# Training
max_iters     = 5000
batch_size    = 64
learning_rate = 3e-4   # AdamW with cosine decay
weight_decay  = 0.1
```

---

## Project Structure

```
mini-gpt/
├── config.py                  # Model & training configuration
├── train.py                   # End-to-end training script
├── inference.py               # Text generation from checkpoints
├── model/
│   ├── embedding.py           # Token embedding
│   ├── positional_embedding.py # Learned position encoding
│   ├── layer_norm.py          # LayerNorm (from scratch)
│   ├── multi_head_attention.py # Multi-head causal self-attention
│   ├── feed_forward.py        # Position-wise FFN
│   ├── transformer_block.py   # Pre-LN Transformer block
│   └── gpt.py                 # Full GPT model
├── tokenizer/
│   ├── tokenizer.py           # Character-level tokenizer
│   └── bpe.py                 # BPE exploration (standalone)
├── training/
│   └── dataset.py             # Data loading & batching
├── utils/
│   └── helpers.py             # Seed, device, checkpoints
├── tests/
│   ├── test_model.py          # Model architecture tests
│   └── test_tokenizer.py      # Tokenizer tests
├── data/
│   └── input.txt              # Tiny Shakespeare (~1MB)
├── requirements.txt
└── .gitignore
```

---

## How It Works

### Training

1. **Load** the Tiny Shakespeare text (~1.1M characters)
2. **Tokenize** into integers using character-level encoding (65 unique chars)
3. **Split** 90/10 into train and validation sets
4. **Sample batches** of random chunks: input = `[t₀, t₁, ..., t₂₅₅]`, target = `[t₁, t₂, ..., t₂₅₆]`
5. **Forward pass** through the Transformer, compute cross-entropy loss
6. **Backpropagate** and update weights with AdamW
7. **Repeat** for 5000 iterations with cosine learning rate decay

### Generation

Autoregressive sampling — generate one token at a time:
1. Feed the prompt through the model
2. Take the logits at the last position
3. Apply temperature scaling and optional top-k filtering
4. Sample from the resulting probability distribution
5. Append the new token and repeat

---

## References

- Vaswani et al., ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (2017)
- Radford et al., ["Language Models are Unsupervised Multitask Learners"](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019)
- Karpathy, [nanoGPT](https://github.com/karpathy/nanoGPT) (2023)
- Karpathy, ["Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) (YouTube, 2023)

---

## License

MIT
