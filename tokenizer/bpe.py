"""
Byte Pair Encoding (BPE) — Learning Script.

This file is a standalone exploration of the BPE algorithm,
NOT used in the main Mini-GPT pipeline (which uses a character-level
tokenizer for Tiny Shakespeare).

BPE iteratively merges the most frequent pair of tokens to build
a subword vocabulary. This is how real tokenizers like GPT-2's
work at scale.

Kept here as a learning artifact to demonstrate understanding
of subword tokenization.

Reference:
    Sennrich et al., "Neural Machine Translation of Rare Words
    with Subword Units" (2016).
"""

from collections import Counter


def get_pair_counts(tokens: list[str]) -> Counter:
    """Count adjacent token pairs in a sequence.

    Args:
        tokens: List of individual tokens (characters or subwords).

    Returns:
        Counter mapping (token_a, token_b) pairs to their frequency.

    Example:
        >>> get_pair_counts(list("banana"))
        Counter({('a', 'n'): 2, ('n', 'a'): 2, ('b', 'a'): 1})
    """
    pair_counts: Counter = Counter()

    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        pair_counts[pair] += 1

    return pair_counts


def merge_pair(tokens: list[str], pair: tuple[str, str]) -> list[str]:
    """Merge all occurrences of a token pair into a single token.

    Args:
        tokens: Current token sequence.
        pair: The (token_a, token_b) pair to merge.

    Returns:
        New token list with the pair merged wherever it appeared.

    Example:
        >>> merge_pair(list("banana"), ('a', 'n'))
        ['b', 'an', 'an', 'a']
    """
    merged = []
    i = 0

    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
            merged.append(tokens[i] + tokens[i + 1])
            i += 2
        else:
            merged.append(tokens[i])
            i += 1

    return merged


# --- Demo ---
if __name__ == "__main__":
    text = "the cat sat on the mat"
    tokens = list(text)
    num_merges = 5

    print(f"Original: {tokens}")
    print(f"Length:   {len(tokens)}\n")

    for step in range(num_merges):
        pairs = get_pair_counts(tokens)
        if not pairs:
            break

        best_pair = max(pairs, key=pairs.get)
        tokens = merge_pair(tokens, best_pair)

        print(f"Step {step + 1}: Merged {best_pair} → '{''.join(best_pair)}'")
        print(f"  Tokens: {tokens}")
        print(f"  Length: {len(tokens)}\n")