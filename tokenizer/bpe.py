from collections import Counter


def get_pair_counts(tokens):

    pair_counts = Counter()

    for i in range(len(tokens) - 1):

        pair = (tokens[i], tokens[i + 1])

        pair_counts[pair] += 1

    return pair_counts
tokens = list("banana")

print(tokens)
pairs = get_pair_counts(tokens)

for pair, count in pairs.items():

    print(pair, count)
best_pair = max(pairs, key=pairs.get)

print(best_pair)