from training.dataset import TextDataset
from tokenizer.tokenizer import BPETokenizer

dataset = TextDataset("data/input.txt")

tokenizer = BPETokenizer()

vocab = tokenizer.build_initial_vocab(dataset.get_text())

print(vocab)