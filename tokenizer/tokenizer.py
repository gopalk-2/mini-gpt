from collections import Counter


class BPETokenizer:

    def __init__(self):

        self.vocab = {}

    def build_initial_vocab(self, text):

        characters = sorted(list(set(text)))

        self.vocab = {

            ch: idx

            for idx, ch in enumerate(characters)

        }

        return self.vocab