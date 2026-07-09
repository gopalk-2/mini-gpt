from pathlib import Path


class TextDataset:

    def __init__(self, file_path):

        self.file_path = Path(file_path)

        self.text = self.load_text()

    def load_text(self):

        with open(self.file_path, "r", encoding="utf-8") as file:
            return file.read()

    def get_text(self):
        return self.text