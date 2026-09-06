import re
import string
from collections import Counter
import torch

PAD_IDX = 0
UNK_IDX = 1


class TextVectorization:
    def __init__(
        self,
        max_tokens,
        output_sequence_length,
        standardize=None,
        special_tokens=None,
    ):
        self.max_tokens = max_tokens
        self.output_sequence_length = output_sequence_length
        self.standardize = standardize
        self.special_tokens = special_tokens or []

        self.vocab = ["", "[UNK]"] + self.special_tokens
        self.word_to_index = {
            word: idx
            for idx, word in enumerate(self.vocab)
        }

    def adapt(self, texts):
        """Строит словарь на основе текстов."""
        counts = Counter()
        for text in texts:
            clean_text = (
                self.standardize(text) if self.standardize else text.lower()
            )
            tokens = clean_text.split()
            counts.update(tokens)

        max_words = self.max_tokens - len(self.vocab)
        for word, _ in counts.most_common(max_words):
            if word not in self.word_to_index:
                self.word_to_index[word] = len(self.vocab)
                self.vocab.append(word)

    def get_vocabulary(self):
        return self.vocab

    def encode(self, text):
        """Преобразует текст в последовательность индексов."""
        clean_text = (
            self.standardize(text) if self.standardize else text.lower()
        )
        tokens = clean_text.split()
        ids = [
            self.word_to_index.get(token, UNK_IDX)
            for token in tokens[:self.output_sequence_length]
        ]
        ids += [PAD_IDX] * (self.output_sequence_length - len(ids))
        return ids

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        encoded_list = [self.encode(text) for text in texts]
        return torch.tensor(encoded_list, dtype=torch.long)
