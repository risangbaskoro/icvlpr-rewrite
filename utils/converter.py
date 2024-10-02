from typing import Sequence

import torch

from .chars import CHARS_DICT, LABELS_DICT


class Converter:
    """Convert sequences of tokens to text."""

    def __init__(self):
        self.corpus_dict = CHARS_DICT
        self.labels_dict = LABELS_DICT

    def to_text(self, sequence: Sequence):
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()
        text = "".join([self.labels_dict[token] for token in sequence])
        return text

    def to_sequence(self, text: str):
        return [self.corpus_dict[char] for char in text]
