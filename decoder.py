import torch

from torch import nn
from torch.nn import functional as F


class GreedyCTCDecoder(nn.Module):
    """Greedy search decoder for CTC.

    Args:
        blank (int): Index of the blank token in the vocabulary.
    """

    def __init__(self, blank: int = 0):
        super().__init__()
        self.blank = blank

    def forward(self, logits: torch.Tensor):
        """Greedy search decoder.

        Args:
            logits (torch.Tensor): Logits of shape (N, C, T).

        Returns:
            Decoded sequence of shape (N, T).
        """
        if logits.dim() != 3:
            raise ValueError("Expected a 3D tensor.")

        logits = logits.permute(2, 0, 1)
        probs = F.log_softmax(logits, dim=2)  # (T, N, C)

        pred_indices = torch.argmax(probs, dim=2)  # (T, N)

        decoded_sequences = []

        for i in range(pred_indices.size(1)):  # Loop over batch dimension (N)
            pred = pred_indices[:, i]  # Get predictions for batch i (shape: T)

            pred = pred.cpu().numpy()

            decoded_sequence = []
            prev_token = None

            for token in pred:
                if token != self.blank and token != prev_token:
                    decoded_sequence.append(token)
                prev_token = token

            decoded_sequences.append(decoded_sequence)

        return decoded_sequences
