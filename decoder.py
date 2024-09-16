import warnings

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


class BeamCTCDecoder(nn.Module):
    """Beam search decoder for CTC.

    Args:
        beam_width (int): Beam width.
        blank (int): Index of the blank token in the vocabulary.
    """

    def __init__(self, beam_width: int = 5, blank: int = 0):
        super().__init__()

        warnings.warn(
            "BeamCTCDecoder is calculated in CPU. It may be slow for training. Use `GreedyCTCDecoder` instead.",
            UserWarning,
        )

        self.beam_width = beam_width
        self.blank = blank

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Beam search decoder.

        Args:
            logits (torch.Tensor): Logits of shape (N, C, T).

        Returns:
            Decoded sequence of shape (N, T).
        """
        if logits.dim() != 3:
            raise ValueError("Expected a 3D tensor.")

        logits = logits.permute(0, 2, 1)  # (N, T, C)
        probs = F.log_softmax(logits, dim=2)  # (N, T, C)

        sequences = self.beam_search(probs)

        return self.decode(sequences)

    def decode(self, sequences: torch.Tensor) -> list:
        decoded_sequence = []
        for seq in sequences:
            prev_token = None
            decoded_seq = []
            for token in seq:
                if token != self.blank and token != prev_token:
                    decoded_seq.append(token.item())
                prev_token = token
            decoded_sequence.append(decoded_seq)

        return decoded_sequence

    def beam_search(self, log_probs: torch.Tensor):
        N, T, C = log_probs.shape
        sequences = []

        for batch_idx in range(N):
            # Initialize the beams with an empty sequence and a log probability of 0
            beams = [
                (torch.tensor([], dtype=torch.long, device=log_probs.device), 0)
            ]  # (sequence, score)

            # Iterate over time steps
            for t in range(T):
                new_beams = []

                # Get the log probabilities at the current time step
                current_log_probs = log_probs[batch_idx, t]  # Shape: (C,)

                # Expand each beam
                for seq, score in beams:
                    # Get the top-k candidates from the current time step
                    topk_log_probs, topk_indices = current_log_probs.topk(
                        self.beam_width
                    )

                    # Create new beams with the top-k candidates
                    for i in range(self.beam_width):
                        new_seq = torch.cat(
                            [seq, topk_indices[i].unsqueeze(0)]
                        )  # Add new character
                        new_score = score + topk_log_probs[i].item()  # Update score
                        new_beams.append((new_seq, new_score))

                # Keep the top beam_width beams
                new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[
                    : self.beam_width
                ]
                beams = new_beams

            # Store the most probable sequence for this batch item
            sequences.append(beams[0][0])  # Take the best beam

        return sequences
