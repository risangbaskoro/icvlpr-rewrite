import torch

from torch import nn


class LetterNumberRecognitionRate(nn.Module):
    """The Letter and Number Recognition Rate.

    This class is used to compute the recognition rate for letter and number recognition task with the following formula:
    :math:`R_{LN} = N_{LN} / (N_{A} \times k)` where :math:`R_{LN}` is the letter and number recognition rate, :math:`N_{LN}`
    is the number of letters and numbers the system can accurately recognise, :math:`N_{A}` is the number of system location
    images with an accurate license plate region, and :math:`k` is the number of letters and numbers in each license plate.

    See: https://doi.org/10.1049/iet-its.2017.0138

    Args:
        decoder (nn.Module): A decoder module that takes logits as input and returns the decoded sequence.
        blank (int): Index of the blank token in the vocabulary.

    Shape:
        pred (Tensor): :math:`(N, C)` where `N` is the batch size and `C` is the number of classes.
        target (Tensor): :math:`(N, C)` where `N` is the batch size and `C` is the number of classes.
    """

    def __init__(
        self,
        decoder: nn.Module = None,
        blank: int = 0,
    ) -> None:
        super().__init__()

        self.decoder = decoder
        self.blank = blank

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the Letter and Number Recognition Rate.

        Args:
            logits (torch.Tensor): Logits of shape (N, C, T).
            target (torch.Tensor): Target tensor of shape (N, T).

        Returns:
            The Letter and Number Recognition Rate.
        """
        if logits.dim() != 3:
            raise ValueError("Expected a 3D tensor for logits.")

        if targets.dim() != 2:
            raise ValueError("Expected a 2D tensor for target.")

        decoded_sequences = self.decoder(logits)
        
        # Convert targets to a list of lists and then remove the blank token elements
        aligned_targets = targets.tolist()
        aligned_targets = [[token for token in sequence if token != self.blank] for sequence in aligned_targets]

        corrects = 0
        lengths = 0

        for decoded_sequence, target_sequence in zip(decoded_sequences, aligned_targets):
            num_correct = 0

            for pred, target in zip(decoded_sequence, target_sequence):
                if pred == target:
                    num_correct += 1

            corrects += num_correct
            lengths += len(target_sequence)

        return corrects / lengths
