import torch
from torch.nn.utils.rnn import pad_sequence


def pad_target_sequence(batch):
    """Collate function for the dataloader.

    Automatically adds padding to the target of each batch.
    """
    # Extract samples and targets from the batch
    samples, targets = zip(*batch)

    # Pad the target sequences to the same length
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)

    # Return padded samples and targets
    return torch.stack(samples), padded_targets
