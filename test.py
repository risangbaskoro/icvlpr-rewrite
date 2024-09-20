from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from dataset import ICVLPDataset
from decoder import GreedyCTCDecoder, BeamCTCDecoder
from model import LPRNet, SpatialTransformerLayer, LocNet


class Converter:
    """Convert sequences of tokens to text."""

    def __init__(self):
        self.corpus_dict = {
            "_": 0,
            "0": 1,
            "1": 2,
            "2": 3,
            "3": 4,
            "4": 5,
            "5": 6,
            "6": 7,
            "7": 8,
            "8": 9,
            "9": 10,
            "A": 11,
            "B": 12,
            "C": 13,
            "D": 14,
            "E": 15,
            "F": 16,
            "G": 17,
            "H": 18,
            "I": 19,
            "J": 20,
            "K": 21,
            "L": 22,
            "M": 23,
            "N": 24,
            "P": 25,
            "Q": 26,
            "R": 27,
            "S": 28,
            "T": 29,
            "U": 30,
            "V": 31,
            "W": 32,
            "X": 33,
            "Y": 34,
            "Z": 35,
        }

        self.labels_dict = {v: k for k, v in self.corpus_dict.items()}

    def to_text(self, sequence: Sequence):
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.tolist()
        text = "".join([self.labels_dict[token] for token in sequence])
        for suffix in text[-3:-1]:
            suffix = suffix.replace("0", "O")
            text = text.replace(suffix, suffix)
        return text

    def to_sequence(self, text: str):
        return [self.corpus_dict[char] for char in text]


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


if __name__ == "__main__":

    img_transforms = transforms.Compose(
        [
            transforms.RandomAffine(
                degrees=(-5, 5),
                translate=(0.07, 0.05),
                scale=(0.7, 1),
                shear=(-10, 10),
            ),
            transforms.RandomPerspective(
                distortion_scale=0.3,
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Resize((24, 94)),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loc = LocNet()
    stn = SpatialTransformerLayer(localization=loc, align_corners=True)

    model = LPRNet(stn=stn)

    model.load_state_dict(torch.load("checkpoints/epoch_1500.pth"))

    decoder = GreedyCTCDecoder(blank=0)

    ds = ICVLPDataset(
        subset="val",
        transform=img_transforms,
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=32, shuffle=True, collate_fn=pad_target_sequence
    )

    converter = Converter()

    images, targets = next(iter(dl))

    with torch.inference_mode():
        probs = model(images)

    preds = decoder(probs)

    for pred, target in zip(preds, targets):
        pred_text = converter.to_text(pred)
        target_text = converter.to_text(target).replace("_", "")

        print(f"{'Prediction':<20}: {pred_text}")
        print(f"{'Target':<20}: {target_text}")
        print()
