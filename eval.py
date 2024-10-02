import argparse
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from dataset import ICVLPDataset
from decoder import GreedyCTCDecoder, BeamCTCDecoder
from metrics import LetterNumberRecognitionRate
from model import LPRNet, SpatialTransformerLayer, LocNet
from utils import Converter, pad_target_sequence



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint", type=str, default=None, help="Path to the checkpoint."
    )
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "-d", "--decoder", type=str, default="greedy", help="Decoder type."
    )
    parser.add_argument(
        "--affine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use affine transformation.",
    )
    args = parser.parse_args()

    assert args.decoder in ["greedy", "beam"], "Invalid decoder type."

    if args.seed is not None:
        torch.manual_seed(args.seed)

    img_transforms = [
        transforms.ToTensor(),
        transforms.Resize((24, 94)),
    ]

    if args.affine:
        img_transforms = [
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

    if args.decoder == "greedy":
        decoder = GreedyCTCDecoder(blank=0)
    else:
        decoder = BeamCTCDecoder(blank=0)

    ds = ICVLPDataset(
        subset="test",
        transform=transforms.Compose(img_transforms),
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_target_sequence
    )

    converter = Converter()

    loc = LocNet()
    stn = SpatialTransformerLayer(localization=loc, align_corners=True)
    model = LPRNet(num_classes=len(ds.corpus_dict), stn=stn)

    recognition_rate = LetterNumberRecognitionRate(decoder=decoder, blank=0)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, weights_only=False))

    model.eval()
    running_accuracy = 0.0
    for images, targets in dl:
        with torch.inference_mode():
            probs = model(images)
        running_accuracy += recognition_rate(probs, targets)

    running_accuracy /= len(dl)
    print(f"Letter and Number Recognition Rate: {running_accuracy:.2%}")
