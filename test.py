import argparse
from typing import Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms

from dataset import ICVLPDataset
from decoder import GreedyCTCDecoder, BeamCTCDecoder
from model import LPRNet, SpatialTransformerLayer, LocNet
from utils import Converter, pad_target_sequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("-d", "--decoder", type=str, default="greedy")
    args = parser.parse_args()

    assert args.decoder in ["greedy", "beam"], "Invalid decoder type."

    if args.seed is not None:
        torch.manual_seed(args.seed)

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

    loc = LocNet()
    stn = SpatialTransformerLayer(localization=loc, align_corners=True)

    if args.decoder == "greedy":
        decoder = GreedyCTCDecoder(blank=0)
    else:
        decoder = BeamCTCDecoder(blank=0)

    ds = ICVLPDataset(
        subset="test",
        transform=img_transforms,
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_target_sequence
    )

    converter = Converter()

    model = LPRNet(num_classes=len(ds.corpus_dict), stn=stn)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, weights_only=False))

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

    from metrics import LetterNumberRecognitionRate

    ret = LetterNumberRecognitionRate(decoder)(probs, targets)
    print(f"Letter and Number Recognition Rate: {ret}")
