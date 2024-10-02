import argparse
import os

import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from dataset import ICVLPDataset, SynthDataset, BikeDataset
from model import LPRNet, SpatialTransformerLayer, LocNet
from utils import pad_target_sequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("-s", "--seed", type=int, default=None)
    args = parser.parse_args()

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

    os.makedirs("stn_results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loc = LocNet()
    stn = SpatialTransformerLayer(localization=loc, align_corners=True)

    model = LPRNet(stn=stn)

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    ds = ICVLPDataset(
        subset="test",
        transform=img_transforms,
        download=True,
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=32, shuffle=False, collate_fn=pad_target_sequence
    )

    with torch.inference_mode():
        images, targets = next(iter(dl))
        results = model.stn(images)

    original_grid = make_grid(images, nrow=8)
    results_grid = make_grid(results, nrow=8)

    original_grid_img = to_pil_image(original_grid)
    results_grid_img = to_pil_image(results_grid)

    for i, (original, result) in enumerate(zip(images, results)):
        to_pil_image(original).save(f"stn_results/original_{i}.jpeg")
        to_pil_image(result).save(f"stn_results/result_{i}.jpeg")

    original_grid_img.save("stn_results/grid_original.jpeg")
    results_grid_img.save("stn_results/grid_results.jpeg")

    to_pil_image(
        make_grid(
            torch.cat(
                [original_grid.unsqueeze_(dim=0), results_grid.unsqueeze_(dim=0)], dim=0
            ),
            nrow=1,
        )
    ).save("stn_results/grid_combined.jpeg")
