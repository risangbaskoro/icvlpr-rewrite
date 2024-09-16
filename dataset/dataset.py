import os
import tarfile

import requests

import numpy as np
import torch

from typing import Callable, Optional, Tuple

from PIL import Image

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import invert, pil_to_tensor

from tqdm import tqdm


class _Dataset(Dataset):
    corpus_dict = {
        "_": 0,
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7,
        "H": 8,
        "I": 9,
        "J": 10,
        "K": 11,
        "L": 12,
        "M": 13,
        "N": 14,
        "O": 15,
        "P": 16,
        "Q": 17,
        "R": 18,
        "S": 19,
        "T": 20,
        "U": 21,
        "V": 22,
        "W": 23,
        "X": 24,
        "Y": 25,
        "Z": 26,
        "1": 27,
        "2": 28,
        "3": 29,
        "4": 30,
        "5": 31,
        "6": 32,
        "7": 33,
        "8": 34,
        "9": 35,
        "0": 36,
    }

    labels_dict = {v: k for k, v in corpus_dict.items()}

    def __init__(
        self,
        root: str = "data",
        subset: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        corpus_dict: dict[str:int] = None,
        version: str = "latest",
    ) -> None:
        assert subset in [
            "train",
            "test",
            "val",
        ], f'Subset must be "train", "test", or "val". Got "{subset}"'

        super().__init__()
        self.root = root
        self.subset = subset
        self.version = version
        self.transform = transform
        self.target_transform = target_transform

        if corpus_dict is not None:
            self.corpus_dict = corpus_dict
            self.labels_dict = {v: k for k, v in corpus_dict.items()}

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. Use download=True to download it")

        self.data, self.targets = self._load_data()

    @property
    def class_folder(self):
        return os.path.join(self.root, self.__class__.__name__)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        target = [self.corpus_dict[char] for char in target]
        target = torch.tensor(target, dtype=torch.long)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _load_data(self):
        images_path = os.path.join(self.class_folder, self.subset)
        images = []
        labels = []
        for filename in os.listdir(images_path):
            # Images
            img_path = os.path.join(images_path, filename)
            img = Image.open(img_path)
            images.append(img)

            # Labels
            label = filename.split(".")[0]
            labels.append(label)

        return images, labels

    def _download(self):
        if self._check_exists():
            return

        os.makedirs(self.class_folder, exist_ok=True)

        for filename in self.resources:
            for mirror in self.mirrors:
                url = f"{mirror}/{filename}"

                try:
                    print(f"Downloading {url}")
                    self._download_archive(
                        url, download_root=self.class_folder, filename=filename
                    )
                except Exception as e:
                    print(f"Failed to download {url} (trying next):\n{e}")
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError(f"Error downloading {filename}")

    @staticmethod
    def _download_archive(url, download_root, filename):
        print(f"Downloading {url} to {download_root} as {filename}")

        fpath = f"{download_root}/{filename}"

        response = requests.get(url, stream=True)

        size = int(response.headers.get("content-length", 0))

        with tqdm(total=size, unit="bit") as progress:
            with open(fpath, "wb") as f:
                for data in response.iter_content(1024):
                    f.write(data)
                    progress.update(len(data))
                f.close()

        print(f"Extracting {filename} to {download_root}")
        try:
            with tarfile.open(fpath, "r:gz") as tar:
                tar.extractall(download_root)
        except Exception as e:
            print(f"Error extracting file: {e}")

    def _check_exists(self):
        return all(
            # Check if archive exists
            os.path.exists(os.path.join(self.class_folder, filename))
            for filename in self.resources
        ) and all(
            # Check if directories exists
            os.path.exists(os.path.join(self.class_folder, filename.split("_")[0]))
            for filename in self.resources
        )


class ICVLPDataset(_Dataset):
    version = "20240905"

    mirrors = ["https://data.risangbaskoro.com/icvlp/master"]

    resources = [
        f"train_{version}.tar.gz",
        f"val_{version}.tar.gz",
        f"test_{version}.tar.gz",
    ]


class BikeDataset(_Dataset):
    version = "20240905"

    mirrors = ["https://data.risangbaskoro.com/icvlp/bikes"]

    resources = [
        f"train_{version}.tar.gz",
        f"val_{version}.tar.gz",
        f"test_{version}.tar.gz",
    ]


class BikeDataset(_Dataset):
    version = "20240905"

    mirrors = ["https://data.risangbaskoro.com/icvlp/bikes"]

    resources = [
        f"train_{version}.tar.gz",
        f"val_{version}.tar.gz",
        f"test_{version}.tar.gz",
    ]


class SynthDataset(_Dataset):
    mirrors = ["https://data.risangbaskoro.com/icvlp/synth"]

    resources = [f"train.tar.gz", f"val.tar.gz", f"test.tar.gz"]
