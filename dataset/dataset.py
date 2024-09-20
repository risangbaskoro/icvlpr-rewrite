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
            label = label.replace("O", "0") # Replace O with 0 (zero)
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
    version = "20240920"

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
