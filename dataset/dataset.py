import os
import tarfile
import sys

import requests

import torch

from typing import Callable, Optional

from PIL import Image

from torch.utils.data import Dataset

from tqdm import tqdm

from utils import CHARS_DICT


class _Dataset(Dataset):
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

        self.corpus_dict = CHARS_DICT if corpus_dict is None else corpus_dict

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


class SynthDataset(_Dataset):
    mirrors = ["https://data.risangbaskoro.com/icvlp/synth"]

    resources = [f"train.tar.gz", f"val.tar.gz", f"test.tar.gz"]
