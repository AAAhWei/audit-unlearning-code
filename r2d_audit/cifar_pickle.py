"""
This module contains a CIFAR dataset that reads the batch files of a pre-split directory.

THIS MODULE IS A RECONSTRUCTION. The code release of the paper imports this module in
datasets.py but does not contain it. We wrote this version from the way the other files
in r2d_audit/ use it. forget_set_design/README.md gives the evidence for each decision.

A directory contains one or more .pkl files. Each file is a tuple (images, labels) or
(images, labels, ids). A dictionary with the keys data/images/x and labels/targets/y is
also correct.
    images  (N, 32, 32, 3) uint8, channels last
    labels  (N,) integer class ids
The dataset reads the files in sorted name order. The rows keep their file order.

Each item is (image, target, identity). main.py, r2d.py and mia.py read these three
values.
If transform is None, the image is the raw pixel array: float32, channels first, values
0 to 255. The audit scripts in evaluation_scripts/ give the model the same input:
`torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)`.
Thus the training and the audit use the same inputs.

If a file contains no ids, each row gets the batch index from the file name. For
example, the rows of batch_00003.pkl get the identity 3. r2d_audit/README.md and
random_forget_runner_all_combinations.load_batch_file use the same rule.
"""

import pickle
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import VisionDataset

_IMAGE_KEYS = ("data", "images", "x")
_LABEL_KEYS = ("labels", "targets", "y")
_ID_KEYS = ("identities", "ids")


def _first_present(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _read_pickle(path, file_pos):
    """
    Read one file.
    Return (images uint8 NHWC, labels int64 (N,), ids int64 (N,)).
    `file_pos` is the position of the file in the directory. It is the identity of
    the rows if the file name contains no batch index.
    """
    with open(path, "rb") as f:
        batch = pickle.load(f)

    ids = None
    if isinstance(batch, (tuple, list)):
        images, labels = batch[0], batch[1]
        if len(batch) > 2:
            ids = batch[2]
    elif isinstance(batch, dict):
        images = _first_present(batch, _IMAGE_KEYS)
        labels = _first_present(batch, _LABEL_KEYS)
        ids = _first_present(batch, _ID_KEYS)
    else:
        raise ValueError(f"The file {path} has an incorrect format: {type(batch)}.")

    images = np.asarray(images)
    # A 3-dimensional array is one sample (H, W, C). Add the batch dimension.
    if images.ndim == 3:
        images = images[np.newaxis, ...]
    if images.ndim != 4:
        raise ValueError(f"The file {path} has an incorrect image shape: {images.shape}.")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.shape[0] != images.shape[0]:
        raise ValueError(
            f"The file {path} has {images.shape[0]} images and {labels.shape[0]} labels.")

    if ids is None:
        m = re.match(r"batch_(\d+)", Path(path).stem)
        ids = np.full(images.shape[0], int(m.group(1)) if m else file_pos, dtype=np.int64)
    else:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    return images, labels, ids


class CIFARPickle(VisionDataset):
    """
    This dataset has the same interface as lacuna.Lacuna100Large.

    `targets` and `identities` contain all rows.
    `indices` selects the rows that the dataset gives.
    `reset()` selects all rows again.

    root: a directory of .pkl files, or one .pkl file.
    """

    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(str(root), transform=transform, target_transform=target_transform)

        root = Path(root)
        files = [root] if root.is_file() else sorted(root.glob("*.pkl"))
        if not files:
            raise FileNotFoundError(f"{root} contains no .pkl file.")

        parts = [_read_pickle(p, i) for i, p in enumerate(files)]
        self.data = np.concatenate([p[0] for p in parts], axis=0)
        self.targets = np.concatenate([p[1] for p in parts], axis=0)
        self.identities = np.concatenate([p[2] for p in parts], axis=0)
        self.indices = np.arange(len(self.targets))

    def __len__(self):
        return len(self.indices)

    def _get(self, index):
        img = self.data[index]
        target = int(self.targets[index])
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is None:
            # Give the raw pixels: float32, channels first, values 0 to 255.
            # The audit scripts use the same input.
            img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
        else:
            img = self.transform(Image.fromarray(img))

        return img, target, int(self.identities[index])

    def __getitem__(self, i):
        return self._get(self.indices[i])

    def get_data_by_index(self, index):
        return self._get(index)

    def reset(self):
        self.indices = np.arange(len(self.targets))
