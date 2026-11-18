"""
CIFAR dataset backed by the batch pickles of a pre-split directory.

RECONSTRUCTION. The paper's release imports this module from datasets.py but does not
ship it. This version was rewritten from how the rest of r2d_audit/ uses it; see
forget_set_design/README.md for the evidence behind each choice.

A folder holds one or more pickles, each an (images, labels) or (images, labels, ids)
tuple (dicts with data/images/x and labels/targets/y keys are accepted too):
    images  (N, 32, 32, 3) uint8, channels last
    labels  (N,) integer class ids
Files are read in sorted filename order and rows keep their order inside each file.

Items are (image, target, identity), the triple main.py, r2d.py and mia.py unpack.
With transform=None the image is the raw pixel array as float32, channels first, values
0-255: exactly what the audit scripts feed the model
(`torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)` in evaluation_scripts/), so
training and auditing see identical inputs.

When a file carries no ids, every row gets the batch index parsed from the file name
(batch_00003.pkl -> 3), the convention of r2d_audit/README.md and of
random_forget_runner_all_combinations.load_batch_file.
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
    """Return (images uint8 NHWC, labels int64 (N,), ids int64 (N,)) of one file."""
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
        raise ValueError(f"Unexpected batch format {type(batch)} in {path}")

    images = np.asarray(images)
    if images.ndim == 3:  # a single (H, W, C) sample
        images = images[np.newaxis, ...]
    if images.ndim != 4:
        raise ValueError(f"Unexpected image shape {images.shape} in {path}")
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.shape[0] != images.shape[0]:
        raise ValueError(f"{path}: {images.shape[0]} images but {labels.shape[0]} labels")

    if ids is None:
        m = re.match(r"batch_(\d+)", Path(path).stem)
        ids = np.full(images.shape[0], int(m.group(1)) if m else file_pos, dtype=np.int64)
    else:
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    return images, labels, ids


class CIFARPickle(VisionDataset):
    """
    Same interface as lacuna.Lacuna100Large: `targets` and `identities` cover every
    stored row, `indices` selects the rows the dataset exposes, `reset()` restores all.

    root: a folder of *.pkl files, or one .pkl file.
    """

    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(str(root), transform=transform, target_transform=target_transform)

        root = Path(root)
        files = [root] if root.is_file() else sorted(root.glob("*.pkl"))
        if not files:
            raise FileNotFoundError(f"No .pkl files under {root}")

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
            # Raw pixels, float32, CHW, 0-255: identical to the audit scripts' inputs.
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
