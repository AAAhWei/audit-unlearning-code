"""
DataLoaders over a pre-split CIFAR directory (train/ val/ test/ retain/ forget/).

RECONSTRUCTION. The paper's release imports this module from datasets.py but does not
ship it. This version was rewritten from how datasets.py and main.py call it; see
forget_set_design/README.md for the evidence behind each choice.

datasets.get_loaders_large calls
    load_cifar_presplit(root, batch_size=..., shuffle=..., seed=..., augment=...)
and falls back to its legacy path when the result is None.

main.py then uses these keys of the returned dict:
    train_loader    training phase: root is a per-run folder whose train/ holds
                    train_combined.pkl (retain batches, then the chosen forget batches)
    valid_loader    never shuffled
    test_loader
    retain_loader   unlearning phase: root is the split itself; main.py trains on this
                    loader when resuming
    forget_loader   every batch file under root/forget
    num_forget_ids  number of points under root/forget (only used in the run name)

No 'forget_ids' key is returned on purpose: main.py would overwrite the runner's
forget_ids.json with it.

Order: files in sorted name order, rows in file order. Loaders shuffle only when
shuffle=True (run_all_combinations_forget_unlearning.sh defaults to false), each with
its own generator so they do not disturb one another. Under full-batch gradient descent
the order only changes floating-point summation order.

Inputs are raw pixels (float32, channels first, 0-255) with no normalisation, matching
the audit scripts; see cifar_pickle.py.
"""

import os

import numpy as np
import torch
import torchvision.transforms as transforms

from cifar_pickle import CIFARPickle

# split folder -> (key in the returned dict, follows `shuffle`, generator offset)
# Offsets 0, 1 and 3 mirror the legacy loaders in datasets.get_loaders_large.
_LOADERS = (
    ("train", "train_loader", True, 0),
    ("val", "valid_loader", False, None),
    ("test", "test_loader", True, 1),
    ("forget", "forget_loader", True, 3),
    ("retain", "retain_loader", True, 4),
)
_REQUIRED = ("train", "val")


def _augment_transform():
    """Crop/flip on the raw 0-255 scale. main.py always passes augment=False."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.PILToTensor(),               # uint8, CHW, 0-255
        transforms.Lambda(lambda t: t.float()),
    ])


def _has_pickles(path):
    return os.path.isdir(path) and any(f.endswith(".pkl") for f in os.listdir(path))


def load_cifar_presplit(root, batch_size=128, shuffle=True, seed=1, augment=False):
    """
    Build the loaders for a pre-split CIFAR root, or return None when root does not
    contain train/ and val/ pickles (the caller then tries its legacy layout).
    """
    root = str(root)
    if not all(_has_pickles(os.path.join(root, s)) for s in _REQUIRED):
        return None

    def _init_fn(worker_id):
        np.random.seed(int(seed) + worker_id)
        torch.manual_seed(int(seed) + worker_id)

    loaders = {}
    for split, key, follows_shuffle, offset in _LOADERS:
        path = os.path.join(root, split)
        if not _has_pickles(path):
            continue
        # Only the training data is ever augmented; retain is what unlearning trains on.
        transform = _augment_transform() if augment and split in ("train", "retain") else None
        dataset = CIFARPickle(root=path, transform=transform)

        do_shuffle = bool(shuffle) and follows_shuffle
        generator = None
        if do_shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(seed) + offset)
        loaders[key] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=do_shuffle,
            worker_init_fn=_init_fn if seed is not None else None,
            generator=generator,
            num_workers=0,
            pin_memory=False,
        )
        print(f"[cifar_presplit] {split}: {len(dataset)} samples from {path}")
        if split == "forget":
            loaders["num_forget_ids"] = len(dataset)

    return loaders
