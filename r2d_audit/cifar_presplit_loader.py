"""
This module creates the DataLoaders for a pre-split CIFAR directory.
The directory contains the subdirectories train/ val/ test/ retain/ forget/.

THIS MODULE IS A RECONSTRUCTION. The code release of the paper imports this module in
datasets.py but does not contain it. We wrote this version from the calls in datasets.py
and main.py. forget_set_design/README.md gives the evidence for each decision.

datasets.get_loaders_large calls
    load_cifar_presplit(root, batch_size=..., shuffle=..., seed=..., augment=...)
If the result is None, datasets.get_loaders_large uses its legacy directory layout.

main.py uses these keys of the returned dictionary:
    train_loader    In the training phase, root is a directory for one run.
                    Its train/ contains train_combined.pkl. That file contains the
                    retain batches first, then the selected forget batches.
    valid_loader    This loader does not shuffle.
    test_loader
    retain_loader   In the unlearning phase, root is the split directory.
                    main.py trains on this loader when it resumes from a checkpoint.
    forget_loader   This loader reads all batch files in root/forget.
    num_forget_ids  The number of points in root/forget. main.py uses this number only
                    in the name of the run.

The dictionary has no 'forget_ids' key. If it had this key, main.py would replace the
forget_ids.json file of the runner.

Order: the dataset reads the files in sorted name order. The rows keep their file order.
A loader shuffles only if shuffle=True. The default of
run_all_combinations_forget_unlearning.sh is false.
Each loader that shuffles has its own random generator. Thus one loader does not change
the order of a different loader.
With full-batch gradient descent, the order changes only the floating-point rounding.

The inputs are raw pixels: float32, channels first, values 0 to 255, no normalisation.
The audit scripts use the same inputs. Refer to cifar_pickle.py.
"""

import os

import numpy as np
import torch
import torchvision.transforms as transforms

from cifar_pickle import CIFARPickle

# Each entry is: (subdirectory, key in the returned dictionary,
#                 the loader obeys the `shuffle` argument, offset of the generator seed).
# The offsets 0, 1 and 3 are the same as in the legacy loaders of
# datasets.get_loaders_large.
_LOADERS = (
    ("train", "train_loader", True, 0),
    ("val", "valid_loader", False, None),
    ("test", "test_loader", True, 1),
    ("forget", "forget_loader", True, 3),
    ("retain", "retain_loader", True, 4),
)
_REQUIRED = ("train", "val")


def _augment_transform():
    """
    Return the crop and flip transform. The output values stay in the range 0 to 255.
    main.py always sets augment=False, thus the pipeline does not use this transform.
    """
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        # PILToTensor gives a uint8 tensor, channels first, values 0 to 255.
        transforms.PILToTensor(),
        transforms.Lambda(lambda t: t.float()),
    ])


def _has_pickles(path):
    """Return True if `path` is a directory that contains a .pkl file."""
    return os.path.isdir(path) and any(f.endswith(".pkl") for f in os.listdir(path))


def load_cifar_presplit(root, batch_size=128, shuffle=True, seed=1, augment=False):
    """
    Create the loaders for a pre-split CIFAR directory.

    Return None if `root` does not contain .pkl files in train/ and val/.
    The caller then uses its legacy directory layout.
    """
    root = str(root)
    if not all(_has_pickles(os.path.join(root, s)) for s in _REQUIRED):
        return None

    def _init_fn(worker_id):
        np.random.seed(int(seed) + worker_id)
        torch.manual_seed(int(seed) + worker_id)

    loaders = {}
    for split, key, obeys_shuffle, offset in _LOADERS:
        path = os.path.join(root, split)
        if not _has_pickles(path):
            continue
        # Augment only the data that the model trains on: train/ and retain/.
        transform = _augment_transform() if augment and split in ("train", "retain") else None
        dataset = CIFARPickle(root=path, transform=transform)

        do_shuffle = bool(shuffle) and obeys_shuffle
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
        print(f"[cifar_presplit] {split}: {len(dataset)} samples in {path}")
        if split == "forget":
            loaders["num_forget_ids"] = len(dataset)

    return loaders
