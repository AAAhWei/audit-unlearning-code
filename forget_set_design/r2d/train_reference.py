#!/usr/bin/env python3
"""
This script trains the reference model for the difficulty scores (spec.md section 2).

The reference model trains on all 45,000 training points of a split directory.
The script saves the weights after each step. compute_scores.py reads these files.

Only --alg r2d is available. The R2D recipe is full-batch gradient descent, thus one
epoch is one step.

The script uses the code of the audit pipeline:
    datasets.get_loaders_large    the data loaders
    models.get_model              the model
    main.run_epoch                one epoch of training or evaluation
The settings and the order of the random draws are the same as in r2d_audit/main.py.
The one difference is the checkpoint frequency. main.py saves the weights each 5 epochs.
This script saves the weights after each epoch.

Output files in --out-dir:
    checkpoints/step_000.pt       the weights before training
    checkpoints/step_<t>.pt       the weights after step t (t = 1 to --epochs)
    reference_meta.json           the settings, the seed and the accuracy values

Example:
    CUDA_VISIBLE_DEVICES=4 python3 forget_set_design/r2d/train_reference.py \\
        --data-dir /tmp2/$USER/audit-unlearning/data/cifar100/data_split/cifar100_uniform_bs_750_seed1 \\
        --out-dir /tmp2/$USER/audit-unlearning/reference/r2d
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
R2D_DIR = REPO / "r2d_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the reference model and save the weights after each step.")
    parser.add_argument("--alg", default="r2d", choices=["r2d", "modelclip"],
                        help="The unlearning algorithm. Only r2d is available.")
    parser.add_argument("--data-dir", required=True,
                        help="The split directory. The model trains on all files in its train/ subdirectory.")
    parser.add_argument("--out-dir", required=True, help="The output directory.")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--model", default="tinynetcifar100")
    parser.add_argument("--epochs", type=int, default=35,
                        help="The number of training steps. The audit trains for 35 steps.")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--micro-batch-size", type=int, default=128,
                        help="The number of points in one forward pass. The step uses the gradient of all points.")
    parser.add_argument("--seed", type=int, default=3,
                        help="The seed. The audit runs use the seed 3.")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false",
                        help="Do not shuffle the rows. The audit runs shuffle the rows.")
    parser.add_argument("--device", default="cuda:0",
                        help="The torch device. Set CUDA_VISIBLE_DEVICES to select the GPU.")
    parser.add_argument("--threads", type=int, default=4, help="The number of CPU threads.")
    return parser.parse_args()


def train_r2d(args: argparse.Namespace) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

    import torch
    import torch.optim as optim

    sys.path.insert(0, str(R2D_DIR))
    import datasets
    import models
    import main as r2d_main
    from utils import manual_seed

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # The sequence below is the same as in main.py: seed, loaders, model, optimizer.
    manual_seed(args.seed)
    loaders = datasets.get_loaders_large(
        args.dataset, num_ids_forget=None, forget_ids=None,
        batch_size=args.micro_batch_size, seed=args.seed, root=args.data_dir,
        augment=False, shuffle=args.shuffle, ood=True, test=False,
    )
    train_loader = loaders["train_loader"]
    n_train = len(train_loader.dataset)
    num_classes = int(max(train_loader.dataset.targets)) + 1

    model = models.get_model(args.model, num_classes=num_classes, filters_percentage=1.0).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0, weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # main.run_epoch reads these fields. batch_size = n_train gives one step for each epoch.
    run_args = SimpleNamespace(
        device=device, quiet=False, no_gradient_estimation=False,
        batch_size=n_train, accumulation_steps=math.ceil(n_train / args.micro_batch_size),
    )
    print(f"[reference] {n_train} training points, {num_classes} classes, "
          f"{run_args.accumulation_steps} forward passes for each step")

    torch.save(model.state_dict(), ckpt_dir / "step_000.pt")
    start = time.time()
    for epoch in range(args.epochs):
        r2d_main.run_epoch(run_args, model, train_loader, criterion, optimizer, epoch, mode="train")
        r2d_main.run_epoch(run_args, model, loaders["valid_loader"], criterion, optimizer, epoch, mode="valid")
        torch.save(model.state_dict(), ckpt_dir / f"step_{epoch + 1:03d}.pt")
    train_seconds = time.time() - start

    # Spec.md section 2.3: record the accuracy of the reference model.
    accuracy = {}
    for name, key in (("train", "train_loader"), ("val", "valid_loader"), ("test", "test_loader")):
        if key in loaders:
            metrics = r2d_main.run_epoch(run_args, model, loaders[key], criterion, optimizer,
                                         args.epochs - 1, mode=f"final_{name}")
            accuracy[name] = 1.0 - metrics.avg["error"]
            print(f"[reference] final {name} accuracy: {100.0 * accuracy[name]:.2f}%")

    meta = {
        "alg": args.alg, "data_dir": str(args.data_dir), "dataset": args.dataset,
        "model": args.model, "steps": args.epochs, "lr": args.lr,
        "micro_batch_size": args.micro_batch_size, "seed": args.seed,
        "shuffle": args.shuffle, "n_train": n_train, "sampling_rate": 1.0,
        "accuracy": accuracy,
        "train_minus_test_accuracy": accuracy.get("train", float("nan")) - accuracy.get("test", float("nan")),
        "train_seconds": round(train_seconds, 1),
    }
    with open(out_dir / "reference_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[reference] The script saved {args.epochs + 1} weight files to {ckpt_dir}.")


def main() -> int:
    args = parse_args()
    if args.alg != "r2d":
        print("ERROR: --alg modelclip is not available. Only r2d is available.", file=sys.stderr)
        return 2
    train_r2d(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
