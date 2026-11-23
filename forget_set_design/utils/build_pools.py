#!/usr/bin/env python3
"""
This script builds a forget pool and its batch assignment from the difficulty scores
(spec.md sections 4.1 and 4.3).

The script reads only the scores file of compute_scores.py. It does not read audit
results (spec.md section 6, rule 1).

Pools:
    hard         the points with the highest score in each class
    easy         the points with the lowest score in each class
    global_hard  the points with the highest score of all points. The classes do not
                 have the same number of points. This pool is not in spec.md.
    random       the points of an assignment file, for example the C1 pool that
                 make_custom_split.py --random --save_assignment wrote. --pool-file
                 gives the file. The script uses the points and ignores the batches.
The hard pool and the easy pool have the same number of points for each class (45 for
4,500 points).
If two points have the same score, the point with the lower global index is first.

Assignments (--assign-seed sets the seed of each random selection):
    rand    Put the pool points into the batches at random.
    label   Put the points of the same superclass into the same batch. CIFAR-100 has
            20 superclasses of 5 classes. The script puts the superclasses in a random
            order, sorts the points by (superclass, class, score) and cuts the list
            into batches. A batch of 750 points contains 3 or 4 superclasses.
    grad    Put the points with a similar gradient direction into the same batch.
            The script does a spherical k-means with clusters of the same size. Each
            cluster is one batch. --directions gives the directions.
    spread  The opposite of grad. The script calculates the grad clusters. Then each
            batch gets the same number of points from each cluster.

Output files in --out-dir:
    assignment_<alg>_<pool>_<assignment>.csv    columns: global_idx, batch_id
    assignment_<alg>_<pool>_<assignment>.json   the settings, the seed and the score
                                                summary of each batch
--out-stem replaces the file name without the extension.
make_custom_split.py --assignment reads the .csv file.

Example:
    python3 forget_set_design/utils/build_pools.py --pool hard \\
        --scores-csv /tmp2/$USER/audit-unlearning/scores/scores_r2d.csv \\
        --out-dir /tmp2/$USER/audit-unlearning/data/cifar100/data_split
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_custom_split import DEFAULT_DATA_DIR  # noqa: E402

POOLS = ("hard", "easy", "global_hard", "random")
ASSIGNMENTS = ("rand", "label", "grad", "spread")
MAX_ITERATIONS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a forget pool and its batch assignment from the scores.")
    parser.add_argument("--scores-csv", required=True, help="The scores file from compute_scores.py.")
    parser.add_argument("--pool", required=True, choices=POOLS,
                        help="hard: the highest scores in each class. easy: the lowest scores in each class. "
                             "global_hard: the highest scores of all points. "
                             "random: the points of --pool-file.")
    parser.add_argument("--pool-file", default=None,
                        help="An assignment file with the column global_idx. Necessary for the random pool.")
    parser.add_argument("--assignment", default="rand", choices=ASSIGNMENTS,
                        help="The rule that puts the pool points into batches.")
    parser.add_argument("--alg", default="r2d", help="The name of the algorithm in the names of the output files.")
    parser.add_argument("--score-column", default="P_hat_a2", help="The column of the scores file to use.")
    parser.add_argument("--n-batches", type=int, default=6, help="The number of forget batches.")
    parser.add_argument("--batch-size", type=int, default=750, help="The number of points in each forget batch.")
    parser.add_argument("--assign-seed", type=int, default=0, help="The seed of the random selections.")
    parser.add_argument("--directions", default=None,
                        help="The directions file from compute_directions.py (.npy). The file with the same "
                             "name and the extension .json must be in the same directory. "
                             "Necessary for grad and spread.")
    parser.add_argument("--cifar-train-file",
                        default=str(DEFAULT_DATA_DIR / "extracted" / "cifar-100-python" / "train"),
                        help="The original CIFAR-100 train file. It gives the superclass of each class. "
                             "Necessary for label.")
    parser.add_argument("--out-dir", required=True, help="The output directory.")
    parser.add_argument("--out-stem", default=None,
                        help="The name of the output files without the extension. "
                             "The default is assignment_<alg>_<pool>_<assignment>.")
    return parser.parse_args()


def read_scores(path: Path, column: str):
    """Read the scores file. Return (global indices, labels, scores)."""
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"global_idx", "label", column} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"ERROR: {path} does not contain the columns {sorted(missing)}.")
        rows = [(int(r["global_idx"]), int(r["label"]), float(r[column])) for r in reader]
    global_idx = np.array([r[0] for r in rows], dtype=np.int64)
    labels = np.array([r[1] for r in rows], dtype=np.int64)
    scores = np.array([r[2] for r in rows], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise SystemExit(f"ERROR: The column {column} of {path} contains values that are not finite.")
    if np.unique(global_idx).shape[0] != global_idx.shape[0]:
        raise SystemExit(f"ERROR: {path} contains the same global index more than one time.")
    return global_idx, labels, scores


def select_pool(global_idx, labels, scores, pool: str, per_class: int) -> np.ndarray:
    """Return the positions of the pool points: `per_class` points for each class."""
    selected = []
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        if members.shape[0] < 2 * per_class:
            raise SystemExit(f"ERROR: Class {c} has {members.shape[0]} points. The hard pool and the easy pool "
                             f"need {2 * per_class} points to have no common point.")
        # Sort by score, then by global index. hard uses the descending score order.
        key = -scores[members] if pool == "hard" else scores[members]
        order = np.lexsort((global_idx[members], key))
        selected.append(members[order[:per_class]])
    return np.concatenate(selected)


def superclass_of_class(path: Path, n_classes: int) -> np.ndarray:
    """Read the original CIFAR-100 train file. Return the superclass of each class."""
    with open(path, "rb") as fh:
        data = pickle.load(fh, encoding="bytes")
    fine = np.asarray(data[b"fine_labels"], dtype=np.int64)
    coarse = np.asarray(data[b"coarse_labels"], dtype=np.int64)
    result = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        values = np.unique(coarse[fine == c])
        if values.shape[0] != 1:
            raise SystemExit(f"ERROR: Class {c} does not have one superclass in {path}.")
        result[c] = values[0]
    return result


def load_directions(path: Path, global_idx: np.ndarray, pool_pos: np.ndarray) -> np.ndarray:
    """Return the unit direction of each pool point, shape [pool points, dimensions], float64."""
    with open(path.with_suffix(".json")) as fh:
        rows = np.array(json.load(fh)["global_idx"], dtype=np.int64)
    if not np.array_equal(rows, global_idx):
        raise SystemExit(f"ERROR: The rows of {path} and the rows of the scores file are not the same points.")
    u = np.load(path, mmap_mode="r")[np.sort(pool_pos)].astype(np.float64)
    u = u[np.argsort(np.argsort(pool_pos))]
    return u / np.linalg.norm(u, axis=1, keepdims=True)


def balanced_kmeans(u: np.ndarray, k: int, size: int, rng: np.random.Generator):
    """
    Do a spherical k-means with exactly `size` points in each cluster.
    Return (cluster of each point, number of iterations, True if the clusters became stable).
    """
    from scipy.optimize import linear_sum_assignment

    def centers_of(cluster: np.ndarray) -> np.ndarray:
        centers = np.stack([u[cluster == j].sum(0) for j in range(k)])
        return centers / np.linalg.norm(centers, axis=1, keepdims=True)

    # Start: an ordinary spherical k-means from k random points.
    centers = u[rng.choice(len(u), size=k, replace=False)]
    for _ in range(MAX_ITERATIONS):
        cluster = (u @ centers.T).argmax(1)
        if np.bincount(cluster, minlength=k).min() == 0:
            break
        centers = centers_of(cluster)

    # Each center occurs `size` times in the cost matrix. Thus each cluster gets `size` points.
    cluster = None
    for iteration in range(1, MAX_ITERATIONS + 1):
        cost = np.repeat(1.0 - u @ centers.T, size, axis=1)
        _, columns = linear_sum_assignment(cost)
        new_cluster = columns // size
        if cluster is not None and np.array_equal(new_cluster, cluster):
            return cluster, iteration, True
        cluster = new_cluster
        centers = centers_of(cluster)
    return cluster, MAX_ITERATIONS, False


def assign_batches(args, pool_pos, global_idx, labels, scores, rng):
    """
    Put the pool points into batches.
    Return (pool positions in the output order, batch of each of these points, notes for the report).
    """
    n_pool, k, size = len(pool_pos), args.n_batches, args.batch_size
    chunks = np.arange(n_pool) // size
    notes = {}

    if args.assignment == "rand":
        # A random order of the pool, then batch 0 gets the first `batch_size` points.
        return pool_pos[rng.permutation(n_pool)], chunks, notes

    if args.assignment == "label":
        superclass = superclass_of_class(Path(args.cifar_train_file), int(labels.max()) + 1)
        position = np.argsort(rng.permutation(superclass.max() + 1))
        coarse = superclass[labels[pool_pos]]
        order = np.lexsort((global_idx[pool_pos], scores[pool_pos], labels[pool_pos], position[coarse]))
        pool_pos, coarse = pool_pos[order], coarse[order]
        notes["superclass_order"] = np.argsort(position).tolist()
        notes["superclasses_in_batch"] = [
            {str(c): int((coarse[chunks == b] == c).sum()) for c in np.unique(coarse[chunks == b])}
            for b in range(k)]
        return pool_pos, chunks, notes

    # grad and spread use the same clusters.
    if not args.directions:
        raise SystemExit(f"ERROR: The assignment {args.assignment} needs --directions.")
    u = load_directions(Path(args.directions), global_idx, pool_pos)
    cluster, iterations, stable = balanced_kmeans(u, k, size, rng)
    notes.update({"directions": str(args.directions), "kmeans_iterations": iterations, "kmeans_stable": stable})
    if not stable:
        print(f"[pools] WARNING: The clusters were not stable after {MAX_ITERATIONS} iterations.")
    if args.assignment == "grad":
        batch = cluster
    else:
        # Deal the points of each cluster to the batches in turn.
        batch = np.zeros(n_pool, dtype=np.int64)
        for j in range(k):
            members = np.where(cluster == j)[0]
            batch[members[rng.permutation(len(members))]] = np.arange(len(members)) % k
    notes["cluster_counts_in_batch"] = [np.bincount(cluster[batch == b], minlength=k).tolist() for b in range(k)]
    order = np.lexsort((global_idx[pool_pos], batch))
    return pool_pos[order], batch[order], notes


def main() -> int:
    args = parse_args()
    global_idx, labels, scores = read_scores(Path(args.scores_csv), args.score_column)

    n_pool = args.n_batches * args.batch_size
    n_classes = np.unique(labels).shape[0]
    if args.pool == "random":
        if not args.pool_file:
            raise SystemExit("ERROR: The random pool needs --pool-file.")
        with open(args.pool_file, newline="") as fh:
            wanted = np.array([int(r["global_idx"]) for r in csv.DictReader(fh)], dtype=np.int64)
        row_of = {int(g): i for i, g in enumerate(global_idx)}
        if len(wanted) != n_pool or len(set(wanted.tolist())) != n_pool or any(int(g) not in row_of for g in wanted):
            raise SystemExit(f"ERROR: {args.pool_file} must contain {n_pool} different points of the scores file.")
        # Use the ascending global index order. Thus the order in the file has no effect.
        pool_pos = np.array([row_of[int(g)] for g in np.sort(wanted)], dtype=np.int64)
        per_class = None
    elif args.pool == "global_hard":
        # Sort all points by descending score, then by global index.
        pool_pos = np.lexsort((global_idx, -scores))[:n_pool]
        per_class = None
    else:
        if n_pool % n_classes:
            raise SystemExit(f"ERROR: The pool size {n_pool} is not a multiple of the number of classes {n_classes}.")
        per_class = n_pool // n_classes
        pool_pos = select_pool(global_idx, labels, scores, args.pool, per_class)
    pool_class_counts = np.bincount(labels[pool_pos], minlength=n_classes)

    rng = np.random.default_rng(args.assign_seed)
    pool_pos, batch_id, notes = assign_batches(args, pool_pos, global_idx, labels, scores, rng)
    if not np.array_equal(np.bincount(batch_id, minlength=args.n_batches),
                          np.full(args.n_batches, args.batch_size)):
        raise SystemExit("ERROR: The batches do not have the same number of points.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_stem or f"assignment_{args.alg}_{args.pool}_{args.assignment}"
    with open(out_dir / f"{stem}.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["global_idx", "batch_id"])
        writer.writerows(zip(global_idx[pool_pos].tolist(), batch_id.tolist()))

    # Spec.md section 4.4, item 4: the score summary and the class counts of each batch.
    def summary(values: np.ndarray) -> dict:
        return {"mean": float(values.mean()), "median": float(np.median(values)),
                "min": float(values.min()), "max": float(values.max())}

    # The rank fraction is 0 for the lowest score of all points and 1 for the highest score.
    rank_fraction = np.argsort(np.argsort(scores)) / (scores.shape[0] - 1)
    batches = []
    for b in range(args.n_batches):
        members = pool_pos[batch_id == b]
        counts = np.bincount(labels[members], minlength=n_classes)
        batches.append({"batch_id": b, "score": summary(scores[members]),
                        "class_count_min": int(counts.min()), "class_count_max": int(counts.max()),
                        "class_counts": counts.tolist()})
    report = {
        "alg": args.alg, "pool": args.pool, "pool_file": args.pool_file, "assignment": args.assignment,
        "scores_csv": str(args.scores_csv), "score_column": args.score_column,
        "assign_seed": args.assign_seed, "n_batches": args.n_batches, "batch_size": args.batch_size,
        "points_per_class": per_class,
        "pool_class_counts": pool_class_counts.tolist(),
        "pool_classes_with_points": int((pool_class_counts > 0).sum()),
        "score_all_points": summary(scores), "score_pool": summary(scores[pool_pos]),
        "rank_fraction_pool_mean": float(rank_fraction[pool_pos].mean()),
        "batches": batches,
    }
    report.update(notes)
    with open(out_dir / f"{stem}.json", "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"[pools] {args.pool} pool: {n_pool} points from {report['pool_classes_with_points']} of {n_classes} "
          f"classes, {pool_class_counts.min()} to {pool_class_counts.max()} points for each class, "
          f"score column {args.score_column}, assignment {args.assignment}, seed {args.assign_seed}")
    print(f"[pools] median score: pool {report['score_pool']['median']:.4g}, "
          f"all points {report['score_all_points']['median']:.4g}. "
          f"Mean rank fraction of the pool: {report['rank_fraction_pool_mean']:.3f}")
    print(f"[pools] The script wrote {out_dir / (stem + '.csv')} and {stem}.json.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
