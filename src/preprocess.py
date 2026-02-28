"""
preprocess.py – Split source_toxic.pt into a train and a val .pt file.

Run this once before training:

    python -m src.preprocess
    python -m src.preprocess --source_path data/source_toxic.pt \\
                             --output_dir  data \\
                             --val_split   0.2 \\
                             --seed        42

Outputs
-------
<output_dir>/train.pt  – dict with keys 'images', 'labels'
<output_dir>/val.pt    – dict with keys 'images', 'labels'
"""

import argparse
import os

import torch


def preprocess(
    source_path: str,
    output_dir: str,
    val_split: float,
    seed: int,
) -> None:
    print(f"Loading source data from: {source_path}")
    data = torch.load(source_path, weights_only=True)
    images = data["images"].float()
    labels = data["labels"].long()
    n_total = len(images)

    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.pt")
    val_path = os.path.join(output_dir, "val.pt")

    torch.save({"images": images[train_idx], "labels": labels[train_idx]}, train_path)
    torch.save({"images": images[val_idx], "labels": labels[val_idx]}, val_path)

    print(f"Saved {n_train} training samples  → {train_path}")
    print(f"Saved {n_val}  validation samples → {val_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split source_toxic.pt into separate train/val .pt files."
    )
    parser.add_argument(
        "--source_path",
        default="data/source_toxic.pt",
        help="Path to source_toxic.pt (default: data/source_toxic.pt)",
    )
    parser.add_argument(
        "--output_dir",
        default="data",
        help="Directory to write train.pt and val.pt (default: data)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Fraction of data held out for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting (default: 42)",
    )
    args = parser.parse_args()

    preprocess(
        source_path=args.source_path,
        output_dir=args.output_dir,
        val_split=args.val_split,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
