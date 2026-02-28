"""
Script to plot accuracy and F1 score from CSV files.

CSV naming convention: q_0.5_k_0.5.csv (params can vary)
CSV columns: epoch, accuracy, F1_score

Outputs:
  - plots/<base_name>_accuracy.png
  - plots/<base_name>_f1_score.png
"""

import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt


def plot_csv(csv_path: str, output_dir: str = "plots") -> None:
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    # Extract params from filename for plot title (e.g. "q=0.5, k=0.5")
    parts = base_name.split("_")
    param_pairs = []
    i = 0
    while i < len(parts) - 1:
        try:
            float(parts[i + 1])
            param_pairs.append(f"{parts[i]}={parts[i+1]}")
            i += 2
        except ValueError:
            i += 1
    title_suffix = ", ".join(param_pairs) if param_pairs else base_name

    # --- Accuracy plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["accuracy"], marker="o", linewidth=2, color="#2196F3")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Accuracy vs Epoch ({title_suffix})", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    acc_path = os.path.join(output_dir, f"{base_name}_accuracy.png")
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {acc_path}")

    # --- F1 Score plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["epoch"], df["F1_score"], marker="s", linewidth=2, color="#FF5722")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title(f"F1 Score vs Epoch ({title_suffix})", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    f1_path = os.path.join(output_dir, f"{base_name}_f1_score.png")
    fig.savefig(f1_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {f1_path}")


def main():
    if len(sys.argv) < 2:
        # No args → process all CSVs in current directory
        csv_files = glob.glob("*.csv")
        if not csv_files:
            print("No CSV files found. Pass CSV paths as arguments or place them in the current directory.")
            sys.exit(1)
    else:
        csv_files = sys.argv[1:]

    for csv_file in csv_files:
        if not os.path.isfile(csv_file):
            print(f"File not found: {csv_file}, skipping.")
            continue
        print(f"\nProcessing: {csv_file}")
        plot_csv(csv_file)


if __name__ == "__main__":
    main()
