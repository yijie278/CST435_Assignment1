#!/usr/bin/env python3
"""Create a 2-D cluster scatter plot from clustered_customers.csv.

Example:
  python plot_clusters.py clustered_customers.csv --x "Annual Income (k$)" --y "Spending Score (1-100)"
Requires matplotlib: python -m pip install matplotlib
"""
import argparse, csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="clustered_customers.csv")
    parser.add_argument("--x", help="Header for horizontal axis (default: first numeric feature)")
    parser.add_argument("--y", help="Header for vertical axis (default: second numeric feature)")
    parser.add_argument("--out", default="customer_clusters.png")
    args = parser.parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("Install matplotlib first: python -m pip install matplotlib")
    with open(args.input, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "cluster" not in reader.fieldnames:
            raise SystemExit("Input must be clustered_customers.csv with a 'cluster' column.")
        rows = list(reader)
        features = [h for h in reader.fieldnames if h != "cluster"]
    if not rows:
        raise SystemExit("No rows to plot.")
    x_name = args.x or (features[0] if features else None)
    y_name = args.y or (features[1] if len(features) > 1 else None)
    if not x_name or not y_name or x_name not in features or y_name not in features:
        raise SystemExit(f"Choose two feature headers from: {', '.join(features)}")
    try:
        xs = [float(r[x_name]) for r in rows]
        ys = [float(r[y_name]) for r in rows]
        labels = [int(r["cluster"]) for r in rows]
    except ValueError as exc:
        raise SystemExit(f"Selected features and cluster labels must be numeric: {exc}")
    plt.figure(figsize=(8, 6))
    for cluster in sorted(set(labels)):
        indices = [i for i, label in enumerate(labels) if label == cluster]
        plt.scatter([xs[i] for i in indices], [ys[i] for i in indices], s=24, alpha=.75, label=f"Cluster {cluster}")
    plt.xlabel(x_name); plt.ylabel(y_name); plt.title("K-Means customer segmentation"); plt.grid(True, alpha=.25); plt.legend(); plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"Saved {args.out} ({len(rows)} rows; {len(set(labels))} clusters).")

if __name__ == "__main__": main()
