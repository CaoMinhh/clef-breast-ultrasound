#!/usr/bin/env python3
"""Plot dataset confusion matrices and prediction probability/confidence."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


DATASETS = {
    "busi": {
        "title": "BUSI",
        "labels": ("normal", "benign", "malignant"),
        "files": {
    "CLEF-Flash / p1": "logs/busi_clef_flash_prompt1.csv",
    "CLEF-Flash / p2": "logs/busi_clef_flash_prompt2.csv",
    "CLEF-Flash / p3": "logs/busi_clef_flash_prompt3.csv",
    "CLEF / p1": "logs/busi_clef_prompt1.csv",
    "CLEF / p2": "logs/busi_clef_prompt2.csv",
    "CLEF / p3": "logs/busi_clef_prompt3.csv",
        },
    },
    "busb": {
        "title": "Dataset B",
        "labels": ("benign", "malignant"),
        "files": {
            "CLEF-Flash / p1": "logs/busb_clef_flash_prompt1.csv",
            "CLEF-Flash / p2": "logs/busb_clef_flash_prompt2.csv",
            "CLEF-Flash / p3": "logs/busb_clef_flash_prompt3.csv",
            "CLEF / p1": "logs/busb_clef_prompt1.csv",
            "CLEF / p2": "logs/busb_clef_prompt2.csv",
            "CLEF / p3": "logs/busb_clef_prompt3.csv",
        },
    },
    "busbra": {
        "title": "BUS-BRA",
        "labels": ("benign", "malignant"),
        "files": {
            "CLEF-Flash / p1": "logs/busbra_clef_flash_prompt1.csv",
            "CLEF-Flash / p2": "logs/busbra_clef_flash_prompt2.csv",
            "CLEF-Flash / p3": "logs/busbra_clef_flash_prompt3.csv",
            "CLEF / p1": "logs/busbra_clef_prompt1.csv",
            "CLEF / p2": "logs/busbra_clef_prompt2.csv",
            "CLEF / p3": "logs/busbra_clef_prompt3.csv",
        },
    },
}


def read_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            raw = json.loads(row["raw"])
            probabilities = raw.get("probabilities", {})
            predicted = row["pred"]
            rows.append({
                "true": row["true"],
                "pred": predicted,
                "pred_prob": float(probabilities.get(predicted, 0.0)),
                "confidence": float(raw.get("confidence", 0.0)),
            })
    return rows


def confusion_matrix(rows, labels):
    matrix = [[0] * len(labels) for _ in labels]
    for row in rows:
        if row["true"] in labels and row["pred"] in labels:
            matrix[labels.index(row["true"])][labels.index(row["pred"])] += 1
    return matrix


def plot_confusion_matrices(data, labels, output):
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    matrices = {name: confusion_matrix(rows, labels) for name, rows in data.items()}
    vmax = max(max(cell for row in matrix for cell in row) for matrix in matrices.values())
    vmax = max(vmax, 1)

    for axis, (name, rows) in zip(axes.flat, data.items()):
        matrix = matrices[name]
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        axis.set_title(f"{name}\nn={len(rows)}")
        axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        axis.set_yticks(range(len(labels)), labels)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for i in range(len(labels)):
            for j in range(len(labels)):
                axis.text(j, i, matrix[i][j], ha="center", va="center",
                          color="white" if matrix[i][j] > vmax / 2 else "black")

    figure.colorbar(image, ax=axes, shrink=0.8, label="Count")
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_probability_confidence(data, dataset_name, output):
    figure, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                                constrained_layout=True)
    colors = plt.cm.tab10.colors

    for color, (name, rows) in zip(colors, data.items()):
        x = range(1, len(rows) + 1)
        axes[0].plot(x, [row["pred_prob"] for row in rows], color=color,
                     linewidth=1, label=name)
        axes[1].plot(x, [row["confidence"] for row in rows], color=color,
                     linewidth=1, label=name)

    axes[0].set_title(f"{dataset_name} predicted-class probability")
    axes[0].set_ylabel("Probability")
    axes[1].set_title(f"{dataset_name} confidence")
    axes[1].set_ylabel("Confidence")
    axes[1].set_xlabel("Image index")
    for axis in axes:
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=9)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="busi")
    parser.add_argument("--out-dir", default="plots/busi")
    parser.add_argument("--csv", nargs="*", help="CSV files; defaults to all runs for the dataset")
    args = parser.parse_args()

    dataset = DATASETS[args.dataset]
    paths = [Path(path) for path in args.csv] if args.csv else [Path(path) for path in dataset["files"].values()]
    names = list(dataset["files"]) if not args.csv else [path.stem for path in paths]
    data = {name: read_csv(path) for name, path in zip(names, paths)}
    if len(data) != 6:
        raise SystemExit("Expected exactly 6 CSV files for the 2x3 comparison plot")

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrices(data, dataset["labels"], output_dir / f"{args.dataset}_confusion_matrices.png")
    plot_probability_confidence(data, dataset["title"], output_dir / f"{args.dataset}_probability_confidence.png")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
