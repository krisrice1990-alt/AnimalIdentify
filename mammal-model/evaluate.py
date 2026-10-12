"""Evaluation: per-class precision/recall/F1 + confusion matrix.

Reads a weights.npz (from train.py) and a feature cache, evaluates on the
val and test splits, writes:
    <out_dir>/metrics.json
    <out_dir>/confusion_matrix.png   (counts + row-normalized)
and prints an honest per-class report.
"""
import argparse
import json
import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CLASSES, N_CLASSES
from train import MLPHead


def prf_per_class(y_true, y_pred):
    rows = []
    for c in range(N_CLASSES):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"class": CLASSES[c], "precision": prec, "recall": rec,
                     "f1": f1, "support": int((y_true == c).sum())})
    return rows


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_cm(cm, out_png, title):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, fmt, ttl in [
        (axes[0], cm, "d", "counts"),
        (axes[1], cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), ".2f",
         "row-normalized (recall)"),
    ]:
        im = ax.imshow(mat, cmap="Blues", vmin=0,
                       vmax=mat.max() if fmt == "d" else 1.0)
        ax.set_xticks(range(N_CLASSES), CLASSES, rotation=30, ha="right")
        ax.set_yticks(range(N_CLASSES), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{title} - {ttl}")
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        fontsize=8, color="white" if mat[i, j] > (mat.max() / 2 if fmt == "d" else 0.5) else "black")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def evaluate_split(model, X, y, name):
    yp, probs = model.predict(X.astype(np.float64))
    return {"split": name,
            "accuracy": float((yp == y).mean()),
            "macro_f1": float(np.mean([r["f1"] for r in prf_per_class(y, yp)])),
            "per_class": prf_per_class(y, yp),
            "confusion_matrix": confusion_matrix(y, yp).tolist(),
            "n": int(len(y))}


def load_model(weights_path):
    z = np.load(weights_path, allow_pickle=True)
    m = MLPHead(int(z["in_dim"]), hidden=int(z["hidden"]))
    m.W1 = z["W1"].astype(np.float64)
    m.b1 = z["b1"].astype(np.float64)
    m.W2 = z["W2"].astype(np.float64)
    m.b2 = z["b2"].astype(np.float64)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default="mammal head")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = load_model(args.weights)
    z = np.load(args.cache, allow_pickle=True)
    X, y, split = z["X"], z["y"], z["split"]
    report = {"title": args.title, "splits": {}}
    for code, name in [(1, "val"), (2, "test")]:
        rep = evaluate_split(model, X[split == code], y[split == code], name)
        report["splits"][name] = rep
        cm = np.array(rep["confusion_matrix"])
        plot_cm(cm, os.path.join(args.out_dir, f"confusion_matrix_{name}.png"),
                f"{args.title} ({name})")

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=1)

    for name in ("val", "test"):
        rep = report["splits"][name]
        print(f"\n===== {name}  acc={rep['accuracy']:.3f}  macro-F1={rep['macro_f1']:.3f}  n={rep['n']} =====")
        print(f"{'class':<12}{'prec':>7}{'rec':>7}{'f1':>7}{'sup':>6}")
        for r in rep["per_class"]:
            print(f"{r['class']:<12}{r['precision']:>7.3f}{r['recall']:>7.3f}"
                  f"{r['f1']:>7.3f}{r['support']:>6d}")
        print("confusion matrix (rows=true, cols=pred):")
        print(" " * 12 + "".join(f"{c[:6]:>7}" for c in CLASSES))
        for i, row in enumerate(rep["confusion_matrix"]):
            print(f"{CLASSES[i]:<12}" + "".join(f"{v:>7d}" for v in row))
    print("\nmetrics ->", os.path.join(args.out_dir, "metrics.json"))


if __name__ == "__main__":
    main()
