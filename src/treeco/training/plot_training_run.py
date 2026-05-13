#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
)


# =========================================================
# Loading
# =========================================================
def load_json(path: Path, required: bool = True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        print(f"[WARNING] Missing optional file: {path}")
        return None

    with open(path, "r") as f:
        return json.load(f)


def load_run(run_path: Path):
    history = load_json(run_path / "history.json", required=True)
    metrics = load_json(run_path / "metrics.json", required=False)
    config = load_json(run_path / "config.json", required=False)
    return metrics, history, config


# =========================================================
# History normalisation
# =========================================================
def normalize_history(history):
    """
    Supports:
    1. Flat dict:
       {
         train_loss: [...],
         val_loss: [...],
         train_acc: [...],
         val_acc: [...],
         val_f1_macro: [...],
         val_recall_macro: [...],
         lr: [...]
       }

    2. Wrapped dict:
       {"history": [...]}

    3. List of epoch dicts:
       [
         {"epoch": 1, "train": {...}, "val": {...}},
         ...
       ]
    """

    if isinstance(history, dict) and "history" in history:
        history = history["history"]

    if isinstance(history, list):
        epochs = []
        train_loss, val_loss = [], []
        train_acc, val_acc = [], []
        val_f1_macro, val_recall_macro = [], []
        lr = []

        for i, h in enumerate(history, start=1):
            train = h.get("train", {})
            val = h.get("val", {})

            epochs.append(h.get("epoch", i))
            train_loss.append(train.get("loss", np.nan))
            val_loss.append(val.get("loss", np.nan))

            train_acc.append(
                train.get("acc", train.get("accuracy", train.get("bal_acc", np.nan)))
            )
            val_acc.append(
                val.get("acc", val.get("accuracy", val.get("bal_acc", np.nan)))
            )

            val_f1_macro.append(
                val.get("f1_macro", val.get("macro_f1", np.nan))
            )
            val_recall_macro.append(
                val.get("recall_macro", val.get("macro_recall", np.nan))
            )

            lr.append(h.get("lr", np.nan))

        return {
            "epochs": np.array(epochs, dtype=float),
            "train_loss": np.array(train_loss, dtype=float),
            "val_loss": np.array(val_loss, dtype=float),
            "train_acc": np.array(train_acc, dtype=float),
            "val_acc": np.array(val_acc, dtype=float),
            "val_f1_macro": np.array(val_f1_macro, dtype=float),
            "val_recall_macro": np.array(val_recall_macro, dtype=float),
            "lr": np.array(lr, dtype=float),
        }

    if isinstance(history, dict) and "train_loss" in history:
        n = len(history.get("train_loss", []))

        return {
            "epochs": np.arange(1, n + 1),
            "train_loss": np.array(history.get("train_loss", [np.nan] * n), dtype=float),
            "val_loss": np.array(history.get("val_loss", [np.nan] * n), dtype=float),
            "train_acc": np.array(history.get("train_acc", [np.nan] * n), dtype=float),
            "val_acc": np.array(history.get("val_acc", [np.nan] * n), dtype=float),
            "val_f1_macro": np.array(history.get("val_f1_macro", [np.nan] * n), dtype=float),
            "val_recall_macro": np.array(history.get("val_recall_macro", [np.nan] * n), dtype=float),
            "lr": np.array(history.get("lr", [np.nan] * n), dtype=float),
        }

    raise ValueError(f"Unknown history format: {type(history)}")


# =========================================================
# Helpers
# =========================================================
def safe_nanargmax(values: np.ndarray) -> int | None:
    if values.size == 0 or np.all(np.isnan(values)):
        return None
    return int(np.nanargmax(values))


def save_figure(fig, save_path: Path, dpi: int = 220):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def get_class_names(config: dict | None):
    if config is None:
        return None

    if "class_names" in config:
        return list(config["class_names"])

    if "height_class_mapping" in config:
        mapping = config["height_class_mapping"]
        return [mapping[k] for k in sorted(mapping.keys(), key=lambda x: int(x))]

    if "diameter_class_mapping" in config:
        mapping = config["diameter_class_mapping"]
        return [mapping[k] for k in sorted(mapping.keys(), key=lambda x: int(x))]

    if "class_mapping" in config:
        mapping = config["class_mapping"]
        return [mapping[k] for k in sorted(mapping.keys(), key=lambda x: int(x))]

    return None


# =========================================================
# Training diagnostics
# =========================================================
def plot_training_diagnostics(
    history,
    title_prefix: str = "Model",
    save_path: Path | None = None,
):
    hist = normalize_history(history)

    epochs = hist["epochs"]
    train_loss = hist["train_loss"]
    val_loss = hist["val_loss"]
    train_acc = hist["train_acc"]
    val_acc = hist["val_acc"]
    val_f1_macro = hist["val_f1_macro"]
    val_recall_macro = hist["val_recall_macro"]
    lr = hist["lr"]

    best_idx = safe_nanargmax(val_f1_macro)
    if best_idx is None:
        best_idx = safe_nanargmax(val_acc)

    best_epoch = int(epochs[best_idx]) if best_idx is not None else None

    acc_gap = train_acc - val_acc

    fig = plt.figure(figsize=(16, 10))

    # ----------------------------
    # Top row: loss
    # ----------------------------
    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax1.plot(epochs, train_loss, marker="o", label="Train loss")
    ax1.plot(epochs, val_loss, marker="o", label="Validation loss")

    if best_epoch is not None:
        ax1.axvline(
            best_epoch,
            linestyle="--",
            label=f"Best epoch {best_epoch}",
        )

    ax1.set_title(f"{title_prefix} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ----------------------------
    # Bottom left: accuracy
    # ----------------------------
    ax2 = plt.subplot2grid((2, 3), (1, 0))
    ax2.plot(epochs, train_acc, marker="o", label="Train accuracy")
    ax2.plot(epochs, val_acc, marker="o", label="Validation accuracy")
    ax2.fill_between(epochs, train_acc, val_acc, alpha=0.12)

    if best_epoch is not None:
        ax2.axvline(best_epoch, linestyle="--")

    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    if not np.all(np.isnan(acc_gap)):
        ax2.text(
            0.03,
            0.05,
            f"Mean accuracy gap = {np.nanmean(acc_gap) * 100:.2f}%",
            transform=ax2.transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # ----------------------------
    # Bottom middle: F1
    # ----------------------------
    ax3 = plt.subplot2grid((2, 3), (1, 1))
    ax3.plot(epochs, val_f1_macro, marker="o", label="Validation macro F1")

    if best_epoch is not None:
        ax3.axvline(best_epoch, linestyle="--")

    ax3.set_title("Validation Macro F1")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("F1-score")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # ----------------------------
    # Bottom right: recall + LR
    # ----------------------------
    ax4 = plt.subplot2grid((2, 3), (1, 2))
    ax4.plot(epochs, val_recall_macro, marker="o", label="Validation macro recall")

    if best_epoch is not None:
        ax4.axvline(best_epoch, linestyle="--")

    ax4.set_title("Validation Macro Recall")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Recall")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper left")

    if not np.all(np.isnan(lr)):
        ax4b = ax4.twinx()
        ax4b.plot(epochs, lr, linestyle=":", label="Learning rate")
        ax4b.set_ylabel("Learning rate")
        ax4b.set_yscale("log")
        ax4b.legend(loc="lower right")

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()


# =========================================================
# Evaluation diagnostics
# =========================================================
def plot_evaluation_diagnostics(
    y_true,
    y_pred,
    class_names,
    title_prefix: str = "Model",
    normalize_cm: bool = True,
    save_path: Path | None = None,
    report_path: Path | None = None,
):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    labels = np.unique(np.concatenate([y_true, y_pred]))

    if class_names is None:
        class_names = [str(x) for x in labels]
    else:
        class_names = list(class_names)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    report_txt = [
        f"Accuracy:      {acc:.4f}",
        f"Macro F1:      {f1_macro:.4f}",
        f"Macro Recall:  {recall_macro:.4f}",
        "",
        "Classification report:",
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=3,
            zero_division=0,
        ),
    ]

    report_str = "\n".join(report_txt)
    print(report_str)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_str)
        print(f"Saved: {report_path}")

    cm = confusion_matrix(y_true, y_pred)

    if normalize_cm:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
        cbar_label = "Proportion of true class"
        annot = np.array([
            [f"{cm[i, j]}\n{cm_display[i, j]:.2f}" for j in range(cm.shape[1])]
            for i in range(cm.shape[0])
        ])
        fmt = ""
    else:
        cm_display = cm
        cbar_label = "Count"
        annot = cm
        fmt = "d"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if sns is not None:
        sns.heatmap(
            cm_display,
            annot=annot,
            fmt=fmt,
            cmap="YlGnBu",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": cbar_label},
            ax=axes[0],
        )
    else:
        im = axes[0].imshow(cm_display)
        fig.colorbar(im, ax=axes[0], label=cbar_label)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                axes[0].text(j, i, annot[i, j], ha="center", va="center")

        axes[0].set_xticks(np.arange(len(class_names)))
        axes[0].set_yticks(np.arange(len(class_names)))
        axes[0].set_xticklabels(class_names)
        axes[0].set_yticklabels(class_names)

    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    axes[0].set_title(f"{title_prefix} — Confusion Matrix")

    per_class_recall = np.divide(
        np.diag(cm),
        cm.sum(axis=1),
        out=np.zeros(cm.shape[0], dtype=float),
        where=cm.sum(axis=1) != 0,
    )

    axes[1].bar(class_names, per_class_recall)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Recall")
    axes[1].set_title(f"{title_prefix} — Per-class Recall")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(True, axis="y", alpha=0.3)

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description="Plot diagnostics for a saved TreeCo training run."
    )

    parser.add_argument(
        "--run",
        required=True,
        help="Path to the saved run folder containing history.json.",
    )

    parser.add_argument(
        "--task_name",
        default=None,
        help="Optional title prefix, e.g. 'Tree height classification'.",
    )

    parser.add_argument(
        "--plots_dirname",
        default="plots",
        help="Name of the plots folder created inside the run directory.",
    )

    args = parser.parse_args()

    run_path = Path(args.run)

    if not run_path.exists():
        raise FileNotFoundError(f"Run path not found: {run_path}")

    plots_dir = run_path / args.plots_dirname
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics, history, config = load_run(run_path)

    run_name = run_path.name
    title_prefix = args.task_name or run_name

    print(f"Loaded run: {run_name}")

    if metrics is not None:
        print(f"Best epoch: {metrics.get('best_epoch')}")
        print(f"Best validation macro F1: {metrics.get('best_val_f1_macro')}")
        print(f"Final validation accuracy: {metrics.get('final_val_accuracy')}")

    if config is not None:
        print(f"Backbone: {config.get('backbone')}")
        print(f"Use depth: {config.get('use_depth')}")
        print(f"Dataset: {config.get('dataset_dir')}")

    plot_training_diagnostics(
        history,
        title_prefix=title_prefix,
        save_path=plots_dir / "training_diagnostics.png",
    )

    val_labels_path = run_path / "val_labels.npy"
    val_preds_path = run_path / "val_preds.npy"

    if val_labels_path.exists() and val_preds_path.exists():
        y_true = np.load(val_labels_path)
        y_pred = np.load(val_preds_path)

        class_names = get_class_names(config)

        plot_evaluation_diagnostics(
            y_true,
            y_pred,
            class_names=class_names,
            title_prefix=title_prefix,
            normalize_cm=True,
            save_path=plots_dir / "evaluation_diagnostics.png",
            report_path=plots_dir / "classification_report.txt",
        )

        print("Saved evaluation diagnostics.")
    else:
        print(
            "No val_labels.npy / val_preds.npy found, so only training diagnostics were generated."
        )

    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()