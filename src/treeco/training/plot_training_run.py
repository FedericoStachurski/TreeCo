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
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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
    if isinstance(history, dict) and "history" in history:
        history = history["history"]

    if isinstance(history, list):
        epochs = []
        train_loss, val_loss = [], []
        train_acc, val_acc = [], []
        val_f1_macro, val_recall_macro = [], []
        train_mae, val_mae = [], []
        train_rmse, val_rmse = [], []
        train_r2, val_r2 = [], []
        lr = []

        for i, h in enumerate(history, start=1):
            train = h.get("train", {})
            val = h.get("val", {})

            epochs.append(h.get("epoch", i))

            train_loss.append(train.get("loss", np.nan))
            val_loss.append(val.get("loss", np.nan))

            train_acc.append(train.get("acc", train.get("accuracy", train.get("bal_acc", np.nan))))
            val_acc.append(val.get("acc", val.get("accuracy", val.get("bal_acc", np.nan))))

            val_f1_macro.append(val.get("f1_macro", val.get("macro_f1", np.nan)))
            val_recall_macro.append(val.get("recall_macro", val.get("macro_recall", np.nan)))

            train_mae.append(train.get("mae", train.get("MAE", np.nan)))
            val_mae.append(val.get("mae", val.get("MAE", np.nan)))

            train_rmse.append(train.get("rmse", train.get("RMSE", np.nan)))
            val_rmse.append(val.get("rmse", val.get("RMSE", np.nan)))

            train_r2.append(train.get("r2", train.get("R2", train.get("R²", np.nan))))
            val_r2.append(val.get("r2", val.get("R2", val.get("R²", np.nan))))

            lr.append(h.get("lr", np.nan))

        return {
            "epochs": np.array(epochs, dtype=float),
            "train_loss": np.array(train_loss, dtype=float),
            "val_loss": np.array(val_loss, dtype=float),
            "train_acc": np.array(train_acc, dtype=float),
            "val_acc": np.array(val_acc, dtype=float),
            "val_f1_macro": np.array(val_f1_macro, dtype=float),
            "val_recall_macro": np.array(val_recall_macro, dtype=float),
            "train_mae": np.array(train_mae, dtype=float),
            "val_mae": np.array(val_mae, dtype=float),
            "train_rmse": np.array(train_rmse, dtype=float),
            "val_rmse": np.array(val_rmse, dtype=float),
            "train_r2": np.array(train_r2, dtype=float),
            "val_r2": np.array(val_r2, dtype=float),
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
            "train_mae": np.array(history.get("train_mae", [np.nan] * n), dtype=float),
            "val_mae": np.array(history.get("val_mae", [np.nan] * n), dtype=float),
            "train_rmse": np.array(history.get("train_rmse", [np.nan] * n), dtype=float),
            "val_rmse": np.array(history.get("val_rmse", [np.nan] * n), dtype=float),
            "train_r2": np.array(history.get("train_r2", [np.nan] * n), dtype=float),
            "val_r2": np.array(history.get("val_r2", [np.nan] * n), dtype=float),
            "lr": np.array(history.get("lr", [np.nan] * n), dtype=float),
        }

    raise ValueError(f"Unknown history format: {type(history)}")


# =========================================================
# Helpers
# =========================================================
def safe_nanargmax(values: np.ndarray) -> int | None:
    if values is None or values.size == 0 or np.all(np.isnan(values)):
        return None
    return int(np.nanargmax(values))


def safe_nanargmin(values: np.ndarray) -> int | None:
    if values is None or values.size == 0 or np.all(np.isnan(values)):
        return None
    return int(np.nanargmin(values))


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


def infer_task_type(config, y_true=None, y_pred=None):
    if config is not None:
        for key in ["task_type", "task", "problem_type", "target"]:
            if key in config:
                value = str(config[key]).lower()
                if any(x in value for x in ["regression", "dbh", "width", "diameter"]):
                    return "regression"
                if any(x in value for x in ["classification", "class", "height"]):
                    return "classification"

        joined = " ".join(str(v).lower() for v in config.values())
        if any(x in joined for x in ["dbh", "width", "diameter", "regression"]):
            return "regression"

    if y_true is not None and y_pred is not None:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if np.issubdtype(y_true.dtype, np.floating) or np.issubdtype(y_pred.dtype, np.floating):
            unique_true = np.unique(y_true[np.isfinite(y_true)])
            if len(unique_true) > 10:
                return "regression"

    return "classification"


# =========================================================
# Training diagnostics
# =========================================================
def plot_training_diagnostics(
    history,
    title_prefix: str = "Model",
    task_type: str = "classification",
    save_path: Path | None = None,
):
    hist = normalize_history(history)

    epochs = hist["epochs"]
    train_loss = hist["train_loss"]
    val_loss = hist["val_loss"]
    lr = hist["lr"]

    train_acc = hist["train_acc"]
    val_acc = hist["val_acc"]
    val_f1_macro = hist["val_f1_macro"]
    val_recall_macro = hist["val_recall_macro"]

    train_mae = hist["train_mae"]
    val_mae = hist["val_mae"]
    train_rmse = hist["train_rmse"]
    val_rmse = hist["val_rmse"]
    train_r2 = hist["train_r2"]
    val_r2 = hist["val_r2"]

    if task_type == "regression":
        best_idx = safe_nanargmin(val_mae)
    else:
        best_idx = safe_nanargmax(val_f1_macro)
        if best_idx is None:
            best_idx = safe_nanargmax(val_acc)

    best_epoch = int(epochs[best_idx]) if best_idx is not None else None

    fig = plt.figure(figsize=(16, 10))

    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax1.plot(epochs, train_loss, marker="o", label="Train loss")
    ax1.plot(epochs, val_loss, marker="o", label="Validation loss")

    if best_epoch is not None:
        ax1.axvline(best_epoch, linestyle="--", label=f"Best epoch {best_epoch}")

    ax1.set_title(f"{title_prefix} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    if task_type == "regression":
        ax2 = plt.subplot2grid((2, 3), (1, 0))
        ax2.plot(epochs, train_mae, marker="o", label="Train MAE")
        ax2.plot(epochs, val_mae, marker="o", label="Validation MAE")

        if best_epoch is not None:
            ax2.axvline(best_epoch, linestyle="--")

        ax2.set_title("MAE")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("MAE (cm)")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        ax3 = plt.subplot2grid((2, 3), (1, 1))
        ax3.plot(epochs, train_rmse, marker="o", label="Train RMSE")
        ax3.plot(epochs, val_rmse, marker="o", label="Validation RMSE")

        if best_epoch is not None:
            ax3.axvline(best_epoch, linestyle="--")

        ax3.set_title("RMSE")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("RMSE (cm)")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        ax4 = plt.subplot2grid((2, 3), (1, 2))
        ax4.plot(epochs, train_r2, marker="o", label="Train R2")
        ax4.plot(epochs, val_r2, marker="o", label="Validation R2")
        ax4.axhline(0, linestyle="--", alpha=0.6)

        if best_epoch is not None:
            ax4.axvline(best_epoch, linestyle="--")

        ax4.set_title("R2")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("R2")
        ax4.grid(True, alpha=0.3)
        ax4.legend(loc="upper left")

        if not np.all(np.isnan(lr)):
            ax4b = ax4.twinx()
            ax4b.plot(epochs, lr, linestyle=":", label="Learning rate")
            ax4b.set_ylabel("Learning rate")
            ax4b.set_yscale("log")
            ax4b.legend(loc="lower right")

    else:
        acc_gap = train_acc - val_acc

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

        ax3 = plt.subplot2grid((2, 3), (1, 1))
        ax3.plot(epochs, val_f1_macro, marker="o", label="Validation macro F1")

        if best_epoch is not None:
            ax3.axvline(best_epoch, linestyle="--")

        ax3.set_title("Validation Macro F1")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("F1-score")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

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
# Regression evaluation diagnostics
# =========================================================
def plot_regression_evaluation_diagnostics(
    y_true,
    y_pred,
    title_prefix: str = "DBH regression",
    save_path: Path | None = None,
    report_path: Path | None = None,
):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        raise ValueError("No finite y_true/y_pred values available for regression plotting.")

    residuals = y_pred - y_true
    abs_errors = np.abs(residuals)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan

    report_str = "\n".join([
        "Regression evaluation",
        f"Samples: {len(y_true)}",
        f"MAE:     {mae:.4f} cm",
        f"RMSE:    {rmse:.4f} cm",
        f"R2:      {r2:.4f}",
        f"Mean residual: {np.mean(residuals):.4f} cm",
        f"Median absolute error: {np.median(abs_errors):.4f} cm",
    ])

    print(report_str)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_str)
        print(f"Saved: {report_path}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    axes[0, 0].scatter(y_true, y_pred, alpha=0.75)
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], linestyle="--")
    axes[0, 0].set_xlabel("True DBH / diameter (cm)")
    axes[0, 0].set_ylabel("Predicted DBH / diameter (cm)")
    axes[0, 0].set_title(f"{title_prefix} — Predicted vs True")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 0].text(
        0.04,
        0.96,
        f"MAE = {mae:.2f} cm\nRMSE = {rmse:.2f} cm\nR2 = {r2:.3f}",
        transform=axes[0, 0].transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    axes[0, 1].scatter(y_true, residuals, alpha=0.75)
    axes[0, 1].axhline(0, linestyle="--")
    axes[0, 1].set_xlabel("True DBH / diameter (cm)")
    axes[0, 1].set_ylabel("Residual: predicted - true (cm)")
    axes[0, 1].set_title("Residuals vs True DBH")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].hist(residuals, bins=20, alpha=0.8)
    axes[1, 0].axvline(0, linestyle="--")
    axes[1, 0].set_xlabel("Residual: predicted - true (cm)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Residual distribution")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(abs_errors, bins=20, alpha=0.8)
    axes[1, 1].set_xlabel("Absolute error (cm)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("Absolute error distribution")
    axes[1, 1].grid(True, alpha=0.3)

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()


# =========================================================
# Classification evaluation diagnostics
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
        help="Optional title prefix, e.g. 'Tree height classification' or 'Tree DBH regression'.",
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

    task_type = infer_task_type(config)
    print(f"Initial detected task type: {task_type}")

    if metrics is not None:
        print(f"Best epoch: {metrics.get('best_epoch')}")

        if "best_val_mae_cm" in metrics:
            print(f"Best validation MAE: {metrics.get('best_val_mae_cm')} cm")
            print(f"Final validation MAE: {metrics.get('final_val_mae_cm')} cm")
            print(f"Final validation RMSE: {metrics.get('final_val_rmse_cm')} cm")
            print(f"Final validation R2: {metrics.get('final_val_r2')}")
            task_type = "regression"
        else:
            print(f"Best validation macro F1: {metrics.get('best_val_f1_macro')}")
            print(f"Final validation accuracy: {metrics.get('final_val_accuracy')}")

    if config is not None:
        print(f"Backbone: {config.get('backbone')}")
        print(f"Use depth: {config.get('use_depth')}")
        print(f"Dataset: {config.get('dataset_dir')}")

    val_labels_path = run_path / "val_labels.npy"
    val_preds_path = run_path / "val_preds.npy"

    if val_labels_path.exists() and val_preds_path.exists():
        y_true = np.load(val_labels_path)
        y_pred = np.load(val_preds_path)

        task_type = infer_task_type(config, y_true=y_true, y_pred=y_pred)
        print(f"Detected task type from predictions: {task_type}")
    else:
        y_true = None
        y_pred = None
        print("No val_labels.npy / val_preds.npy found.")

    plot_training_diagnostics(
        history,
        title_prefix=title_prefix,
        task_type=task_type,
        save_path=plots_dir / "training_diagnostics.png",
    )

    if y_true is not None and y_pred is not None:
        if task_type == "regression":
            plot_regression_evaluation_diagnostics(
                y_true,
                y_pred,
                title_prefix=title_prefix,
                save_path=plots_dir / "regression_evaluation_diagnostics.png",
                report_path=plots_dir / "regression_report.txt",
            )

            print("Saved regression evaluation diagnostics.")

        else:
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

            print("Saved classification evaluation diagnostics.")
    else:
        print("Only training diagnostics were generated.")

    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()