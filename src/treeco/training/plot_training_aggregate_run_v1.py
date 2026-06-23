#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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
# Pretty plotting style
# =========================================================
TREECO = {
    "blue": "#005F73",
    "cyan": "#0A9396",
    "mint": "#94D2BD",
    "cream": "#E9D8A6",
    "orange": "#EE9B00",
    "rust": "#CA6702",
    "red": "#BB3E03",
    "dark_red": "#9B2226",
    "ink": "#1F2933",
    "grid": "#CBD5E1",
}

TREECO_CMAP = LinearSegmentedColormap.from_list(
    "treeco_cmap",
    [TREECO["blue"], TREECO["cyan"], TREECO["mint"], TREECO["cream"], TREECO["orange"], TREECO["red"]],
)


def apply_treeco_style():
    """Publication-ish matplotlib style without requiring a local LaTeX install."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FBFCFD",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#334155",
            "axes.linewidth": 1.1,
            "axes.grid": True,
            "grid.color": TREECO["grid"],
            "grid.alpha": 0.45,
            "grid.linewidth": 0.8,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#CBD5E1",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "mathtext.fontset": "stix",
            "mathtext.default": "regular",
        }
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
# Generic helpers
# =========================================================
def safe_nanargmax(values: np.ndarray) -> int | None:
    if values is None or values.size == 0 or np.all(np.isnan(values)):
        return None
    return int(np.nanargmax(values))


def safe_nanargmin(values: np.ndarray) -> int | None:
    if values is None or values.size == 0 or np.all(np.isnan(values)):
        return None
    return int(np.nanargmin(values))


def save_figure(fig, save_path: Path, dpi: int = 260):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def norm_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lookup = {norm_col(c): c for c in df.columns}
    for cand in candidates:
        key = norm_col(cand)
        if key in lookup:
            return lookup[key]
    return None


def numeric_series(values):
    return pd.to_numeric(values, errors="coerce")


def mode_value(values):
    s = pd.Series(values).dropna()
    if len(s) == 0:
        return np.nan
    modes = s.mode()
    if len(modes) == 0:
        return np.nan
    return modes.iloc[0]


def get_class_names(config: dict | None):
    if config is None:
        return None

    if "class_names" in config:
        return list(config["class_names"])

    for key in ["height_class_mapping", "height_mapping", "diameter_class_mapping", "class_mapping"]:
        if key in config:
            mapping = config[key]
            return [mapping[k] for k in sorted(mapping.keys(), key=lambda x: int(x))]

    return None


def label_names_for_present_labels(labels, class_names):
    if class_names is None:
        return [str(x) for x in labels]

    names = []
    for label in labels:
        try:
            i = int(label)
            if 0 <= i < len(class_names):
                names.append(str(class_names[i]))
            else:
                names.append(str(label))
        except Exception:
            names.append(str(label))
    return names


def infer_task_type(config, y_true=None, y_pred=None):
    """
    Robust TreeCo task detection.

    Important:
    DBH/width/diameter can be input features.
    They should only imply regression if the TARGET/task is DBH/width regression,
    not if the config merely contains DBH as an auxiliary input.
    """

    config = config or {}

    # --------------------------------------------------
    # 1. Explicit flags should win
    # --------------------------------------------------
    if bool(config.get("classification", False)):
        return "classification"

    explicit_task_type = str(config.get("task_type", "")).lower()
    problem_type = str(config.get("problem_type", "")).lower()

    if explicit_task_type in {"classification", "multiclass_classification"}:
        return "classification"

    if problem_type in {"classification", "multiclass_classification"}:
        return "classification"

    if explicit_task_type in {"regression", "continuous_regression"}:
        return "regression"

    if problem_type in {"regression", "continuous_regression"}:
        return "regression"

    # --------------------------------------------------
    # 2. Task name: classification should be checked first
    # --------------------------------------------------
    task = str(config.get("task", "")).lower()
    target = str(config.get("target", "")).lower()
    target_col = str(config.get("target_column", "")).lower()

    task_text = " ".join([task, target, target_col])

    if any(x in task_text for x in ["classification", "class", "height_class"]):
        return "classification"

    if any(x in task_text for x in ["height"]):
        # In your TreeCo scripts, height currently means height class.
        return "classification"

    if any(x in task_text for x in ["regression"]):
        return "regression"

    # Only treat DBH/width/diameter as regression if they are the TARGET,
    # not merely present somewhere in the config.
    if any(x in target for x in ["dbh", "width", "diameter"]):
        return "regression"

    if any(x in target_col for x in ["dbh", "width", "diameter"]):
        return "regression"

    # --------------------------------------------------
    # 3. Classification config keys
    # --------------------------------------------------
    classification_keys = [
        "num_classes",
        "num_height_classes",
        "height_mapping",
        "height_class_mapping",
        "class_mapping",
    ]

    if any(k in config for k in classification_keys):
        return "classification"

    # --------------------------------------------------
    # 4. Prediction fallback
    # --------------------------------------------------
    if y_true is not None and y_pred is not None:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)

        finite_true = y_true_flat[pd.notna(y_true_flat)]
        finite_pred = y_pred_flat[pd.notna(y_pred_flat)]

        if len(finite_true) > 0 and len(finite_pred) > 0:
            true_integer_like = np.all(np.isclose(finite_true, np.round(finite_true)))
            pred_integer_like = np.all(np.isclose(finite_pred, np.round(finite_pred)))

            unique_all = np.unique(np.concatenate([finite_true, finite_pred]))

            # Height classes are numeric IDs 0,1,2,3.
            # Numeric IDs do NOT mean regression.
            if true_integer_like and pred_integer_like and len(unique_all) <= 20:
                return "classification"

            if len(np.unique(finite_true)) > 20:
                return "regression"

    return "classification"


# =========================================================
# Prediction table / tree ID loading
# =========================================================
def load_optional_npy(path: Path):
    if path.exists():
        return np.load(path, allow_pickle=True)
    return None


def find_prediction_table(run_path: Path, explicit_path: str | None = None) -> pd.DataFrame | None:
    candidates = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))

    candidates.extend(
        [
            run_path / "val_predictions.csv",
            run_path / "validation_predictions.csv",
            run_path / "val_pred_df.csv",
            run_path / "pred_df.csv",
            run_path / "predictions.csv",
            run_path / "val_results.csv",
            run_path / "results.csv",
            run_path / "val_predictions.parquet",
            run_path / "validation_predictions.parquet",
            run_path / "predictions.parquet",
        ]
    )

    for path in candidates:
        if path.exists():
            print(f"Loaded prediction table: {path}")
            if path.suffix.lower() == ".parquet":
                return pd.read_parquet(path)
            return pd.read_csv(path)

    print("No prediction table found. Tree-level aggregation will need tree IDs from .npy files.")
    return None


def find_tree_ids_array(run_path: Path):
    for name in [
        "val_tree_ids.npy",
        "val_root_ids.npy",
        "tree_ids.npy",
        "root_ids.npy",
        "val_ids.npy",
        "ids.npy",
    ]:
        arr = load_optional_npy(run_path / name)
        if arr is not None:
            print(f"Loaded tree IDs from: {run_path / name}")
            return arr
    return None


def infer_tree_id_col(df: pd.DataFrame, explicit: str | None = None) -> str | None:
    if explicit is not None:
        if explicit in df.columns:
            return explicit
        raise ValueError(f"Requested --tree_id_col '{explicit}' was not found in prediction table.")

    candidates = [
        "ROOT_ID",
        "root_id",
        "TREE_ID",
        "tree_id",
        "SOURCE_ID",
        "source_id",
        "ID",
        "id",
        "ENTRY_ID",
        "entry_id",
        "RECORD_ID",
        "record_id",
        "original_id",
        "ORIGINAL_ID",
    ]
    return find_col(df, candidates)


def infer_regression_y_cols(df: pd.DataFrame):
    true_candidates = [
        "y_true",
        "true",
        "label",
        "target",
        "target_cm",
        "true_cm",
        "DBH_CM",
        "dbh_cm",
        "TRUE_DBH_CM",
        "true_dbh_cm",
        "DBH_TRUE_CM",
        "dbh_true_cm",
        "DIAMETER_CM",
        "diameter_cm",
        "TRUE_DIAMETER_CM",
        "true_diameter_cm",
        "WIDTH_CM",
        "width_cm",
    ]
    pred_candidates = [
        "y_pred",
        "pred",
        "prediction",
        "predicted",
        "pred_cm",
        "PRED_DBH_CM",
        "pred_dbh_cm",
        "DBH_PRED_CM",
        "dbh_pred_cm",
        "PREDICTED_DBH_CM",
        "predicted_dbh_cm",
        "PRED_DIAMETER_CM",
        "pred_diameter_cm",
        "pred_width_cm",
    ]

    true_col = find_col(df, true_candidates)
    pred_col = find_col(df, pred_candidates)

    if true_col is None:
        for c in df.columns:
            n = norm_col(c)
            if ("true" in n or "label" in n or "target" in n) and any(x in n for x in ["dbh", "diameter", "width", "cm"]):
                true_col = c
                break

    if pred_col is None:
        for c in df.columns:
            n = norm_col(c)
            if ("pred" in n or "prediction" in n) and any(x in n for x in ["dbh", "diameter", "width", "cm"]):
                pred_col = c
                break

    return true_col, pred_col


def infer_classification_y_cols(df: pd.DataFrame):
    true_candidates = [
        "y_true",
        "true",
        "label",
        "target",
        "class_idx",
        "true_class",
        "true_class_idx",
        "HEIGHT_CLASS_IDX",
        "height_class_idx",
        "DIAMETER_CLASS_IDX",
        "diameter_class_idx",
    ]
    pred_candidates = [
        "y_pred",
        "pred",
        "prediction",
        "predicted",
        "pred_class",
        "pred_class_idx",
        "PRED_CLASS_IDX",
        "predicted_class_idx",
    ]
    return find_col(df, true_candidates), find_col(df, pred_candidates)


def infer_probability_columns(df: pd.DataFrame) -> list[str]:
    prob_cols = []
    for c in df.columns:
        n = norm_col(c)
        if n.startswith("prob") or n.startswith("classprob") or n.startswith("pclass"):
            if pd.api.types.is_numeric_dtype(df[c]):
                prob_cols.append(c)

    def prob_index(c):
        digits = "".join(ch for ch in str(c) if ch.isdigit())
        return int(digits) if digits else 10**9

    return sorted(prob_cols, key=prob_index)


def build_eval_dataframe(
    run_path: Path,
    task_type: str,
    y_true=None,
    y_pred=None,
    probabilities=None,
    predictions_csv: str | None = None,
    tree_id_col: str | None = None,
):
    """
    Returns df, tree_id_col, true_col, pred_col, prob_cols.

    This lets the script aggregate even when labels/preds are stored in .npy
    and tree IDs are stored in either a CSV or val_tree_ids.npy.
    """
    df = find_prediction_table(run_path, predictions_csv)

    if df is None:
        if y_true is None or y_pred is None:
            return None, None, None, None, []

        df = pd.DataFrame(
            {
                "_y_true": np.asarray(y_true).reshape(-1),
                "_y_pred": np.asarray(y_pred).reshape(-1),
            }
        )

        ids = find_tree_ids_array(run_path)
        if ids is not None and len(ids) == len(df):
            df["_tree_id"] = ids
            tree_id_col = "_tree_id"
        elif ids is not None:
            print(
                f"[WARNING] Found tree ID array with length {len(ids)}, "
                f"but predictions have length {len(df)}. Ignoring IDs."
            )

        if probabilities is not None and len(probabilities) == len(df):
            for j in range(probabilities.shape[1]):
                df[f"prob_{j}"] = probabilities[:, j]

    else:
        # If CSV does not contain predictions/labels, append the .npy arrays if lengths match.
        if y_true is not None and len(y_true) == len(df):
            df["_y_true_from_npy"] = np.asarray(y_true).reshape(-1)
        if y_pred is not None and len(y_pred) == len(df):
            df["_y_pred_from_npy"] = np.asarray(y_pred).reshape(-1)
        if probabilities is not None and len(probabilities) == len(df):
            for j in range(probabilities.shape[1]):
                df[f"prob_{j}"] = probabilities[:, j]

    inferred_tree_id_col = infer_tree_id_col(df, tree_id_col)

    if task_type == "regression":
        true_col, pred_col = infer_regression_y_cols(df)
    else:
        true_col, pred_col = infer_classification_y_cols(df)

    # Prefer the exact .npy values if available and the table's own column inference failed.
    if true_col is None and "_y_true_from_npy" in df.columns:
        true_col = "_y_true_from_npy"
    if pred_col is None and "_y_pred_from_npy" in df.columns:
        pred_col = "_y_pred_from_npy"

    if true_col is None and "_y_true" in df.columns:
        true_col = "_y_true"
    if pred_col is None and "_y_pred" in df.columns:
        pred_col = "_y_pred"

    prob_cols = infer_probability_columns(df) if task_type == "classification" else []

    return df, inferred_tree_id_col, true_col, pred_col, prob_cols


# =========================================================
# Metric helpers
# =========================================================
def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "r2": np.nan, "bias": np.nan, "medae": np.nan}

    residuals = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
        "bias": float(np.mean(residuals)),
        "medae": float(np.median(np.abs(residuals))),
    }


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = pd.notna(y_true) & pd.notna(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {"n": 0, "accuracy": np.nan, "f1_macro": np.nan, "recall_macro": np.nan}

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# =========================================================
# Tree-level aggregation
# =========================================================
def aggregate_regression_by_tree(
    df: pd.DataFrame,
    tree_id_col: str,
    true_col: str,
    pred_col: str,
    agg: str = "mean",
) -> pd.DataFrame:
    tmp = df[[tree_id_col, true_col, pred_col]].copy()
    tmp[true_col] = numeric_series(tmp[true_col])
    tmp[pred_col] = numeric_series(tmp[pred_col])
    tmp = tmp.dropna(subset=[tree_id_col, true_col, pred_col])

    if len(tmp) == 0:
        raise ValueError("No valid rows available for tree-level regression aggregation.")

    pred_func = "mean" if agg == "mean" else "median"

    out = (
        tmp.groupby(tree_id_col)
        .agg(
            y_true=(true_col, "median"),
            y_pred=(pred_col, pred_func),
            y_pred_std=(pred_col, "std"),
            n_images=(pred_col, "size"),
        )
        .reset_index()
    )
    out["y_pred_std"] = out["y_pred_std"].fillna(0.0)
    out["abs_error"] = np.abs(out["y_pred"] - out["y_true"])
    out["residual"] = out["y_pred"] - out["y_true"]
    return out


def aggregate_classification_by_tree(
    df: pd.DataFrame,
    tree_id_col: str,
    true_col: str,
    pred_col: str,
    prob_cols: list[str] | None = None,
) -> pd.DataFrame:
    tmp_cols = [tree_id_col, true_col, pred_col]
    prob_cols = prob_cols or []
    tmp_cols += prob_cols

    tmp = df[tmp_cols].copy().dropna(subset=[tree_id_col, true_col, pred_col])
    if len(tmp) == 0:
        raise ValueError("No valid rows available for tree-level classification aggregation.")

    if prob_cols:
        rows = []
        for tree_id, g in tmp.groupby(tree_id_col):
            probs = g[prob_cols].to_numpy(dtype=float)
            mean_probs = np.nanmean(probs, axis=0)
            pred_class = int(np.nanargmax(mean_probs))
            rows.append(
                {
                    tree_id_col: tree_id,
                    "y_true": mode_value(g[true_col]),
                    "y_pred": pred_class,
                    "n_images": len(g),
                    **{f"mean_prob_{j}": mean_probs[j] for j in range(len(mean_probs))},
                }
            )
        return pd.DataFrame(rows)

    out = (
        tmp.groupby(tree_id_col)
        .agg(
            y_true=(true_col, mode_value),
            y_pred=(pred_col, mode_value),
            n_images=(pred_col, "size"),
        )
        .reset_index()
    )
    return out


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

    fig = plt.figure(figsize=(17, 10.5))
    fig.suptitle(title_prefix, fontsize=17, fontweight="bold", y=1.015)

    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
    ax1.plot(epochs, train_loss, marker="o", ms=4, lw=2.0, color=TREECO["blue"], label=r"Train loss")
    ax1.plot(epochs, val_loss, marker="o", ms=4, lw=2.0, color=TREECO["orange"], label=r"Validation loss")

    if best_epoch is not None:
        ax1.axvline(best_epoch, linestyle="--", lw=1.7, color=TREECO["dark_red"], label=rf"Best epoch ${best_epoch}$")

    ax1.set_title(r"Loss curve")
    ax1.set_xlabel(r"Epoch")
    ax1.set_ylabel(r"Loss")
    ax1.legend(loc="best")

    if task_type == "regression":
        ax2 = plt.subplot2grid((2, 3), (1, 0))
        ax2.plot(epochs, train_mae, marker="o", ms=4, lw=2.0, color=TREECO["blue"], label=r"Train MAE")
        ax2.plot(epochs, val_mae, marker="o", ms=4, lw=2.0, color=TREECO["orange"], label=r"Validation MAE")
        if best_epoch is not None:
            ax2.axvline(best_epoch, linestyle="--", lw=1.5, color=TREECO["dark_red"])
        ax2.set_title(r"Mean absolute error")
        ax2.set_xlabel(r"Epoch")
        ax2.set_ylabel(r"$\mathrm{MAE}\;[\mathrm{cm}]$")
        ax2.legend(loc="best")

        ax3 = plt.subplot2grid((2, 3), (1, 1))
        ax3.plot(epochs, train_rmse, marker="o", ms=4, lw=2.0, color=TREECO["blue"], label=r"Train RMSE")
        ax3.plot(epochs, val_rmse, marker="o", ms=4, lw=2.0, color=TREECO["orange"], label=r"Validation RMSE")
        if best_epoch is not None:
            ax3.axvline(best_epoch, linestyle="--", lw=1.5, color=TREECO["dark_red"])
        ax3.set_title(r"Root mean squared error")
        ax3.set_xlabel(r"Epoch")
        ax3.set_ylabel(r"$\mathrm{RMSE}\;[\mathrm{cm}]$")
        ax3.legend(loc="best")

        ax4 = plt.subplot2grid((2, 3), (1, 2))
        ax4.plot(epochs, train_r2, marker="o", ms=4, lw=2.0, color=TREECO["blue"], label=r"Train $R^2$")
        ax4.plot(epochs, val_r2, marker="o", ms=4, lw=2.0, color=TREECO["orange"], label=r"Validation $R^2$")
        ax4.axhline(0, linestyle="--", lw=1.2, color=TREECO["cyan"], alpha=0.65)
        if best_epoch is not None:
            ax4.axvline(best_epoch, linestyle="--", lw=1.5, color=TREECO["dark_red"])
        ax4.set_title(r"Explained variance")
        ax4.set_xlabel(r"Epoch")
        ax4.set_ylabel(r"$R^2$")
        ax4.legend(loc="upper left")

        if not np.all(np.isnan(lr)):
            ax4b = ax4.twinx()
            ax4b.plot(epochs, lr, linestyle=":", lw=2.0, color=TREECO["dark_red"], label=r"Learning rate")
            ax4b.set_ylabel(r"Learning rate, $\eta$")
            ax4b.set_yscale("log")
            ax4b.legend(loc="lower right")

    else:
        acc_gap = train_acc - val_acc

        ax2 = plt.subplot2grid((2, 3), (1, 0))
        ax2.plot(epochs, train_acc, marker="o", ms=4, lw=2.0, color=TREECO["blue"], label=r"Train accuracy")
        ax2.plot(epochs, val_acc, marker="o", ms=4, lw=2.0, color=TREECO["orange"], label=r"Validation accuracy")
        ax2.fill_between(epochs, train_acc, val_acc, color=TREECO["mint"], alpha=0.25)
        if best_epoch is not None:
            ax2.axvline(best_epoch, linestyle="--", lw=1.5, color=TREECO["dark_red"])
        ax2.set_title(r"Accuracy")
        ax2.set_xlabel(r"Epoch")
        ax2.set_ylabel(r"Accuracy")
        ax2.legend(loc="best")

        if not np.all(np.isnan(acc_gap)):
            ax2.text(
                0.03,
                0.05,
                rf"Mean gap $= {np.nanmean(acc_gap) * 100:.2f}\%$",
                transform=ax2.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.95),
            )

        ax3 = plt.subplot2grid((2, 3), (1, 1))
        ax3.plot(epochs, val_f1_macro, marker="o", ms=4, lw=2.0, color=TREECO["cyan"], label=r"Validation macro $F_1$")
        if best_epoch is not None:
            ax3.axvline(best_epoch, linestyle="--", lw=1.5, color=TREECO["dark_red"])
        ax3.set_title(r"Macro $F_1$")
        ax3.set_xlabel(r"Epoch")
        ax3.set_ylabel(r"$F_1$")
        ax3.legend(loc="best")

        ax4 = plt.subplot2grid((2, 3), (1, 2))
        ax4.plot(epochs, val_recall_macro, marker="o", ms=4, lw=2.0, color=TREECO["rust"], label=r"Validation macro recall")
        if best_epoch is not None:
            ax4.axvline(best_epoch, linestyle="--", lw=1.5, color=TREECO["dark_red"])
        ax4.set_title(r"Macro recall")
        ax4.set_xlabel(r"Epoch")
        ax4.set_ylabel(r"Recall")
        ax4.legend(loc="upper left")

        if not np.all(np.isnan(lr)):
            ax4b = ax4.twinx()
            ax4b.plot(epochs, lr, linestyle=":", lw=2.0, color=TREECO["dark_red"], label=r"Learning rate")
            ax4b.set_ylabel(r"Learning rate, $\eta$")
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
    level_name: str = "Image level",
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
    metrics = regression_metrics(y_true, y_pred)

    report_str = "\n".join(
        [
            f"Regression evaluation: {level_name}",
            f"Samples: {metrics['n']}",
            f"MAE:     {metrics['mae']:.4f} cm",
            f"RMSE:    {metrics['rmse']:.4f} cm",
            f"R2:      {metrics['r2']:.4f}",
            f"Mean residual / bias: {metrics['bias']:.4f} cm",
            f"Median absolute error: {metrics['medae']:.4f} cm",
        ]
    )

    print(report_str)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            f.write(report_str)
        print(f"Saved: {report_path}")

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.8))
    fig.suptitle(f"{title_prefix} — {level_name}", fontsize=16, fontweight="bold", y=1.02)

    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.04 * (max_val - min_val + 1e-9)
    lims = [min_val - pad, max_val + pad]

    sc = axes[0, 0].scatter(
        y_true,
        y_pred,
        c=abs_errors,
        cmap=TREECO_CMAP,
        s=48,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.45,
    )
    axes[0, 0].plot(lims, lims, linestyle="--", lw=1.8, color=TREECO["dark_red"], label=r"$\hat{y}=y$")
    axes[0, 0].set_xlim(lims)
    axes[0, 0].set_ylim(lims)
    axes[0, 0].set_xlabel(r"Observed DBH, $y$ $[\mathrm{cm}]$")
    axes[0, 0].set_ylabel(r"Predicted DBH, $\hat{y}$ $[\mathrm{cm}]$")
    axes[0, 0].set_title(r"Predicted vs observed")
    axes[0, 0].legend(loc="lower right")
    cbar = fig.colorbar(sc, ax=axes[0, 0])
    cbar.set_label(r"$|\hat{y}-y|$ $[\mathrm{cm}]$")

    axes[0, 0].text(
        0.04,
        0.96,
        "\n".join(
            [
                rf"$N={metrics['n']}$",
                rf"$\mathrm{{MAE}}={metrics['mae']:.2f}\,\mathrm{{cm}}$",
                rf"$\mathrm{{RMSE}}={metrics['rmse']:.2f}\,\mathrm{{cm}}$",
                rf"$R^2={metrics['r2']:.3f}$",
            ]
        ),
        transform=axes[0, 0].transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CBD5E1", alpha=0.94),
    )

    sc2 = axes[0, 1].scatter(
        y_true,
        residuals,
        c=residuals,
        cmap="coolwarm",
        s=48,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.45,
    )
    axes[0, 1].axhline(0, linestyle="--", lw=1.8, color=TREECO["ink"])
    axes[0, 1].set_xlabel(r"Observed DBH, $y$ $[\mathrm{cm}]$")
    axes[0, 1].set_ylabel(r"Residual, $\hat{y}-y$ $[\mathrm{cm}]$")
    axes[0, 1].set_title(r"Residuals vs observed DBH")
    cbar2 = fig.colorbar(sc2, ax=axes[0, 1])
    cbar2.set_label(r"$\hat{y}-y$ $[\mathrm{cm}]$")

    axes[1, 0].hist(residuals, bins=24, color=TREECO["cyan"], alpha=0.86, edgecolor="white")
    axes[1, 0].axvline(0, linestyle="--", lw=1.8, color=TREECO["ink"])
    axes[1, 0].axvline(np.mean(residuals), linestyle="-", lw=2.0, color=TREECO["dark_red"], label=rf"Mean $={np.mean(residuals):.2f}$ cm")
    axes[1, 0].set_xlabel(r"Residual, $\hat{y}-y$ $[\mathrm{cm}]$")
    axes[1, 0].set_ylabel(r"Count")
    axes[1, 0].set_title(r"Residual distribution")
    axes[1, 0].legend(loc="best")

    axes[1, 1].hist(abs_errors, bins=24, color=TREECO["orange"], alpha=0.88, edgecolor="white")
    axes[1, 1].axvline(metrics["mae"], linestyle="-", lw=2.0, color=TREECO["dark_red"], label=rf"MAE $={metrics['mae']:.2f}$ cm")
    axes[1, 1].set_xlabel(r"Absolute error, $|\hat{y}-y|$ $[\mathrm{cm}]$")
    axes[1, 1].set_ylabel(r"Count")
    axes[1, 1].set_title(r"Absolute error distribution")
    axes[1, 1].legend(loc="best")

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()

    return metrics


# =========================================================
# Classification evaluation diagnostics
# =========================================================
def plot_classification_evaluation_diagnostics(
    y_true,
    y_pred,
    class_names,
    title_prefix: str = "Model",
    level_name: str = "Image level",
    normalize_cm: bool = True,
    save_path: Path | None = None,
    report_path: Path | None = None,
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = pd.notna(y_true) & pd.notna(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    labels = np.unique(np.concatenate([y_true, y_pred]))
    names = label_names_for_present_labels(labels, class_names)

    metrics = classification_metrics(y_true, y_pred)

    report_txt = [
        f"Classification evaluation: {level_name}",
        f"Samples:       {metrics['n']}",
        f"Accuracy:      {metrics['accuracy']:.4f}",
        f"Macro F1:      {metrics['f1_macro']:.4f}",
        f"Macro Recall:  {metrics['recall_macro']:.4f}",
        "",
        "Classification report:",
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=names,
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

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize_cm:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_display = np.divide(
            cm.astype(float),
            row_sums,
            out=np.zeros_like(cm, dtype=float),
            where=row_sums != 0,
        )
        cbar_label = r"Proportion of true class"
        annot = np.array(
            [[f"{cm[i, j]}\n{cm_display[i, j]:.2f}" for j in range(cm.shape[1])] for i in range(cm.shape[0])]
        )
    else:
        cm_display = cm.astype(float)
        cbar_label = r"Count"
        annot = cm.astype(str)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), gridspec_kw={"width_ratios": [1.15, 0.9]})
    fig.suptitle(f"{title_prefix} — {level_name}", fontsize=16, fontweight="bold", y=1.04)

    im = axes[0].imshow(cm_display, cmap=TREECO_CMAP, aspect="auto")
    cbar = fig.colorbar(im, ax=axes[0])
    cbar.set_label(cbar_label)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm_display[i, j]
            text_color = "white" if value > np.nanmax(cm_display) * 0.55 else TREECO["ink"]
            axes[0].text(j, i, annot[i, j], ha="center", va="center", fontsize=10, color=text_color)

    axes[0].set_xticks(np.arange(len(names)))
    axes[0].set_yticks(np.arange(len(names)))
    axes[0].set_xticklabels(names, rotation=35, ha="right")
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel(r"Predicted class, $\hat{c}$")
    axes[0].set_ylabel(r"True class, $c$")
    axes[0].set_title(r"Confusion matrix")
    axes[0].grid(False)

    per_class_recall = np.divide(
        np.diag(cm),
        cm.sum(axis=1),
        out=np.zeros(cm.shape[0], dtype=float),
        where=cm.sum(axis=1) != 0,
    )

    bar_colors = TREECO_CMAP(np.linspace(0.08, 0.92, len(names)))
    axes[1].bar(names, per_class_recall, color=bar_colors, edgecolor="white", linewidth=0.8)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel(r"Recall")
    axes[1].set_title(r"Per-class recall")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(True, axis="y", alpha=0.35)
    axes[1].text(
        0.04,
        0.95,
        "\n".join(
            [
                rf"$N={metrics['n']}$",
                rf"$\mathrm{{Acc}}={metrics['accuracy']:.3f}$",
                rf"$F_1^{{macro}}={metrics['f1_macro']:.3f}$",
                rf"$\mathrm{{Recall}}^{{macro}}={metrics['recall_macro']:.3f}$",
            ]
        ),
        transform=axes[1].transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CBD5E1", alpha=0.94),
    )

    if save_path is not None:
        save_figure(fig, save_path)
    else:
        plt.show()

    return metrics


# Backward-compatible alias with your original function name.
def plot_evaluation_diagnostics(*args, **kwargs):
    return plot_classification_evaluation_diagnostics(*args, **kwargs)


# =========================================================
# Individual-vs-tree comparison plots
# =========================================================
def plot_regression_metric_comparison(image_metrics, tree_metrics, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.25, 0.75]})
    fig.suptitle(r"Image-level vs tree-level aggregate performance", fontsize=15, fontweight="bold", y=1.03)

    error_names = [r"MAE", r"RMSE"]
    image_errors = [image_metrics["mae"], image_metrics["rmse"]]
    tree_errors = [tree_metrics["mae"], tree_metrics["rmse"]]

    x = np.arange(len(error_names))
    width = 0.36
    axes[0].bar(x - width / 2, image_errors, width, color=TREECO["blue"], label=r"Image level")
    axes[0].bar(x + width / 2, tree_errors, width, color=TREECO["orange"], label=r"Tree aggregate")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(error_names)
    axes[0].set_ylabel(r"Error $[\mathrm{cm}]$")
    axes[0].set_title(r"Error metrics")
    axes[0].legend(loc="best")

    for xi, v in zip(x - width / 2, image_errors):
        axes[0].text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    for xi, v in zip(x + width / 2, tree_errors):
        axes[0].text(xi, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    axes[1].bar([0], [image_metrics["r2"]], width=0.5, color=TREECO["blue"], label=r"Image level")
    axes[1].bar([1], [tree_metrics["r2"]], width=0.5, color=TREECO["orange"], label=r"Tree aggregate")
    axes[1].axhline(0, linestyle="--", color=TREECO["ink"], lw=1.2, alpha=0.65)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels([r"Image", r"Tree"])
    axes[1].set_ylabel(r"$R^2$")
    axes[1].set_title(r"Explained variance")

    for xi, v in zip([0, 1], [image_metrics["r2"], tree_metrics["r2"]]):
        axes[1].text(xi, v, f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)

    save_figure(fig, save_path)


def plot_classification_metric_comparison(image_metrics, tree_metrics, save_path: Path):
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    fig.suptitle(r"Image-level vs tree-level aggregate performance", fontsize=15, fontweight="bold", y=1.03)

    names = [r"Accuracy", r"Macro $F_1$", r"Macro recall"]
    image_values = [image_metrics["accuracy"], image_metrics["f1_macro"], image_metrics["recall_macro"]]
    tree_values = [tree_metrics["accuracy"], tree_metrics["f1_macro"], tree_metrics["recall_macro"]]

    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, image_values, width, color=TREECO["blue"], label=r"Image level")
    ax.bar(x + width / 2, tree_values, width, color=TREECO["orange"], label=r"Tree aggregate")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel(r"Score")
    ax.set_title(r"Classification metrics")
    ax.legend(loc="best")

    for xi, v in zip(x - width / 2, image_values):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    for xi, v in zip(x + width / 2, tree_values):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=10)

    save_figure(fig, save_path)


def plot_tree_group_size_distribution(tree_df: pd.DataFrame, save_path: Path, title_prefix: str):
    if "n_images" not in tree_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    counts = tree_df["n_images"].astype(int).to_numpy()
    bins = np.arange(0.5, counts.max() + 1.5, 1)
    ax.hist(counts, bins=bins, color=TREECO["cyan"], edgecolor="white", alpha=0.9)
    ax.set_xticks(np.arange(1, counts.max() + 1))
    ax.set_xlabel(r"Validation images per tree")
    ax.set_ylabel(r"Number of trees")
    ax.set_title(f"{title_prefix} — Tree aggregate group sizes")
    ax.text(
        0.97,
        0.95,
        "\n".join(
            [
                rf"Trees $= {len(tree_df)}$",
                rf"Images $= {int(counts.sum())}$",
                rf"Mean $= {counts.mean():.2f}$",
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CBD5E1", alpha=0.94),
    )
    save_figure(fig, save_path)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Plot diagnostics for a saved TreeCo training run.")

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

    parser.add_argument(
        "--predictions_csv",
        default=None,
        help=(
            "Optional validation prediction table containing one row per validation image. "
            "Needed for tree-level aggregation unless val_tree_ids.npy exists."
        ),
    )

    parser.add_argument(
        "--tree_id_col",
        default=None,
        help=(
            "Optional tree/group ID column for aggregation, e.g. ROOT_ID. "
            "If omitted, the script tries ROOT_ID, TREE_ID, SOURCE_ID, ID, etc."
        ),
    )

    parser.add_argument(
        "--regression_agg",
        default="mean",
        choices=["mean", "median"],
        help="How to aggregate image-level regression predictions back to tree level.",
    )

    parser.add_argument(
        "--no_tree_aggregate",
        action="store_true",
        help="Disable tree-level aggregation even if tree IDs are available.",
    )

    args = parser.parse_args()
    apply_treeco_style()

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
        print(f"Input mode: {config.get('input_mode')}")
        print(f"Image source: {config.get('image_source')}")
        print(f"Dataset: {config.get('dataset_dir')}")

    val_labels_path = run_path / "val_labels.npy"
    val_preds_path = run_path / "val_preds.npy"
    val_probs_path = run_path / "val_probs.npy"

    if val_labels_path.exists() and val_preds_path.exists():
        y_true = np.load(val_labels_path, allow_pickle=True)
        y_pred = np.load(val_preds_path, allow_pickle=True)
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)

        task_type = infer_task_type(config, y_true=y_true, y_pred=y_pred)
        print(f"Detected task type from predictions: {task_type}")
    else:
        y_true = None
        y_pred = None
        print("No val_labels.npy / val_preds.npy found.")

    probabilities = None
    if val_probs_path.exists():
        probabilities = np.load(val_probs_path, allow_pickle=True)
        print(f"Loaded class probabilities from: {val_probs_path}")

    plot_training_diagnostics(
        history,
        title_prefix=title_prefix,
        task_type=task_type,
        save_path=plots_dir / "training_diagnostics.png",
    )

    if y_true is None or y_pred is None:
        print("Only training diagnostics were generated because validation labels/predictions were not found.")
        print(f"Plots saved to: {plots_dir}")
        return

    # -------------------------
    # Image-level plots
    # -------------------------
    if task_type == "regression":
        image_metrics = plot_regression_evaluation_diagnostics(
            y_true,
            y_pred,
            title_prefix=title_prefix,
            level_name="Image level",
            save_path=plots_dir / "regression_evaluation_image_level.png",
            report_path=plots_dir / "regression_report_image_level.txt",
        )
        print("Saved image-level regression evaluation diagnostics.")
    else:
        class_names = get_class_names(config)
        image_metrics = plot_classification_evaluation_diagnostics(
            y_true,
            y_pred,
            class_names=class_names,
            title_prefix=title_prefix,
            level_name="Image level",
            normalize_cm=True,
            save_path=plots_dir / "classification_evaluation_image_level.png",
            report_path=plots_dir / "classification_report_image_level.txt",
        )
        print("Saved image-level classification evaluation diagnostics.")

    # -------------------------
    # Tree-level aggregate plots
    # -------------------------
    if args.no_tree_aggregate:
        print("Tree-level aggregation disabled with --no_tree_aggregate.")
        print(f"Plots saved to: {plots_dir}")
        return

    df_eval, detected_tree_id_col, true_col, pred_col, prob_cols = build_eval_dataframe(
        run_path=run_path,
        task_type=task_type,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        predictions_csv=args.predictions_csv,
        tree_id_col=args.tree_id_col,
    )

    if df_eval is None or detected_tree_id_col is None:
        print("[WARNING] Could not generate tree-level aggregate plots because no tree ID column/array was found.")
        print("          Provide --predictions_csv with ROOT_ID/TREE_ID, or save val_tree_ids.npy in the run folder.")
        print(f"Plots saved to: {plots_dir}")
        return

    if true_col is None or pred_col is None:
        print("[WARNING] Could not infer true/prediction columns for aggregation.")
        print("          If using a CSV, include y_true/y_pred columns or rely on val_labels.npy/val_preds.npy with matching row order.")
        print(f"Plots saved to: {plots_dir}")
        return

    print(f"Tree ID column for aggregation: {detected_tree_id_col}")
    print(f"True column for aggregation:    {true_col}")
    print(f"Pred column for aggregation:    {pred_col}")

    if task_type == "regression":
        tree_df = aggregate_regression_by_tree(
            df_eval,
            tree_id_col=detected_tree_id_col,
            true_col=true_col,
            pred_col=pred_col,
            agg=args.regression_agg,
        )
        tree_csv = plots_dir / "tree_level_aggregate_predictions.csv"
        tree_df.to_csv(tree_csv, index=False)
        print(f"Saved: {tree_csv}")

        tree_metrics = plot_regression_evaluation_diagnostics(
            tree_df["y_true"],
            tree_df["y_pred"],
            title_prefix=title_prefix,
            level_name=f"Tree aggregate level ({args.regression_agg})",
            save_path=plots_dir / "regression_evaluation_tree_aggregate_level.png",
            report_path=plots_dir / "regression_report_tree_aggregate_level.txt",
        )

        plot_regression_metric_comparison(
            image_metrics,
            tree_metrics,
            save_path=plots_dir / "regression_image_vs_tree_aggregate_metrics.png",
        )
        plot_tree_group_size_distribution(
            tree_df,
            save_path=plots_dir / "tree_aggregate_group_sizes.png",
            title_prefix=title_prefix,
        )
        print("Saved tree-level aggregate regression diagnostics.")

    else:
        tree_df = aggregate_classification_by_tree(
            df_eval,
            tree_id_col=detected_tree_id_col,
            true_col=true_col,
            pred_col=pred_col,
            prob_cols=prob_cols,
        )
        tree_csv = plots_dir / "tree_level_aggregate_predictions.csv"
        tree_df.to_csv(tree_csv, index=False)
        print(f"Saved: {tree_csv}")

        class_names = get_class_names(config)
        tree_metrics = plot_classification_evaluation_diagnostics(
            tree_df["y_true"],
            tree_df["y_pred"],
            class_names=class_names,
            title_prefix=title_prefix,
            level_name="Tree aggregate level",
            normalize_cm=True,
            save_path=plots_dir / "classification_evaluation_tree_aggregate_level.png",
            report_path=plots_dir / "classification_report_tree_aggregate_level.txt",
        )

        plot_classification_metric_comparison(
            image_metrics,
            tree_metrics,
            save_path=plots_dir / "classification_image_vs_tree_aggregate_metrics.png",
        )
        plot_tree_group_size_distribution(
            tree_df,
            save_path=plots_dir / "tree_aggregate_group_sizes.png",
            title_prefix=title_prefix,
        )
        print("Saved tree-level aggregate classification diagnostics.")

    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
