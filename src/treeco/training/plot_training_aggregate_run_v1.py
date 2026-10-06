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
# Plot style
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
    [
        TREECO["blue"],
        TREECO["cyan"],
        TREECO["mint"],
        TREECO["cream"],
        TREECO["orange"],
        TREECO["red"],
    ],
)


def apply_treeco_style():
    """Publication-style matplotlib defaults without requiring LaTeX."""
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


def load_optional_npy(path: Path):
    if path.exists():
        return np.load(path, allow_pickle=True)
    return None


# =========================================================
# Generic helpers
# =========================================================

def save_figure(fig, save_path: Path, dpi: int = 260):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def norm_col(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


def find_col(
    df: pd.DataFrame,
    candidates: Iterable[str],
) -> str | None:
    lookup = {
        norm_col(c): c
        for c in df.columns
    }

    for cand in candidates:
        key = norm_col(cand)
        if key in lookup:
            return lookup[key]

    return None


def numeric_series(values):
    return pd.to_numeric(
        values,
        errors="coerce",
    )


def mode_value(values):
    s = pd.Series(values).dropna()

    if len(s) == 0:
        return np.nan

    modes = s.mode()

    if len(modes) == 0:
        return np.nan

    return modes.iloc[0]


def safe_nanargmax(values: np.ndarray) -> int | None:
    if (
        values is None
        or values.size == 0
        or np.all(np.isnan(values))
    ):
        return None

    return int(
        np.nanargmax(values)
    )


def safe_nanargmin(values: np.ndarray) -> int | None:
    if (
        values is None
        or values.size == 0
        or np.all(np.isnan(values))
    ):
        return None

    return int(
        np.nanargmin(values)
    )


# =========================================================
# Task / column inference
# =========================================================

def infer_task_type(
    config,
    y_true=None,
    y_pred=None,
):
    """
    Detect regression vs classification.

    Explicit task settings win. DBH/diameter/width imply regression
    only when they describe the target, rather than an auxiliary input.
    """

    config = config or {}

    if bool(config.get("classification", False)):
        return "classification"

    explicit_task_type = str(
        config.get("task_type", "")
    ).lower()

    problem_type = str(
        config.get("problem_type", "")
    ).lower()

    if explicit_task_type in {
        "classification",
        "multiclass_classification",
    }:
        return "classification"

    if problem_type in {
        "classification",
        "multiclass_classification",
    }:
        return "classification"

    if explicit_task_type in {
        "regression",
        "continuous_regression",
    }:
        return "regression"

    if problem_type in {
        "regression",
        "continuous_regression",
    }:
        return "regression"

    task = str(
        config.get("task", "")
    ).lower()

    target = str(
        config.get("target", "")
    ).lower()

    target_col = str(
        config.get("target_column", "")
    ).lower()

    task_text = " ".join(
        [
            task,
            target,
            target_col,
        ]
    )

    if any(
        x in task_text
        for x in [
            "classification",
            "height_class",
        ]
    ):
        return "classification"

    if "height" in task_text:
        return "classification"

    if "regression" in task_text:
        return "regression"

    if any(
        x in target
        for x in [
            "dbh",
            "diameter",
            "width",
        ]
    ):
        return "regression"

    if any(
        x in target_col
        for x in [
            "dbh",
            "diameter",
            "width",
        ]
    ):
        return "regression"

    classification_keys = [
        "num_classes",
        "num_height_classes",
        "height_mapping",
        "height_class_mapping",
        "class_mapping",
    ]

    if any(
        k in config
        for k in classification_keys
    ):
        return "classification"

    if (
        y_true is not None
        and y_pred is not None
    ):
        yt = np.asarray(
            y_true
        ).reshape(-1)

        yp = np.asarray(
            y_pred
        ).reshape(-1)

        finite_true = yt[
            pd.notna(yt)
        ]

        finite_pred = yp[
            pd.notna(yp)
        ]

        if (
            len(finite_true) > 0
            and len(finite_pred) > 0
        ):
            true_integer_like = np.all(
                np.isclose(
                    finite_true,
                    np.round(finite_true),
                )
            )

            pred_integer_like = np.all(
                np.isclose(
                    finite_pred,
                    np.round(finite_pred),
                )
            )

            unique_all = np.unique(
                np.concatenate(
                    [
                        finite_true,
                        finite_pred,
                    ]
                )
            )

            if (
                true_integer_like
                and pred_integer_like
                and len(unique_all) <= 20
            ):
                return "classification"

            if len(
                np.unique(
                    finite_true
                )
            ) > 20:
                return "regression"

    return "classification"


def infer_tree_id_col(
    df: pd.DataFrame,
    explicit: str | None = None,
) -> str | None:

    if explicit is not None:
        if explicit in df.columns:
            return explicit

        raise ValueError(
            f"Requested --tree_id_col '{explicit}' "
            "was not found in prediction table."
        )

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

    return find_col(
        df,
        candidates,
    )


def infer_regression_y_cols(
    df: pd.DataFrame,
):
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

    true_col = find_col(
        df,
        true_candidates,
    )

    pred_col = find_col(
        df,
        pred_candidates,
    )

    if true_col is None:
        for c in df.columns:
            n = norm_col(c)

            if (
                (
                    "true" in n
                    or "label" in n
                    or "target" in n
                )
                and any(
                    x in n
                    for x in [
                        "dbh",
                        "diameter",
                        "width",
                        "cm",
                    ]
                )
            ):
                true_col = c
                break

    if pred_col is None:
        for c in df.columns:
            n = norm_col(c)

            if (
                (
                    "pred" in n
                    or "prediction" in n
                )
                and any(
                    x in n
                    for x in [
                        "dbh",
                        "diameter",
                        "width",
                        "cm",
                    ]
                )
            ):
                pred_col = c
                break

    return (
        true_col,
        pred_col,
    )


def infer_classification_y_cols(
    df: pd.DataFrame,
):
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

    return (
        find_col(
            df,
            true_candidates,
        ),
        find_col(
            df,
            pred_candidates,
        ),
    )


def infer_probability_columns(
    df: pd.DataFrame,
) -> list[str]:

    prob_cols = []

    for c in df.columns:
        n = norm_col(c)

        if (
            n.startswith("prob")
            or n.startswith("classprob")
            or n.startswith("pclass")
        ):
            if pd.api.types.is_numeric_dtype(
                df[c]
            ):
                prob_cols.append(c)

    def prob_index(c):
        digits = "".join(
            ch
            for ch in str(c)
            if ch.isdigit()
        )

        return (
            int(digits)
            if digits
            else 10**9
        )

    return sorted(
        prob_cols,
        key=prob_index,
    )


# =========================================================
# Prediction table / evaluation dataframe
# =========================================================

def find_prediction_table(
    run_path: Path,
    explicit_path: str | None = None,
) -> pd.DataFrame | None:

    candidates = []

    if explicit_path is not None:
        candidates.append(
            Path(explicit_path)
        )

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

            print(
                f"Loaded prediction table: {path}"
            )

            if (
                path.suffix.lower()
                == ".parquet"
            ):
                return pd.read_parquet(
                    path
                )

            return pd.read_csv(
                path
            )

    print(
        "No prediction table found."
    )

    return None


def find_tree_ids_array(
    run_path: Path,
):
    for name in [
        "val_tree_ids.npy",
        "val_root_ids.npy",
        "tree_ids.npy",
        "root_ids.npy",
        "val_ids.npy",
        "ids.npy",
    ]:
        path = run_path / name
        arr = load_optional_npy(
            path
        )

        if arr is not None:
            print(
                f"Loaded tree IDs from: {path}"
            )

            return arr

    return None


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
    Return:
        df,
        inferred_tree_id_col,
        true_col,
        pred_col,
        prob_cols
    """

    df = find_prediction_table(
        run_path,
        predictions_csv,
    )

    if df is None:

        if (
            y_true is None
            or y_pred is None
        ):
            return (
                None,
                None,
                None,
                None,
                [],
            )

        df = pd.DataFrame(
            {
                "_y_true": np.asarray(
                    y_true
                ).reshape(-1),

                "_y_pred": np.asarray(
                    y_pred
                ).reshape(-1),
            }
        )

        ids = find_tree_ids_array(
            run_path
        )

        if (
            ids is not None
            and len(ids) == len(df)
        ):
            df["_tree_id"] = ids
            tree_id_col = "_tree_id"

        elif ids is not None:
            print(
                "[WARNING] Found a tree-ID array "
                f"with length {len(ids)}, but "
                f"predictions have length {len(df)}."
            )

        if (
            probabilities is not None
            and len(probabilities) == len(df)
        ):
            for j in range(
                probabilities.shape[1]
            ):
                df[
                    f"prob_{j}"
                ] = probabilities[:, j]

    else:

        if (
            y_true is not None
            and len(y_true) == len(df)
        ):
            df[
                "_y_true_from_npy"
            ] = np.asarray(
                y_true
            ).reshape(-1)

        if (
            y_pred is not None
            and len(y_pred) == len(df)
        ):
            df[
                "_y_pred_from_npy"
            ] = np.asarray(
                y_pred
            ).reshape(-1)

        if (
            probabilities is not None
            and len(probabilities) == len(df)
        ):
            for j in range(
                probabilities.shape[1]
            ):
                df[
                    f"prob_{j}"
                ] = probabilities[:, j]

    inferred_tree_id_col = (
        infer_tree_id_col(
            df,
            tree_id_col,
        )
    )

    if task_type == "regression":
        true_col, pred_col = (
            infer_regression_y_cols(
                df
            )
        )
    else:
        true_col, pred_col = (
            infer_classification_y_cols(
                df
            )
        )

    if (
        true_col is None
        and "_y_true_from_npy"
        in df.columns
    ):
        true_col = "_y_true_from_npy"

    if (
        pred_col is None
        and "_y_pred_from_npy"
        in df.columns
    ):
        pred_col = "_y_pred_from_npy"

    if (
        true_col is None
        and "_y_true"
        in df.columns
    ):
        true_col = "_y_true"

    if (
        pred_col is None
        and "_y_pred"
        in df.columns
    ):
        pred_col = "_y_pred"

    prob_cols = (
        infer_probability_columns(df)
        if task_type
        == "classification"
        else []
    )

    return (
        df,
        inferred_tree_id_col,
        true_col,
        pred_col,
        prob_cols,
    )


# =========================================================
# History normalisation
# =========================================================

def normalize_history(
    history,
):
    if (
        isinstance(history, dict)
        and "history" in history
    ):
        history = history[
            "history"
        ]

    if isinstance(
        history,
        list,
    ):
        epochs = []

        train_loss = []
        val_loss = []

        train_acc = []
        val_acc = []

        val_f1_macro = []
        val_recall_macro = []

        train_mae = []
        val_mae = []

        train_rmse = []
        val_rmse = []

        train_r2 = []
        val_r2 = []

        lr = []

        for i, h in enumerate(
            history,
            start=1,
        ):
            train = h.get(
                "train",
                {},
            )

            val = h.get(
                "val",
                {},
            )

            epochs.append(
                h.get(
                    "epoch",
                    i,
                )
            )

            train_loss.append(
                train.get(
                    "loss",
                    np.nan,
                )
            )

            val_loss.append(
                val.get(
                    "loss",
                    np.nan,
                )
            )

            train_acc.append(
                train.get(
                    "acc",
                    train.get(
                        "accuracy",
                        train.get(
                            "bal_acc",
                            np.nan,
                        ),
                    ),
                )
            )

            val_acc.append(
                val.get(
                    "acc",
                    val.get(
                        "accuracy",
                        val.get(
                            "bal_acc",
                            np.nan,
                        ),
                    ),
                )
            )

            val_f1_macro.append(
                val.get(
                    "f1_macro",
                    val.get(
                        "macro_f1",
                        np.nan,
                    ),
                )
            )

            val_recall_macro.append(
                val.get(
                    "recall_macro",
                    val.get(
                        "macro_recall",
                        np.nan,
                    ),
                )
            )

            train_mae.append(
                train.get(
                    "mae",
                    train.get(
                        "MAE",
                        np.nan,
                    ),
                )
            )

            val_mae.append(
                val.get(
                    "mae",
                    val.get(
                        "MAE",
                        np.nan,
                    ),
                )
            )

            train_rmse.append(
                train.get(
                    "rmse",
                    train.get(
                        "RMSE",
                        np.nan,
                    ),
                )
            )

            val_rmse.append(
                val.get(
                    "rmse",
                    val.get(
                        "RMSE",
                        np.nan,
                    ),
                )
            )

            train_r2.append(
                train.get(
                    "r2",
                    train.get(
                        "R2",
                        train.get(
                            "R²",
                            np.nan,
                        ),
                    ),
                )
            )

            val_r2.append(
                val.get(
                    "r2",
                    val.get(
                        "R2",
                        val.get(
                            "R²",
                            np.nan,
                        ),
                    ),
                )
            )

            lr.append(
                h.get(
                    "lr",
                    np.nan,
                )
            )

        return {
            "epochs": np.asarray(
                epochs,
                dtype=float,
            ),
            "train_loss": np.asarray(
                train_loss,
                dtype=float,
            ),
            "val_loss": np.asarray(
                val_loss,
                dtype=float,
            ),
            "train_acc": np.asarray(
                train_acc,
                dtype=float,
            ),
            "val_acc": np.asarray(
                val_acc,
                dtype=float,
            ),
            "val_f1_macro": np.asarray(
                val_f1_macro,
                dtype=float,
            ),
            "val_recall_macro": np.asarray(
                val_recall_macro,
                dtype=float,
            ),
            "train_mae": np.asarray(
                train_mae,
                dtype=float,
            ),
            "val_mae": np.asarray(
                val_mae,
                dtype=float,
            ),
            "train_rmse": np.asarray(
                train_rmse,
                dtype=float,
            ),
            "val_rmse": np.asarray(
                val_rmse,
                dtype=float,
            ),
            "train_r2": np.asarray(
                train_r2,
                dtype=float,
            ),
            "val_r2": np.asarray(
                val_r2,
                dtype=float,
            ),
            "lr": np.asarray(
                lr,
                dtype=float,
            ),
        }

    if (
        isinstance(history, dict)
        and "train_loss"
        in history
    ):
        n = len(
            history.get(
                "train_loss",
                [],
            )
        )

        return {
            "epochs": np.arange(
                1,
                n + 1,
            ),
            "train_loss": np.asarray(
                history.get(
                    "train_loss",
                    [np.nan] * n,
                ),
                dtype=float,
            ),
            "val_loss": np.asarray(
                history.get(
                    "val_loss",
                    [np.nan] * n,
                ),
                dtype=float,
            ),
            "train_acc": np.asarray(
                history.get(
                    "train_acc",
                    [np.nan] * n,
                ),
                dtype=float,
            ),
            "val_acc": np.asarray(
                history.get(
                    "val_acc",
                    [np.nan] * n,
                ),
                dtype=float,
            ),
            "val_f1_macro": np.asarray(
                history.get(
                    "val_f1_macro",
                    [np.nan] * n,
                ),
                dtype=float,
            ),
            "val_recall_macro": np.asarray(
                history.get(
                    "val_recall_macro",
                    [np.nan] * n,
                ),
                dtype=float,
            ),
            "train_mae": np.asarray(
                history.get(
                    "train_mae",
                    [np.nan] * n,
                ),
                dtype=float,
            ),
            "val_mae": np.asarray(
                history.get(
                    "val_mae",
                    [np.nan] * n,
                ),
                dtype=float,
            ),
            "train_rmse": np.asarray(
                history.get(
                    "train_rmse",
                    [np.nan] * n,
                ),
                dtype=float,
            ),
            "val_rmse": np.asarray(
                history.get(
                    "val_rmse",
                    [np.nan] * n,
                ),
                dtype=float,
            ),
            "train_r2": np.asarray(
                history.get(
                    "train_r2",
                    [np.nan] * n,
                ),
                dtype=float,
            ),
            "val_r2": np.asarray(
                history.get(
                    "val_r2",
                    [np.nan] * n,
                ),
                dtype=float,
            ),
            "lr": np.asarray(
                history.get(
                    "lr",
                    [np.nan] * n,
                ),
                dtype=float,
            ),
        }

    raise ValueError(
        f"Unknown history format: "
        f"{type(history)}"
    )


# =========================================================
# Metrics
# =========================================================

def regression_metrics(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "bias": np.nan,
            "medae": np.nan,
        }

    residuals = (
        y_pred - y_true
    )

    return {
        "n": int(
            len(y_true)
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse": float(
            mean_squared_error(
                y_true,
                y_pred,
            ) ** 0.5
        ),
        "r2": (
            float(
                r2_score(
                    y_true,
                    y_pred,
                )
            )
            if len(y_true) > 1
            else np.nan
        ),
        "bias": float(
            np.mean(
                residuals
            )
        ),
        "medae": float(
            np.median(
                np.abs(
                    residuals
                )
            )
        ),
    }


def classification_metrics(
    y_true,
    y_pred,
):
    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    mask = (
        pd.notna(y_true)
        & pd.notna(y_pred)
    )

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "n": 0,
            "accuracy": np.nan,
            "f1_macro": np.nan,
            "recall_macro": np.nan,
        }

    return {
        "n": int(
            len(y_true)
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
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
    """
    Aggregate all image-level regression predictions back to one
    prediction per tree.

    agg:
        "mean"   -> mean image prediction
        "median" -> median image prediction

    Uncertainty is estimated AFTER this aggregation from the
    aggregate true-vs-predicted relationship.
    """

    if agg not in {
        "mean",
        "median",
    }:
        raise ValueError(
            "Regression aggregation must be "
            "'mean' or 'median'."
        )

    tmp = df[
        [
            tree_id_col,
            true_col,
            pred_col,
        ]
    ].copy()

    tmp[true_col] = numeric_series(
        tmp[true_col]
    )

    tmp[pred_col] = numeric_series(
        tmp[pred_col]
    )

    tmp = tmp.dropna(
        subset=[
            tree_id_col,
            true_col,
            pred_col,
        ]
    )

    if len(tmp) == 0:
        raise ValueError(
            "No valid rows available for "
            "tree-level regression aggregation."
        )

    out = (
        tmp.groupby(
            tree_id_col
        )
        .agg(
            y_true=(
                true_col,
                "median",
            ),
            y_pred=(
                pred_col,
                agg,
            ),
            y_pred_std=(
                pred_col,
                "std",
            ),
            n_images=(
                pred_col,
                "size",
            ),
        )
        .reset_index()
    )

    out[
        "y_pred_std"
    ] = out[
        "y_pred_std"
    ].fillna(0.0)

    out[
        "abs_error"
    ] = np.abs(
        out["y_pred"]
        - out["y_true"]
    )

    out[
        "residual"
    ] = (
        out["y_pred"]
        - out["y_true"]
    )

    return out


def aggregate_classification_by_tree(
    df: pd.DataFrame,
    tree_id_col: str,
    true_col: str,
    pred_col: str,
    prob_cols: list[str] | None = None,
) -> pd.DataFrame:

    prob_cols = (
        prob_cols
        or []
    )

    tmp_cols = [
        tree_id_col,
        true_col,
        pred_col,
        *prob_cols,
    ]

    tmp = (
        df[
            tmp_cols
        ]
        .copy()
        .dropna(
            subset=[
                tree_id_col,
                true_col,
                pred_col,
            ]
        )
    )

    if len(tmp) == 0:
        raise ValueError(
            "No valid rows available for "
            "tree-level classification aggregation."
        )

    if prob_cols:
        rows = []

        for (
            tree_id,
            group,
        ) in tmp.groupby(
            tree_id_col
        ):

            probs = group[
                prob_cols
            ].to_numpy(
                dtype=float
            )

            mean_probs = (
                np.nanmean(
                    probs,
                    axis=0,
                )
            )

            pred_class = int(
                np.nanargmax(
                    mean_probs
                )
            )

            row = {
                tree_id_col: tree_id,
                "y_true": mode_value(
                    group[
                        true_col
                    ]
                ),
                "y_pred": pred_class,
                "n_images": len(
                    group
                ),
            }

            for j in range(
                len(mean_probs)
            ):
                row[
                    f"mean_prob_{j}"
                ] = mean_probs[j]

            rows.append(
                row
            )

        return pd.DataFrame(
            rows
        )

    return (
        tmp.groupby(
            tree_id_col
        )
        .agg(
            y_true=(
                true_col,
                mode_value,
            ),
            y_pred=(
                pred_col,
                mode_value,
            ),
            n_images=(
                pred_col,
                "size",
            ),
        )
        .reset_index()
    )


# =========================================================
# Training plots
# =========================================================

def plot_training_diagnostics(
    history,
    title_prefix: str = "Model",
    task_type: str = "classification",
    save_path: Path | None = None,
):
    hist = normalize_history(
        history
    )

    epochs = hist[
        "epochs"
    ]

    train_loss = hist[
        "train_loss"
    ]

    val_loss = hist[
        "val_loss"
    ]

    lr = hist[
        "lr"
    ]

    if task_type == "regression":
        best_idx = safe_nanargmin(
            hist["val_mae"]
        )
    else:
        best_idx = safe_nanargmax(
            hist["val_f1_macro"]
        )

        if best_idx is None:
            best_idx = safe_nanargmax(
                hist["val_acc"]
            )

    best_epoch = (
        int(
            epochs[
                best_idx
            ]
        )
        if best_idx
        is not None
        else None
    )

    fig = plt.figure(
        figsize=(
            17,
            10.5,
        )
    )

    fig.suptitle(
        title_prefix,
        fontsize=17,
        fontweight="bold",
        y=1.015,
    )

    ax1 = plt.subplot2grid(
        (
            2,
            3,
        ),
        (
            0,
            0,
        ),
        colspan=3,
    )

    ax1.plot(
        epochs,
        train_loss,
        marker="o",
        ms=4,
        lw=2.0,
        color=TREECO[
            "blue"
        ],
        label="Train loss",
    )

    ax1.plot(
        epochs,
        val_loss,
        marker="o",
        ms=4,
        lw=2.0,
        color=TREECO[
            "orange"
        ],
        label="Validation loss",
    )

    if best_epoch is not None:
        ax1.axvline(
            best_epoch,
            linestyle="--",
            lw=1.7,
            color=TREECO[
                "dark_red"
            ],
            label=(
                f"Best epoch "
                f"{best_epoch}"
            ),
        )

    ax1.set_title(
        "Loss curve"
    )
    ax1.set_xlabel(
        "Epoch"
    )
    ax1.set_ylabel(
        "Loss"
    )
    ax1.legend(
        loc="best"
    )

    if task_type == "regression":

        train_mae = hist[
            "train_mae"
        ]
        val_mae = hist[
            "val_mae"
        ]

        train_rmse = hist[
            "train_rmse"
        ]
        val_rmse = hist[
            "val_rmse"
        ]

        train_r2 = hist[
            "train_r2"
        ]
        val_r2 = hist[
            "val_r2"
        ]

        ax2 = plt.subplot2grid(
            (
                2,
                3,
            ),
            (
                1,
                0,
            ),
        )

        ax2.plot(
            epochs,
            train_mae,
            marker="o",
            ms=4,
            lw=2.0,
            color=TREECO[
                "blue"
            ],
            label="Train MAE",
        )

        ax2.plot(
            epochs,
            val_mae,
            marker="o",
            ms=4,
            lw=2.0,
            color=TREECO[
                "orange"
            ],
            label="Validation MAE",
        )

        if best_epoch is not None:
            ax2.axvline(
                best_epoch,
                linestyle="--",
                lw=1.5,
                color=TREECO[
                    "dark_red"
                ],
            )

        ax2.set_title(
            "Mean absolute error"
        )
        ax2.set_xlabel(
            "Epoch"
        )
        ax2.set_ylabel(
            "MAE [cm]"
        )
        ax2.legend(
            loc="best"
        )

        ax3 = plt.subplot2grid(
            (
                2,
                3,
            ),
            (
                1,
                1,
            ),
        )

        ax3.plot(
            epochs,
            train_rmse,
            marker="o",
            ms=4,
            lw=2.0,
            color=TREECO[
                "blue"
            ],
            label="Train RMSE",
        )

        ax3.plot(
            epochs,
            val_rmse,
            marker="o",
            ms=4,
            lw=2.0,
            color=TREECO[
                "orange"
            ],
            label="Validation RMSE",
        )

        if best_epoch is not None:
            ax3.axvline(
                best_epoch,
                linestyle="--",
                lw=1.5,
                color=TREECO[
                    "dark_red"
                ],
            )

        ax3.set_title(
            "Root mean squared error"
        )
        ax3.set_xlabel(
            "Epoch"
        )
        ax3.set_ylabel(
            "RMSE [cm]"
        )
        ax3.legend(
            loc="best"
        )

        ax4 = plt.subplot2grid(
            (
                2,
                3,
            ),
            (
                1,
                2,
            ),
        )

        ax4.plot(
            epochs,
            train_r2,
            marker="o",
            ms=4,
            lw=2.0,
            color=TREECO[
                "blue"
            ],
            label=r"Train $R^2$",
        )

        ax4.plot(
            epochs,
            val_r2,
            marker="o",
            ms=4,
            lw=2.0,
            color=TREECO[
                "orange"
            ],
            label=r"Validation $R^2$",
        )

        ax4.axhline(
            0,
            linestyle="--",
            lw=1.2,
            color=TREECO[
                "cyan"
            ],
            alpha=0.65,
        )

        if best_epoch is not None:
            ax4.axvline(
                best_epoch,
                linestyle="--",
                lw=1.5,
                color=TREECO[
                    "dark_red"
                ],
            )

        ax4.set_title(
            "Explained variance"
        )
        ax4.set_xlabel(
            "Epoch"
        )
        ax4.set_ylabel(
            r"$R^2$"
        )
        ax4.legend(
            loc="upper left"
        )

        if not np.all(
            np.isnan(lr)
        ):
            ax4b = (
                ax4.twinx()
            )

            ax4b.plot(
                epochs,
                lr,
                linestyle=":",
                lw=2.0,
                color=TREECO[
                    "dark_red"
                ],
                label="Learning rate",
            )

            ax4b.set_ylabel(
                "Learning rate"
            )
            ax4b.set_yscale(
                "log"
            )
            ax4b.legend(
                loc="lower right"
            )

    else:

        train_acc = hist[
            "train_acc"
        ]
        val_acc = hist[
            "val_acc"
        ]

        val_f1_macro = hist[
            "val_f1_macro"
        ]

        val_recall_macro = hist[
            "val_recall_macro"
        ]

        ax2 = plt.subplot2grid(
            (
                2,
                3,
            ),
            (
                1,
                0,
            ),
        )

        ax2.plot(
            epochs,
            train_acc,
            marker="o",
            ms=4,
            lw=2.0,
            color=TREECO[
                "blue"
            ],
            label="Train accuracy",
        )

        ax2.plot(
            epochs,
            val_acc,
            marker="o",
            ms=4,
            lw=2.0,
            color=TREECO[
                "orange"
            ],
            label="Validation accuracy",
        )

        ax2.set_title(
            "Accuracy"
        )
        ax2.set_xlabel(
            "Epoch"
        )
        ax2.set_ylabel(
            "Accuracy"
        )
        ax2.legend(
            loc="best"
        )

        ax3 = plt.subplot2grid(
            (
                2,
                3,
            ),
            (
                1,
                1,
            ),
        )

        ax3.plot(
            epochs,
            val_f1_macro,
            marker="o",
            ms=4,
            lw=2.0,
            color=TREECO[
                "cyan"
            ],
            label="Validation macro F1",
        )

        ax3.set_title(
            "Macro F1"
        )
        ax3.set_xlabel(
            "Epoch"
        )
        ax3.set_ylabel(
            "F1"
        )
        ax3.legend(
            loc="best"
        )

        ax4 = plt.subplot2grid(
            (
                2,
                3,
            ),
            (
                1,
                2,
            ),
        )

        ax4.plot(
            epochs,
            val_recall_macro,
            marker="o",
            ms=4,
            lw=2.0,
            color=TREECO[
                "rust"
            ],
            label="Validation macro recall",
        )

        ax4.set_title(
            "Macro recall"
        )
        ax4.set_xlabel(
            "Epoch"
        )
        ax4.set_ylabel(
            "Recall"
        )
        ax4.legend(
            loc="best"
        )

    if save_path is not None:
        save_figure(
            fig,
            save_path,
        )
    else:
        plt.show()


# =========================================================
# Regression evaluation plot
# =========================================================

def plot_regression_evaluation_diagnostics(
    y_true,
    y_pred,
    title_prefix: str,
    level_name: str,
    save_path: Path,
    report_path: Path,
):

    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    mask = (
        np.isfinite(y_true)
        & np.isfinite(y_pred)
    )

    y_true = y_true[
        mask
    ]

    y_pred = y_pred[
        mask
    ]

    if len(y_true) == 0:
        raise ValueError(
            "No finite y_true/y_pred "
            "values available."
        )

    residuals = (
        y_pred - y_true
    )

    abs_errors = np.abs(
        residuals
    )

    metrics = regression_metrics(
        y_true,
        y_pred,
    )

    report_str = "\n".join(
        [
            (
                f"Regression evaluation: "
                f"{level_name}"
            ),
            (
                f"Samples: "
                f"{metrics['n']}"
            ),
            (
                f"MAE:     "
                f"{metrics['mae']:.4f} cm"
            ),
            (
                f"RMSE:    "
                f"{metrics['rmse']:.4f} cm"
            ),
            (
                f"R2:      "
                f"{metrics['r2']:.4f}"
            ),
            (
                "Mean residual / bias: "
                f"{metrics['bias']:.4f} cm"
            ),
            (
                "Median absolute error: "
                f"{metrics['medae']:.4f} cm"
            ),
        ]
    )

    print(
        report_str
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        report_str
    )

    print(
        f"Saved: {report_path}"
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(
            14.5,
            10.8,
        ),
    )

    fig.suptitle(
        (
            f"{title_prefix} — "
            f"{level_name}"
        ),
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    min_val = float(
        min(
            np.min(y_true),
            np.min(y_pred),
        )
    )

    max_val = float(
        max(
            np.max(y_true),
            np.max(y_pred),
        )
    )

    pad = (
        0.04
        * (
            max_val
            - min_val
            + 1e-9
        )
    )

    lims = [
        min_val - pad,
        max_val + pad,
    ]

    sc = axes[
        0,
        0,
    ].scatter(
        y_true,
        y_pred,
        c=abs_errors,
        cmap=TREECO_CMAP,
        s=48,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.45,
    )

    axes[
        0,
        0,
    ].plot(
        lims,
        lims,
        linestyle="--",
        lw=1.8,
        color=TREECO[
            "dark_red"
        ],
        label=r"$\hat{y}=y$",
    )

    axes[
        0,
        0,
    ].set_xlim(
        lims
    )

    axes[
        0,
        0,
    ].set_ylim(
        lims
    )

    axes[
        0,
        0,
    ].set_xlabel(
        "Observed DBH [cm]"
    )

    axes[
        0,
        0,
    ].set_ylabel(
        "Predicted DBH [cm]"
    )

    axes[
        0,
        0,
    ].set_title(
        "Predicted vs observed"
    )

    axes[
        0,
        0,
    ].legend(
        loc="lower right"
    )

    cbar = fig.colorbar(
        sc,
        ax=axes[
            0,
            0,
        ],
    )

    cbar.set_label(
        "Absolute error [cm]"
    )

    axes[
        0,
        0,
    ].text(
        0.04,
        0.96,
        "\n".join(
            [
                (
                    f"N = "
                    f"{metrics['n']}"
                ),
                (
                    f"MAE = "
                    f"{metrics['mae']:.2f} cm"
                ),
                (
                    f"RMSE = "
                    f"{metrics['rmse']:.2f} cm"
                ),
                (
                    f"R² = "
                    f"{metrics['r2']:.3f}"
                ),
            ]
        ),
        transform=axes[
            0,
            0,
        ].transAxes,
        va="top",
        bbox=dict(
            boxstyle=(
                "round,pad=0.4"
            ),
            facecolor="white",
            edgecolor="#CBD5E1",
            alpha=0.94,
        ),
    )

    axes[
        0,
        1,
    ].scatter(
        y_true,
        residuals,
        c=residuals,
        cmap="coolwarm",
        s=48,
        alpha=0.86,
        edgecolor="white",
        linewidth=0.45,
    )

    axes[
        0,
        1,
    ].axhline(
        0,
        linestyle="--",
        lw=1.8,
        color=TREECO[
            "ink"
        ],
    )

    axes[
        0,
        1,
    ].set_xlabel(
        "Observed DBH [cm]"
    )

    axes[
        0,
        1,
    ].set_ylabel(
        "Residual, prediction - truth [cm]"
    )

    axes[
        0,
        1,
    ].set_title(
        "Residuals vs observed DBH"
    )

    axes[
        1,
        0,
    ].hist(
        residuals,
        bins=24,
        color=TREECO[
            "cyan"
        ],
        alpha=0.86,
        edgecolor="white",
    )

    axes[
        1,
        0,
    ].axvline(
        0,
        linestyle="--",
        lw=1.8,
        color=TREECO[
            "ink"
        ],
    )

    axes[
        1,
        0,
    ].axvline(
        np.mean(
            residuals
        ),
        linestyle="-",
        lw=2.0,
        color=TREECO[
            "dark_red"
        ],
        label=(
            f"Mean = "
            f"{np.mean(residuals):.2f} cm"
        ),
    )

    axes[
        1,
        0,
    ].set_xlabel(
        "Residual [cm]"
    )

    axes[
        1,
        0,
    ].set_ylabel(
        "Count"
    )

    axes[
        1,
        0,
    ].set_title(
        "Residual distribution"
    )

    axes[
        1,
        0,
    ].legend(
        loc="best"
    )

    axes[
        1,
        1,
    ].hist(
        abs_errors,
        bins=24,
        color=TREECO[
            "orange"
        ],
        alpha=0.88,
        edgecolor="white",
    )

    axes[
        1,
        1,
    ].axvline(
        metrics[
            "mae"
        ],
        linestyle="-",
        lw=2.0,
        color=TREECO[
            "dark_red"
        ],
        label=(
            f"MAE = "
            f"{metrics['mae']:.2f} cm"
        ),
    )

    axes[
        1,
        1,
    ].set_xlabel(
        "Absolute error [cm]"
    )

    axes[
        1,
        1,
    ].set_ylabel(
        "Count"
    )

    axes[
        1,
        1,
    ].set_title(
        "Absolute error distribution"
    )

    axes[
        1,
        1,
    ].legend(
        loc="best"
    )

    save_figure(
        fig,
        save_path,
    )

    return metrics


# =========================================================
# Classification evaluation
# =========================================================

def get_class_names(
    config: dict | None,
):
    if config is None:
        return None

    if "class_names" in config:
        return list(
            config[
                "class_names"
            ]
        )

    for key in [
        "height_class_mapping",
        "height_mapping",
        "diameter_class_mapping",
        "class_mapping",
    ]:
        if key in config:
            mapping = config[
                key
            ]

            return [
                mapping[k]
                for k in sorted(
                    mapping.keys(),
                    key=lambda x: int(x),
                )
            ]

    return None


def label_names_for_present_labels(
    labels,
    class_names,
):
    if class_names is None:
        return [
            str(x)
            for x in labels
        ]

    names = []

    for label in labels:
        try:
            i = int(
                label
            )

            if (
                0 <= i
                < len(class_names)
            ):
                names.append(
                    str(
                        class_names[i]
                    )
                )
            else:
                names.append(
                    str(label)
                )

        except Exception:
            names.append(
                str(label)
            )

    return names


def plot_classification_evaluation_diagnostics(
    y_true,
    y_pred,
    class_names,
    title_prefix: str,
    level_name: str,
    save_path: Path,
    report_path: Path,
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    mask = (
        pd.notna(y_true)
        & pd.notna(y_pred)
    )

    y_true = y_true[
        mask
    ]

    y_pred = y_pred[
        mask
    ]

    labels = np.unique(
        np.concatenate(
            [
                y_true,
                y_pred,
            ]
        )
    )

    names = (
        label_names_for_present_labels(
            labels,
            class_names,
        )
    )

    metrics = (
        classification_metrics(
            y_true,
            y_pred,
        )
    )

    report = (
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=names,
            digits=3,
            zero_division=0,
        )
    )

    report_str = "\n".join(
        [
            (
                f"Classification evaluation: "
                f"{level_name}"
            ),
            (
                f"Samples: "
                f"{metrics['n']}"
            ),
            (
                f"Accuracy: "
                f"{metrics['accuracy']:.4f}"
            ),
            (
                f"Macro F1: "
                f"{metrics['f1_macro']:.4f}"
            ),
            (
                f"Macro Recall: "
                f"{metrics['recall_macro']:.4f}"
            ),
            "",
            report,
        ]
    )

    print(
        report_str
    )

    report_path.write_text(
        report_str
    )

    print(
        f"Saved: {report_path}"
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    row_sums = cm.sum(
        axis=1,
        keepdims=True,
    )

    cm_display = np.divide(
        cm.astype(float),
        row_sums,
        out=np.zeros_like(
            cm,
            dtype=float,
        ),
        where=row_sums != 0,
    )

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            6.8,
        )
    )

    im = ax.imshow(
        cm_display,
        cmap=TREECO_CMAP,
        aspect="auto",
    )

    fig.colorbar(
        im,
        ax=ax,
        label=(
            "Proportion of true class"
        ),
    )

    for i in range(
        cm.shape[0]
    ):
        for j in range(
            cm.shape[1]
        ):
            ax.text(
                j,
                i,
                (
                    f"{cm[i, j]}\n"
                    f"{cm_display[i, j]:.2f}"
                ),
                ha="center",
                va="center",
            )

    ax.set_xticks(
        np.arange(
            len(names)
        )
    )

    ax.set_yticks(
        np.arange(
            len(names)
        )
    )

    ax.set_xticklabels(
        names,
        rotation=35,
        ha="right",
    )

    ax.set_yticklabels(
        names
    )

    ax.set_xlabel(
        "Predicted class"
    )

    ax.set_ylabel(
        "True class"
    )

    ax.set_title(
        (
            f"{title_prefix} — "
            f"{level_name}"
        )
    )

    save_figure(
        fig,
        save_path,
    )

    return metrics


# =========================================================
# Image-vs-tree metric comparison
# =========================================================

def plot_regression_metric_comparison(
    image_metrics,
    tree_metrics,
    save_path: Path,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            12.5,
            5.2,
        ),
        gridspec_kw={
            "width_ratios": [
                1.25,
                0.75,
            ]
        },
    )

    fig.suptitle(
        "Image-level vs tree-level aggregate performance",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )

    names = [
        "MAE",
        "RMSE",
    ]

    image_errors = [
        image_metrics[
            "mae"
        ],
        image_metrics[
            "rmse"
        ],
    ]

    tree_errors = [
        tree_metrics[
            "mae"
        ],
        tree_metrics[
            "rmse"
        ],
    ]

    x = np.arange(
        len(names)
    )

    width = 0.36

    axes[
        0
    ].bar(
        x - width / 2,
        image_errors,
        width,
        color=TREECO[
            "blue"
        ],
        label="Image level",
    )

    axes[
        0
    ].bar(
        x + width / 2,
        tree_errors,
        width,
        color=TREECO[
            "orange"
        ],
        label="Tree aggregate",
    )

    axes[
        0
    ].set_xticks(
        x
    )

    axes[
        0
    ].set_xticklabels(
        names
    )

    axes[
        0
    ].set_ylabel(
        "Error [cm]"
    )

    axes[
        0
    ].set_title(
        "Error metrics"
    )

    axes[
        0
    ].legend(
        loc="best"
    )

    axes[
        1
    ].bar(
        [
            0,
            1,
        ],
        [
            image_metrics[
                "r2"
            ],
            tree_metrics[
                "r2"
            ],
        ],
        width=0.5,
        color=[
            TREECO[
                "blue"
            ],
            TREECO[
                "orange"
            ],
        ],
    )

    axes[
        1
    ].axhline(
        0,
        linestyle="--",
        color=TREECO[
            "ink"
        ],
        lw=1.2,
        alpha=0.65,
    )

    axes[
        1
    ].set_xticks(
        [
            0,
            1,
        ]
    )

    axes[
        1
    ].set_xticklabels(
        [
            "Image",
            "Tree",
        ]
    )

    axes[
        1
    ].set_ylabel(
        r"$R^2$"
    )

    axes[
        1
    ].set_title(
        "Explained variance"
    )

    save_figure(
        fig,
        save_path,
    )


def plot_tree_group_size_distribution(
    tree_df: pd.DataFrame,
    save_path: Path,
    title_prefix: str,
):
    if (
        "n_images"
        not in tree_df.columns
    ):
        return

    counts = tree_df[
        "n_images"
    ].astype(
        int
    ).to_numpy()

    fig, ax = plt.subplots(
        figsize=(
            8.5,
            5.2,
        )
    )

    bins = np.arange(
        0.5,
        counts.max() + 1.5,
        1,
    )

    ax.hist(
        counts,
        bins=bins,
        color=TREECO[
            "cyan"
        ],
        edgecolor="white",
        alpha=0.9,
    )

    ax.set_xlabel(
        "Validation images per tree"
    )

    ax.set_ylabel(
        "Number of trees"
    )

    ax.set_title(
        (
            f"{title_prefix} — "
            "tree aggregate group sizes"
        )
    )

    ax.text(
        0.97,
        0.95,
        "\n".join(
            [
                (
                    f"Trees = "
                    f"{len(tree_df)}"
                ),
                (
                    f"Images = "
                    f"{int(counts.sum())}"
                ),
                (
                    f"Mean = "
                    f"{counts.mean():.2f}"
                ),
            ]
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(
            boxstyle=(
                "round,pad=0.4"
            ),
            facecolor="white",
            edgecolor="#CBD5E1",
            alpha=0.94,
        ),
    )

    save_figure(
        fig,
        save_path,
    )


# =========================================================
# FINAL TREE-LEVEL UNCERTAINTY MODEL
# =========================================================

def build_smooth_tree_sigma_lookup(
    tree_df: pd.DataFrame,
    bandwidth_cm: float = 12.0,
    n_grid: int = 300,
    min_effective_n: float = 8.0,
) -> pd.DataFrame:
    """
    Build a smooth prediction-dependent uncertainty relationship
    from ALL aggregate validation trees.

    For each aggregate predicted DBH x:

        residual = observed DBH - predicted DBH

    Nearby aggregate trees receive Gaussian weights.

    The lookup estimates:
        local bias     = E[residual | predicted DBH]
        local sigma    = SD[residual | predicted DBH]
        expected truth = prediction + local bias

    The plotted 1-sigma and 2-sigma regions are empirical local
    standard-deviation bands. They are not automatically formal
    68% / 95% confidence intervals.

    Because this final descriptive lookup is fitted to all aggregate
    validation trees, its coverage on those same trees is in-sample
    descriptive coverage rather than held-out generalisation coverage.
    """

    tmp = tree_df[
        [
            "y_true",
            "y_pred",
        ]
    ].copy()

    tmp[
        "y_true"
    ] = pd.to_numeric(
        tmp[
            "y_true"
        ],
        errors="coerce",
    )

    tmp[
        "y_pred"
    ] = pd.to_numeric(
        tmp[
            "y_pred"
        ],
        errors="coerce",
    )

    tmp = tmp.dropna()

    if len(tmp) < 10:
        raise ValueError(
            "Too few aggregate trees "
            "to estimate a useful "
            "prediction-dependent uncertainty curve."
        )

    y_true = tmp[
        "y_true"
    ].to_numpy(
        dtype=float
    )

    y_pred = tmp[
        "y_pred"
    ].to_numpy(
        dtype=float
    )

    residual = (
        y_true - y_pred
    )

    x_grid = np.linspace(
        float(
            y_pred.min()
        ),
        float(
            y_pred.max()
        ),
        int(
            max(
                n_grid,
                50,
            )
        ),
    )

    mean_residual = np.full(
        len(x_grid),
        np.nan,
        dtype=float,
    )

    std_residual = np.full(
        len(x_grid),
        np.nan,
        dtype=float,
    )

    effective_n = np.zeros(
        len(x_grid),
        dtype=float,
    )

    for i, x in enumerate(
        x_grid
    ):

        weights = np.exp(
            -0.5
            * (
                (
                    y_pred - x
                )
                / bandwidth_cm
            ) ** 2
        )

        weight_sum = np.sum(
            weights
        )

        weight_sq_sum = np.sum(
            weights ** 2
        )

        if (
            weight_sum <= 0
            or weight_sq_sum <= 0
        ):
            continue

        n_eff = (
            weight_sum ** 2
            / weight_sq_sum
        )

        effective_n[i] = (
            n_eff
        )

        mu = np.sum(
            weights
            * residual
        ) / weight_sum

        variance = np.sum(
            weights
            * (
                residual
                - mu
            ) ** 2
        ) / weight_sum

        # Small weighted-sample correction.
        if n_eff > 1.0:
            variance *= (
                n_eff
                / (
                    n_eff
                    - 1.0
                )
            )

        sigma = np.sqrt(
            max(
                variance,
                0.0,
            )
        )

        mean_residual[i] = (
            mu
        )

        std_residual[i] = (
            sigma
        )

    centre = (
        x_grid
        + mean_residual
    )

    lower_1sigma = (
        centre
        - std_residual
    )

    upper_1sigma = (
        centre
        + std_residual
    )

    lower_2sigma = (
        centre
        - 2.0
        * std_residual
    )

    upper_2sigma = (
        centre
        + 2.0
        * std_residual
    )

    # Physical DBH lower bound.
    lower_1sigma = np.maximum(
        lower_1sigma,
        0.0,
    )

    lower_2sigma = np.maximum(
        lower_2sigma,
        0.0,
    )

    unreliable = (
        effective_n
        < min_effective_n
    )

    # Hide poorly supported edge regions.
    centre[
        unreliable
    ] = np.nan

    lower_1sigma[
        unreliable
    ] = np.nan

    upper_1sigma[
        unreliable
    ] = np.nan

    lower_2sigma[
        unreliable
    ] = np.nan

    upper_2sigma[
        unreliable
    ] = np.nan

    return pd.DataFrame(
        {
            "predicted_dbh_cm": (
                x_grid
            ),
            "local_bias_cm": (
                mean_residual
            ),
            "sigma_cm": (
                std_residual
            ),
            "two_sigma_cm": (
                2.0
                * std_residual
            ),
            "expected_true_dbh_cm": (
                centre
            ),
            "lower_1sigma_cm": (
                lower_1sigma
            ),
            "upper_1sigma_cm": (
                upper_1sigma
            ),
            "lower_2sigma_cm": (
                lower_2sigma
            ),
            "upper_2sigma_cm": (
                upper_2sigma
            ),
            "effective_n": (
                effective_n
            ),
        }
    )


def _interp_finite(
    x_query,
    x_grid,
    values,
):
    """
    Interpolate while ignoring NaN grid regions.
    Returns NaN if fewer than two finite grid points exist.
    """

    x_query = np.asarray(
        x_query,
        dtype=float,
    )

    x_grid = np.asarray(
        x_grid,
        dtype=float,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    valid = (
        np.isfinite(x_grid)
        & np.isfinite(values)
    )

    if valid.sum() < 2:
        return np.full(
            len(x_query),
            np.nan,
            dtype=float,
        )

    out = np.interp(
        x_query,
        x_grid[valid],
        values[valid],
        left=np.nan,
        right=np.nan,
    )

    return out


def sigma_band_coverage(
    tree_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
):
    """
    Descriptive same-sample coverage of the smooth sigma bands.
    """

    y_true = tree_df[
        "y_true"
    ].to_numpy(
        dtype=float
    )

    y_pred = tree_df[
        "y_pred"
    ].to_numpy(
        dtype=float
    )

    x = lookup_df[
        "predicted_dbh_cm"
    ].to_numpy(
        dtype=float
    )

    lower_1 = _interp_finite(
        y_pred,
        x,
        lookup_df[
            "lower_1sigma_cm"
        ].to_numpy(
            dtype=float
        ),
    )

    upper_1 = _interp_finite(
        y_pred,
        x,
        lookup_df[
            "upper_1sigma_cm"
        ].to_numpy(
            dtype=float
        ),
    )

    lower_2 = _interp_finite(
        y_pred,
        x,
        lookup_df[
            "lower_2sigma_cm"
        ].to_numpy(
            dtype=float
        ),
    )

    upper_2 = _interp_finite(
        y_pred,
        x,
        lookup_df[
            "upper_2sigma_cm"
        ].to_numpy(
            dtype=float
        ),
    )

    valid_1 = (
        np.isfinite(y_true)
        & np.isfinite(lower_1)
        & np.isfinite(upper_1)
    )

    valid_2 = (
        np.isfinite(y_true)
        & np.isfinite(lower_2)
        & np.isfinite(upper_2)
    )

    coverage_1 = (
        float(
            np.mean(
                (
                    y_true[
                        valid_1
                    ]
                    >= lower_1[
                        valid_1
                    ]
                )
                & (
                    y_true[
                        valid_1
                    ]
                    <= upper_1[
                        valid_1
                    ]
                )
            )
        )
        if valid_1.any()
        else np.nan
    )

    coverage_2 = (
        float(
            np.mean(
                (
                    y_true[
                        valid_2
                    ]
                    >= lower_2[
                        valid_2
                    ]
                )
                & (
                    y_true[
                        valid_2
                    ]
                    <= upper_2[
                        valid_2
                    ]
                )
            )
        )
        if valid_2.any()
        else np.nan
    )

    return {
        "coverage_1sigma": (
            coverage_1
        ),
        "coverage_2sigma": (
            coverage_2
        ),
        "n_1sigma": int(
            valid_1.sum()
        ),
        "n_2sigma": int(
            valid_2.sum()
        ),
    }


def plot_tree_aggregate_sigma_uncertainty(
    tree_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    save_path: Path,
    METRICS: dict | None = None
):
    """
    Plot all aggregate trees on true-vs-predicted axes with smooth
    local 1-sigma and 2-sigma empirical residual bands.
    """

    y_true = tree_df[
        "y_true"
    ].to_numpy(
        dtype=float
    )

    y_pred = tree_df[
        "y_pred"
    ].to_numpy(
        dtype=float
    )

    x = lookup_df[
            "predicted_dbh_cm"
        ].to_numpy(
            dtype=float
        )

    # =====================================================
    # Global unweighted linear fit
    # observed DBH = intercept + slope * predicted DBH
    # =====================================================

    valid_fit = (
        np.isfinite(y_pred)
        & np.isfinite(y_true)
    )

    fit_pred = y_pred[valid_fit]
    fit_true = y_true[valid_fit]

    slope, intercept = np.polyfit(
        fit_pred,
        fit_true,
        deg=1,
    )

    global_fit = (
        intercept
        + slope * x
    )

    global_fit_r2 = r2_score(
        fit_true,
        intercept + slope * fit_pred,
    )

    print()
    print("Global unweighted aggregate regression")
    print("--------------------------------------")
    print(f"Slope:     {slope:.4f}")
    print(f"Intercept: {intercept:.4f} cm")
    print(f"R2:        {global_fit_r2:.4f}")


    centre = lookup_df[
        "expected_true_dbh_cm"
    ].to_numpy(
        dtype=float
    )

    lower_1 = lookup_df[
        "lower_1sigma_cm"
    ].to_numpy(
        dtype=float
    )

    upper_1 = lookup_df[
        "upper_1sigma_cm"
    ].to_numpy(
        dtype=float
    )

    lower_2 = lookup_df[
        "lower_2sigma_cm"
    ].to_numpy(
        dtype=float
    )

    upper_2 = lookup_df[
        "upper_2sigma_cm"
    ].to_numpy(
        dtype=float
    )

    coverage = sigma_band_coverage(
        tree_df,
        lookup_df,
    )

    fig, ax = plt.subplots(
        figsize=(
            10.5,
            8.0,
        )
    )

    # 2 sigma outer band.
    label_2 = r"$\pm2\sigma$"

    if np.isfinite(
        coverage[
            "coverage_2sigma"
        ]
    ):
        label_2 += (
            " "
            f"(descriptive coverage "
            f"{coverage['coverage_2sigma'] * 100:.1f}%)"
        )

    ax.fill_between(
        x,
        lower_2,
        upper_2,
        color=TREECO[
            "cream"
        ],
        alpha=0.38,
        label=label_2,
    )

    # 1 sigma inner band.
    label_1 = r"$\pm1\sigma$"

    if np.isfinite(
        coverage[
            "coverage_1sigma"
        ]
    ):
        label_1 += (
            " "
            f"(descriptive coverage "
            f"{coverage['coverage_1sigma'] * 100:.1f}%)"
        )

    ax.fill_between(
        x,
        lower_1,
        upper_1,
        color=TREECO[
            "mint"
        ],
        alpha=0.58,
        label=label_1,
    )

    # Local expected true DBH / local bias trend.
    ax.plot(
        x,
        centre,
        color=TREECO[
            "blue"
        ],
        linewidth=4.6,
        label=(
            "Local mean observed DBH"
        ),
    )

    ax.plot(
        x,
        intercept + slope * x,
        color="green",
        linestyle = "--",
        linewidth=4.6,
        label=f"Observed best fit over validation:  {intercept:.4f} + {slope:.4f} * y_pred, ",
    )

    # Actual aggregate trees.
    ax.scatter(
        y_pred,
        y_true,
        s=62,
        alpha=0.78,
        color="red",
        edgecolor="black",
        linewidth=0.6,
        label=(
            f"Aggregate trees "
            f"(N={len(tree_df)})"
        ),
        zorder=5,
    )
    ax.scatter([],[], label = f"MAE:     "
                    f"{METRICS['mae']:.4f} cm", color = 'none')

    # Perfect prediction.
    finite = np.concatenate(
        [
            y_true[
                np.isfinite(
                    y_true
                )
            ],
            y_pred[
                np.isfinite(
                    y_pred
                )
            ],
            upper_2[
                np.isfinite(
                    upper_2
                )
            ],
        ]
    )

    lim_min = max(
        0.0,
        float(
            np.min(
                finite
            )
        ),
    )

    lim_max = float(
        np.max(
            finite
        )
    )

    ax.plot(
        [
            lim_min,
            lim_max,
        ],
        [
            lim_min,
            lim_max,
        ],
        linestyle=":",
        linewidth=2.0,
        color=TREECO[
            "dark_red"
        ],
        label="Perfect prediction",
    )

    ax.set_xlabel(
        (
            "Aggregate predicted DBH, "
            r"$\hat{y}$ [cm]"
        )
    )

    ax.set_ylabel(
        (
            "Observed DBH, "
            r"$y$ [cm]"
        )
    )

    ax.set_title(
        (
            "TreeCo aggregate DBH prediction "
            "with local empirical uncertainty"
        )
    )

    ax.set_xlim(
        [
            10.0,
            150.0,
        ]
    )

    ax.set_ylim(
            [
                1.0,
                150.0,
            ]
        )
    # ax.set_yscale("log")
    # ax.set_xscale("log")

    ax.legend(
        loc="best",
        fontsize=9,
    )

    save_figure(
        fig,
        save_path,
    )


def plot_tree_sigma_vs_prediction(
    lookup_df: pd.DataFrame,
    save_path: Path,
):
    """
    Simple 2-D uncertainty lookup plot:

        x = aggregate predicted DBH
        y = local residual uncertainty

    Shows both 1 sigma and 2 sigma.
    """

    x = lookup_df[
        "predicted_dbh_cm"
    ].to_numpy(
        dtype=float
    )

    sigma = lookup_df[
        "sigma_cm"
    ].to_numpy(
        dtype=float
    )

    two_sigma = lookup_df[
        "two_sigma_cm"
    ].to_numpy(
        dtype=float
    )

    n_eff = lookup_df[
        "effective_n"
    ].to_numpy(
        dtype=float
    )

    reliable = np.isfinite(
        lookup_df[
            "expected_true_dbh_cm"
        ].to_numpy(
            dtype=float
        )
    )

    sigma_plot = np.where(
        reliable,
        sigma,
        np.nan,
    )

    two_sigma_plot = np.where(
        reliable,
        two_sigma,
        np.nan,
    )

    fig, ax = plt.subplots(
        figsize=(
            10.0,
            6.0,
        )
    )

    ax.plot(
        x,
        sigma_plot,
        color=TREECO[
            "blue"
        ],
        linewidth=2.6,
        label=r"$1\sigma$",
    )

    ax.plot(
        x,
        two_sigma_plot,
        color=TREECO[
            "orange"
        ],
        linewidth=2.6,
        label=r"$2\sigma$",
    )

    ax.fill_between(
        x,
        sigma_plot,
        two_sigma_plot,
        color=TREECO[
            "cream"
        ],
        alpha=0.25,
    )

    ax.set_xlabel(
        (
            "Aggregate predicted DBH, "
            r"$\hat{y}$ [cm]"
        )
    )

    ax.set_ylabel(
        "Local empirical uncertainty [cm]"
    )

    ax.set_title(
        (
            "TreeCo aggregate DBH uncertainty "
            "vs predicted diameter"
        )
    )

    ax.legend(
        loc="best"
    )

    # Show local support on a secondary axis.
    ax2 = ax.twinx()

    ax2.plot(
        x,
        n_eff,
        linestyle=":",
        linewidth=1.4,
        color=TREECO[
            "ink"
        ],
        alpha=0.45,
        label="Effective local N",
    )

    ax2.set_ylabel(
        "Effective local number of trees"
    )

    save_figure(
        fig,
        save_path,
    )


def save_sigma_summary(
    tree_df: pd.DataFrame,
    lookup_df: pd.DataFrame,
    save_path: Path,
):
    """
    Save a compact text summary of the final all-tree empirical
    uncertainty relationship.
    """

    coverage = sigma_band_coverage(
        tree_df,
        lookup_df,
    )

    metrics = regression_metrics(
        tree_df[
            "y_true"
        ],
        tree_df[
            "y_pred"
        ],
    )

    sigma = lookup_df[
        "sigma_cm"
    ].to_numpy(
        dtype=float
    )

    finite_sigma = sigma[
        np.isfinite(
            sigma
        )
    ]

    lines = [
        "TreeCo aggregate DBH empirical uncertainty",
        "==========================================",
        "",
        (
            "This uncertainty curve is fitted to all "
            "aggregate validation trees."
        ),
        (
            "Coverage below is therefore descriptive "
            "same-sample coverage, not held-out coverage."
        ),
        "",
        (
            f"Aggregate trees: "
            f"{len(tree_df)}"
        ),
        (
            f"Aggregate MAE: "
            f"{metrics['mae']:.4f} cm"
        ),
        (
            f"Aggregate RMSE: "
            f"{metrics['rmse']:.4f} cm"
        ),
        (
            f"Aggregate R2: "
            f"{metrics['r2']:.4f}"
        ),
        "",
        (
            f"1-sigma descriptive coverage: "
            f"{coverage['coverage_1sigma'] * 100:.1f}%"
            if np.isfinite(
                coverage[
                    "coverage_1sigma"
                ]
            )
            else (
                "1-sigma descriptive coverage: "
                "n/a"
            )
        ),
        (
            f"2-sigma descriptive coverage: "
            f"{coverage['coverage_2sigma'] * 100:.1f}%"
            if np.isfinite(
                coverage[
                    "coverage_2sigma"
                ]
            )
            else (
                "2-sigma descriptive coverage: "
                "n/a"
            )
        ),
        (
            f"Median local sigma: "
            f"{np.median(finite_sigma):.2f} cm"
            if len(
                finite_sigma
            )
            else (
                "Median local sigma: n/a"
            )
        ),
    ]

    save_path.write_text(
        "\n".join(
            lines
        )
    )

    print(
        f"Saved: {save_path}"
    )


# =========================================================
# Main
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Plot diagnostics for a saved "
            "TreeCo training run."
        )
    )

    parser.add_argument(
        "--run",
        required=True,
        help=(
            "Path to the saved run folder "
            "containing history.json."
        ),
    )

    parser.add_argument(
        "--task_name",
        default=None,
        help=(
            "Optional title prefix."
        ),
    )

    parser.add_argument(
        "--plots_dirname",
        default="plots",
        help=(
            "Name of plots folder created "
            "inside the run directory."
        ),
    )

    parser.add_argument(
        "--predictions_csv",
        default=None,
        help=(
            "Optional validation prediction "
            "table containing one row per "
            "validation image."
        ),
    )

    parser.add_argument(
        "--tree_id_col",
        default=None,
        help=(
            "Optional tree/group ID column, "
            "e.g. ROOT_ID."
        ),
    )

    parser.add_argument(
        "--regression_agg",
        default="mean",
        choices=[
            "mean",
            "median",
        ],
        help=(
            "How image-level regression "
            "predictions are aggregated to "
            "one prediction per tree."
        ),
    )

    parser.add_argument(
        "--sigma_bandwidth_cm",
        type=float,
        default=12.0,
        help=(
            "Gaussian smoothing bandwidth in cm "
            "for the final tree-level uncertainty "
            "curve. Larger values are smoother."
        ),
    )

    parser.add_argument(
        "--sigma_min_effective_n",
        type=float,
        default=8.0,
        help=(
            "Minimum effective local number of "
            "trees required to display sigma bands."
        ),
    )

    parser.add_argument(
        "--no_tree_aggregate",
        action="store_true",
        help=(
            "Disable tree-level aggregation."
        ),
    )

    args = parser.parse_args()

    if args.sigma_bandwidth_cm <= 0:
        raise ValueError(
            "--sigma_bandwidth_cm must be > 0."
        )

    if args.sigma_min_effective_n <= 1:
        raise ValueError(
            "--sigma_min_effective_n must be > 1."
        )

    apply_treeco_style()

    run_path = Path(
        args.run
    )

    if not run_path.exists():
        raise FileNotFoundError(
            f"Run path not found: "
            f"{run_path}"
        )

    plots_dir = (
        run_path
        / args.plots_dirname
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics, history, config = (
        load_run(
            run_path
        )
    )

    run_name = (
        run_path.name
    )

    title_prefix = (
        args.task_name
        or run_name
    )

    print(
        f"Loaded run: "
        f"{run_name}"
    )

    task_type = infer_task_type(
        config
    )

    print(
        f"Initial detected task type: "
        f"{task_type}"
    )

    if metrics is not None:

        print(
            f"Best epoch: "
            f"{metrics.get('best_epoch')}"
        )

        if (
            "best_val_mae_cm"
            in metrics
        ):
            print(
                "Best validation MAE: "
                f"{metrics.get('best_val_mae_cm')} cm"
            )

            print(
                "Final validation MAE: "
                f"{metrics.get('final_val_mae_cm')} cm"
            )

            print(
                "Final validation RMSE: "
                f"{metrics.get('final_val_rmse_cm')} cm"
            )

            print(
                "Final validation R2: "
                f"{metrics.get('final_val_r2')}"
            )

            task_type = (
                "regression"
            )

        else:
            print(
                "Best validation macro F1: "
                f"{metrics.get('best_val_f1_macro')}"
            )

            print(
                "Final validation accuracy: "
                f"{metrics.get('final_val_accuracy')}"
            )

    if config is not None:
        print(
            f"Backbone: "
            f"{config.get('backbone')}"
        )

        print(
            f"Input mode: "
            f"{config.get('input_mode')}"
        )

        print(
            f"Image source: "
            f"{config.get('image_source')}"
        )

        print(
            f"Dataset: "
            f"{config.get('dataset_dir')}"
        )

    val_labels_path = (
        run_path
        / "val_labels.npy"
    )

    val_preds_path = (
        run_path
        / "val_preds.npy"
    )

    val_probs_path = (
        run_path
        / "val_probs.npy"
    )

    if (
        val_labels_path.exists()
        and val_preds_path.exists()
    ):
        y_true = np.load(
            val_labels_path,
            allow_pickle=True,
        )

        y_pred = np.load(
            val_preds_path,
            allow_pickle=True,
        )

        y_true = np.asarray(
            y_true
        ).reshape(-1)

        y_pred = np.asarray(
            y_pred
        ).reshape(-1)

        task_type = infer_task_type(
            config,
            y_true=y_true,
            y_pred=y_pred,
        )

        print(
            "Detected task type from predictions: "
            f"{task_type}"
        )

    else:
        y_true = None
        y_pred = None

        print(
            "No val_labels.npy / "
            "val_preds.npy found."
        )

    probabilities = None

    if val_probs_path.exists():
        probabilities = np.load(
            val_probs_path,
            allow_pickle=True,
        )

        print(
            "Loaded class probabilities from: "
            f"{val_probs_path}"
        )

    # -----------------------------------------------------
    # Training diagnostics
    # -----------------------------------------------------

    plot_training_diagnostics(
        history,
        title_prefix=title_prefix,
        task_type=task_type,
        save_path=(
            plots_dir
            / "training_diagnostics.png"
        ),
    )

    if (
        y_true is None
        or y_pred is None
    ):
        print(
            "Only training diagnostics were generated "
            "because validation labels/predictions "
            "were not found."
        )

        print(
            f"Plots saved to: "
            f"{plots_dir}"
        )

        return

    # -----------------------------------------------------
    # Image-level evaluation
    # -----------------------------------------------------

    if task_type == "regression":

        image_metrics = (
            plot_regression_evaluation_diagnostics(
                y_true,
                y_pred,
                title_prefix=title_prefix,
                level_name="Image level",
                save_path=(
                    plots_dir
                    / "regression_evaluation_image_level.png"
                ),
                report_path=(
                    plots_dir
                    / "regression_report_image_level.txt"
                ),
            )
        )

    else:

        class_names = (
            get_class_names(
                config
            )
        )

        image_metrics = (
            plot_classification_evaluation_diagnostics(
                y_true,
                y_pred,
                class_names=class_names,
                title_prefix=title_prefix,
                level_name="Image level",
                save_path=(
                    plots_dir
                    / "classification_evaluation_image_level.png"
                ),
                report_path=(
                    plots_dir
                    / "classification_report_image_level.txt"
                ),
            )
        )

    if args.no_tree_aggregate:
        print(
            "Tree-level aggregation disabled "
            "with --no_tree_aggregate."
        )

        print(
            f"Plots saved to: "
            f"{plots_dir}"
        )

        return

    # -----------------------------------------------------
    # Build evaluation dataframe with tree IDs
    # -----------------------------------------------------

    (
        df_eval,
        detected_tree_id_col,
        true_col,
        pred_col,
        prob_cols,
    ) = build_eval_dataframe(
        run_path=run_path,
        task_type=task_type,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        predictions_csv=(
            args.predictions_csv
        ),
        tree_id_col=(
            args.tree_id_col
        ),
    )

    if (
        df_eval is None
        or detected_tree_id_col
        is None
    ):
        print(
            "[WARNING] Could not generate "
            "tree-level aggregate plots "
            "because no tree ID was found."
        )

        return

    if (
        true_col is None
        or pred_col is None
    ):
        print(
            "[WARNING] Could not infer "
            "true/prediction columns."
        )

        return

    print(
        "Tree ID column for aggregation: "
        f"{detected_tree_id_col}"
    )

    print(
        "True column for aggregation:    "
        f"{true_col}"
    )

    print(
        "Pred column for aggregation:    "
        f"{pred_col}"
    )

    # =====================================================
    # REGRESSION
    # =====================================================

    if task_type == "regression":

        # -------------------------------------------------
        # Aggregate ALL validation images to ALL trees
        # -------------------------------------------------

        tree_df_all = (
            aggregate_regression_by_tree(
                df_eval,
                tree_id_col=(
                    detected_tree_id_col
                ),
                true_col=true_col,
                pred_col=pred_col,
                agg=(
                    args.regression_agg
                ),
            )
        )

        print()
        print(
            "All validation trees — aggregate regression"
        )
        print(
            "-------------------------------------------"
        )
        print(
            f"Aggregation: "
            f"{args.regression_agg}"
        )
        print(
            f"Trees: "
            f"{len(tree_df_all)}"
        )
        print(
            f"Images: "
            f"{int(tree_df_all['n_images'].sum())}"
        )

        tree_csv = (
            plots_dir
            / (
                "tree_level_aggregate_predictions_"
                "ALL.csv"
            )
        )

        tree_df_all.to_csv(
            tree_csv,
            index=False,
        )

        print(
            f"Saved: "
            f"{tree_csv}"
        )

        # -------------------------------------------------
        # Standard tree-level regression diagnostics
        # -------------------------------------------------

        tree_metrics = (
            plot_regression_evaluation_diagnostics(
                tree_df_all[
                    "y_true"
                ],
                tree_df_all[
                    "y_pred"
                ],
                title_prefix=(
                    title_prefix
                ),
                level_name=(
                    "Tree aggregate level "
                    f"(all validation trees; "
                    f"{args.regression_agg})"
                ),
                save_path=(
                    plots_dir
                    / (
                        "regression_evaluation_"
                        "tree_aggregate_ALL.png"
                    )
                ),
                report_path=(
                    plots_dir
                    / (
                        "regression_report_"
                        "tree_aggregate_ALL.txt"
                    )
                ),
            )
        )

        # -------------------------------------------------
        # Image vs aggregate metrics
        # -------------------------------------------------

        plot_regression_metric_comparison(
            image_metrics,
            tree_metrics,
            save_path=(
                plots_dir
                / (
                    "regression_image_vs_"
                    "tree_aggregate_ALL_metrics.png"
                )
            ),
        )

        # -------------------------------------------------
        # Images per tree
        # -------------------------------------------------

        plot_tree_group_size_distribution(
            tree_df_all,
            save_path=(
                plots_dir
                / "tree_aggregate_ALL_group_sizes.png"
            ),
            title_prefix=(
                title_prefix
            ),
        )

        # =================================================
        # FINAL ALL-TREE SMOOTH 1σ / 2σ UNCERTAINTY
        # =================================================

        tree_sigma_lookup = (
            build_smooth_tree_sigma_lookup(
                tree_df_all,
                bandwidth_cm=(
                    args.sigma_bandwidth_cm
                ),
                n_grid=300,
                min_effective_n=(
                    args.sigma_min_effective_n
                ),
            )
        )

        sigma_lookup_csv = (
            plots_dir
            / (
                "tree_aggregate_ALL_"
                "sigma_uncertainty_lookup.csv"
            )
        )

        tree_sigma_lookup.to_csv(
            sigma_lookup_csv,
            index=False,
        )

        print(
            f"Saved: "
            f"{sigma_lookup_csv}"
        )

        plot_tree_aggregate_sigma_uncertainty(
            tree_df_all,
            tree_sigma_lookup,
            save_path=(
                plots_dir
                / (
                    "tree_aggregate_ALL_"
                    "true_vs_predicted_sigma.png"
                )
            ),
            METRICS=tree_metrics
        )

        plot_tree_sigma_vs_prediction(
            tree_sigma_lookup,
            save_path=(
                plots_dir
                / (
                    "tree_aggregate_ALL_"
                    "sigma_vs_prediction.png"
                )
            ),
        )

        save_sigma_summary(
            tree_df_all,
            tree_sigma_lookup,
            save_path=(
                plots_dir
                / (
                    "tree_aggregate_ALL_"
                    "sigma_uncertainty_report.txt"
                )
            ),
        )

        print()
        print(
            "Final uncertainty model"
        )
        print(
            "-----------------------"
        )
        print(
            "Uses ALL aggregate validation trees."
        )
        print(
            "x-axis: aggregate predicted DBH."
        )
        print(
            "Local uncertainty: Gaussian-weighted "
            "residual standard deviation."
        )
        print(
            "Saved 1-sigma / 2-sigma lookup for "
            "later DBH inference."
        )

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    else:

        tree_df = (
            aggregate_classification_by_tree(
                df_eval,
                tree_id_col=(
                    detected_tree_id_col
                ),
                true_col=true_col,
                pred_col=pred_col,
                prob_cols=prob_cols,
            )
        )

        tree_csv = (
            plots_dir
            / "tree_level_aggregate_predictions_ALL.csv"
        )

        tree_df.to_csv(
            tree_csv,
            index=False,
        )

        print(
            f"Saved: "
            f"{tree_csv}"
        )

        class_names = (
            get_class_names(
                config
            )
        )

        plot_classification_evaluation_diagnostics(
            tree_df[
                "y_true"
            ],
            tree_df[
                "y_pred"
            ],
            class_names=class_names,
            title_prefix=title_prefix,
            level_name=(
                "Tree aggregate level "
                "(all validation trees)"
            ),
            save_path=(
                plots_dir
                / (
                    "classification_evaluation_"
                    "tree_aggregate_ALL.png"
                )
            ),
            report_path=(
                plots_dir
                / (
                    "classification_report_"
                    "tree_aggregate_ALL.txt"
                )
            ),
        )

    print()
    print(
        f"Plots saved to: "
        f"{plots_dir}"
    )


if __name__ == "__main__":
    main()
