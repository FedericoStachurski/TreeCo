from __future__ import annotations

import numpy as np
import pandas as pd


# =========================================================
# Shared
# =========================================================

def conformal_quantile(scores, alpha):
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]

    if len(scores) == 0:
        return np.nan

    scores = np.sort(scores)
    n = len(scores)

    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    k = min(k, n)

    return float(scores[k - 1])


def build_quantile_table(scores, coverages=(0.80, 0.90, 0.95)):
    rows = []

    for coverage in coverages:
        alpha = 1.0 - coverage
        q = conformal_quantile(scores, alpha)

        rows.append({
            "coverage": coverage,
            "alpha": alpha,
            "q": q,
            "n_calibration": len(scores),
        })

    return pd.DataFrame(rows)


# =========================================================
# Regression
# =========================================================

def interp_finite(x_query, x_grid, values):
    x_query = np.asarray(x_query, dtype=float)
    x_grid = np.asarray(x_grid, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = np.isfinite(x_grid) & np.isfinite(values)

    if valid.sum() < 2:
        return np.full(len(x_query), np.nan)

    return np.interp(
        x_query,
        x_grid[valid],
        values[valid],
        left=np.nan,
        right=np.nan,
    )


def build_regression_conformal_scores(calibration_df, sigma_lookup):
    y_true = calibration_df["y_true"].to_numpy(dtype=float)
    y_pred = calibration_df["y_pred"].to_numpy(dtype=float)

    x = sigma_lookup["predicted_dbh_cm"].to_numpy(dtype=float)

    bias = interp_finite(
        y_pred,
        x,
        sigma_lookup["local_bias_cm"].to_numpy(dtype=float),
    )

    sigma = interp_finite(
        y_pred,
        x,
        sigma_lookup["sigma_cm"].to_numpy(dtype=float),
    )

    centre = y_pred + bias
    abs_error = np.abs(y_true - centre)

    score = np.divide(
        abs_error,
        sigma,
        out=np.full_like(abs_error, np.nan),
        where=np.isfinite(sigma) & (sigma > 0),
    )

    out = calibration_df.copy()
    out["local_bias_cm"] = bias
    out["local_sigma_cm"] = sigma
    out["conformal_centre_cm"] = centre
    out["abs_error_cm"] = abs_error
    out["conformal_score"] = score

    return out[np.isfinite(out["conformal_score"])].copy()


def add_regression_conformal_intervals(df, sigma_lookup, quantiles_df):
    out = df.copy()

    pred = out["y_pred"].to_numpy(dtype=float)
    x = sigma_lookup["predicted_dbh_cm"].to_numpy(dtype=float)

    bias = interp_finite(pred, x, sigma_lookup["local_bias_cm"])
    sigma = interp_finite(pred, x, sigma_lookup["sigma_cm"])

    centre = pred + bias

    out["local_bias_cm"] = bias
    out["local_sigma_cm"] = sigma
    out["conformal_centre_cm"] = centre

    for _, row in quantiles_df.iterrows():
        label = int(round(float(row["coverage"]) * 100))
        width = float(row["q"]) * sigma

        out[f"lower_{label}_cm"] = np.maximum(0.0, centre - width)
        out[f"upper_{label}_cm"] = centre + width

    return out


# =========================================================
# Classification
# =========================================================

def build_classification_conformal_scores(
    calibration_df,
    true_col="y_true",
    prob_prefix="mean_prob_",
):
    prob_cols = sorted(
        [c for c in calibration_df.columns if c.startswith(prob_prefix)],
        key=lambda c: int(c.split("_")[-1]),
    )

    if not prob_cols:
        raise ValueError("No tree-level probability columns found.")

    probs = calibration_df[prob_cols].to_numpy(dtype=float)
    y_true = calibration_df[true_col].to_numpy(dtype=int)

    if probs.shape[0] != len(y_true):
        raise ValueError("Probability and target lengths do not match.")

    true_prob = probs[np.arange(len(y_true)), y_true]

    # Simple split-conformal classification score
    scores = 1.0 - true_prob

    out = calibration_df.copy()
    out["true_class_probability"] = true_prob
    out["conformal_score"] = scores

    return out


def add_classification_prediction_sets(
    df,
    quantiles_df,
    prob_prefix="mean_prob_",
):
    out = df.copy()

    prob_cols = sorted(
        [c for c in out.columns if c.startswith(prob_prefix)],
        key=lambda c: int(c.split("_")[-1]),
    )

    probs = out[prob_cols].to_numpy(dtype=float)

    for _, row in quantiles_df.iterrows():
        coverage = float(row["coverage"])
        q = float(row["q"])

        label = int(round(coverage * 100))
        threshold = 1.0 - q

        prediction_sets = []

        for p in probs:
            classes = np.where(p >= threshold)[0].tolist()
            prediction_sets.append(classes)

        out[f"prediction_set_{label}"] = [
            ",".join(map(str, classes))
            for classes in prediction_sets
        ]

        out[f"prediction_set_size_{label}"] = [
            len(classes)
            for classes in prediction_sets
        ]

    return out