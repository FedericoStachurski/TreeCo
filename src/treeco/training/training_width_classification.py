from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# ============================================================
# Input modes
# ============================================================

INPUT_CHANNELS = {
    # RGB / grayscale only
    "rgb": 3,
    "gray": 1,

    # Depth
    "rgb_depth": 4,
    "gray_depth": 2,

    # SAM1 logits
    "rgb_sam": 4,
    "gray_sam": 2,
    "rgb_sam_depth": 5,
    "gray_sam_depth": 3,

    # SAM3 masks
    "rgb_sam3": 4,
    "gray_sam3": 2,
    "rgb_sam3_depth": 5,
    "gray_sam3_depth": 3,

    # Compact pseudo-image mode:
    #   channel 0 = grayscale RGB
    #   channel 1 = SAM3 mask
    #   channel 2 = grayscale RGB * SAM3 mask
    "gray_sam3_overlay": 3,
}

RGB_MODES = {m for m in INPUT_CHANNELS if m.startswith("rgb")}
GRAY_MODES = {m for m in INPUT_CHANNELS if m.startswith("gray")}


# ============================================================
# Species helpers
# ============================================================

UNKNOWN_STRINGS = {
    "", "nan", "none", "null", "unknown", "unk", "not known",
    "not sure", "unsure", "n/a", "na"
}


def clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def is_unknownish(value) -> bool:
    return clean_text(value).lower() in UNKNOWN_STRINGS


def infer_species_label(row: pd.Series) -> str:
    direct_cols = [
        "TREE_SPECIES_LABEL", "SPECIES_LABEL", "SPECIES",
        "species", "tree_species",
    ]

    for col in direct_cols:
        if col in row.index:
            value = clean_text(row[col])
            if not is_unknownish(value):
                return value

    tree_type = clean_text(row["TREE_TYPE"]) if "TREE_TYPE" in row.index else ""
    other_tree = clean_text(row["OTHER_TREE"]) if "OTHER_TREE" in row.index else ""

    if tree_type.lower() in {"other", "others", "other tree", "other_tree"}:
        if not is_unknownish(other_tree):
            return other_tree
        return "Unknown"

    if not is_unknownish(tree_type):
        return tree_type

    if not is_unknownish(other_tree):
        return other_tree

    return "Unknown"


def add_species_input_labels(
    df: pd.DataFrame,
    min_species_count: int,
    include_unknown: bool = False,
) -> pd.DataFrame:
    df = df.copy()
    df["SPECIES_LABEL"] = df.apply(infer_species_label, axis=1).apply(clean_text)

    unknown_mask = (
        df["SPECIES_LABEL"].apply(is_unknownish)
        | (df["SPECIES_LABEL"].str.lower() == "unknown")
    )

    if include_unknown:
        df.loc[unknown_mask, "SPECIES_LABEL"] = "Unknown"
    else:
        df.loc[unknown_mask, "SPECIES_LABEL"] = "Rare/Other"

    species_tree_counts = (
        df.groupby("SPECIES_LABEL")["ID"]
        .nunique()
        .sort_values(ascending=False)
    )

    rare_species = species_tree_counts[
        species_tree_counts < min_species_count
    ].index

    df.loc[df["SPECIES_LABEL"].isin(rare_species), "SPECIES_LABEL"] = "Rare/Other"

    labels = sorted(df["SPECIES_LABEL"].unique(), key=lambda x: str(x).lower())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    df["SPECIES_INPUT_IDX"] = df["SPECIES_LABEL"].map(label_to_idx).astype(int)

    print("\nSpecies INPUT mapping:")
    for label, idx in label_to_idx.items():
        n_trees = df.loc[df["SPECIES_LABEL"] == label, "ID"].nunique()
        n_images = int((df["SPECIES_LABEL"] == label).sum())
        print(f"  {idx}: {label} | trees={n_trees}, images={n_images}")

    return df.reset_index(drop=True)


def make_species_input_mapping(df: pd.DataFrame) -> dict[str, str]:
    if "SPECIES_INPUT_IDX" not in df.columns:
        return {"0": "No species input"}

    mapping_df = (
        df[["SPECIES_INPUT_IDX", "SPECIES_LABEL"]]
        .drop_duplicates()
        .sort_values("SPECIES_INPUT_IDX")
    )

    return {
        str(int(row["SPECIES_INPUT_IDX"])): str(row["SPECIES_LABEL"])
        for _, row in mapping_df.iterrows()
    }


# ============================================================
# DBH label helpers
# ============================================================

DBH_FINE_BINS = [0, 15, 30, 45, 60, 80, 100, np.inf]
DBH_FINE_LABELS = [
    "0-15 cm",
    "15-30 cm",
    "30-45 cm",
    "45-60 cm",
    "60-80 cm",
    "80-100 cm",
    "100+ cm",
]

DBH_COARSE_BINS = [0, 30, 60, 100, np.inf]
DBH_COARSE_LABELS = ["0-30 cm", "30-60 cm", "60-100 cm", "100+ cm"]

COARSE_INTERVALS = {
    "0-30 cm": (0.0, 30.0),
    "30-60 cm": (30.0, 60.0),
    "60-100 cm": (60.0, 100.0),
    "100+ cm": (100.0, np.inf),
}

COARSE_TO_FINE = {
    "0-30 cm": ["0-15 cm", "15-30 cm"],
    "30-60 cm": ["30-45 cm", "45-60 cm"],
    "60-100 cm": ["60-80 cm", "80-100 cm"],
    "100+ cm": ["100+ cm"],
}


def get_dbh_scheme_bins(dbh_class_scheme: str) -> list[float]:
    if dbh_class_scheme == "fine":
        return DBH_FINE_BINS
    if dbh_class_scheme == "coarse":
        return DBH_COARSE_BINS
    raise ValueError(f"Unknown dbh_class_scheme: {dbh_class_scheme}")


def get_dbh_scheme_labels(dbh_class_scheme: str) -> list[str]:
    if dbh_class_scheme == "fine":
        return DBH_FINE_LABELS
    if dbh_class_scheme == "coarse":
        return DBH_COARSE_LABELS
    raise ValueError(f"Unknown dbh_class_scheme: {dbh_class_scheme}")


TAPE_METHODS = {"tape measure", "observation", "observed", "measurement", "measured"}
ESTIMATION_METHODS = {"estimation", "estimate", "estimated"}


def numeric_from_series(series: pd.Series) -> pd.Series:
    s = series.astype("string")
    s = (
        s.str.replace("cm", "", case=False, regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    numeric = pd.to_numeric(s, errors="coerce")

    if numeric.notna().sum() == 0:
        extracted = s.str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0]
        numeric = pd.to_numeric(extracted, errors="coerce")

    return numeric


def clean_trunk_size_label(series: pd.Series) -> pd.Series:
    s = (
        series.astype("string")
        .str.strip()
        .str.lower()
        .str.replace("–", "-", regex=False)
        .str.replace("—", "-", regex=False)
        .str.replace("−", "-", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"\s*-\s*", "-", regex=True)
        .str.replace(r"\s*\+\s*", "+", regex=True)
    )

    trunk_map = {
        "0-30 cm": "0-30 cm",
        "0-30cm": "0-30 cm",
        "30-60 cm": "30-60 cm",
        "30-60cm": "30-60 cm",
        "60-100 cm": "60-100 cm",
        "60-100cm": "60-100 cm",
        "100+ cm": "100+ cm",
        "100+cm": "100+ cm",
        "+100 cm": "100+ cm",
        "+100cm": "100+ cm",
        ">100 cm": "100+ cm",
        ">100cm": "100+ cm",
    }

    return s.map(trunk_map)


def sample_truncated_normal(
    mu: float,
    sigma: float,
    lo: float,
    hi: float,
    rng: np.random.Generator,
    n: int,
) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=float)

    sigma = max(float(sigma), 1e-6)
    samples: list[float] = []

    # Rejection sampling is fine here because the intervals are wide.
    while len(samples) < n:
        batch = rng.normal(mu, sigma, size=max(256, n * 8))

        if np.isfinite(hi):
            batch = batch[(batch >= lo) & (batch < hi)]
        else:
            batch = batch[batch >= lo]

        samples.extend(batch.tolist())

    return np.array(samples[:n], dtype=float)


def add_dbh_classification_labels(
    df: pd.DataFrame,
    tree_w: float,
    use_estimates: bool,
    estimate_strategy: str,
    stochastic_seed: int,
    estimate_min_obs: int,
    estimate_sigma_fallback: float,
    dbh_class_scheme: str = "fine",
) -> pd.DataFrame:
    """
    Adds DBH target labels for DBH classification.

    dbh_class_scheme="fine":
        Target classes are:
            0-15, 15-30, 30-45, 45-60, 60-80, 80-100, 100+ cm.
        Tape-measure rows are exact fine labels.
        Estimation rows are weak labels converted from TREE_TRUNK_SIZE using
        stochastic/uniform/midpoint imputation inside the coarse interval.

    dbh_class_scheme="coarse":
        Target classes are:
            0-30, 30-60, 60-100, 100+ cm.
        Tape-measure rows are binned into the same coarse classes.
        Estimation rows are used directly from TREE_TRUNK_SIZE, so no stochastic
        split is needed.

    Tape-measure sample weight = 1.0.
    Estimation sample weight = tree_w.
    """

    df = df.copy()

    required = ["TREE_CIRCUMFERENCE_METHOD", "CIRCUMFERENCE_IN_CM", "TREE_TRUNK_SIZE"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Manifest missing DBH source columns: {missing}. "
            "Expected TREE_CIRCUMFERENCE_METHOD, CIRCUMFERENCE_IN_CM, TREE_TRUNK_SIZE."
        )

    if not (0.0 <= tree_w <= 1.0):
        raise ValueError("--tree_w should usually be in [0, 1].")

    target_bins = get_dbh_scheme_bins(dbh_class_scheme)
    target_labels = get_dbh_scheme_labels(dbh_class_scheme)

    df["TREE_CIRCUMFERENCE_METHOD_CLEAN"] = (
        df["TREE_CIRCUMFERENCE_METHOD"]
        .astype("string")
        .str.strip()
    )

    method_lower = df["TREE_CIRCUMFERENCE_METHOD_CLEAN"].str.lower()
    df["METHOD_GROUP"] = pd.Series(np.nan, index=df.index, dtype="string")
    df.loc[method_lower.isin(TAPE_METHODS), "METHOD_GROUP"] = "Tape measure"
    df.loc[method_lower.isin(ESTIMATION_METHODS), "METHOD_GROUP"] = "Estimation"

    df["CIRCUMFERENCE_IN_CM_NUM"] = numeric_from_series(df["CIRCUMFERENCE_IN_CM"])
    df["DBH_CM_REAL"] = df["CIRCUMFERENCE_IN_CM_NUM"] / np.pi

    df["DBH_COARSE_FROM_ESTIMATE"] = clean_trunk_size_label(df["TREE_TRUNK_SIZE"])

    df["DBH_COARSE_FROM_NUMERIC"] = pd.cut(
        df["DBH_CM_REAL"],
        bins=DBH_COARSE_BINS,
        labels=DBH_COARSE_LABELS,
        include_lowest=True,
        right=False,
    )

    df["DBH_FINE_FROM_NUMERIC"] = pd.cut(
        df["DBH_CM_REAL"],
        bins=DBH_FINE_BINS,
        labels=DBH_FINE_LABELS,
        include_lowest=True,
        right=False,
    )

    # Start with real tape-measure DBH values where available.
    df["DBH_CM_TARGET"] = pd.Series(np.nan, index=df.index, dtype="float64")
    df["DBH_FINE_CLASS"] = pd.Series(np.nan, index=df.index, dtype="string")
    df["DBH_COARSE_CLASS"] = pd.Series(np.nan, index=df.index, dtype="string")
    df["DBH_LABEL_SOURCE"] = pd.Series(np.nan, index=df.index, dtype="string")
    df["DBH_SAMPLE_WEIGHT"] = pd.Series(np.nan, index=df.index, dtype="float64")

    if dbh_class_scheme == "fine":
        target_from_numeric = df["DBH_FINE_FROM_NUMERIC"]
    elif dbh_class_scheme == "coarse":
        target_from_numeric = df["DBH_COARSE_FROM_NUMERIC"]
    else:
        raise ValueError(f"Unknown dbh_class_scheme: {dbh_class_scheme}")

    tape_valid = (
        df["METHOD_GROUP"].eq("Tape measure")
        & df["DBH_CM_REAL"].notna()
        & np.isfinite(df["DBH_CM_REAL"])
        & (df["DBH_CM_REAL"] > 0)
        & target_from_numeric.notna()
    )

    df.loc[tape_valid, "DBH_CM_TARGET"] = df.loc[tape_valid, "DBH_CM_REAL"]
    df.loc[tape_valid, "DBH_FINE_CLASS"] = target_from_numeric.loc[tape_valid].astype(str)
    df.loc[tape_valid, "DBH_COARSE_CLASS"] = df.loc[tape_valid, "DBH_COARSE_FROM_NUMERIC"].astype(str)
    df.loc[tape_valid, "DBH_LABEL_SOURCE"] = "Tape measure"
    df.loc[tape_valid, "DBH_SAMPLE_WEIGHT"] = 1.0

    # Then optionally add weak estimated DBH rows.
    if use_estimates:
        est_valid = (
            df["METHOD_GROUP"].eq("Estimation")
            & df["DBH_COARSE_FROM_ESTIMATE"].notna()
        )

        # In coarse mode, TREE_TRUNK_SIZE already matches the target classes.
        # No stochastic splitting is needed.
        if dbh_class_scheme == "coarse":
            for coarse_lab, (lo, hi) in COARSE_INTERVALS.items():
                mask = est_valid & df["DBH_COARSE_FROM_ESTIMATE"].eq(coarse_lab)
                n_est = int(mask.sum())

                if n_est == 0:
                    continue

                if np.isfinite(hi):
                    representative_dbh = (lo + hi) / 2.0
                else:
                    representative_dbh = lo + estimate_sigma_fallback

                df.loc[mask, "DBH_CM_TARGET"] = representative_dbh
                df.loc[mask, "DBH_FINE_CLASS"] = coarse_lab
                df.loc[mask, "DBH_COARSE_CLASS"] = coarse_lab
                df.loc[mask, "DBH_LABEL_SOURCE"] = "Estimation"
                df.loc[mask, "DBH_SAMPLE_WEIGHT"] = float(tree_w)

        # In fine mode, estimates are coarse labels, so we impute a plausible
        # hidden DBH inside the coarse interval and then bin it into fine classes.
        else:
            rng = np.random.default_rng(stochastic_seed)

            # Observed tape-measure distribution used to guide stochastic imputation.
            observed_real = df.loc[tape_valid, ["DBH_CM_REAL", "DBH_COARSE_FROM_NUMERIC"]].copy()

            for coarse_lab, (lo, hi) in COARSE_INTERVALS.items():
                mask = est_valid & df["DBH_COARSE_FROM_ESTIMATE"].eq(coarse_lab)
                n_est = int(mask.sum())

                if n_est == 0:
                    continue

                if np.isfinite(hi):
                    obs_vals = observed_real.loc[
                        observed_real["DBH_CM_REAL"].between(lo, hi, inclusive="left"),
                        "DBH_CM_REAL",
                    ].dropna()
                else:
                    obs_vals = observed_real.loc[
                        observed_real["DBH_CM_REAL"] >= lo,
                        "DBH_CM_REAL",
                    ].dropna()

                if estimate_strategy == "midpoint":
                    if np.isfinite(hi):
                        sampled = np.full(n_est, (lo + hi) / 2.0)
                    else:
                        sampled = np.full(n_est, lo + estimate_sigma_fallback)

                elif estimate_strategy == "uniform":
                    if np.isfinite(hi):
                        sampled = rng.uniform(lo, hi, size=n_est)
                    else:
                        # Open-ended interval: use an exponential tail above 100 cm.
                        sampled = lo + rng.exponential(scale=estimate_sigma_fallback, size=n_est)

                elif estimate_strategy == "stochastic":
                    if len(obs_vals) >= estimate_min_obs:
                        mu = float(obs_vals.mean())
                        sigma = float(obs_vals.std())
                        if not np.isfinite(sigma) or sigma <= 0:
                            sigma = (hi - lo) / 4.0 if np.isfinite(hi) else estimate_sigma_fallback
                    else:
                        if np.isfinite(hi):
                            mu = (lo + hi) / 2.0
                            sigma = (hi - lo) / 4.0
                        else:
                            mu = lo + estimate_sigma_fallback
                            sigma = estimate_sigma_fallback

                    sampled = sample_truncated_normal(mu, sigma, lo, hi, rng, n_est)

                else:
                    raise ValueError(f"Unknown estimate_strategy: {estimate_strategy}")

                fine_class = pd.cut(
                    sampled,
                    bins=DBH_FINE_BINS,
                    labels=DBH_FINE_LABELS,
                    include_lowest=True,
                    right=False,
                ).astype(str)

                df.loc[mask, "DBH_CM_TARGET"] = sampled
                df.loc[mask, "DBH_FINE_CLASS"] = np.asarray(fine_class, dtype=object)
                df.loc[mask, "DBH_COARSE_CLASS"] = coarse_lab
                df.loc[mask, "DBH_LABEL_SOURCE"] = "Estimation"
                df.loc[mask, "DBH_SAMPLE_WEIGHT"] = float(tree_w)

    df["DBH_CLASS_SCHEME"] = dbh_class_scheme

    # Final categorical target.  The column is still named DBH_FINE_CLASS for
    # backward compatibility with the rest of the training script; in coarse
    # mode it contains the 4 coarse labels.
    df["DBH_FINE_CLASS"] = pd.Categorical(
        df["DBH_FINE_CLASS"],
        categories=target_labels,
        ordered=True,
    )

    df["DBH_FINE_CLASS_IDX"] = df["DBH_FINE_CLASS"].cat.codes
    df.loc[df["DBH_FINE_CLASS_IDX"] == -1, "DBH_FINE_CLASS_IDX"] = np.nan

    before = len(df)
    df = df.dropna(subset=["DBH_FINE_CLASS_IDX", "DBH_SAMPLE_WEIGHT"]).copy()
    after = len(df)

    df["DBH_FINE_CLASS_IDX"] = df["DBH_FINE_CLASS_IDX"].astype(int)
    df["DBH_SAMPLE_WEIGHT"] = df["DBH_SAMPLE_WEIGHT"].astype(float)

    print("\nDBH label construction:")
    print(f"  Rows with usable DBH target: {after} / {before}")
    print(f"  Use estimates: {use_estimates}")
    print(f"  Estimate strategy: {estimate_strategy if use_estimates else 'none'}")
    print(f"  Estimation sample weight tree_w: {tree_w}")

    print("\nDBH target counts by source:")
    print(pd.crosstab(df["DBH_FINE_CLASS"], df["DBH_LABEL_SOURCE"]).reindex(target_labels, fill_value=0))

    print("\nDBH weighted counts by class:")
    weighted_counts = df.groupby("DBH_FINE_CLASS", observed=False)["DBH_SAMPLE_WEIGHT"].sum().reindex(target_labels, fill_value=0)
    print(weighted_counts)

    if df.empty:
        raise RuntimeError("No rows left after DBH target filtering.")

    return df.reset_index(drop=True)


def make_dbh_mapping(dbh_class_scheme: str = "fine") -> dict[str, str]:
    labels = get_dbh_scheme_labels(dbh_class_scheme)
    return {str(i): lab for i, lab in enumerate(labels)}


def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray, distance: int = 1) -> float:
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)) <= distance))


def weighted_class_weights(
    y: np.ndarray,
    sample_weights: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    counts = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        counts[cls] = sample_weights[y == cls].sum()

    if np.any(counts <= 0):
        missing = np.where(counts <= 0)[0].tolist()
        raise RuntimeError(
            f"At least one DBH class has zero weighted training samples: {missing}. "
            "Use a larger training set, change bins, or adjust validation split."
        )

    total = counts.sum()
    weights = total / (num_classes * counts)
    return weights.astype(np.float32)


# ============================================================
# Losses with per-sample weighting
# ============================================================

class FocalLossPerSample(nn.Module):
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma) * ce


def weighted_mean_loss(loss_per_sample: torch.Tensor, sample_weights: torch.Tensor) -> torch.Tensor:
    sample_weights = sample_weights.to(dtype=loss_per_sample.dtype)
    return (loss_per_sample * sample_weights).sum() / sample_weights.sum().clamp_min(1e-8)


# ============================================================
# Dataset
# ============================================================

class TreeDBHClassificationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        input_mode: str = "rgb_sam3",
        image_source: str = "full",
        train: bool = True,
        use_species: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.input_mode = input_mode
        self.image_source = image_source
        self.train = train
        self.use_species = use_species

        if input_mode not in INPUT_CHANNELS:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        self.use_rgb = input_mode in RGB_MODES
        self.use_gray = input_mode in GRAY_MODES
        self.use_depth = "depth" in input_mode
        self.use_sam3 = "sam3" in input_mode
        self.use_sam = ("sam" in input_mode) and ("sam3" not in input_mode)
        self.use_overlay = input_mode == "gray_sam3_overlay"

        if image_source == "crop":
            self.rgb_col = "RGB_CROP_PATH"
        elif image_source == "full":
            self.rgb_col = "ORIGINAL_RGB_PATH"
        else:
            raise ValueError(f"Unknown image_source: {image_source}")

        self.color_jitter = transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        )

        self.rgb_erasing = transforms.RandomErasing(
            p=0.20,
            scale=(0.02, 0.10),
            ratio=(0.3, 3.3),
            value="random",
        )

    def __len__(self) -> int:
        return len(self.df)

    def _load_single_channel_image(self, path: str) -> Image.Image:
        try:
            arr = np.load(path).astype(np.float32)
            arr = np.squeeze(arr)

            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {arr.shape}")

            amin = np.nanmin(arr)
            amax = np.nanmax(arr)
            arr = (arr - amin) / (amax - amin + 1e-8)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")

        except Exception:
            img = Image.fromarray(
                np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            ).convert("L")

        return img

    def _apply_shared_geometric_transforms(
        self,
        rgb: Image.Image,
        single_channels: list[Image.Image],
    ):
        rgb = TF.resize(rgb, [self.image_size, self.image_size])
        single_channels = [
            TF.resize(ch, [self.image_size, self.image_size])
            for ch in single_channels
        ]

        if self.train:
            if random.random() < 0.5:
                rgb = TF.hflip(rgb)
                single_channels = [TF.hflip(ch) for ch in single_channels]

            angle = random.uniform(-8, 8)

            rgb = TF.rotate(
                rgb,
                angle,
                interpolation=TF.InterpolationMode.BILINEAR,
                fill=0,
            )

            single_channels = [
                TF.rotate(
                    ch,
                    angle,
                    interpolation=TF.InterpolationMode.BILINEAR,
                    fill=0,
                )
                for ch in single_channels
            ]

        return rgb, single_channels

    def _rgb_to_tensor(self, rgb: Image.Image) -> torch.Tensor:
        if self.train:
            rgb = self.color_jitter(rgb)

        rgb_t = TF.to_tensor(rgb)
        rgb_t = TF.normalize(
            rgb_t,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if self.train:
            rgb_t = self.rgb_erasing(rgb_t)

        return rgb_t

    def _gray_to_tensor(self, rgb: Image.Image) -> torch.Tensor:
        gray = TF.rgb_to_grayscale(rgb, num_output_channels=1)
        gray_t = TF.to_tensor(gray)
        gray_t = TF.normalize(gray_t, mean=[0.5], std=[0.25])
        return gray_t

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        rgb = Image.open(row[self.rgb_col]).convert("RGB")

        single_channels = []

        if self.use_sam:
            single_channels.append(self._load_single_channel_image(row["SAM_LOGITS_PATH"]))

        if self.use_sam3:
            single_channels.append(self._load_single_channel_image(row["SAM3_MASK_PATH"]))

        if self.use_depth:
            single_channels.append(self._load_single_channel_image(row["DEPTH_PATH"]))

        rgb, single_channels = self._apply_shared_geometric_transforms(rgb, single_channels)

        if self.use_overlay:
            if len(single_channels) < 1:
                raise RuntimeError("gray_sam3_overlay requires SAM3_MASK_PATH.")

            gray_img = TF.rgb_to_grayscale(rgb, num_output_channels=1)
            gray_t = TF.to_tensor(gray_img).to(dtype=torch.float32)
            sam3_t = TF.to_tensor(single_channels[0]).to(dtype=torch.float32)
            sam3_t = torch.clamp(sam3_t, 0.0, 1.0)
            overlay_t = gray_t * sam3_t

            x = torch.cat([gray_t, sam3_t, overlay_t], dim=0)
            x = TF.normalize(x, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])

        else:
            if self.use_rgb:
                base_t = self._rgb_to_tensor(rgb)
            else:
                base_t = self._gray_to_tensor(rgb)

            channels = [base_t]
            for ch in single_channels:
                channels.append(TF.to_tensor(ch).to(dtype=base_t.dtype))

            x = torch.cat(channels, dim=0)

        if self.use_species:
            species = torch.tensor(int(row["SPECIES_INPUT_IDX"]), dtype=torch.long)
        else:
            species = torch.tensor(0, dtype=torch.long)

        y = torch.tensor(int(row["DBH_FINE_CLASS_IDX"]), dtype=torch.long)
        sample_weight = torch.tensor(float(row["DBH_SAMPLE_WEIGHT"]), dtype=torch.float32)

        return x, species, y, sample_weight


# ============================================================
# Model
# ============================================================

class ResNetDBHWithSpecies(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_dbh_classes: int,
        in_channels: int,
        input_mode: str,
        use_species: bool,
        num_species: int,
        dropout_rate: float = 0.1,
        species_embedding_dim: int = 16,
    ):
        super().__init__()

        backbone = backbone.lower()

        weights_map = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
            "resnet101": models.ResNet101_Weights.DEFAULT,
        }

        if backbone not in weights_map:
            raise ValueError(f"Unsupported backbone: {backbone}")

        resnet = getattr(models, backbone)(weights=weights_map[backbone])

        if in_channels != 3:
            old_conv = resnet.conv1
            resnet.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

            with torch.no_grad():
                if input_mode in RGB_MODES and in_channels >= 3:
                    resnet.conv1.weight[:, :3] = old_conv.weight
                    for c in range(3, in_channels):
                        resnet.conv1.weight[:, c:c + 1] = old_conv.weight.mean(dim=1, keepdim=True)
                else:
                    mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    for c in range(in_channels):
                        resnet.conv1.weight[:, c:c + 1] = mean_weight

        image_feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()

        self.image_encoder = resnet
        self.use_species = use_species

        extra_dim = 0
        if use_species:
            self.species_embedding = nn.Embedding(
                num_embeddings=num_species,
                embedding_dim=species_embedding_dim,
            )
            extra_dim += species_embedding_dim
        else:
            self.species_embedding = None

        classifier_in = image_feature_dim + extra_dim
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(classifier_in, num_dbh_classes),
        )

    def forward(self, x: torch.Tensor, species: torch.Tensor | None = None) -> torch.Tensor:
        features = [self.image_encoder(x)]

        if self.use_species:
            if species is None:
                raise ValueError("use_species=True but species is None.")
            features.append(self.species_embedding(species))

        features = torch.cat(features, dim=1)
        return self.classifier(features)


def build_model(
    backbone: str,
    num_dbh_classes: int,
    in_channels: int,
    input_mode: str,
    device: torch.device,
    use_species: bool,
    num_species: int,
    dropout_rate: float,
    species_embedding_dim: int,
) -> nn.Module:
    model = ResNetDBHWithSpecies(
        backbone=backbone,
        num_dbh_classes=num_dbh_classes,
        in_channels=in_channels,
        input_mode=input_mode,
        use_species=use_species,
        num_species=num_species,
        dropout_rate=dropout_rate,
        species_embedding_dim=species_embedding_dim,
    )
    return model.to(device)


# ============================================================
# Training / validation
# ============================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    use_species: bool,
    optimizer=None,
    scheduler=None,
    scheduler_step_per_batch: bool = False,
):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss_num = 0.0
    total_loss_den = 0.0
    all_preds = []
    all_labels = []
    all_logits = []
    all_weights = []

    for x, species, y, sample_w in loader:
        x = x.to(device, non_blocking=True)
        species = species.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        sample_w = sample_w.to(device, non_blocking=True).float()

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            logits = model(x, species if use_species else None)
            loss_per_sample = criterion(logits, y)
            loss = weighted_mean_loss(loss_per_sample, sample_w)

            if train_mode:
                loss.backward()
                optimizer.step()

                if scheduler is not None and scheduler_step_per_batch:
                    scheduler.step()

        preds = logits.argmax(dim=1)

        total_loss_num += float((loss_per_sample.detach() * sample_w).sum().cpu())
        total_loss_den += float(sample_w.sum().detach().cpu())

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())
        all_weights.extend(sample_w.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())

    all_preds = np.asarray(all_preds)
    all_labels = np.asarray(all_labels)
    all_weights = np.asarray(all_weights, dtype=float)

    all_logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 0), dtype=np.float32)
    avg_loss = total_loss_num / max(total_loss_den, 1e-8)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) else 0.0
    adj_acc = adjacent_accuracy(all_labels, all_preds) if len(all_labels) else 0.0

    return avg_loss, acc, adj_acc, all_preds, all_labels, all_logits_np, all_weights


# ============================================================
# Manifest loading / filtering
# ============================================================

def find_dataset_dir(dataset_name: str | None, dataset_path: str | None) -> Path:
    if dataset_path is not None:
        p = Path(dataset_path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset path not found: {p}")
        return p

    if dataset_name is None:
        raise ValueError("Provide either --dataset_path or --dataset_name")

    matches = sorted([p for p in Path(".").glob(f"{dataset_name}_*") if p.is_dir()])
    if not matches:
        raise FileNotFoundError(f"No dataset folders found for pattern {dataset_name}_* under current directory")

    return matches[-1]


def load_manifest(dataset_dir: Path, manifest_path: str | None = None) -> pd.DataFrame:
    if manifest_path is not None:
        path = Path(manifest_path)
    else:
        path = dataset_dir / "manifests" / "tree_dataset_manifest.csv"

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    print("\nLoading manifest:")
    print(f"  {path}")

    df = pd.read_csv(path)

    required = ["ID", "RGB_CROP_PATH"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    optional_cols = [
        "ORIGINAL_RGB_PATH",
        "SAM_LOGITS_PATH",
        "SAM3_MASK_PATH",
        "DEPTH_PATH",
        "TREE_TYPE",
        "OTHER_TREE",
        "TREE_SPECIES_LABEL",
        "SPECIES_LABEL",
        "IMAGE_ID",
        "TREE_CIRCUMFERENCE_METHOD",
        "CIRCUMFERENCE_IN_CM",
        "TREE_TRUNK_SIZE",
    ]

    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan

    if "TRAINABLE" in df.columns:
        df = df[
            df["TRAINABLE"].astype(str).str.lower().isin(["true", "1", "yes"])
        ].copy()

    print("\nCircumference/DBH columns found:")
    width_like_cols = [
        c for c in df.columns
        if "DBH" in c.upper()
        or "WIDTH" in c.upper()
        or "CIRCUMFERENCE" in c.upper()
        or "TRUNK_SIZE" in c.upper()
    ]

    for c in width_like_cols:
        valid_n = pd.to_numeric(df[c], errors="coerce").notna().sum()
        non_na = df[c].notna().sum()
        print(f"  {c} | non-null={non_na}, numeric={valid_n}")

    return df.reset_index(drop=True)


def filter_manifest_for_inputs(df: pd.DataFrame, input_mode: str, image_source: str) -> pd.DataFrame:
    df = df.copy()

    if image_source == "crop":
        rgb_col = "RGB_CROP_PATH"
    elif image_source == "full":
        rgb_col = "ORIGINAL_RGB_PATH"
    else:
        raise ValueError(f"Unknown image_source: {image_source}")

    before = len(df)

    df = df[df[rgb_col].notna()].copy()
    df = df[df[rgb_col].apply(lambda p: Path(str(p)).exists())].copy()

    if "depth" in input_mode:
        df = df[df["DEPTH_PATH"].notna()].copy()
        df = df[df["DEPTH_PATH"].apply(lambda p: Path(str(p)).exists())].copy()

    if ("sam" in input_mode) and ("sam3" not in input_mode):
        df = df[df["SAM_LOGITS_PATH"].notna()].copy()
        df = df[df["SAM_LOGITS_PATH"].apply(lambda p: Path(str(p)).exists())].copy()

    if "sam3" in input_mode:
        df = df[df["SAM3_MASK_PATH"].notna()].copy()
        df = df[df["SAM3_MASK_PATH"].apply(lambda p: Path(str(p)).exists())].copy()

    df = df.drop_duplicates(subset=[rgb_col]).reset_index(drop=True)

    print("\nInput filtering:")
    print(f"  input_mode={input_mode}, image_source={image_source}")
    print(f"  rows kept: {len(df)} / {before}")

    return df


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_name", type=str, default=None)
    ap.add_argument("--dataset_path", type=str, default=None)
    ap.add_argument("--manifest_path", type=str, default=None)

    ap.add_argument("--out_dir", type=str, default="TreeCo/models")
    ap.add_argument("--run_name", type=str, default=None)

    ap.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
    )

    ap.add_argument(
        "--input_mode",
        type=str,
        default="rgb_sam3",
        choices=list(INPUT_CHANNELS.keys()),
    )

    ap.add_argument("--image_source", type=str, default="full", choices=["crop", "full"])
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dropout_rate", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument(
        "--criterion",
        type=str,
        default="weighted_ce",
        choices=["cross_entropy", "weighted_ce", "focal"],
    )
    ap.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "plateau", "cosine", "step", "onecycle"],
    )
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--focal_gamma", type=float, default=2.0)

    ap.add_argument(
        "--tree_w",
        type=float,
        default=0.4,
        help="Sample weight assigned to stochastic/estimated DBH rows. Tape measure rows always use weight 1.0.",
    )
    ap.add_argument(
        "--no_estimates",
        action="store_true",
        help="Drop TREE_TRUNK_SIZE estimation rows and train only on tape-measure DBH labels.",
    )
    ap.add_argument(
        "--estimate_strategy",
        type=str,
        default="stochastic",
        choices=["stochastic", "uniform", "midpoint"],
        help="How to convert coarse TREE_TRUNK_SIZE estimates into fine DBH classes.",
    )
    ap.add_argument(
        "--stochastic_seed",
        type=int,
        default=None,
        help="Seed for stochastic DBH imputation. Defaults to --random_state.",
    )
    ap.add_argument("--estimate_min_obs", type=int, default=5)
    ap.add_argument("--estimate_sigma_fallback", type=float, default=15.0)

    ap.add_argument(
        "--dbh_class_scheme",
        type=str,
        default="fine",
        choices=["fine", "coarse"],
        help=(
            "DBH target class scheme. "
            "fine = 7 classes: 0-15, 15-30, 30-45, 45-60, 60-80, 80-100, 100+. "
            "coarse = 4 classes: 0-30, 30-60, 60-100, 100+."
        ),
    )

    ap.add_argument("--include_unknown", action="store_true")
    ap.add_argument("--min_species_count", type=int, default=5)
    ap.add_argument("--no_species", action="store_true")
    ap.add_argument("--species_embedding_dim", type=int, default=16)

    args = ap.parse_args()

    seed_everything(args.random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_species = not args.no_species
    use_estimates = not args.no_estimates
    stochastic_seed = args.random_state if args.stochastic_seed is None else args.stochastic_seed

    dataset_dir = find_dataset_dir(args.dataset_name, args.dataset_path)
    out_root = Path(args.out_dir)

    df = load_manifest(dataset_dir=dataset_dir, manifest_path=args.manifest_path)
    df = filter_manifest_for_inputs(df, input_mode=args.input_mode, image_source=args.image_source)

    df = add_dbh_classification_labels(
        df,
        tree_w=args.tree_w,
        use_estimates=use_estimates,
        estimate_strategy=args.estimate_strategy,
        stochastic_seed=stochastic_seed,
        estimate_min_obs=args.estimate_min_obs,
        estimate_sigma_fallback=args.estimate_sigma_fallback,
        dbh_class_scheme=args.dbh_class_scheme,
    )

    # Species input is added after image/DBH filtering so the mapping matches the actual rows.
    df = add_species_input_labels(
        df,
        min_species_count=args.min_species_count,
        include_unknown=args.include_unknown,
    )

    dbh_labels = get_dbh_scheme_labels(args.dbh_class_scheme)
    num_dbh_classes = len(dbh_labels)
    num_species = int(df["SPECIES_INPUT_IDX"].nunique()) if use_species else 1

    print("\nDataset summary:")
    print(f"  dataset_dir: {dataset_dir}")
    print(f"  device: {device}")
    print(f"  input_mode: {args.input_mode}")
    print(f"  image_source: {args.image_source}")
    print(f"  use_species: {use_species}")
    print(f"  use_estimates: {use_estimates}")
    print(f"  tree_w: {args.tree_w}")
    print(f"  dbh_class_scheme: {args.dbh_class_scheme}")
    print(f"  total images with DBH target: {len(df)}")
    print(f"  unique trees with DBH target: {df['ID'].nunique()}")

    print("\nFull DBH target class counts:")
    print(df["DBH_FINE_CLASS"].value_counts().sort_index())

    # --------------------------------------------------------
    # Tree-level split based only on real tape-measure labels.
    # Validation is tape-measure only.
    # --------------------------------------------------------
    tape_df = df[df["DBH_LABEL_SOURCE"].eq("Tape measure")].copy()

    if tape_df.empty:
        raise RuntimeError("No tape-measure DBH rows available. Cannot build reliable validation set.")

    tree_df = (
        tape_df.groupby("ID")["DBH_FINE_CLASS_IDX"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
    )

    tree_counts = tree_df["DBH_FINE_CLASS_IDX"].value_counts().sort_index()
    print("\nTree-level real/tape DBH class counts for stratified split:")
    print(tree_counts)

    if tree_counts.min() < 2:
        raise RuntimeError(
            "At least one real/tape DBH class has fewer than 2 trees. "
            "Cannot do a stratified tree-level split. Consider merging bins or removing rare classes."
        )

    n_val_requested = int(round(len(tree_df) * args.val_size))
    if n_val_requested < num_dbh_classes:
        raise RuntimeError(
            f"Validation split too small for stratification: {n_val_requested} val trees "
            f"for {num_dbh_classes} DBH classes. Increase --val_size."
        )

    train_tree_ids, val_tree_ids = train_test_split(
        tree_df["ID"],
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=tree_df["DBH_FINE_CLASS_IDX"],
    )

    train_tree_ids = set(train_tree_ids)
    val_tree_ids = set(val_tree_ids)

    # Training uses all usable rows except validation trees.
    # This includes estimate-only trees, but never rows from validation tree IDs.
    train_df = df[~df["ID"].isin(val_tree_ids)].copy().reset_index(drop=True)

    # Validation uses only real tape-measure labels from held-out trees.
    val_df = df[
        df["ID"].isin(val_tree_ids)
        & df["DBH_LABEL_SOURCE"].eq("Tape measure")
    ].copy().reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise RuntimeError("Empty train or validation split after filtering.")

    print("\nSplit summary:")
    print(f"  Train trees: {train_df['ID'].nunique()}")
    print(f"  Val trees:   {val_df['ID'].nunique()}")
    print(f"  Train images: {len(train_df)}")
    print(f"  Val images:   {len(val_df)}")

    print("\nTrain DBH counts by source:")
    print(pd.crosstab(train_df["DBH_FINE_CLASS"], train_df["DBH_LABEL_SOURCE"]).reindex(dbh_labels, fill_value=0))

    print("\nValidation DBH counts, tape-measure only:")
    print(val_df["DBH_FINE_CLASS"].value_counts().sort_index())

    train_ds = TreeDBHClassificationDataset(
        train_df,
        image_size=args.image_size,
        input_mode=args.input_mode,
        image_source=args.image_source,
        train=True,
        use_species=use_species,
    )

    val_ds = TreeDBHClassificationDataset(
        val_df,
        image_size=args.image_size,
        input_mode=args.input_mode,
        image_source=args.image_source,
        train=False,
        use_species=use_species,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    in_channels = INPUT_CHANNELS[args.input_mode]

    model = build_model(
        backbone=args.backbone,
        num_dbh_classes=num_dbh_classes,
        in_channels=in_channels,
        input_mode=args.input_mode,
        device=device,
        use_species=use_species,
        num_species=num_species,
        dropout_rate=args.dropout_rate,
        species_embedding_dim=args.species_embedding_dim,
    )

    if args.criterion in {"weighted_ce", "focal"}:
        class_weights_np = weighted_class_weights(
            y=train_df["DBH_FINE_CLASS_IDX"].to_numpy(),
            sample_weights=train_df["DBH_SAMPLE_WEIGHT"].to_numpy(),
            num_classes=num_dbh_classes,
        )
    else:
        class_weights_np = np.ones(num_dbh_classes, dtype=np.float32)

    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    print("\nDBH class weights:")
    for i, w in enumerate(class_weights_np):
        print(f"  class {i} ({dbh_labels[i]}): {w:.4f}")

    if args.criterion == "cross_entropy":
        criterion = nn.CrossEntropyLoss(
            reduction="none",
            label_smoothing=args.label_smoothing,
        )
    elif args.criterion == "weighted_ce":
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="none",
            label_smoothing=args.label_smoothing,
        )
    elif args.criterion == "focal":
        criterion = FocalLossPerSample(
            alpha=class_weights,
            gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing,
        )
    else:
        raise ValueError(f"Unknown criterion: {args.criterion}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=4,
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=args.epochs * len(train_loader),
        )
    else:
        scheduler = None

    models_root = out_root / "tree_dbh_class_models"
    models_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_name is not None:
        run_name = f"{args.run_name}_{timestamp}"
    else:
        species_tag = "species" if use_species else "noSpecies"
        est_tag = f"estW{args.tree_w:g}" if use_estimates else "tapeOnly"
        run_name = f"dbhclass_{args.dbh_class_scheme}_{args.backbone}_{args.input_mode}_{args.image_source}_{species_tag}_{est_tag}_{timestamp}"

    run_dir = models_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_dir / "best_model.pth"
    last_model_path = run_dir / "last_model.pth"
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.json"
    metrics_path = run_dir / "metrics.json"
    val_predictions_path = run_dir / "val_predictions.csv"
    train_manifest_path = run_dir / "train_manifest.csv"
    val_manifest_path = run_dir / "val_manifest.csv"

    dbh_mapping = make_dbh_mapping(args.dbh_class_scheme)
    species_mapping = make_species_input_mapping(df)

    config = {
        "task": "dbh_classification_with_weak_estimates",
        "dataset_dir": str(dataset_dir),
        "manifest_path": args.manifest_path,
        "backbone": args.backbone,
        "num_dbh_classes": num_dbh_classes,
        "dbh_class_scheme": args.dbh_class_scheme,
        "dbh_mapping": dbh_mapping,
        "in_channels": in_channels,
        "input_mode": args.input_mode,
        "image_source": args.image_source,
        "use_depth": "depth" in args.input_mode,
        "use_sam": ("sam" in args.input_mode) and ("sam3" not in args.input_mode),
        "use_sam3": "sam3" in args.input_mode,
        "use_species": use_species,
        "num_species_inputs": num_species,
        "species_mapping": species_mapping,
        "species_embedding_dim": args.species_embedding_dim,
        "tree_w": args.tree_w,
        "use_estimates": use_estimates,
        "estimate_strategy": args.estimate_strategy if use_estimates else None,
        "stochastic_seed": stochastic_seed if use_estimates else None,
        "estimate_min_obs": args.estimate_min_obs,
        "estimate_sigma_fallback": args.estimate_sigma_fallback,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "dropout_rate": args.dropout_rate,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "val_size": args.val_size,
        "random_state": args.random_state,
        "num_workers": args.num_workers,
        "criterion": args.criterion,
        "label_smoothing": args.label_smoothing,
        "focal_gamma": args.focal_gamma,
        "scheduler": args.scheduler,
        "class_weights": class_weights_np.tolist(),
        "device": str(device),
        "validation_policy": "tree-level split; validation uses tape-measure rows only",
        "min_species_count_for_rare_group": args.min_species_count,
        "include_unknown_species_group": args.include_unknown,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    train_df.to_csv(train_manifest_path, index=False)
    val_df.to_csv(val_manifest_path, index=False)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_adjacent_acc": [],
        "val_adjacent_acc": [],
        "val_f1_macro": [],
        "val_recall_macro": [],
        "lr": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1
    best_val_preds = None
    best_val_labels = None
    best_val_logits = None

    print(f"\nSaving DBH classification run to: {run_dir}")

    for epoch in range(args.epochs):
        train_loss, train_acc, train_adj, _, _, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            use_species=use_species,
            optimizer=optimizer,
            scheduler=scheduler if args.scheduler == "onecycle" else None,
            scheduler_step_per_batch=args.scheduler == "onecycle",
        )

        val_loss, val_acc, val_adj, val_preds, val_labels, val_logits, _ = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_species=use_species,
            optimizer=None,
        )

        val_f1_macro = f1_score(
            val_labels,
            val_preds,
            average="macro",
            zero_division=0,
            labels=np.arange(num_dbh_classes),
        )

        val_recall_macro = recall_score(
            val_labels,
            val_preds,
            average="macro",
            zero_division=0,
            labels=np.arange(num_dbh_classes),
        )

        if scheduler is not None and args.scheduler != "onecycle":
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_f1_macro)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["train_adjacent_acc"].append(float(train_adj))
        history["val_adjacent_acc"].append(float(val_adj))
        history["val_f1_macro"].append(float(val_f1_macro))
        history["val_recall_macro"].append(float(val_recall_macro))
        history["lr"].append(float(current_lr))

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_adj={train_adj:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_adj={val_adj:.4f} | "
            f"val_f1={val_f1_macro:.4f} val_recall={val_recall_macro:.4f} | "
            f"lr={current_lr:.4e}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "task": "dbh_classification_with_weak_estimates",
            "backbone": args.backbone,
            "num_dbh_classes": num_dbh_classes,
            "dbh_class_scheme": args.dbh_class_scheme,
            "dbh_mapping": dbh_mapping,
            "in_channels": in_channels,
            "input_mode": args.input_mode,
            "image_source": args.image_source,
            "image_size": args.image_size,
            "use_species": use_species,
            "num_species_inputs": num_species,
            "species_mapping": species_mapping,
            "species_embedding_dim": args.species_embedding_dim,
            "tree_w": args.tree_w,
            "use_estimates": use_estimates,
            "estimate_strategy": args.estimate_strategy if use_estimates else None,
            "stochastic_seed": stochastic_seed if use_estimates else None,
            "val_accuracy": float(val_acc),
            "val_adjacent_accuracy": float(val_adj),
            "val_f1_macro": float(val_f1_macro),
            "val_recall_macro": float(val_recall_macro),
            "scheduler": args.scheduler,
        }

        torch.save(checkpoint, last_model_path)

        if val_f1_macro > best_val_f1:
            best_val_f1 = float(val_f1_macro)
            best_epoch = epoch + 1
            best_val_preds = val_preds.copy()
            best_val_labels = val_labels.copy()
            best_val_logits = val_logits.copy()
            torch.save(checkpoint, best_model_path)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    if best_val_preds is not None:
        val_preds = best_val_preds
        val_labels = best_val_labels
        val_logits = best_val_logits

    idx_to_dbh = {i: lab for i, lab in enumerate(dbh_labels)}

    val_pred_df = val_df.copy()
    val_pred_df["TRUE_DBH_CLASS_IDX"] = val_labels
    val_pred_df["PRED_DBH_CLASS_IDX"] = val_preds
    val_pred_df["TRUE_DBH_CLASS_STR"] = [idx_to_dbh[int(i)] for i in val_labels]
    val_pred_df["PRED_DBH_CLASS_STR"] = [idx_to_dbh[int(i)] for i in val_preds]
    val_pred_df["ABS_CLASS_ERROR"] = np.abs(val_pred_df["TRUE_DBH_CLASS_IDX"] - val_pred_df["PRED_DBH_CLASS_IDX"])
    val_pred_df["ADJACENT_CORRECT"] = val_pred_df["ABS_CLASS_ERROR"] <= 1

    if val_logits is not None and len(val_logits) == len(val_pred_df):
        val_probs = torch.softmax(torch.tensor(val_logits, dtype=torch.float32), dim=1).numpy()

        for k in range(num_dbh_classes):
            val_pred_df[f"RESNET_LOGIT_{k}"] = val_logits[:, k]
            val_pred_df[f"RESNET_PROB_{k}"] = val_probs[:, k]

        np.save(run_dir / "val_logits.npy", val_logits)
        np.save(run_dir / "val_probs.npy", val_probs)

    val_pred_df.to_csv(val_predictions_path, index=False)

    best_idx = max(best_epoch - 1, 0)
    cm = confusion_matrix(
        val_labels,
        val_preds,
        labels=np.arange(num_dbh_classes),
    )

    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(history["val_loss"][best_idx]),
        "best_val_accuracy": float(history["val_acc"][best_idx]),
        "best_val_adjacent_accuracy": float(history["val_adjacent_acc"][best_idx]),
        "best_val_f1_macro": float(best_val_f1),
        "best_val_recall_macro": float(history["val_recall_macro"][best_idx]),
        "final_val_accuracy": float(history["val_acc"][-1]),
        "final_val_adjacent_accuracy": float(history["val_adjacent_acc"][-1]),
        "final_val_f1_macro": float(history["val_f1_macro"][-1]),
        "final_val_recall_macro": float(history["val_recall_macro"][-1]),
        "n_train_images": int(len(train_df)),
        "n_val_images": int(len(val_df)),
        "n_train_trees": int(train_df["ID"].nunique()),
        "n_val_trees": int(val_df["ID"].nunique()),
        "num_dbh_classes": int(num_dbh_classes),
        "use_species": bool(use_species),
        "num_species_inputs": int(num_species),
        "tree_w": float(args.tree_w),
        "use_estimates": bool(use_estimates),
        "estimate_strategy": args.estimate_strategy if use_estimates else None,
        "validation_policy": "tape-measure-only validation",
        "confusion_matrix": cm.tolist(),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    np.save(run_dir / "val_labels.npy", val_labels)
    np.save(run_dir / "val_preds.npy", val_preds)
    np.save(run_dir / "confusion_matrix.npy", cm)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro F1: {best_val_f1:.4f}")
    print(f"Best val adjacent accuracy: {history['val_adjacent_acc'][best_idx]:.4f}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved last model to: {last_model_path}")
    print(f"Saved validation predictions to: {val_predictions_path}")


if __name__ == "__main__":
    main()
