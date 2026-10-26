#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
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

import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    recall_score,
)

from treeco.image_models.encoder import TreeCoEncoder


# =========================================================
# Reproducibility
# =========================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# Input modes
# =========================================================


INPUT_CHANNELS = {
    "rgb": 3,
    "rgb_depth": 4,
    "rgb_sam": 4,
    "rgb_sam_depth": 5,
    "rgb_sam3": 4,
    "rgb_sam3_depth": 5,
}

SPECIES_UNKNOWN = "Unknown"
SPECIES_RARE = "Rare/Other"


# =========================================================
# Generic helpers
# =========================================================


def _is_missing_text(value) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {
        "nan", "none", "null", "na", "n/a", "unknown", "unk",
        "not known", "not sure", "unsure",
    }


def find_column_case_insensitive(df: pd.DataFrame, requested: str) -> str | None:
    requested_lower = requested.lower()
    for col in df.columns:
        if col.lower() == requested_lower:
            return col
    return None


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        found = find_column_case_insensitive(df, candidate)
        if found is not None:
            return found
    return None


def numeric_from_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        extracted = series.astype(str).str.extract(
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        )[0]
        numeric = pd.to_numeric(extracted, errors="coerce")
    return numeric


# =========================================================
# Height labels
# =========================================================


def add_height_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Height class is the target.

    Expected classes are usually:
        0-5
        5-10
        10-15
        15+

    Existing class indices are remapped to contiguous 0..K-1 while
    preserving their original ordering.
    """
    df = df.copy()
    required = ["HEIGHT_CLASS_STR", "HEIGHT_CLASS_IDX"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing height columns: {missing}")

    df = df[
        df["HEIGHT_CLASS_STR"].notna()
        & df["HEIGHT_CLASS_IDX"].notna()
    ].copy()

    df["HEIGHT_CLASS_IDX_ORIGINAL"] = pd.to_numeric(
        df["HEIGHT_CLASS_IDX"], errors="raise"
    ).astype(int)

    class_order = (
        df[["HEIGHT_CLASS_IDX_ORIGINAL", "HEIGHT_CLASS_STR"]]
        .drop_duplicates()
        .sort_values("HEIGHT_CLASS_IDX_ORIGINAL")
    )

    # Make sure one original index maps to one label.
    duplicate_original = class_order["HEIGHT_CLASS_IDX_ORIGINAL"].duplicated(keep=False)
    if duplicate_original.any():
        raise ValueError(
            "A HEIGHT_CLASS_IDX maps to more than one HEIGHT_CLASS_STR. "
            "Please clean the manifest first."
        )

    remap = {
        int(old_idx): int(new_idx)
        for new_idx, old_idx in enumerate(class_order["HEIGHT_CLASS_IDX_ORIGINAL"])
    }

    df["HEIGHT_CLASS_IDX"] = (
        df["HEIGHT_CLASS_IDX_ORIGINAL"].map(remap).astype(int)
    )

    print("\nHeight class remapping:")
    for _, row in class_order.iterrows():
        old_idx = int(row["HEIGHT_CLASS_IDX_ORIGINAL"])
        new_idx = remap[old_idx]
        print(f"  original {old_idx} -> train {new_idx}: {row['HEIGHT_CLASS_STR']}")

    return df.reset_index(drop=True)


def make_height_mapping(df: pd.DataFrame) -> dict[str, str]:
    mapping_df = (
        df[["HEIGHT_CLASS_IDX", "HEIGHT_CLASS_STR"]]
        .drop_duplicates()
        .sort_values("HEIGHT_CLASS_IDX")
    )
    return {
        str(int(row["HEIGHT_CLASS_IDX"])): str(row["HEIGHT_CLASS_STR"])
        for _, row in mapping_df.iterrows()
    }


# =========================================================
# Species helpers - vocabulary is fitted on TRAINING only
# =========================================================


def clean_species_label(row: pd.Series) -> str:
    direct_cols = [
        "TREE_SPECIES_RAW",
        "TREE_SPECIES_LABEL",
        "SPECIES_LABEL",
        "SPECIES",
        "species",
        "tree_species",
    ]

    for col in direct_cols:
        if col in row.index and not _is_missing_text(row[col]):
            raw = row[col]
            break
    else:
        tree_type = row.get("TREE_TYPE", np.nan)
        other_tree = row.get("OTHER_TREE", np.nan)

        if _is_missing_text(tree_type):
            raw = other_tree
        elif str(tree_type).strip().lower() in {"other", "others", "other tree", "other_tree"}:
            raw = other_tree
        else:
            raw = tree_type

    if _is_missing_text(raw):
        return SPECIES_UNKNOWN

    label = str(raw).strip()
    label = re.sub(r"\s*\([^)]*\)", "", label)
    label = label.replace(",", " ")
    label = re.sub(r"\s+", " ", label).strip()
    label = label.strip(" .;:-_/")

    if len(label) < 3:
        return SPECIES_UNKNOWN

    label = label.title().replace(" X ", " x ")
    return label if not _is_missing_text(label) else SPECIES_UNKNOWN


def prepare_species_encoding(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    min_tree_count: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """
    Fit species vocabulary on TRAINING TREES only.

    Rare training species and validation-only species become Rare/Other.
    Unknown remains its own category so species dropout can map to it.
    """
    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df["TREE_SPECIES_RAW"] = train_df.apply(clean_species_label, axis=1)
    val_df["TREE_SPECIES_RAW"] = val_df.apply(clean_species_label, axis=1)

    # Count independent trees, not image rows.
    train_tree_species = (
        train_df.groupby("ID")["TREE_SPECIES_RAW"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else SPECIES_UNKNOWN)
    )
    counts = train_tree_species.value_counts(dropna=False)

    keep_species = {
        species
        for species, count in counts.items()
        if species != SPECIES_UNKNOWN and int(count) >= int(min_tree_count)
    }

    def map_label(raw) -> str:
        if _is_missing_text(raw) or str(raw).strip() == SPECIES_UNKNOWN:
            return SPECIES_UNKNOWN
        raw = str(raw).strip()
        return raw if raw in keep_species else SPECIES_RARE

    train_df["TREE_SPECIES_LABEL"] = train_df["TREE_SPECIES_RAW"].apply(map_label)
    val_df["TREE_SPECIES_LABEL"] = val_df["TREE_SPECIES_RAW"].apply(map_label)

    classes = [SPECIES_UNKNOWN, SPECIES_RARE]
    classes += sorted(
        c for c in train_df["TREE_SPECIES_LABEL"].unique()
        if c not in {SPECIES_UNKNOWN, SPECIES_RARE}
    )
    species_to_idx = {species: idx for idx, species in enumerate(classes)}

    fallback = species_to_idx[SPECIES_RARE]
    train_df["TREE_SPECIES_IDX"] = (
        train_df["TREE_SPECIES_LABEL"].map(species_to_idx).fillna(fallback).astype(int)
    )
    val_df["TREE_SPECIES_IDX"] = (
        val_df["TREE_SPECIES_LABEL"].map(species_to_idx).fillna(fallback).astype(int)
    )

    return train_df, val_df, species_to_idx


# =========================================================
# DBH feature helpers
# =========================================================


PREDICTED_DBH_CANDIDATES = [
    "DBH_CM_TREE_PRED",
    "TREE_AGG_PRED_DBH_CM",
    "AGG_PRED_DBH_CM",
    "PRED_DBH_CM",
    "PREDICTED_DBH_CM",
    "INFERRED_DBH_CM",
    "DBH_PRED_CM",
    "MEAN_PRED_DBH_CM",
    "TREE_PRED_DBH_CM",
]

FINAL_MIXED_DBH_CANDIDATES = [
    "DBH_CM_FINAL",
    "FINAL_DBH_CM",
    "DBH_FINAL_CM",
    "DBH_MEASURED_OR_INFERRED_CM",
    "MEASURED_OR_INFERRED_DBH_CM",
    "DBH_INFERRED_OR_MEASURED_CM",
]

OBSERVED_DBH_CANDIDATES = [
    "DBH_CM_OBSERVED",
    "DBH_CM",
]

OBSERVED_CIRCUMFERENCE_CANDIDATES = [
    "CIRCUMFERENCE_CM_OBSERVED",
    "CIRCUMFERENCE_CM_CLEAN",
    "CIRCUMFERENCE_IN_CM",
    "TREE_CIRCUMFERENCE_CM",
    "CIRCUMFERENCE_CM",
    "TREE_CIRCUMFERENCE_IN_CM",
]


def add_dbh_feature(
    df: pd.DataFrame,
    dbh_source: str,
    dbh_column: str = "auto",
    dbh_column_is_circumference: bool = False,
    tree_agg: str = "median",
) -> tuple[pd.DataFrame, str, bool]:
    """
    Add DBH_FEATURE_CM and make it one consistent scalar per tree.

    dbh_source:
        none      -> caller should not invoke this function
        predicted -> inferred/predicted DBH columns only
        observed  -> observed DBH, or observed circumference / pi
        auto      -> final/mixed, then predicted, then observed

    An explicit --dbh_column overrides automatic column selection.
    """
    if dbh_source not in {"predicted", "observed", "auto"}:
        raise ValueError(f"Invalid dbh_source for add_dbh_feature: {dbh_source}")
    if tree_agg not in {"mean", "median"}:
        raise ValueError("dbh_tree_agg must be mean or median")

    out = df.copy()
    source_col = None
    source_is_circumference = False

    if dbh_column != "auto":
        source_col = find_column_case_insensitive(out, dbh_column)
        if source_col is None:
            raise ValueError(
                f"--dbh_column '{dbh_column}' not found. Available columns:\n{list(out.columns)}"
            )
        source_is_circumference = bool(dbh_column_is_circumference)
    else:
        if dbh_source == "predicted":
            source_col = first_existing_column(out, PREDICTED_DBH_CANDIDATES)
            if source_col is None:
                raise ValueError(
                    "dbh_source='predicted' but no predicted DBH column was found. "
                    "Use --use_inferred_widths / --final_width_manifest_path, or specify --dbh_column."
                )

        elif dbh_source == "observed":
            source_col = first_existing_column(out, OBSERVED_DBH_CANDIDATES)
            if source_col is None:
                source_col = first_existing_column(out, OBSERVED_CIRCUMFERENCE_CANDIDATES)
                source_is_circumference = source_col is not None
            if source_col is None:
                raise ValueError(
                    "dbh_source='observed' but no observed DBH/circumference column was found."
                )

        else:  # auto
            source_col = first_existing_column(out, FINAL_MIXED_DBH_CANDIDATES)
            if source_col is None:
                source_col = first_existing_column(out, PREDICTED_DBH_CANDIDATES)
            if source_col is None:
                source_col = first_existing_column(out, OBSERVED_DBH_CANDIDATES)
            if source_col is None:
                source_col = first_existing_column(out, OBSERVED_CIRCUMFERENCE_CANDIDATES)
                source_is_circumference = source_col is not None
            if source_col is None:
                raise ValueError("Could not find any DBH/circumference column for dbh_source='auto'.")

    values = numeric_from_series(out[source_col])
    if source_is_circumference:
        values = values / np.pi

    out["DBH_FEATURE_CM_RAW"] = values
    valid = (
        out["DBH_FEATURE_CM_RAW"].notna()
        & np.isfinite(out["DBH_FEATURE_CM_RAW"])
        & (out["DBH_FEATURE_CM_RAW"] > 0)
    )
    before = len(out)
    out = out[valid].copy()

    agg_func = "mean" if tree_agg == "mean" else "median"
    tree_dbh = (
        out.groupby("ID", as_index=False)["DBH_FEATURE_CM_RAW"]
        .agg(agg_func)
        .rename(columns={"DBH_FEATURE_CM_RAW": "DBH_FEATURE_CM"})
    )
    out = out.drop(columns=["DBH_FEATURE_CM_RAW"]).merge(
        tree_dbh, on="ID", how="inner", validate="many_to_one"
    )

    print(
        f"\nUsing DBH input: source={dbh_source}, column={source_col}, "
        f"tree_agg={tree_agg}"
        + (" (circumference / pi)" if source_is_circumference else "")
    )
    print(f"Rows with valid DBH feature: {len(out)} / {before}")

    return out.reset_index(drop=True), source_col, source_is_circumference


def filter_common_dbh_ablation_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Restrict to trees with BOTH a valid predicted DBH and a valid observed DBH.

    This is useful when dbh_source is swept over none/predicted/observed: every
    trial then evaluates the same underlying tree cohort rather than allowing
    missing DBH values to change the split.
    """
    out = df.copy()

    pred_col = first_existing_column(out, PREDICTED_DBH_CANDIDATES)
    obs_col = first_existing_column(out, OBSERVED_DBH_CANDIDATES)
    obs_is_circ = False

    if obs_col is None:
        obs_col = first_existing_column(out, OBSERVED_CIRCUMFERENCE_CANDIDATES)
        obs_is_circ = obs_col is not None

    if pred_col is None or obs_col is None:
        raise ValueError(
            "--ablation_common_dbh_cohort requires both a predicted DBH column "
            "and an observed DBH/circumference column in the loaded manifest. "
            f"Found predicted={pred_col!r}, observed={obs_col!r}."
        )

    pred = numeric_from_series(out[pred_col])
    obs = numeric_from_series(out[obs_col])
    if obs_is_circ:
        obs = obs / np.pi

    valid_row = (
        pred.notna() & np.isfinite(pred) & (pred > 0)
        & obs.notna() & np.isfinite(obs) & (obs > 0)
    )

    valid_ids = out.loc[valid_row, "ID"].dropna().unique()
    before_trees = out["ID"].nunique()
    out = out[out["ID"].isin(valid_ids)].copy().reset_index(drop=True)

    print(
        f"\nCommon DBH ablation cohort: {out['ID'].nunique()} / {before_trees} trees "
        f"with both predicted ({pred_col}) and observed ({obs_col}) DBH available."
    )
    return out


def fit_dbh_normalisation(train_df: pd.DataFrame, transform: str) -> tuple[float, float]:
    # Fit on one value per training tree so trees with many images do not dominate.
    values = (
        train_df.groupby("ID")["DBH_FEATURE_CM"].median().astype(float).to_numpy()
    )
    if transform == "log1p_zscore":
        values = np.log1p(values)
    elif transform != "zscore":
        raise ValueError(f"Unknown DBH transform: {transform}")

    mean = float(np.mean(values))
    std = float(np.std(values))
    if std < 1e-8:
        std = 1.0
    return mean, std


def apply_dbh_normalisation(
    df: pd.DataFrame,
    mean: float,
    std: float,
    transform: str,
) -> pd.DataFrame:
    out = df.copy()
    values = out["DBH_FEATURE_CM"].astype(float).to_numpy()
    if transform == "log1p_zscore":
        values = np.log1p(values)
    elif transform != "zscore":
        raise ValueError(f"Unknown DBH transform: {transform}")
    out["DBH_NORM"] = ((values - mean) / std).astype(np.float32)
    return out


# =========================================================
# Training weights: height frequency + N images per tree
# =========================================================


def add_height_training_weights(
    df: pd.DataFrame,
    power: float = 0.5,
    max_weight: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Height-class frequency weighting based on UNIQUE TRAINING TREES.

    For class c:
        raw_weight_c = (max_class_count / class_count_c) ** power

    Weight is clipped to max_weight and broadcast to all images from a tree.
    """
    if power < 0:
        raise ValueError("height_weight_power must be >= 0")
    if max_weight <= 0:
        raise ValueError("height_weight_max must be > 0")

    out = df.copy()
    tree_info = (
        out.groupby("ID", as_index=False)
        .agg(
            HEIGHT_CLASS_IDX=("HEIGHT_CLASS_IDX", lambda s: int(s.mode().iloc[0])),
            HEIGHT_CLASS_STR=("HEIGHT_CLASS_STR", lambda s: str(s.mode().iloc[0])),
            N_IMAGES_PER_TREE=("ID", "size"),
        )
    )

    counts = (
        tree_info.groupby(["HEIGHT_CLASS_IDX", "HEIGHT_CLASS_STR"], as_index=False)
        .size()
        .rename(columns={"size": "N_TREES"})
    )
    max_count = float(counts["N_TREES"].max())
    counts["HEIGHT_TRAIN_WEIGHT"] = (
        (max_count / counts["N_TREES"].astype(float)) ** float(power)
    ).clip(upper=float(max_weight))

    tree_info = tree_info.merge(
        counts[["HEIGHT_CLASS_IDX", "HEIGHT_TRAIN_WEIGHT"]],
        on="HEIGHT_CLASS_IDX",
        how="left",
        validate="many_to_one",
    )

    out = out.merge(
        tree_info[["ID", "N_IMAGES_PER_TREE", "HEIGHT_TRAIN_WEIGHT"]],
        on="ID",
        how="left",
        validate="many_to_one",
    )
    out["HEIGHT_TRAIN_WEIGHT"] = out["HEIGHT_TRAIN_WEIGHT"].astype(np.float32)
    return out, counts


def add_nimages_training_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Give every image weight 1/N_images_for_tree."""
    out = df.copy()
    if "N_IMAGES_PER_TREE" not in out.columns:
        out["N_IMAGES_PER_TREE"] = out.groupby("ID")["ID"].transform("size").astype(int)
    out["NIMAGES_TRAIN_WEIGHT"] = (
        1.0 / out["N_IMAGES_PER_TREE"].astype(float)
    ).astype(np.float32)
    return out


def combine_training_weights(
    df: pd.DataFrame,
    use_height_weights: bool,
    use_nimages_weights: bool,
    final_weight_max: float | None,
) -> pd.DataFrame:
    """
    Combine enabled components multiplicatively, then normalise image weights
    to mean 1. Validation is always unweighted.
    """
    out = df.copy()
    final_weight = np.ones(len(out), dtype=np.float64)

    if use_height_weights:
        final_weight *= out["HEIGHT_TRAIN_WEIGHT"].to_numpy(dtype=np.float64)
    if use_nimages_weights:
        final_weight *= out["NIMAGES_TRAIN_WEIGHT"].to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(final_weight)) or np.any(final_weight <= 0):
        raise ValueError("Final training weights must be finite and > 0")

    final_weight /= final_weight.mean()

    if final_weight_max is not None:
        if float(final_weight_max) <= 0:
            raise ValueError("final_weight_max must be > 0")
        final_weight = np.minimum(final_weight, float(final_weight_max))

    out["TRAIN_WEIGHT"] = final_weight.astype(np.float32)
    return out


def print_training_weight_diagnostics(
    df: pd.DataFrame,
    use_height_weights: bool,
    use_nimages_weights: bool,
) -> None:
    print("\n" + "=" * 64)
    print("TRAINING WEIGHT DIAGNOSTICS")
    print("=" * 64)
    print(f"Height weighting:  {use_height_weights}")
    print(f"N-images weighting:{use_nimages_weights}")

    for col in ["HEIGHT_TRAIN_WEIGHT", "NIMAGES_TRAIN_WEIGHT", "TRAIN_WEIGHT"]:
        if col in df.columns:
            print(f"\n{col}")
            print(df[col].describe())

    if "TRAIN_WEIGHT" in df.columns:
        tree_totals = (
            df.groupby("ID", as_index=False)
            .agg(
                HEIGHT_CLASS_IDX=("HEIGHT_CLASS_IDX", "first"),
                N_IMAGES=("ID", "size"),
                TOTAL_TRAIN_WEIGHT=("TRAIN_WEIGHT", "sum"),
                MEAN_TRAIN_WEIGHT=("TRAIN_WEIGHT", "mean"),
            )
        )
        print("\nPer-tree total training-weight summary")
        print(tree_totals["TOTAL_TRAIN_WEIGHT"].describe())
        print("\nHighest-weight training trees")
        print(
            tree_totals.sort_values("TOTAL_TRAIN_WEIGHT", ascending=False)
            .head(10)
            .to_string(index=False)
        )


# =========================================================
# Losses
# =========================================================


class FocalLoss(nn.Module):
    """Multiclass focal loss returning one loss per sample."""

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        true_prob = torch.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)
        focal = (1.0 - true_prob).pow(self.gamma) * ce
        if self.alpha is not None:
            focal = focal * self.alpha[targets]
        return focal


# =========================================================
# Dataset
# =========================================================


class TreeHeightDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int,
        input_mode: str,
        image_source: str,
        train: bool,
        use_dbh: bool,
        use_species: bool,
        species_dropout: float = 0.0,
        species_unknown_idx: int = 0,
        sample_weight_col: str | None = None,
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)
        self.input_mode = input_mode
        self.image_source = image_source
        self.train = bool(train)
        self.use_dbh = bool(use_dbh)
        self.use_species = bool(use_species)
        self.species_dropout = float(species_dropout)
        self.species_unknown_idx = int(species_unknown_idx)
        self.sample_weight_col = sample_weight_col

        if input_mode not in INPUT_CHANNELS:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        self.use_depth = input_mode in {"rgb_depth", "rgb_sam_depth", "rgb_sam3_depth"}
        self.use_sam = input_mode in {"rgb_sam", "rgb_sam_depth"}
        self.use_sam3 = input_mode in {"rgb_sam3", "rgb_sam3_depth"}

        if image_source == "crop":
            self.rgb_col = "RGB_CROP_PATH"
        elif image_source == "full":
            self.rgb_col = "ORIGINAL_RGB_PATH"
        else:
            raise ValueError(f"Unknown image_source: {image_source}")

        if self.use_dbh and "DBH_NORM" not in self.df.columns:
            raise ValueError("use_dbh=True but DBH_NORM is missing")
        if self.use_species and "TREE_SPECIES_IDX" not in self.df.columns:
            raise ValueError("use_species=True but TREE_SPECIES_IDX is missing")
        if self.sample_weight_col is not None and self.sample_weight_col not in self.df.columns:
            raise ValueError(f"Sample weight column {self.sample_weight_col!r} is missing")

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
            if arr.ndim == 3:
                arr = np.squeeze(arr)
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got {arr.shape}")
            amin, amax = np.nanmin(arr), np.nanmax(arr)
            arr = (arr - amin) / (amax - amin + 1e-8)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            return Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
        except Exception:
            return Image.fromarray(
                np.zeros((self.image_size, self.image_size), dtype=np.uint8)
            ).convert("L")

    def _apply_shared_geometric_transforms(
        self,
        rgb: Image.Image,
        single_channels: list[Image.Image],
    ) -> tuple[Image.Image, list[Image.Image]]:
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
        x = TF.to_tensor(rgb)
        x = TF.normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.train:
            x = self.rgb_erasing(x)
        return x

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rgb = Image.open(row[self.rgb_col]).convert("RGB")
        single_channels: list[Image.Image] = []

        if self.use_sam:
            single_channels.append(self._load_single_channel_image(row["SAM_LOGITS_PATH"]))
        if self.use_sam3:
            single_channels.append(self._load_single_channel_image(row["SAM3_MASK_PATH"]))
        if self.use_depth:
            single_channels.append(self._load_single_channel_image(row["DEPTH_PATH"]))

        rgb, single_channels = self._apply_shared_geometric_transforms(rgb, single_channels)
        rgb_tensor = self._rgb_to_tensor(rgb)
        channels = [rgb_tensor]
        channels.extend(TF.to_tensor(ch).to(dtype=rgb_tensor.dtype) for ch in single_channels)
        x = torch.cat(channels, dim=0)

        dbh = torch.tensor(
            [float(row["DBH_NORM"])] if self.use_dbh else [0.0],
            dtype=torch.float32,
        )

        species_idx = 0
        if self.use_species:
            species_idx = int(row["TREE_SPECIES_IDX"])
            if self.train and self.species_dropout > 0 and random.random() < self.species_dropout:
                species_idx = self.species_unknown_idx
        species = torch.tensor(species_idx, dtype=torch.long)

        y = torch.tensor(int(row["HEIGHT_CLASS_IDX"]), dtype=torch.long)
        weight = torch.tensor(
            float(row[self.sample_weight_col]) if self.sample_weight_col is not None else 1.0,
            dtype=torch.float32,
        )
        return x, dbh, species, y, weight


# =========================================================
# Image encoders
# =========================================================


class TreeCoMultimodalResNetEncoder(nn.Module):
    """
    Optional TreeCo modality encoder stem copied conceptually from the DBH model:

        RGB   -> TreeCoEncoder (32 channels)
        mask  -> TreeCoEncoder (8 channels, if present)
        depth -> TreeCoEncoder (16 channels, if present)
        concat -> fusion -> pretrained ResNet layer1..4 -> GAP
    """

    def __init__(self, backbone: str, input_mode: str):
        super().__init__()
        self.input_mode = input_mode
        self.use_mask = input_mode in {"rgb_sam", "rgb_sam_depth", "rgb_sam3", "rgb_sam3_depth"}
        self.use_depth = input_mode in {"rgb_depth", "rgb_sam_depth", "rgb_sam3_depth"}

        self.rgb_encoder = TreeCoEncoder(
            in_channels=3,
            hidden_channels=32,
            out_channels=32,
            first_kernel_size=7,
        )
        fusion_channels = 32

        if self.use_mask:
            self.mask_encoder = TreeCoEncoder(
                in_channels=1,
                hidden_channels=8,
                out_channels=8,
                first_kernel_size=5,
            )
            fusion_channels += 8
        else:
            self.mask_encoder = None

        if self.use_depth:
            self.depth_encoder = TreeCoEncoder(
                in_channels=1,
                hidden_channels=16,
                out_channels=16,
                first_kernel_size=5,
            )
            fusion_channels += 16
        else:
            self.depth_encoder = None

        self.fusion_encoder = nn.Sequential(
            nn.Conv2d(fusion_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        weights_map = {
            "resnet18": models.ResNet18_Weights.DEFAULT,
            "resnet34": models.ResNet34_Weights.DEFAULT,
            "resnet50": models.ResNet50_Weights.DEFAULT,
            "resnet101": models.ResNet101_Weights.DEFAULT,
        }
        backbone = backbone.lower()
        if backbone not in weights_map:
            raise ValueError(f"Unsupported backbone: {backbone}")

        resnet = getattr(models, backbone)(weights=weights_map[backbone])
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_feat_dim = resnet.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_channels = 3 + int(self.use_mask) + int(self.use_depth)
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"{self.input_mode} expects {expected_channels} channels, got {x.shape[1]}"
            )

        encoded = [self.rgb_encoder(x[:, 0:3])]
        channel_idx = 3

        if self.use_mask:
            encoded.append(self.mask_encoder(x[:, channel_idx:channel_idx + 1]))
            channel_idx += 1
        if self.use_depth:
            encoded.append(self.depth_encoder(x[:, channel_idx:channel_idx + 1]))

        z = self.fusion_encoder(torch.cat(encoded, dim=1))
        z = self.layer1(z)
        z = self.layer2(z)
        z = self.layer3(z)
        z = self.layer4(z)
        z = self.pool(z)
        return torch.flatten(z, 1)


def build_standard_resnet_encoder(
    backbone: str,
    in_channels: int,
) -> tuple[nn.Module, int]:
    weights_map = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "resnet101": models.ResNet101_Weights.DEFAULT,
    }
    backbone = backbone.lower()
    if backbone not in weights_map:
        raise ValueError(f"Unsupported backbone: {backbone}")

    model = getattr(models, backbone)(weights=weights_map[backbone])

    if in_channels != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            model.conv1.weight[:, :3] = old_conv.weight
            for c in range(3, in_channels):
                model.conv1.weight[:, c:c + 1] = old_conv.weight.mean(dim=1, keepdim=True)

    feat_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feat_dim


# =========================================================
# Height classifier: image + optional DBH + optional species
# =========================================================


class TreeHeightClassifier(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        image_feat_dim: int,
        num_height_classes: int,
        use_dbh: bool,
        dbh_hidden_dim: int,
        use_species: bool,
        num_species: int,
        species_embedding_dim: int,
        dropout_rate: float,
        fusion_hidden_dim: int,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.use_dbh = bool(use_dbh)
        self.use_species = bool(use_species)

        extra_dim = 0

        if self.use_dbh:
            self.dbh_encoder = nn.Sequential(
                nn.Linear(1, dbh_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(dbh_hidden_dim, dbh_hidden_dim),
                nn.ReLU(inplace=True),
            )
            extra_dim += dbh_hidden_dim
        else:
            self.dbh_encoder = None

        if self.use_species:
            self.species_embedding = nn.Embedding(
                num_embeddings=num_species,
                embedding_dim=species_embedding_dim,
            )
            extra_dim += species_embedding_dim
        else:
            self.species_embedding = None

        classifier_in = image_feat_dim + extra_dim
        if fusion_hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(classifier_in, fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(fusion_hidden_dim, num_height_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(classifier_in, num_height_classes),
            )

    def forward(
        self,
        x: torch.Tensor,
        dbh: torch.Tensor | None = None,
        species: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = [self.image_encoder(x)]

        if self.use_dbh:
            if dbh is None:
                raise ValueError("use_dbh=True but dbh is None")
            features.append(self.dbh_encoder(dbh))

        if self.use_species:
            if species is None:
                raise ValueError("use_species=True but species is None")
            features.append(self.species_embedding(species))

        return self.classifier(torch.cat(features, dim=1))


def build_model(
    backbone: str,
    input_mode: str,
    use_encoder: bool,
    num_height_classes: int,
    use_dbh: bool,
    dbh_hidden_dim: int,
    use_species: bool,
    num_species: int,
    species_embedding_dim: int,
    dropout_rate: float,
    fusion_hidden_dim: int,
    device: torch.device,
) -> nn.Module:
    if use_encoder:
        image_encoder = TreeCoMultimodalResNetEncoder(
            backbone=backbone,
            input_mode=input_mode,
        )
        image_feat_dim = image_encoder.image_feat_dim
    else:
        image_encoder, image_feat_dim = build_standard_resnet_encoder(
            backbone=backbone,
            in_channels=INPUT_CHANNELS[input_mode],
        )

    model = TreeHeightClassifier(
        image_encoder=image_encoder,
        image_feat_dim=image_feat_dim,
        num_height_classes=num_height_classes,
        use_dbh=use_dbh,
        dbh_hidden_dim=dbh_hidden_dim,
        use_species=use_species,
        num_species=num_species,
        species_embedding_dim=species_embedding_dim,
        dropout_rate=dropout_rate,
        fusion_hidden_dim=fusion_hidden_dim,
    )
    return model.to(device)


# =========================================================
# Metrics and aggregation
# =========================================================


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = np.arange(num_classes)

    if len(y_true) == 0:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "f1_macro": np.nan,
            "recall_macro": np.nan,
            "qwk": np.nan,
            "mae_class": np.nan,
            "within_one_class": np.nan,
        }

    try:
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except Exception:
        qwk = np.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "recall_macro": float(recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "qwk": qwk,
        "mae_class": float(np.mean(np.abs(y_pred - y_true))),
        "within_one_class": float(np.mean(np.abs(y_pred - y_true) <= 1)),
    }


def aggregate_tree_probabilities(
    df: pd.DataFrame,
    logits: np.ndarray,
    num_classes: int,
) -> pd.DataFrame:
    if len(df) != len(logits):
        raise ValueError(
            f"Cannot aggregate tree probabilities: len(df)={len(df)}, len(logits)={len(logits)}"
        )

    probs = torch.softmax(torch.as_tensor(logits, dtype=torch.float32), dim=1).numpy()
    tmp = df[["ID", "HEIGHT_CLASS_IDX", "HEIGHT_CLASS_STR"]].copy().reset_index(drop=True)
    for k in range(num_classes):
        tmp[f"PROB_{k}"] = probs[:, k]

    rows = []
    for tree_id, group in tmp.groupby("ID", sort=False):
        true_class = int(group["HEIGHT_CLASS_IDX"].mode().iloc[0])
        mean_probs = group[[f"PROB_{k}" for k in range(num_classes)]].mean().to_numpy(dtype=float)
        pred_class = int(np.argmax(mean_probs))

        row = {
            "ID": tree_id,
            "y_true": true_class,
            "y_pred": pred_class,
            "HEIGHT_CLASS_STR": str(group["HEIGHT_CLASS_STR"].mode().iloc[0]),
            "n_images": int(len(group)),
        }
        for k in range(num_classes):
            row[f"mean_prob_{k}"] = float(mean_probs[k])
        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# Epoch runner
# =========================================================


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_dbh: bool,
    use_species: bool,
    num_classes: int,
    optimizer=None,
    scheduler=None,
    scheduler_step_per_batch: bool = False,
    use_sample_weights: bool = False,
):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss_num = 0.0
    total_loss_den = 0.0
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []

    for x, dbh, species, y, sample_weight in loader:
        x = x.to(device, non_blocking=True)
        dbh = dbh.to(device, non_blocking=True)
        species = species.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        sample_weight = sample_weight.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            logits = model(
                x,
                dbh if use_dbh else None,
                species if use_species else None,
            )
            per_sample_loss = criterion(logits, y)
            if per_sample_loss.ndim != 1:
                per_sample_loss = per_sample_loss.view(per_sample_loss.size(0), -1).mean(dim=1)

            if use_sample_weights and train_mode:
                w = sample_weight.to(dtype=per_sample_loss.dtype)
                weighted_sum = (per_sample_loss * w).sum()
                weight_sum = w.sum().clamp_min(1e-8)
                loss = weighted_sum / weight_sum
                total_loss_num += float(weighted_sum.detach().item())
                total_loss_den += float(weight_sum.detach().item())
            else:
                loss = per_sample_loss.mean()
                total_loss_num += float(per_sample_loss.detach().sum().item())
                total_loss_den += float(per_sample_loss.numel())

            if train_mode:
                loss.backward()
                optimizer.step()
                if scheduler is not None and scheduler_step_per_batch:
                    scheduler.step()

        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(y.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds) if all_preds else np.empty(0, dtype=int)
    y_true = np.concatenate(all_labels) if all_labels else np.empty(0, dtype=int)
    logits_np = (
        np.concatenate(all_logits, axis=0)
        if all_logits
        else np.empty((0, num_classes), dtype=np.float32)
    )
    metrics = classification_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = float(total_loss_num / max(total_loss_den, 1e-8))
    return metrics, y_pred, y_true, logits_np


# =========================================================
# Manifest loading
# =========================================================


def find_dataset_dir(
    out_root: Path,
    dataset_name: str | None,
    dataset_path: str | None,
) -> Path:
    if dataset_path is not None:
        p = Path(dataset_path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Dataset path not found: {p}")
        return p

    if dataset_name is None:
        raise ValueError("Provide either --dataset_path or --dataset_name")

    matches = sorted([p for p in out_root.glob(f"{dataset_name}_*") if p.is_dir()])
    if not matches:
        raise FileNotFoundError(
            f"No dataset folders found for pattern {dataset_name}_* under {out_root}"
        )
    return matches[-1]


def find_latest_width_manifest(dataset_dir: Path) -> Path | None:
    root = dataset_dir / "manifests"
    candidates = list(root.glob(
        "width_inference_*/tree_dataset_manifest_with_final_widths_*.csv"
    ))
    candidates += list(root.glob("tree_dataset_manifest_with_final_widths_*.csv"))
    return sorted(candidates)[-1] if candidates else None


def find_latest_tree_level_width_features(dataset_dir: Path) -> Path | None:
    root = dataset_dir / "manifests"
    candidates = list(root.glob(
        "width_inference_*/tree_level_width_features_for_height_training_*.csv"
    ))
    candidates += list(root.glob("tree_level_width_features_for_height_training_*.csv"))
    return sorted(candidates)[-1] if candidates else None


def load_manifest(
    dataset_dir: Path,
    use_inferred_widths: bool,
    final_width_manifest_path: str | None,
) -> pd.DataFrame:
    base_manifest_path = dataset_dir / "manifests" / "tree_dataset_manifest.csv"
    if not base_manifest_path.exists():
        raise FileNotFoundError(f"Base manifest not found: {base_manifest_path}")

    if use_inferred_widths:
        if final_width_manifest_path is not None:
            path = Path(final_width_manifest_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"final width manifest not found: {path}")
            print(f"\nLoading explicit final-width manifest:\n  {path}")
            df = pd.read_csv(path)
        else:
            width_manifest = find_latest_width_manifest(dataset_dir)
            if width_manifest is not None:
                print(f"\nLoading latest final-width image-level manifest:\n  {width_manifest}")
                df = pd.read_csv(width_manifest)
            else:
                tree_width_path = find_latest_tree_level_width_features(dataset_dir)
                if tree_width_path is None:
                    raise FileNotFoundError(
                        "Could not find inferred-width outputs under manifests/width_inference_*."
                    )
                print(f"\nMerging tree-level width features:\n  {tree_width_path}")
                df_base = pd.read_csv(base_manifest_path)
                df_width = pd.read_csv(tree_width_path)
                if "ID" not in df_base.columns or "ID" not in df_width.columns:
                    raise ValueError("Both base and width files require an ID column")
                width_cols = [
                    c for c in df_width.columns
                    if c == "ID"
                    or "DBH" in c.upper()
                    or "WIDTH" in c.upper()
                    or "CIRCUMFERENCE" in c.upper()
                ]
                df = df_base.merge(
                    df_width[width_cols].drop_duplicates("ID"),
                    on="ID",
                    how="left",
                    suffixes=("", "_WIDTHFEATURE"),
                )
    else:
        print(f"\nLoading base manifest:\n  {base_manifest_path}")
        df = pd.read_csv(base_manifest_path)

    required = ["ID", "RGB_CROP_PATH"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    optional_cols = [
        "ORIGINAL_RGB_PATH", "SAM_LOGITS_PATH", "SAM3_MASK_PATH", "DEPTH_PATH",
        "TREE_TYPE", "OTHER_TREE", "TREE_SPECIES_LABEL", "SPECIES_LABEL", "IMAGE_ID",
    ]
    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan

    if "TRAINABLE" in df.columns:
        df = df[
            df["TRAINABLE"].astype(str).str.lower().isin(["true", "1", "yes"])
        ].copy()

    return df.reset_index(drop=True)


def filter_manifest_for_inputs(
    df: pd.DataFrame,
    input_mode: str,
    image_source: str,
    ablation_common_cohort: bool,
) -> pd.DataFrame:
    """
    If ablation_common_cohort=True, always require SAM3 + depth so the sweep
    compares rgb/rgb_depth/rgb_sam3/rgb_sam3_depth on the same image cohort.
    """
    out = df.copy()

    rgb_col = "RGB_CROP_PATH" if image_source == "crop" else "ORIGINAL_RGB_PATH"
    out = out[out[rgb_col].notna()].copy()
    out = out[out[rgb_col].apply(lambda p: Path(str(p)).exists())].copy()

    require_depth = input_mode in {"rgb_depth", "rgb_sam_depth", "rgb_sam3_depth"}
    require_sam = input_mode in {"rgb_sam", "rgb_sam_depth"}
    require_sam3 = input_mode in {"rgb_sam3", "rgb_sam3_depth"}

    if ablation_common_cohort:
        require_depth = True
        require_sam3 = True

    if require_depth:
        out = out[out["DEPTH_PATH"].notna()].copy()
        out = out[out["DEPTH_PATH"].apply(lambda p: Path(str(p)).exists())].copy()
    if require_sam:
        out = out[out["SAM_LOGITS_PATH"].notna()].copy()
        out = out[out["SAM_LOGITS_PATH"].apply(lambda p: Path(str(p)).exists())].copy()
    if require_sam3:
        out = out[out["SAM3_MASK_PATH"].notna()].copy()
        out = out[out["SAM3_MASK_PATH"].apply(lambda p: Path(str(p)).exists())].copy()

    out = out.drop_duplicates(subset=[rgb_col]).reset_index(drop=True)
    return out


# =========================================================
# Main
# =========================================================


def main() -> None:
    ap = argparse.ArgumentParser()

    # Data/output/W&B
    ap.add_argument("--dataset_name", type=str, default=None)
    ap.add_argument("--dataset_path", type=str, default=None)
    ap.add_argument(
        "--out_dir",
        type=str,
        default="TreeCo/models/tree_height_models",
        help="Directory that directly receives height run folders.",
    )
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--wandb_project", type=str, default="treeco-height")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument(
        "--wandb_mode", type=str, default="online",
        choices=["online", "offline", "disabled"],
    )

    # Images / backbone
    ap.add_argument(
        "--backbone", type=str, default="resnet50",
        choices=["resnet18", "resnet34", "resnet50", "resnet101"],
    )
    ap.add_argument(
        "--input_mode", type=str, default="rgb_sam3_depth",
        choices=list(INPUT_CHANNELS.keys()),
    )
    ap.add_argument("--image_source", type=str, default="full", choices=["crop", "full"])
    ap.add_argument(
        "--use_encoder", action="store_true",
        help="Use separate TreeCo modality encoders before pretrained ResNet blocks.",
    )
    ap.add_argument(
        "--ablation_common_cohort", action="store_true",
        help=(
            "Always require SAM3+depth files even when a run uses fewer channels. "
            "Useful for fair rgb/rgb_depth/rgb_sam3/rgb_sam3_depth ablations."
        ),
    )
    ap.add_argument(
        "--ablation_common_dbh_cohort", action="store_true",
        help=(
            "Restrict every trial to trees with both predicted and observed DBH. "
            "Use this when sweeping dbh_source over none/predicted/observed."
        ),
    )

    # Width manifest / DBH scalar branch
    ap.add_argument("--use_inferred_widths", action="store_true")
    ap.add_argument("--final_width_manifest_path", type=str, default=None)
    ap.add_argument(
        "--dbh_source", type=str, default="none",
        choices=["none", "auto", "predicted", "observed"],
        help="DBH metadata ablation. 'none' disables the DBH branch.",
    )
    ap.add_argument(
        "--use_DBH", action="store_true",
        help="Legacy alias: if set while --dbh_source=none, uses dbh_source=auto.",
    )
    ap.add_argument("--dbh_column", type=str, default="auto")
    ap.add_argument("--dbh_column_is_circumference", action="store_true")
    ap.add_argument("--dbh_tree_agg", type=str, default="median", choices=["mean", "median"])
    ap.add_argument(
        "--dbh_transform", type=str, default="log1p_zscore",
        choices=["zscore", "log1p_zscore"],
    )
    ap.add_argument("--dbh_hidden_dim", type=int, default=32)

    # Species branch
    ap.add_argument("--use_species", action="store_true")
    ap.add_argument("--species_embedding_dim", type=int, default=16)
    ap.add_argument("--species_min_count", type=int, default=5)
    ap.add_argument("--species_dropout", type=float, default=0.10)

    # Loss weighting
    ap.add_argument(
        "--w_height", action="store_true",
        help="Height-class inverse-frequency weighting from unique TRAINING trees.",
    )
    ap.add_argument(
        "--w_nimages", action="store_true",
        help="Multiply each image by 1/N_images_for_tree.",
    )
    ap.add_argument("--height_weight_power", type=float, default=0.5)
    ap.add_argument("--height_weight_max", type=float, default=3.0)
    ap.add_argument("--final_weight_max", type=float, default=6.0)

    # Training hyperparameters
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--dropout_rate", type=float, default=0.1)
    ap.add_argument(
        "--fusion_hidden_dim", type=int, default=128,
        help="Hidden width after image/DBH/species fusion. Set 0 for a single linear head.",
    )
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--val_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=55)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument(
        "--criterion", type=str, default="cross_entropy",
        choices=["cross_entropy", "weighted_ce", "focal"],
    )
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument(
        "--scheduler", type=str, default="plateau",
        choices=["none", "plateau", "cosine", "step", "onecycle"],
    )
    ap.add_argument(
        "--selection_metric", type=str, default="tree_f1_macro",
        choices=["tree_f1_macro", "image_f1_macro"],
        help="Metric used for best checkpoint, plateau scheduler and early stopping.",
    )
    ap.add_argument("--early_stopping_patience", type=int, default=15)
    ap.add_argument(
        "--early_stopping_min_delta", type=float, default=0.002,
        help="Minimum F1 improvement required to reset early-stopping patience.",
    )

    args = ap.parse_args()

    # Initialise W&B first so sweep config can override CLI defaults.
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        config=vars(args),
        name=args.run_name,
    )
    cfg = wandb.config

    sweep_params = [
        "dataset_name", "dataset_path", "out_dir",
        "backbone", "input_mode", "image_source", "use_encoder",
        "ablation_common_cohort", "ablation_common_dbh_cohort",
        "use_inferred_widths", "final_width_manifest_path",
        "dbh_source", "use_DBH", "dbh_column", "dbh_column_is_circumference",
        "dbh_tree_agg", "dbh_transform", "dbh_hidden_dim",
        "use_species", "species_embedding_dim", "species_min_count", "species_dropout",
        "w_height", "w_nimages", "height_weight_power", "height_weight_max",
        "final_weight_max",
        "image_size", "batch_size", "dropout_rate", "fusion_hidden_dim",
        "epochs", "lr", "weight_decay", "val_size", "random_state", "num_workers",
        "criterion", "label_smoothing", "focal_gamma", "scheduler",
        "selection_metric", "early_stopping_patience", "early_stopping_min_delta",
    ]
    for name in sweep_params:
        if name in cfg:
            setattr(args, name, cfg[name])

    # Legacy compatibility.
    if bool(args.use_DBH) and args.dbh_source == "none":
        args.dbh_source = "auto"

    use_dbh = args.dbh_source != "none"
    use_species = bool(args.use_species)
    use_height_weights = bool(args.w_height)
    use_nimages_weights = bool(args.w_nimages)
    use_sample_weights = use_height_weights or use_nimages_weights

    if use_height_weights and args.criterion in {"weighted_ce", "focal"}:
        print(
            "\n[WARNING] w_height is enabled together with criterion-level class weighting. "
            "This intentionally applies stronger class balancing. For a cleaner ablation, "
            "compare cross_entropy+w_height against weighted_ce without w_height."
        )

    seed_everything(int(args.random_state))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = find_dataset_dir(
        out_root=Path("."),
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
    )
    df = load_manifest(
        dataset_dir=dataset_dir,
        use_inferred_widths=bool(args.use_inferred_widths),
        final_width_manifest_path=args.final_width_manifest_path,
    )
    df = filter_manifest_for_inputs(
        df,
        input_mode=args.input_mode,
        image_source=args.image_source,
        ablation_common_cohort=bool(args.ablation_common_cohort),
    )
    df = add_height_labels(df)

    if bool(args.ablation_common_dbh_cohort):
        df = filter_common_dbh_ablation_cohort(df)

    dbh_source_col = None
    dbh_source_is_circumference = False
    dbh_mean = None
    dbh_std = None

    if use_dbh:
        df, dbh_source_col, dbh_source_is_circumference = add_dbh_feature(
            df,
            dbh_source=args.dbh_source,
            dbh_column=args.dbh_column,
            dbh_column_is_circumference=bool(args.dbh_column_is_circumference),
            tree_agg=args.dbh_tree_agg,
        )

    if df.empty:
        raise RuntimeError("No valid rows remain after filtering.")

    # ---------------------------------------------------------
    # Tree-level stratified split - prevents image leakage
    # ---------------------------------------------------------
    tree_df = (
        df.groupby("ID", as_index=False)
        .agg(HEIGHT_CLASS_IDX=("HEIGHT_CLASS_IDX", lambda s: int(s.mode().iloc[0])))
    )
    height_tree_counts = tree_df["HEIGHT_CLASS_IDX"].value_counts().sort_index()
    num_height_classes = int(df["HEIGHT_CLASS_IDX"].nunique())
    n_val_requested = int(round(len(tree_df) * float(args.val_size)))

    print("\nTree-level HEIGHT class counts:")
    print(height_tree_counts)

    if height_tree_counts.min() < 2:
        raise RuntimeError("At least one height class has fewer than 2 trees.")
    if n_val_requested < num_height_classes:
        raise RuntimeError(
            f"Validation split would have {n_val_requested} trees for {num_height_classes} classes. "
            "Increase --val_size."
        )

    train_tree_ids, val_tree_ids = train_test_split(
        tree_df["ID"],
        test_size=float(args.val_size),
        random_state=int(args.random_state),
        stratify=tree_df["HEIGHT_CLASS_IDX"],
    )
    train_df = df[df["ID"].isin(train_tree_ids)].copy().reset_index(drop=True)
    val_df = df[df["ID"].isin(val_tree_ids)].copy().reset_index(drop=True)

    # DBH normalisation is learned from training trees only.
    if use_dbh:
        dbh_mean, dbh_std = fit_dbh_normalisation(train_df, args.dbh_transform)
        train_df = apply_dbh_normalisation(train_df, dbh_mean, dbh_std, args.dbh_transform)
        val_df = apply_dbh_normalisation(val_df, dbh_mean, dbh_std, args.dbh_transform)

    # Species vocabulary is learned from training trees only.
    species_to_idx = None
    if use_species:
        train_df, val_df, species_to_idx = prepare_species_encoding(
            train_df,
            val_df,
            min_tree_count=int(args.species_min_count),
        )

    # ---------------------------------------------------------
    # Training weights - computed AFTER split, training only
    # ---------------------------------------------------------
    height_weight_summary = None
    if use_height_weights:
        train_df, height_weight_summary = add_height_training_weights(
            train_df,
            power=float(args.height_weight_power),
            max_weight=float(args.height_weight_max),
        )
    if use_nimages_weights:
        train_df = add_nimages_training_weights(train_df)
    if use_sample_weights:
        train_df = combine_training_weights(
            train_df,
            use_height_weights=use_height_weights,
            use_nimages_weights=use_nimages_weights,
            final_weight_max=(
                float(args.final_weight_max)
                if args.final_weight_max is not None
                else None
            ),
        )
        print_training_weight_diagnostics(
            train_df,
            use_height_weights=use_height_weights,
            use_nimages_weights=use_nimages_weights,
        )

    print("\n" + "=" * 64)
    print("HEIGHT TRAINING SETUP")
    print("=" * 64)
    print(f"Dataset:                 {dataset_dir}")
    print(f"Device:                  {device}")
    print(f"Backbone:                {args.backbone}")
    print(f"Input mode:              {args.input_mode}")
    print(f"Image source:            {args.image_source}")
    print(f"Use modality encoder:    {bool(args.use_encoder)}")
    print(f"Common ablation cohort:  {bool(args.ablation_common_cohort)}")
    print(f"Common DBH cohort:       {bool(args.ablation_common_dbh_cohort)}")
    print(f"DBH source:              {args.dbh_source}")
    print(f"Use species:             {use_species}")
    print(f"w_height:                {use_height_weights}")
    print(f"w_nimages:               {use_nimages_weights}")
    print(f"Train trees/images:      {train_df['ID'].nunique()} / {len(train_df)}")
    print(f"Val trees/images:        {val_df['ID'].nunique()} / {len(val_df)}")

    if use_species:
        print(f"Species categories:      {len(species_to_idx)}")
        print(train_df["TREE_SPECIES_LABEL"].value_counts().head(20))

    # ---------------------------------------------------------
    # Datasets/loaders
    # ---------------------------------------------------------
    species_unknown_idx = (
        species_to_idx[SPECIES_UNKNOWN] if species_to_idx is not None else 0
    )
    train_ds = TreeHeightDataset(
        train_df,
        image_size=int(args.image_size),
        input_mode=args.input_mode,
        image_source=args.image_source,
        train=True,
        use_dbh=use_dbh,
        use_species=use_species,
        species_dropout=float(args.species_dropout) if use_species else 0.0,
        species_unknown_idx=species_unknown_idx,
        sample_weight_col="TRAIN_WEIGHT" if use_sample_weights else None,
    )
    val_ds = TreeHeightDataset(
        val_df,
        image_size=int(args.image_size),
        input_mode=args.input_mode,
        image_source=args.image_source,
        train=False,
        use_dbh=use_dbh,
        use_species=use_species,
        species_dropout=0.0,
        species_unknown_idx=species_unknown_idx,
        sample_weight_col=None,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=torch.cuda.is_available(),
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = build_model(
        backbone=args.backbone,
        input_mode=args.input_mode,
        use_encoder=bool(args.use_encoder),
        num_height_classes=num_height_classes,
        use_dbh=use_dbh,
        dbh_hidden_dim=int(args.dbh_hidden_dim),
        use_species=use_species,
        num_species=len(species_to_idx) if species_to_idx is not None else 1,
        species_embedding_dim=int(args.species_embedding_dim),
        dropout_rate=float(args.dropout_rate),
        fusion_hidden_dim=int(args.fusion_hidden_dim),
        device=device,
    )

    # Criterion-level class weights use UNIQUE TRAINING TREES.
    train_tree_targets = (
        train_df.groupby("ID")["HEIGHT_CLASS_IDX"]
        .agg(lambda s: int(s.mode().iloc[0]))
        .to_numpy(dtype=int)
    )
    class_counts = np.bincount(train_tree_targets, minlength=num_height_classes).astype(float)
    if np.any(class_counts <= 0):
        raise RuntimeError(f"At least one class is absent from training split: {class_counts}")
    class_weights_np = len(train_tree_targets) / (num_height_classes * class_counts)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)

    if args.criterion == "cross_entropy":
        criterion = nn.CrossEntropyLoss(
            reduction="none",
            label_smoothing=float(args.label_smoothing),
        )
    elif args.criterion == "weighted_ce":
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction="none",
            label_smoothing=float(args.label_smoothing),
        )
    elif args.criterion == "focal":
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=float(args.focal_gamma),
            label_smoothing=float(args.label_smoothing),
        )
    else:
        raise ValueError(f"Unknown criterion: {args.criterion}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=4
        )
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(args.epochs)
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )
    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(args.lr),
            total_steps=int(args.epochs) * len(train_loader),
        )

    # ---------------------------------------------------------
    # Run/output names
    # ---------------------------------------------------------
    out_root = Path(args.out_dir).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.run_name:
        run_name = f"{args.run_name}_{timestamp}"
    else:
        tags = [
            "height",
            args.backbone,
            args.input_mode,
            args.image_source,
            "encoder" if args.use_encoder else "noEncoder",
            f"dbh-{args.dbh_source}" if use_dbh else "noDBH",
            "species" if use_species else "noSpecies",
        ]
        if use_height_weights:
            tags.append("wHeight")
        if use_nimages_weights:
            tags.append("wNimages")
        run_name = "_".join(tags) + f"_{timestamp}"

    wandb_run.name = run_name
    run_dir = out_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_dir / "best_model.pth"
    last_model_path = run_dir / "last_model.pth"
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.json"
    metrics_path = run_dir / "metrics.json"
    val_predictions_path = run_dir / "val_predictions.csv"
    val_tree_predictions_path = run_dir / "val_tree_predictions.csv"
    species_mapping_path = run_dir / "species_to_idx.json"

    if species_to_idx is not None:
        with open(species_mapping_path, "w") as f:
            json.dump(species_to_idx, f, indent=4)

    if use_sample_weights:
        weight_cols = [
            c for c in [
                "ID", "IMAGE_ID", "HEIGHT_CLASS_IDX", "HEIGHT_CLASS_STR",
                "N_IMAGES_PER_TREE", "HEIGHT_TRAIN_WEIGHT",
                "NIMAGES_TRAIN_WEIGHT", "TRAIN_WEIGHT",
            ]
            if c in train_df.columns
        ]
        train_df[weight_cols].to_csv(run_dir / "training_sample_weights.csv", index=False)
        if height_weight_summary is not None:
            height_weight_summary.to_csv(run_dir / "height_weight_summary.csv", index=False)

    height_mapping = make_height_mapping(df)
    config = {
        "task": "height_classification",
        "dataset_dir": str(dataset_dir),
        "run_name": run_name,
        "backbone": args.backbone,
        "input_mode": args.input_mode,
        "image_source": args.image_source,
        "in_channels": int(INPUT_CHANNELS[args.input_mode]),
        "use_encoder": bool(args.use_encoder),
        "ablation_common_cohort": bool(args.ablation_common_cohort),
        "ablation_common_dbh_cohort": bool(args.ablation_common_dbh_cohort),
        "dbh_source": args.dbh_source,
        "use_dbh": bool(use_dbh),
        "dbh_source_col": dbh_source_col,
        "dbh_source_is_circumference": bool(dbh_source_is_circumference),
        "dbh_tree_agg": args.dbh_tree_agg if use_dbh else None,
        "dbh_transform": args.dbh_transform if use_dbh else None,
        "dbh_mean": dbh_mean,
        "dbh_std": dbh_std,
        "dbh_hidden_dim": int(args.dbh_hidden_dim) if use_dbh else None,
        "use_species": bool(use_species),
        "species_embedding_dim": int(args.species_embedding_dim) if use_species else None,
        "species_min_count": int(args.species_min_count) if use_species else None,
        "species_dropout": float(args.species_dropout) if use_species else None,
        "num_species": len(species_to_idx) if species_to_idx is not None else None,
        "height_mapping": height_mapping,
        "w_height": bool(use_height_weights),
        "w_nimages": bool(use_nimages_weights),
        "height_weight_power": float(args.height_weight_power),
        "height_weight_max": float(args.height_weight_max),
        "final_weight_max": float(args.final_weight_max) if args.final_weight_max is not None else None,
        "criterion": args.criterion,
        "label_smoothing": float(args.label_smoothing),
        "focal_gamma": float(args.focal_gamma),
        "class_weights_unique_trees": class_weights_np.tolist(),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "dropout_rate": float(args.dropout_rate),
        "fusion_hidden_dim": int(args.fusion_hidden_dim),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "scheduler": args.scheduler,
        "val_size": float(args.val_size),
        "random_state": int(args.random_state),
        "num_workers": int(args.num_workers),
        "selection_metric": args.selection_metric,
        "early_stopping_patience": int(args.early_stopping_patience),
        "early_stopping_min_delta": float(args.early_stopping_min_delta),
        "n_train_images": int(len(train_df)),
        "n_val_images": int(len(val_df)),
        "n_train_trees": int(train_df["ID"].nunique()),
        "n_val_trees": int(val_df["ID"].nunique()),
        "num_height_classes": int(num_height_classes),
        "device": str(device),
        "use_inferred_widths": bool(args.use_inferred_widths),
        "final_width_manifest_path": args.final_width_manifest_path,
    }

    wandb.config.update(config, allow_val_change=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    # Save exact split IDs for reproducibility.
    pd.DataFrame({"ID": list(train_tree_ids)}).to_csv(run_dir / "train_tree_ids.csv", index=False)
    pd.DataFrame({"ID": list(val_tree_ids)}).to_csv(run_dir / "val_tree_ids.csv", index=False)

    # W&B metric definitions are helpful when using this in sweeps.
    try:
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
    except Exception:
        pass

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1_macro": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1_macro": [],
        "val_recall_macro": [],
        "val_tree_accuracy": [],
        "val_tree_f1_macro": [],
        "val_tree_recall_macro": [],
        "val_tree_qwk": [],
        "val_tree_mae_class": [],
        "val_tree_within_one_class": [],
        "lr": [],
    }

    best_selection_value = -np.inf
    best_epoch = -1
    best_val_logits = None
    best_val_preds = None
    best_val_labels = None
    best_tree_df = None
    best_image_f1 = -np.inf
    best_tree_f1 = -np.inf

    early_stop_best = -np.inf
    epochs_without_improvement = 0
    stopped_early = False

    print(f"\nSaving training run to: {run_dir}")

    for epoch in range(int(args.epochs)):
        train_metrics, _, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            use_dbh=use_dbh,
            use_species=use_species,
            num_classes=num_height_classes,
            optimizer=optimizer,
            scheduler=scheduler if args.scheduler == "onecycle" else None,
            scheduler_step_per_batch=args.scheduler == "onecycle",
            use_sample_weights=use_sample_weights,
        )

        val_metrics, val_preds, val_labels, val_logits = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_dbh=use_dbh,
            use_species=use_species,
            num_classes=num_height_classes,
            optimizer=None,
            use_sample_weights=False,
        )

        tree_pred_df = aggregate_tree_probabilities(
            val_df,
            val_logits,
            num_classes=num_height_classes,
        )
        tree_metrics = classification_metrics(
            tree_pred_df["y_true"].to_numpy(),
            tree_pred_df["y_pred"].to_numpy(),
            num_classes=num_height_classes,
        )

        selection_value = (
            tree_metrics["f1_macro"]
            if args.selection_metric == "tree_f1_macro"
            else val_metrics["f1_macro"]
        )

        if scheduler is not None and args.scheduler != "onecycle":
            if args.scheduler == "plateau":
                scheduler.step(selection_value)
            else:
                scheduler.step()

        current_lr = float(optimizer.param_groups[0]["lr"])
        best_image_f1 = max(best_image_f1, val_metrics["f1_macro"])
        best_tree_f1 = max(best_tree_f1, tree_metrics["f1_macro"])

        history["train_loss"].append(float(train_metrics["loss"]))
        history["train_accuracy"].append(float(train_metrics["accuracy"]))
        history["train_f1_macro"].append(float(train_metrics["f1_macro"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        history["val_f1_macro"].append(float(val_metrics["f1_macro"]))
        history["val_recall_macro"].append(float(val_metrics["recall_macro"]))
        history["val_tree_accuracy"].append(float(tree_metrics["accuracy"]))
        history["val_tree_f1_macro"].append(float(tree_metrics["f1_macro"]))
        history["val_tree_recall_macro"].append(float(tree_metrics["recall_macro"]))
        history["val_tree_qwk"].append(float(tree_metrics["qwk"]))
        history["val_tree_mae_class"].append(float(tree_metrics["mae_class"]))
        history["val_tree_within_one_class"].append(float(tree_metrics["within_one_class"]))
        history["lr"].append(current_lr)

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_metrics["loss"],
            "train/accuracy": train_metrics["accuracy"],
            "train/f1_macro": train_metrics["f1_macro"],
            "val/loss": val_metrics["loss"],
            "val/accuracy": val_metrics["accuracy"],
            "val/image_f1_macro": val_metrics["f1_macro"],
            "val/image_recall_macro": val_metrics["recall_macro"],
            "val/tree_accuracy": tree_metrics["accuracy"],
            "val/tree_balanced_accuracy": tree_metrics["balanced_accuracy"],
            "val/tree_f1_macro": tree_metrics["f1_macro"],
            "val/tree_recall_macro": tree_metrics["recall_macro"],
            "val/tree_qwk": tree_metrics["qwk"],
            "val/tree_mae_class": tree_metrics["mae_class"],
            "val/tree_within_one_class": tree_metrics["within_one_class"],
            "val/selection_metric": selection_value,
            "val/best_image_f1_macro": best_image_f1,
            "val/best_tree_f1_macro": best_tree_f1,
            "learning_rate": current_lr,
        })

        print(
            f"Epoch {epoch + 1:03d}/{int(args.epochs)} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"train_F1={train_metrics['f1_macro']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_img_F1={val_metrics['f1_macro']:.4f} "
            f"val_tree_F1={tree_metrics['f1_macro']:.4f} "
            f"val_tree_QWK={tree_metrics['qwk']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "task": "height_classification",
            "backbone": args.backbone,
            "input_mode": args.input_mode,
            "image_source": args.image_source,
            "in_channels": int(INPUT_CHANNELS[args.input_mode]),
            "image_size": int(args.image_size),
            "num_height_classes": int(num_height_classes),
            "height_mapping": height_mapping,
            "use_encoder": bool(args.use_encoder),
            "use_dbh": bool(use_dbh),
            "dbh_source": args.dbh_source,
            "dbh_source_col": dbh_source_col,
            "dbh_source_is_circumference": bool(dbh_source_is_circumference),
            "dbh_tree_agg": args.dbh_tree_agg if use_dbh else None,
            "dbh_transform": args.dbh_transform if use_dbh else None,
            "dbh_mean": dbh_mean,
            "dbh_std": dbh_std,
            "dbh_hidden_dim": int(args.dbh_hidden_dim) if use_dbh else None,
            "use_species": bool(use_species),
            "num_species": len(species_to_idx) if species_to_idx is not None else None,
            "species_embedding_dim": int(args.species_embedding_dim) if use_species else None,
            "species_mapping": species_to_idx,
            "fusion_hidden_dim": int(args.fusion_hidden_dim),
            "dropout_rate": float(args.dropout_rate),
            "criterion": args.criterion,
            "w_height": bool(use_height_weights),
            "w_nimages": bool(use_nimages_weights),
            "height_weight_power": float(args.height_weight_power),
            "height_weight_max": float(args.height_weight_max),
            "final_weight_max": float(args.final_weight_max) if args.final_weight_max is not None else None,
            "selection_metric": args.selection_metric,
            "selection_value": float(selection_value),
            "val_image_accuracy": float(val_metrics["accuracy"]),
            "val_image_f1_macro": float(val_metrics["f1_macro"]),
            "val_tree_accuracy": float(tree_metrics["accuracy"]),
            "val_tree_f1_macro": float(tree_metrics["f1_macro"]),
            "val_tree_qwk": float(tree_metrics["qwk"]),
            "scheduler": args.scheduler,
        }
        torch.save(checkpoint, last_model_path)

        # Always save the mathematically best checkpoint.
        if selection_value > best_selection_value:
            best_selection_value = float(selection_value)
            best_epoch = epoch + 1
            best_val_preds = val_preds.copy()
            best_val_labels = val_labels.copy()
            best_val_logits = val_logits.copy()
            best_tree_df = tree_pred_df.copy()
            torch.save(checkpoint, best_model_path)

            wandb.run.summary["best_epoch"] = best_epoch
            wandb.run.summary["best_selection_metric"] = best_selection_value
            wandb.run.summary["best_val_image_f1_macro"] = float(val_metrics["f1_macro"])
            wandb.run.summary["best_val_tree_f1_macro"] = float(tree_metrics["f1_macro"])
            wandb.run.summary["best_val_tree_qwk"] = float(tree_metrics["qwk"])

        # Early stopping uses min_delta only to reset patience.
        if int(args.early_stopping_patience) > 0:
            significant_improvement = (
                selection_value > early_stop_best + float(args.early_stopping_min_delta)
            )
            if significant_improvement:
                early_stop_best = float(selection_value)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            print(
                f"Early stopping: {epochs_without_improvement}/"
                f"{int(args.early_stopping_patience)}"
            )

            if epochs_without_improvement >= int(args.early_stopping_patience):
                stopped_early = True
                print("\n" + "=" * 64)
                print("EARLY STOPPING")
                print("=" * 64)
                print(
                    f"No {args.selection_metric} improvement of at least "
                    f"{float(args.early_stopping_min_delta):.4f} for "
                    f"{int(args.early_stopping_patience)} epochs."
                )
                print(f"Best epoch: {best_epoch}")
                print(f"Best selection value: {best_selection_value:.4f}")
                print("=" * 64 + "\n")
                break

    # ---------------------------------------------------------
    # Save best-epoch outputs
    # ---------------------------------------------------------
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    if best_val_logits is None:
        raise RuntimeError("No best validation outputs were captured")

    idx_to_height = {int(k): v for k, v in height_mapping.items()}
    val_probs = torch.softmax(
        torch.tensor(best_val_logits, dtype=torch.float32), dim=1
    ).numpy()

    val_pred_df = val_df.copy().reset_index(drop=True)
    val_pred_df["TRUE_HEIGHT_CLASS_IDX"] = best_val_labels
    val_pred_df["PRED_HEIGHT_CLASS_IDX"] = best_val_preds
    val_pred_df["TRUE_HEIGHT_CLASS_STR"] = [idx_to_height[int(i)] for i in best_val_labels]
    val_pred_df["PRED_HEIGHT_CLASS_STR"] = [idx_to_height[int(i)] for i in best_val_preds]

    for k in range(num_height_classes):
        val_pred_df[f"RESNET_LOGIT_{k}"] = best_val_logits[:, k]
        val_pred_df[f"RESNET_PROB_{k}"] = val_probs[:, k]

    val_pred_df.to_csv(val_predictions_path, index=False)
    best_tree_df.to_csv(val_tree_predictions_path, index=False)

    np.save(run_dir / "val_logits.npy", best_val_logits)
    np.save(run_dir / "val_probs.npy", val_probs)
    np.save(run_dir / "val_labels.npy", best_val_labels)
    np.save(run_dir / "val_preds.npy", best_val_preds)

    best_idx = max(best_epoch - 1, 0)
    metrics = {
        "best_epoch": int(best_epoch),
        "best_selection_metric": float(best_selection_value),
        "selection_metric": args.selection_metric,
        "best_val_loss": float(history["val_loss"][best_idx]),
        "best_val_accuracy": float(history["val_accuracy"][best_idx]),
        "best_val_f1_macro": float(history["val_f1_macro"][best_idx]),
        "best_val_recall_macro": float(history["val_recall_macro"][best_idx]),
        "best_val_tree_accuracy": float(history["val_tree_accuracy"][best_idx]),
        "best_val_tree_f1_macro": float(history["val_tree_f1_macro"][best_idx]),
        "best_val_tree_recall_macro": float(history["val_tree_recall_macro"][best_idx]),
        "best_val_tree_qwk": float(history["val_tree_qwk"][best_idx]),
        "best_val_tree_mae_class": float(history["val_tree_mae_class"][best_idx]),
        "best_val_tree_within_one_class": float(history["val_tree_within_one_class"][best_idx]),
        "final_val_accuracy": float(history["val_accuracy"][-1]),
        "final_val_f1_macro": float(history["val_f1_macro"][-1]),
        "final_val_tree_f1_macro": float(history["val_tree_f1_macro"][-1]),
        "epochs_ran": int(len(history["val_loss"])),
        "stopped_early": bool(stopped_early),
        "n_train_images": int(len(train_df)),
        "n_val_images": int(len(val_df)),
        "n_train_trees": int(train_df["ID"].nunique()),
        "n_val_trees": int(val_df["ID"].nunique()),
        "num_height_classes": int(num_height_classes),
        "use_encoder": bool(args.use_encoder),
        "use_dbh": bool(use_dbh),
        "dbh_source": args.dbh_source,
        "use_species": bool(use_species),
        "w_height": bool(use_height_weights),
        "w_nimages": bool(use_nimages_weights),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    wandb.run.summary.update(metrics)

    # Useful W&B tables.
    try:
        wandb.log({
            "validation_predictions": wandb.Table(dataframe=val_pred_df),
            "validation_tree_predictions": wandb.Table(dataframe=best_tree_df),
        })
    except Exception:
        pass

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best {args.selection_metric}: {best_selection_value:.4f}")
    print(f"Best image macro F1: {metrics['best_val_f1_macro']:.4f}")
    print(f"Best tree macro F1:  {metrics['best_val_tree_f1_macro']:.4f}")
    print(f"Best tree QWK:       {metrics['best_val_tree_qwk']:.4f}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved validation predictions to: {val_predictions_path}")
    print(f"Saved tree predictions to: {val_tree_predictions_path}")

    wandb.finish()


if __name__ == "__main__":
    main()