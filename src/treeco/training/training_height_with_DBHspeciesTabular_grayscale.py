#!/usr/bin/env python3
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
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, recall_score, accuracy_score


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

    # Compact image-only mode for late fusion experiments:
    #   channel 0 = grayscale RGB
    #   channel 1 = SAM3 mask
    #   channel 2 = grayscale RGB * SAM3 mask
    "gray_sam3_overlay": 3,
}


# =========================================================
# Losses
# =========================================================

class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()


# =========================================================
# Text / species helpers
# =========================================================

UNKNOWN_STRINGS = {
    "",
    "nan",
    "none",
    "null",
    "unknown",
    "unk",
    "not known",
    "not sure",
    "unsure",
    "n/a",
    "na",
}


def clean_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def is_unknownish(value) -> bool:
    s = clean_text(value).lower()
    return s in UNKNOWN_STRINGS


def infer_species_label(row: pd.Series) -> str:
    """
    Species is used as an INPUT feature, not as the target.
    """

    direct_cols = [
        "TREE_SPECIES_LABEL",
        "SPECIES_LABEL",
        "SPECIES",
        "species",
        "tree_species",
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
    """
    Add species as an INPUT feature.

    Rare species are grouped into Rare/Other instead of being dropped.
    This is important because the target is height class, not species.
    """

    df = df.copy()

    df["SPECIES_LABEL"] = df.apply(infer_species_label, axis=1)
    df["SPECIES_LABEL"] = df["SPECIES_LABEL"].apply(clean_text)

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
    mapping_df = (
        df[["SPECIES_INPUT_IDX", "SPECIES_LABEL"]]
        .drop_duplicates()
        .sort_values("SPECIES_INPUT_IDX")
    )

    return {
        str(int(row["SPECIES_INPUT_IDX"])): str(row["SPECIES_LABEL"])
        for _, row in mapping_df.iterrows()
    }


# =========================================================
# Height label helpers
# =========================================================

def add_height_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Height class is the TARGET.

    Expected target classes:
        0-5
        5-10
        10-15
        15+
    """

    df = df.copy()

    required = [
        "HEIGHT_CLASS_STR",
        "HEIGHT_CLASS_IDX",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Manifest missing height columns: {missing}")

    df = df[
        df["HEIGHT_CLASS_STR"].notna()
        & df["HEIGHT_CLASS_IDX"].notna()
    ].copy()

    df["HEIGHT_CLASS_IDX_ORIGINAL"] = df["HEIGHT_CLASS_IDX"].astype(int)

    class_order = (
        df[["HEIGHT_CLASS_IDX_ORIGINAL", "HEIGHT_CLASS_STR"]]
        .drop_duplicates()
        .sort_values("HEIGHT_CLASS_IDX_ORIGINAL")
    )

    remap = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(class_order["HEIGHT_CLASS_IDX_ORIGINAL"])
    }

    df["HEIGHT_CLASS_IDX"] = (
        df["HEIGHT_CLASS_IDX_ORIGINAL"]
        .map(remap)
        .astype(int)
    )

    print("\nHeight class remapping:")
    for _, row in class_order.iterrows():
        old_idx = int(row["HEIGHT_CLASS_IDX_ORIGINAL"])
        new_idx = remap[old_idx]
        label = row["HEIGHT_CLASS_STR"]
        print(f"  original {old_idx} -> train {new_idx}: {label}")

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
# DBH helpers
# =========================================================

DBH_COLUMN_CANDIDATES = [
    # Preferred final measured/inferred DBH
    "DBH_CM_FINAL",
    "FINAL_DBH_CM",
    "DBH_FINAL_CM",
    "DBH_MEASURED_OR_INFERRED_CM",
    "MEASURED_OR_INFERRED_DBH_CM",
    "DBH_INFERRED_OR_MEASURED_CM",

    # Tree-level inferred DBH
    "DBH_CM_TREE_PRED",
    "TREE_AGG_PRED_DBH_CM",
    "AGG_PRED_DBH_CM",
    "PRED_DBH_CM",
    "PREDICTED_DBH_CM",
    "INFERRED_DBH_CM",
    "DBH_PRED_CM",
    "MEAN_PRED_DBH_CM",
    "TREE_PRED_DBH_CM",

    # Observed-only DBH comes later
    "DBH_CM_OBSERVED",
    "DBH_CM",
]

CIRCUMFERENCE_COLUMN_CANDIDATES = [
    "CIRCUMFERENCE_IN_CM_FINAL",
    "CIRCUMFERENCE_CM_FINAL",
    "CIRCUMFERENCE_CM_TREE_PRED",
    "CIRCUMFERENCE_CM_OBSERVED",
    "CIRCUMFERENCE_CM_CLEAN",
    "CIRCUMFERENCE_IN_CM",
    "TREE_CIRCUMFERENCE_CM",
    "CIRCUMFERENCE_CM",
    "TREE_CIRCUMFERENCE_IN_CM",
]


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
        extracted = (
            series.astype(str)
            .str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0]
        )
        numeric = pd.to_numeric(extracted, errors="coerce")

    return numeric


def add_dbh_column(
    df: pd.DataFrame,
    dbh_column: str,
    dbh_column_is_circumference: bool,
) -> tuple[pd.DataFrame, str, bool]:
    """
    Adds DBH_FEATURE_CM.

    If --dbh_column auto:
        1. searches for final/inferred DBH columns first
        2. falls back to circumference columns and converts circumference / pi
    """

    df = df.copy()

    source_col = None
    source_is_circumference = False

    if dbh_column != "auto":
        source_col = find_column_case_insensitive(df, dbh_column)

        if source_col is None:
            raise ValueError(
                f"--dbh_column '{dbh_column}' was not found in manifest.\n"
                f"Available columns:\n{list(df.columns)}"
            )

        source_is_circumference = dbh_column_is_circumference

    else:
        source_col = first_existing_column(df, DBH_COLUMN_CANDIDATES)

        if source_col is None:
            source_col = first_existing_column(df, CIRCUMFERENCE_COLUMN_CANDIDATES)
            source_is_circumference = True

        if source_col is None:
            raise ValueError(
                "Could not automatically find a DBH column.\n"
                "Use --dbh_column YOUR_COLUMN_NAME.\n"
                f"Tried DBH columns: {DBH_COLUMN_CANDIDATES}\n"
                f"Tried circumference columns: {CIRCUMFERENCE_COLUMN_CANDIDATES}"
            )

    values = numeric_from_series(df[source_col])

    if source_is_circumference:
        values = values / np.pi

    df["DBH_FEATURE_CM"] = values

    before = len(df)

    df = df[
        df["DBH_FEATURE_CM"].notna()
        & np.isfinite(df["DBH_FEATURE_CM"])
        & (df["DBH_FEATURE_CM"] > 0)
    ].copy()

    after = len(df)

    print(
        f"\nUsing DBH INPUT from column: {source_col} "
        f"{'(converted circumference / pi)' if source_is_circumference else ''}"
    )
    print(f"Rows with valid DBH: {after} / {before}")

    if df.empty:
        raise RuntimeError("No rows left after filtering valid DBH values.")

    return df.reset_index(drop=True), source_col, source_is_circumference


def fit_dbh_normalisation(
    train_df: pd.DataFrame,
    transform: str,
) -> tuple[float, float]:
    values = train_df["DBH_FEATURE_CM"].astype(float).to_numpy()

    if transform == "log1p_zscore":
        values = np.log1p(values)
    elif transform == "zscore":
        values = values
    else:
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
    df = df.copy()

    values = df["DBH_FEATURE_CM"].astype(float).to_numpy()

    if transform == "log1p_zscore":
        values = np.log1p(values)
    elif transform == "zscore":
        values = values
    else:
        raise ValueError(f"Unknown DBH transform: {transform}")

    df["DBH_NORM"] = (values - mean) / std

    return df


# =========================================================
# Dataset
# =========================================================

class TreeHeightSpeciesDBHDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int = 224,
        input_mode: str = "rgb",
        image_source: str = "full",
        train: bool = True,
        use_dbh: bool = False,
        use_species: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.input_mode = input_mode
        self.image_source = image_source
        self.train = train
        self.use_dbh = use_dbh
        self.use_species = use_species

        if input_mode not in INPUT_CHANNELS:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        self.use_depth = input_mode in {
            "rgb_depth",
            "rgb_sam_depth",
            "rgb_sam3_depth",
        }

        self.use_sam = input_mode in {
            "rgb_sam",
            "rgb_sam_depth",
        }

        self.use_sam3 = input_mode in {
            "rgb_sam3",
            "rgb_sam3_depth",
            "gray_sam3_overlay",
        }

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

            if arr.ndim == 3:
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

        rgb_tensor = TF.to_tensor(rgb)

        rgb_tensor = TF.normalize(
            rgb_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        if self.train:
            rgb_tensor = self.rgb_erasing(rgb_tensor)

        return rgb_tensor

    def _single_to_tensor(self, img: Image.Image, dtype: torch.dtype) -> torch.Tensor:
        return TF.to_tensor(img).to(dtype=dtype)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        rgb = Image.open(row[self.rgb_col]).convert("RGB")

        single_channels = []

        if self.use_sam:
            single_channels.append(
                self._load_single_channel_image(row["SAM_LOGITS_PATH"])
            )

        if self.use_sam3:
            single_channels.append(
                self._load_single_channel_image(row["SAM3_MASK_PATH"])
            )

        if self.use_depth:
            single_channels.append(
                self._load_single_channel_image(row["DEPTH_PATH"])
            )

        rgb, single_channels = self._apply_shared_geometric_transforms(
            rgb,
            single_channels,
        )

        # ---------------------------------------------------------
        # Special compact image-only mode for late fusion:
        #   channel 0 = grayscale RGB
        #   channel 1 = SAM3 mask
        #   channel 2 = grayscale RGB * SAM3 mask
        # ---------------------------------------------------------
        if self.input_mode == "gray_sam3_overlay":
            if len(single_channels) < 1:
                raise RuntimeError("gray_sam3_overlay requires SAM3_MASK_PATH.")

            sam3_img = single_channels[0]
            gray_img = TF.rgb_to_grayscale(rgb, num_output_channels=1)

            gray_t = TF.to_tensor(gray_img).to(dtype=torch.float32)
            sam3_t = TF.to_tensor(sam3_img).to(dtype=torch.float32)

            sam3_t = torch.clamp(sam3_t, 0.0, 1.0)
            overlay_t = gray_t * sam3_t

            x = torch.cat(
                [
                    gray_t,
                    sam3_t,
                    overlay_t,
                ],
                dim=0,
            )

            # These are not ImageNet RGB channels, so use a simple shared
            # pseudo-channel normalisation instead of RGB ImageNet stats.
            x = TF.normalize(
                x,
                mean=[0.5, 0.5, 0.5],
                std=[0.25, 0.25, 0.25],
            )

            if self.use_dbh:
                dbh = torch.tensor([float(row["DBH_NORM"])], dtype=torch.float32)
            else:
                dbh = torch.zeros(1, dtype=torch.float32)

            if self.use_species:
                species = torch.tensor(
                    int(row["SPECIES_INPUT_IDX"]),
                    dtype=torch.long,
                )
            else:
                species = torch.tensor(0, dtype=torch.long)

            # IMPORTANT: target is HEIGHT class, not species.
            y = int(row["HEIGHT_CLASS_IDX"])

            return x, dbh, species, y

        rgb_tensor = self._rgb_to_tensor(rgb)

        channels = [rgb_tensor]

        for ch in single_channels:
            channels.append(
                self._single_to_tensor(
                    ch,
                    dtype=rgb_tensor.dtype,
                )
            )

        x = torch.cat(channels, dim=0)

        if self.use_dbh:
            dbh = torch.tensor([float(row["DBH_NORM"])], dtype=torch.float32)
        else:
            dbh = torch.zeros(1, dtype=torch.float32)

        if self.use_species:
            species = torch.tensor(
                int(row["SPECIES_INPUT_IDX"]),
                dtype=torch.long,
            )
        else:
            species = torch.tensor(0, dtype=torch.long)

        # IMPORTANT: target is HEIGHT class, not species.
        y = int(row["HEIGHT_CLASS_IDX"])

        return x, dbh, species, y


# =========================================================
# Model: ResNet image branch + DBH + species inputs
# =========================================================

class ResNetHeightWithSpeciesDBH(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_height_classes: int,
        in_channels: int,
        input_mode: str,
        use_dbh: bool,
        use_species: bool,
        num_species: int,
        dropout_rate: float = 0.1,
        dbh_hidden_dim: int = 32,
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

        if in_channels != 3 or input_mode == "gray_sam3_overlay":
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
                if input_mode == "gray_sam3_overlay":
                    # Pseudo-RGB channels are not natural RGB. Initialise all
                    # three channels from the mean ImageNet RGB filter.
                    mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    for c in range(in_channels):
                        resnet.conv1.weight[:, c:c + 1] = mean_weight
                else:
                    resnet.conv1.weight[:, :3] = old_conv.weight

                    for c in range(3, in_channels):
                        resnet.conv1.weight[:, c:c + 1] = old_conv.weight.mean(
                            dim=1,
                            keepdim=True,
                        )

        image_feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()

        self.image_encoder = resnet
        self.use_dbh = use_dbh
        self.use_species = use_species

        extra_dim = 0

        if use_dbh:
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
                raise ValueError("use_dbh=True but dbh is None.")
            features.append(self.dbh_encoder(dbh))

        if self.use_species:
            if species is None:
                raise ValueError("use_species=True but species is None.")
            features.append(self.species_embedding(species))

        features = torch.cat(features, dim=1)
        logits = self.classifier(features)

        return logits


def build_model(
    backbone: str,
    num_height_classes: int,
    in_channels: int,
    input_mode: str,
    device: torch.device,
    use_dbh: bool,
    use_species: bool,
    num_species: int,
    dropout_rate: float,
    dbh_hidden_dim: int,
    species_embedding_dim: int,
) -> nn.Module:
    model = ResNetHeightWithSpeciesDBH(
        backbone=backbone,
        num_height_classes=num_height_classes,
        in_channels=in_channels,
        input_mode=input_mode,
        use_dbh=use_dbh,
        use_species=use_species,
        num_species=num_species,
        dropout_rate=dropout_rate,
        dbh_hidden_dim=dbh_hidden_dim,
        species_embedding_dim=species_embedding_dim,
    )

    return model.to(device)


# =========================================================
# Training
# =========================================================

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    use_dbh: bool,
    use_species: bool,
    optimizer=None,
    scheduler=None,
    scheduler_step_per_batch: bool = False,
):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_logits = []

    for x, dbh, species, y in loader:
        x = x.to(device, non_blocking=True)
        dbh = dbh.to(device, non_blocking=True)
        species = species.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            logits = model(
                x,
                dbh if use_dbh else None,
                species if use_species else None,
            )

            loss = criterion(logits, y)

            if train_mode:
                loss.backward()
                optimizer.step()

                if scheduler is not None and scheduler_step_per_batch:
                    scheduler.step()

        preds = logits.argmax(dim=1)

        total_loss += loss.item() * x.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if all_logits:
        all_logits = np.concatenate(all_logits, axis=0)
    else:
        all_logits = np.empty((0, 0), dtype=np.float32)

    avg_loss = total_loss / max(len(all_labels), 1)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0

    return avg_loss, acc, all_preds, all_labels, all_logits


# =========================================================
# Manifest loading
# =========================================================

def find_dataset_dir(
    out_root: Path,
    dataset_name: str | None,
    dataset_path: str | None,
) -> Path:
    if dataset_path is not None:
        p = Path(dataset_path)
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
    manifest_root = dataset_dir / "manifests"

    candidates = []

    candidates.extend(
        manifest_root.glob(
            "width_inference_*/tree_dataset_manifest_with_final_widths_*.csv"
        )
    )

    candidates.extend(
        manifest_root.glob(
            "tree_dataset_manifest_with_final_widths_*.csv"
        )
    )

    candidates = sorted(candidates)

    if not candidates:
        return None

    return candidates[-1]


def find_latest_tree_level_width_features(dataset_dir: Path) -> Path | None:
    manifest_root = dataset_dir / "manifests"

    candidates = []

    candidates.extend(
        manifest_root.glob(
            "width_inference_*/tree_level_width_features_for_height_training_*.csv"
        )
    )

    candidates.extend(
        manifest_root.glob(
            "tree_level_width_features_for_height_training_*.csv"
        )
    )

    candidates = sorted(candidates)

    if not candidates:
        return None

    return candidates[-1]


def load_manifest(
    dataset_dir: Path,
    use_inferred_widths: bool = False,
    final_width_manifest_path: str | None = None,
) -> pd.DataFrame:
    base_manifest_path = dataset_dir / "manifests" / "tree_dataset_manifest.csv"

    if not base_manifest_path.exists():
        raise FileNotFoundError(f"Base manifest not found: {base_manifest_path}")

    if use_inferred_widths:
        if final_width_manifest_path is not None:
            manifest_path = Path(final_width_manifest_path)

            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"--final_width_manifest_path not found: {manifest_path}"
                )

            print("\nLoading explicit final-width manifest:")
            print(f"  {manifest_path}")

            df = pd.read_csv(manifest_path)

        else:
            width_manifest_path = find_latest_width_manifest(dataset_dir)

            if width_manifest_path is not None:
                print("\nLoading latest final-width image-level manifest:")
                print(f"  {width_manifest_path}")

                df = pd.read_csv(width_manifest_path)

            else:
                print(
                    "\nNo image-level tree_dataset_manifest_with_final_widths_*.csv found."
                )
                print("Trying fallback: merge tree-level width features onto base manifest.")

                tree_width_path = find_latest_tree_level_width_features(dataset_dir)

                if tree_width_path is None:
                    raise FileNotFoundError(
                        "Could not find inferred-width outputs.\n"
                        "Expected one of:\n"
                        "  manifests/width_inference_*/tree_dataset_manifest_with_final_widths_*.csv\n"
                        "  manifests/width_inference_*/tree_level_width_features_for_height_training_*.csv\n"
                    )

                print("Loading base manifest:")
                print(f"  {base_manifest_path}")
                print("Loading tree-level width features:")
                print(f"  {tree_width_path}")

                df_base = pd.read_csv(base_manifest_path)
                df_width = pd.read_csv(tree_width_path)

                if "ID" not in df_base.columns:
                    raise ValueError("Base manifest is missing ID column.")

                if "ID" not in df_width.columns:
                    raise ValueError("Tree-level width features file is missing ID column.")

                width_cols = [
                    c for c in df_width.columns
                    if c == "ID"
                    or "DBH" in c.upper()
                    or "WIDTH" in c.upper()
                    or "CIRCUMFERENCE" in c.upper()
                ]

                df_width = df_width[width_cols].drop_duplicates(subset=["ID"])

                df = df_base.merge(
                    df_width,
                    on="ID",
                    how="left",
                    suffixes=("", "_WIDTHFEATURE"),
                )

                print("Merged width feature columns:")
                for c in width_cols:
                    if c != "ID":
                        print(f"  {c}")

    else:
        print("\nLoading base manifest:")
        print(f"  {base_manifest_path}")

        df = pd.read_csv(base_manifest_path)

    required = [
        "ID",
        "RGB_CROP_PATH",
    ]

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
    ]

    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan

    if "TRAINABLE" in df.columns:
        df = df[
            df["TRAINABLE"]
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes"])
        ].copy()

    print("\nDBH / width-like columns available in loaded manifest:")
    width_like_cols = [
        c for c in df.columns
        if "DBH" in c.upper()
        or "WIDTH" in c.upper()
        or "CIRCUMFERENCE" in c.upper()
    ]

    if width_like_cols:
        for c in width_like_cols:
            valid_n = pd.to_numeric(df[c], errors="coerce").notna().sum()
            print(f"  {c} | numeric values: {valid_n}")
    else:
        print("  None found.")

    return df.reset_index(drop=True)


def filter_manifest_for_inputs(
    df: pd.DataFrame,
    input_mode: str,
    image_source: str,
) -> pd.DataFrame:
    df = df.copy()

    if image_source == "crop":
        rgb_col = "RGB_CROP_PATH"
    elif image_source == "full":
        rgb_col = "ORIGINAL_RGB_PATH"
    else:
        raise ValueError(f"Unknown image_source: {image_source}")

    df = df[df[rgb_col].notna()].copy()
    df = df[df[rgb_col].apply(lambda p: Path(str(p)).exists())].copy()

    if input_mode in {"rgb_depth", "rgb_sam_depth", "rgb_sam3_depth"}:
        df = df[df["DEPTH_PATH"].notna()].copy()
        df = df[df["DEPTH_PATH"].apply(lambda p: Path(str(p)).exists())].copy()

    if input_mode in {"rgb_sam", "rgb_sam_depth"}:
        df = df[df["SAM_LOGITS_PATH"].notna()].copy()
        df = df[df["SAM_LOGITS_PATH"].apply(lambda p: Path(str(p)).exists())].copy()

    if input_mode in {"rgb_sam3", "rgb_sam3_depth", "gray_sam3_overlay"}:
        df = df[df["SAM3_MASK_PATH"].notna()].copy()
        df = df[df["SAM3_MASK_PATH"].apply(lambda p: Path(str(p)).exists())].copy()

    df = df.drop_duplicates(subset=[rgb_col]).reset_index(drop=True)

    return df


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_name", type=str, default=None)
    ap.add_argument("--dataset_path", type=str, default=None)

    ap.add_argument(
        "--out_dir",
        type=str,
        default="TreeCo/models",
    )

    ap.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional custom run name.",
    )

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

    ap.add_argument(
        "--image_source",
        type=str,
        default="full",
        choices=["crop", "full"],
    )

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
        "--use_inferred_widths",
        action="store_true",
        help=(
            "Load the width-inference manifest with measured/inferred final widths "
            "instead of the base tree_dataset_manifest.csv."
        ),
    )

    ap.add_argument(
        "--final_width_manifest_path",
        type=str,
        default=None,
        help=(
            "Optional explicit path to tree_dataset_manifest_with_final_widths_*.csv. "
            "If omitted, the latest one under manifests/width_inference_* is used."
        ),
    )

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

    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--focal_gamma", type=float, default=2.0)

    ap.add_argument(
        "--include_unknown",
        action="store_true",
        help="Keep Unknown as a valid species input group.",
    )

    ap.add_argument(
        "--min_species_count",
        type=int,
        default=5,
        help="Minimum number of unique trees before a species is grouped as Rare/Other.",
    )

    ap.add_argument(
        "--no_species",
        action="store_true",
        help="Disable species input branch.",
    )

    ap.add_argument(
        "--species_embedding_dim",
        type=int,
        default=16,
        help="Embedding dimension for species input.",
    )

    ap.add_argument(
        "--use_DBH",
        action="store_true",
        help="Use measured or inferred DBH as an extra scalar input.",
    )

    ap.add_argument(
        "--dbh_column",
        type=str,
        default="auto",
        help="Column containing measured/inferred DBH in cm. Use auto to search common names.",
    )

    ap.add_argument(
        "--dbh_column_is_circumference",
        action="store_true",
        help="Treat --dbh_column as circumference in cm and convert to DBH via circumference / pi.",
    )

    ap.add_argument(
        "--dbh_transform",
        type=str,
        default="log1p_zscore",
        choices=["zscore", "log1p_zscore"],
    )

    ap.add_argument(
        "--dbh_hidden_dim",
        type=int,
        default=32,
    )

    args = ap.parse_args()

    seed_everything(args.random_state)

    use_species = not args.no_species

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_dir)

    dataset_dir = find_dataset_dir(
        out_root=Path("."),
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
    )

    df = load_manifest(
        dataset_dir=dataset_dir,
        use_inferred_widths=args.use_inferred_widths,
        final_width_manifest_path=args.final_width_manifest_path,
    )

    df = filter_manifest_for_inputs(
        df,
        input_mode=args.input_mode,
        image_source=args.image_source,
    )

    # TARGET labels
    df = add_height_labels(df)

    # Optional DBH input
    dbh_source_col = None
    dbh_source_is_circumference = False
    dbh_mean = None
    dbh_std = None

    if args.use_DBH:
        df, dbh_source_col, dbh_source_is_circumference = add_dbh_column(
            df,
            dbh_column=args.dbh_column,
            dbh_column_is_circumference=args.dbh_column_is_circumference,
        )

    # Species input, grouped after DBH/image/height filtering
    df = add_species_input_labels(
        df,
        min_species_count=args.min_species_count,
        include_unknown=args.include_unknown,
    )

    if df.empty:
        raise RuntimeError(
            f"No valid training rows after filtering for "
            f"input_mode={args.input_mode}, image_source={args.image_source}."
        )

    print(f"\nUsing dataset: {dataset_dir}")
    print(f"Device: {device}")
    print(f"Input mode: {args.input_mode}")
    print(f"Image source: {args.image_source}")
    print(f"Use species input: {use_species}")
    print(f"Use DBH input: {args.use_DBH}")
    print(f"Total images: {len(df)}")
    print(f"Unique trees: {df['ID'].nunique()}")

    print("\nFull HEIGHT class counts:")
    print(df["HEIGHT_CLASS_STR"].value_counts().sort_index())

    print("\nFull species input counts:")
    print(df["SPECIES_LABEL"].value_counts().sort_index())

    # ---------------------------------------------------
    # Tree-level split by HEIGHT class
    # ---------------------------------------------------

    tree_df = (
        df.groupby("ID")["HEIGHT_CLASS_IDX"]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
    )

    height_tree_counts = tree_df["HEIGHT_CLASS_IDX"].value_counts().sort_index()
    min_trees_per_height_class = height_tree_counts.min()

    print("\nTree-level HEIGHT class counts:")
    print(height_tree_counts)

    if min_trees_per_height_class < 2:
        raise RuntimeError(
            "At least one height class has fewer than 2 trees. "
            "Cannot perform stratified tree-level split."
        )

    num_height_classes = int(df["HEIGHT_CLASS_IDX"].nunique())
    n_val_requested = int(round(len(tree_df) * args.val_size))

    if n_val_requested < num_height_classes:
        raise RuntimeError(
            f"Validation split too small for stratification. "
            f"val trees would be {n_val_requested}, but there are "
            f"{num_height_classes} height classes. "
            f"Increase --val_size."
        )

    train_tree_ids, val_tree_ids = train_test_split(
        tree_df["ID"],
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=tree_df["HEIGHT_CLASS_IDX"],
    )

    train_df = df[df["ID"].isin(train_tree_ids)].copy().reset_index(drop=True)
    val_df = df[df["ID"].isin(val_tree_ids)].copy().reset_index(drop=True)

    if args.use_DBH:
        dbh_mean, dbh_std = fit_dbh_normalisation(
            train_df,
            transform=args.dbh_transform,
        )

        train_df = apply_dbh_normalisation(
            train_df,
            mean=dbh_mean,
            std=dbh_std,
            transform=args.dbh_transform,
        )

        val_df = apply_dbh_normalisation(
            val_df,
            mean=dbh_mean,
            std=dbh_std,
            transform=args.dbh_transform,
        )

        print("\nDBH normalization fitted on training split only:")
        print(f"  transform: {args.dbh_transform}")
        print(f"  mean: {dbh_mean:.6f}")
        print(f"  std:  {dbh_std:.6f}")

    print("\nUnique trees:")
    print(f"Train trees: {train_df['ID'].nunique()}")
    print(f"Val trees:   {val_df['ID'].nunique()}")

    print("\nImages:")
    print(f"Train images: {len(train_df)}")
    print(f"Val images:   {len(val_df)}")

    print("\nTrain HEIGHT class counts:")
    print(train_df["HEIGHT_CLASS_STR"].value_counts().sort_index())

    print("\nValidation HEIGHT class counts:")
    print(val_df["HEIGHT_CLASS_STR"].value_counts().sort_index())

    print("\nTrain species input counts:")
    print(train_df["SPECIES_LABEL"].value_counts().sort_index())

    print("\nValidation species input counts:")
    print(val_df["SPECIES_LABEL"].value_counts().sort_index())

    train_ds = TreeHeightSpeciesDBHDataset(
        train_df,
        image_size=args.image_size,
        input_mode=args.input_mode,
        image_source=args.image_source,
        train=True,
        use_dbh=args.use_DBH,
        use_species=use_species,
    )

    val_ds = TreeHeightSpeciesDBHDataset(
        val_df,
        image_size=args.image_size,
        input_mode=args.input_mode,
        image_source=args.image_source,
        train=False,
        use_dbh=args.use_DBH,
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
    num_species = int(df["SPECIES_INPUT_IDX"].nunique())

    model = build_model(
        backbone=args.backbone,
        num_height_classes=num_height_classes,
        in_channels=in_channels,
        input_mode=args.input_mode,
        device=device,
        use_dbh=args.use_DBH,
        use_species=use_species,
        num_species=num_species,
        dropout_rate=args.dropout_rate,
        dbh_hidden_dim=args.dbh_hidden_dim,
        species_embedding_dim=args.species_embedding_dim,
    )

    classes_np = np.arange(num_height_classes)

    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=classes_np,
        y=train_df["HEIGHT_CLASS_IDX"].to_numpy(),
    )

    class_weights = torch.tensor(
        class_weights_np,
        dtype=torch.float32,
        device=device,
    )

    print("\nHeight class weights:")
    for i, w in enumerate(class_weights_np):
        print(f"  class {i}: {w:.4f}")

    if args.criterion == "cross_entropy":
        criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing,
        )

    elif args.criterion == "weighted_ce":
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )

    elif args.criterion == "focal":
        criterion = FocalLoss(
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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
        )

    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5,
        )

    elif args.scheduler == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=args.epochs * len(train_loader),
        )

    else:
        scheduler = None

    models_root = out_root / "tree_height_models"
    models_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.run_name is not None:
        run_name = f"{args.run_name}_{timestamp}"
    else:
        species_tag = "species" if use_species else "noSpecies"
        dbh_tag = "withDBH" if args.use_DBH else "noDBH"
        run_name = (
            f"height_{args.backbone}_"
            f"{args.input_mode}_{args.image_source}_{species_tag}_{dbh_tag}_{timestamp}"
        )

    run_dir = models_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = run_dir / "best_model.pth"
    last_model_path = run_dir / "last_model.pth"
    config_path = run_dir / "config.json"
    history_path = run_dir / "history.json"
    metrics_path = run_dir / "metrics.json"
    val_predictions_path = run_dir / "val_predictions.csv"

    height_mapping = make_height_mapping(df)
    species_mapping = make_species_input_mapping(df)

    config = {
        "task": "height_classification_with_species_dbh",
        "dataset_dir": str(dataset_dir),
        "backbone": args.backbone,
        "num_height_classes": num_height_classes,
        "num_species_inputs": num_species,
        "in_channels": in_channels,
        "input_mode": args.input_mode,
        "image_source": args.image_source,
        "use_depth": args.input_mode in {"rgb_depth", "rgb_sam_depth", "rgb_sam3_depth"},
        "use_sam": args.input_mode in {"rgb_sam", "rgb_sam_depth"},
        "use_sam3": args.input_mode in {"rgb_sam3", "rgb_sam3_depth", "gray_sam3_overlay"},
        "use_species": use_species,
        "species_embedding_dim": args.species_embedding_dim,
        "use_DBH": args.use_DBH,
        "dbh_source_col": dbh_source_col,
        "dbh_source_is_circumference": dbh_source_is_circumference,
        "dbh_transform": args.dbh_transform if args.use_DBH else None,
        "dbh_mean": dbh_mean,
        "dbh_std": dbh_std,
        "dbh_hidden_dim": args.dbh_hidden_dim,
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
        "scheduler": args.scheduler,
        "device": str(device),
        "height_mapping": height_mapping,
        "species_mapping": species_mapping,
        "min_species_count_for_rare_group": args.min_species_count,
        "include_unknown_species_group": args.include_unknown,
        "use_inferred_widths": args.use_inferred_widths,
        "final_width_manifest_path": args.final_width_manifest_path,
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1_macro": [],
        "val_recall_macro": [],
        "lr": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1
    best_val_preds = None
    best_val_labels = None
    best_val_logits = None

    print(f"\nSaving training run to: {run_dir}")

    for epoch in range(args.epochs):
        train_loss, train_acc, _, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            use_dbh=args.use_DBH,
            use_species=use_species,
            optimizer=optimizer,
            scheduler=scheduler if args.scheduler == "onecycle" else None,
            scheduler_step_per_batch=args.scheduler == "onecycle",
        )

        val_loss, val_acc, val_preds, val_labels, val_logits = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_dbh=args.use_DBH,
            use_species=use_species,
            optimizer=None,
        )

        val_f1_macro = f1_score(
            val_labels,
            val_preds,
            average="macro",
            zero_division=0,
            labels=np.arange(num_height_classes),
        )

        val_recall_macro = recall_score(
            val_labels,
            val_preds,
            average="macro",
            zero_division=0,
            labels=np.arange(num_height_classes),
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
        history["val_f1_macro"].append(float(val_f1_macro))
        history["val_recall_macro"].append(float(val_recall_macro))
        history["lr"].append(float(current_lr))

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"val_f1={val_f1_macro:.4f} val_recall={val_recall_macro:.4f} | "
            f"lr={current_lr:.4e}"
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "task": "height_classification_with_species_dbh",
            "backbone": args.backbone,
            "num_height_classes": num_height_classes,
            "num_species_inputs": num_species,
            "in_channels": in_channels,
            "input_mode": args.input_mode,
            "image_source": args.image_source,
            "image_size": args.image_size,
            "use_species": use_species,
            "species_embedding_dim": args.species_embedding_dim,
            "use_DBH": args.use_DBH,
            "dbh_source_col": dbh_source_col,
            "dbh_source_is_circumference": dbh_source_is_circumference,
            "dbh_transform": args.dbh_transform if args.use_DBH else None,
            "dbh_mean": dbh_mean,
            "dbh_std": dbh_std,
            "dbh_hidden_dim": args.dbh_hidden_dim,
            "height_mapping": height_mapping,
            "species_mapping": species_mapping,
            "val_accuracy": float(val_acc),
            "val_f1_macro": float(val_f1_macro),
            "val_recall_macro": float(val_recall_macro),
            "scheduler": args.scheduler,
        }

        torch.save(checkpoint, last_model_path)

        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_epoch = epoch + 1
            best_val_preds = val_preds.copy()
            best_val_labels = val_labels.copy()
            best_val_logits = val_logits.copy()
            torch.save(checkpoint, best_model_path)

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    # Use the validation outputs from the best epoch for val_predictions.csv.
    # This is important for late fusion, because the probabilities should match
    # the same epoch that produced best_model.pth.
    if best_val_preds is not None:
        val_preds = best_val_preds
        val_labels = best_val_labels
        val_logits = best_val_logits

    idx_to_height = {
        int(k): v
        for k, v in height_mapping.items()
    }

    val_pred_df = val_df.copy()
    val_pred_df["TRUE_HEIGHT_CLASS_IDX"] = val_labels
    val_pred_df["PRED_HEIGHT_CLASS_IDX"] = val_preds
    val_pred_df["TRUE_HEIGHT_CLASS_STR"] = [
        idx_to_height[int(i)] for i in val_labels
    ]
    val_pred_df["PRED_HEIGHT_CLASS_STR"] = [
        idx_to_height[int(i)] for i in val_preds
    ]

    if val_logits is not None and len(val_logits) == len(val_pred_df):
        val_probs = torch.softmax(
            torch.tensor(val_logits, dtype=torch.float32),
            dim=1,
        ).numpy()

        for k in range(num_height_classes):
            val_pred_df[f"RESNET_LOGIT_{k}"] = val_logits[:, k]
            val_pred_df[f"RESNET_PROB_{k}"] = val_probs[:, k]

        np.save(run_dir / "val_logits.npy", val_logits)
        np.save(run_dir / "val_probs.npy", val_probs)

    val_pred_df.to_csv(val_predictions_path, index=False)

    best_idx = max(best_epoch - 1, 0)

    metrics = {
        "best_epoch": int(best_epoch),
        "best_val_loss": float(history["val_loss"][best_idx]),
        "best_val_accuracy": float(history["val_acc"][best_idx]),
        "best_val_f1_macro": float(best_val_f1),
        "best_val_recall_macro": float(history["val_recall_macro"][best_idx]),
        "final_val_accuracy": float(history["val_acc"][-1]),
        "final_val_f1_macro": float(history["val_f1_macro"][-1]),
        "final_val_recall_macro": float(history["val_recall_macro"][-1]),
        "n_train_images": int(len(train_df)),
        "n_val_images": int(len(val_df)),
        "n_train_trees": int(train_df["ID"].nunique()),
        "n_val_trees": int(val_df["ID"].nunique()),
        "num_height_classes": int(num_height_classes),
        "num_species_inputs": int(num_species),
        "use_species": bool(use_species),
        "use_DBH": bool(args.use_DBH),
        "dbh_source_col": dbh_source_col,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    np.save(run_dir / "val_labels.npy", val_labels)
    np.save(run_dir / "val_preds.npy", val_preds)

    print("\nTraining complete.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val macro F1: {best_val_f1:.4f}")
    print(f"Saved best model to: {best_model_path}")
    print(f"Saved last model to: {last_model_path}")
    print(f"Saved validation predictions to: {val_predictions_path}")


if __name__ == "__main__":
    main()
