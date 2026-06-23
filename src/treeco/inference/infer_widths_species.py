#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms.functional as TF


# =========================================================
# Constants / labels
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
# General helpers
# =========================================================
def _is_missing_text(value: Any) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    if s == "":
        return True
    return s.lower() in {"nan", "none", "null", "na", "n/a"}


def file_exists(path: Any) -> bool:
    if path is None:
        return False
    try:
        if pd.isna(path):
            return False
    except Exception:
        pass
    return Path(str(path)).exists()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_json(path: Path, required: bool = True):
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


# =========================================================
# Species cleaning: must match training script
# =========================================================
def clean_species_label(row: pd.Series) -> str:
    """
    Creates one cleaned species label from TREE_TYPE and OTHER_TREE.

    This mirrors the species cleaning in the DBH training script:
    - If TREE_TYPE is "Other", use OTHER_TREE.
    - Otherwise use TREE_TYPE.
    - Missing / unclear values become Unknown.
    - Scientific names in brackets are removed.
    """
    tree_type = row.get("TREE_TYPE", np.nan)
    other_tree = row.get("OTHER_TREE", np.nan)

    if _is_missing_text(tree_type):
        raw = other_tree
    elif str(tree_type).strip().lower() == "other":
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

    label = label.title()
    label = label.replace(" X ", " x ")

    manual_map = {
        "Swe Wh": SPECIES_UNKNOWN,
        "N/A": SPECIES_UNKNOWN,
        "Na": SPECIES_UNKNOWN,
        "Unknown": SPECIES_UNKNOWN,
        "Highclere Holly": "Highclere Holly",
        "Swedish Whitebeam": "Swedish Whitebeam",
        "Purple Crabapple": "Purple Crabapple",
    }

    label = manual_map.get(label, label)

    if _is_missing_text(label):
        return SPECIES_UNKNOWN

    return label


def load_species_mapping(run_path: Path, config: dict) -> dict[str, int] | None:
    candidate_paths = []

    config_mapping = config.get("species_mapping_path")
    if config_mapping:
        candidate_paths.append(Path(config_mapping))

    candidate_paths.append(run_path / "species_to_idx.json")

    for p in candidate_paths:
        if p.exists():
            mapping = load_json(p, required=True)
            mapping = {str(k): int(v) for k, v in mapping.items()}
            print(f"Loaded species mapping: {p}")
            return mapping

    return None


def apply_species_mapping(df: pd.DataFrame, species_to_idx: dict[str, int]) -> pd.DataFrame:
    df = df.copy()

    if "TREE_TYPE" not in df.columns:
        df["TREE_TYPE"] = np.nan
    if "OTHER_TREE" not in df.columns:
        df["OTHER_TREE"] = np.nan

    df["TREE_SPECIES_RAW"] = df.apply(clean_species_label, axis=1)

    unknown_idx = species_to_idx.get(SPECIES_UNKNOWN, 0)
    rare_idx = species_to_idx.get(SPECIES_RARE, unknown_idx)

    def to_label(raw: Any) -> str:
        if _is_missing_text(raw):
            return SPECIES_UNKNOWN
        raw = str(raw).strip()
        if raw in species_to_idx:
            return raw
        return SPECIES_RARE if SPECIES_RARE in species_to_idx else SPECIES_UNKNOWN

    df["TREE_SPECIES_LABEL"] = df["TREE_SPECIES_RAW"].apply(to_label)
    df["TREE_SPECIES_IDX"] = df["TREE_SPECIES_LABEL"].map(species_to_idx)
    df["TREE_SPECIES_IDX"] = df["TREE_SPECIES_IDX"].fillna(rare_idx).astype(int)

    return df


# =========================================================
# Model definitions: match training script exactly
# =========================================================
def build_resnet_encoder(
    backbone: str,
    in_channels: int,
    device: torch.device,
) -> tuple[nn.Module, int]:
    """
    Same structure as training build_resnet_encoder, but uses weights=None
    because all learned weights are loaded from the checkpoint.
    """
    backbone = str(backbone).lower()

    if backbone == "resnet18":
        model = models.resnet18(weights=None)
    elif backbone == "resnet34":
        model = models.resnet34(weights=None)
    elif backbone == "resnet50":
        model = models.resnet50(weights=None)
    elif backbone == "resnet101":
        model = models.resnet101(weights=None)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

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

    image_feat_dim = model.fc.in_features
    model.fc = nn.Identity()

    return model.to(device), image_feat_dim


class ResNetDBHWithSpecies(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        image_feat_dim: int,
        num_species: int,
        species_emb_dim: int = 16,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.image_encoder = image_encoder

        self.species_embedding = nn.Embedding(
            num_embeddings=num_species,
            embedding_dim=species_emb_dim,
        )

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(image_feat_dim + species_emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, species_idx: torch.Tensor) -> torch.Tensor:
        image_feat = self.image_encoder(x)
        species_feat = self.species_embedding(species_idx)
        fused = torch.cat([image_feat, species_feat], dim=1)
        return self.head(fused)


def build_resnet(
    backbone: str,
    in_channels: int,
    device: torch.device,
    dropout_rate: float = 0.1,
) -> nn.Module:
    image_encoder, image_feat_dim = build_resnet_encoder(
        backbone=backbone,
        in_channels=in_channels,
        device=device,
    )

    image_encoder.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(image_feat_dim, 1),
    )

    return image_encoder.to(device)


def build_resnet_with_species(
    backbone: str,
    in_channels: int,
    device: torch.device,
    num_species: int,
    species_emb_dim: int = 16,
    dropout_rate: float = 0.1,
) -> nn.Module:
    image_encoder, image_feat_dim = build_resnet_encoder(
        backbone=backbone,
        in_channels=in_channels,
        device=device,
    )

    model = ResNetDBHWithSpecies(
        image_encoder=image_encoder,
        image_feat_dim=image_feat_dim,
        num_species=num_species,
        species_emb_dim=species_emb_dim,
        dropout_rate=dropout_rate,
    )

    return model.to(device)


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def load_model_run(run_path: Path, device: torch.device):
    config_path = run_path / "config.json"
    checkpoint_path = run_path / "best_model.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best_model.pth: {checkpoint_path}")

    config = load_json(config_path, required=True)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = _strip_module_prefix(state_dict)

    is_species_model = bool(
        config.get("use_species", False)
        or checkpoint.get("use_species", False)
        or any(k.startswith("species_embedding.") for k in state_dict.keys())
    )

    backbone = config.get("backbone", checkpoint.get("backbone", "resnet18"))
    dropout_rate = float(config.get("dropout_rate", 0.1))
    input_mode = config.get("input_mode", checkpoint.get("input_mode", "rgb"))

    # Infer in_channels from checkpoint when possible; this avoids config drift.
    if is_species_model and "image_encoder.conv1.weight" in state_dict:
        in_channels = int(state_dict["image_encoder.conv1.weight"].shape[1])
    elif "conv1.weight" in state_dict:
        in_channels = int(state_dict["conv1.weight"].shape[1])
    else:
        in_channels = int(config.get("in_channels", INPUT_CHANNELS.get(input_mode, 3)))

    if is_species_model:
        species_to_idx = load_species_mapping(run_path, config)
        if species_to_idx is None:
            raise FileNotFoundError(
                "This is a species-aware checkpoint, but species_to_idx.json could not be found. "
                f"Expected {run_path / 'species_to_idx.json'} or config['species_mapping_path']."
            )

        if "species_embedding.weight" in state_dict:
            num_species = int(state_dict["species_embedding.weight"].shape[0])
            species_emb_dim = int(state_dict["species_embedding.weight"].shape[1])
        else:
            num_species = int(config.get("num_species", len(species_to_idx)))
            species_emb_dim = int(config.get("species_emb_dim", 16))

        print("Detected species-aware DBH model.")
        print(f"Backbone: {backbone}")
        print(f"Input channels: {in_channels}")
        print(f"Species categories: {num_species}")
        print(f"Species embedding dim: {species_emb_dim}")

        model = build_resnet_with_species(
            backbone=backbone,
            in_channels=in_channels,
            device=device,
            num_species=num_species,
            species_emb_dim=species_emb_dim,
            dropout_rate=dropout_rate,
        )
    else:
        species_to_idx = None

        print("Detected plain image-only DBH model.")
        print(f"Backbone: {backbone}")
        print(f"Input channels: {in_channels}")

        model = build_resnet(
            backbone=backbone,
            in_channels=in_channels,
            device=device,
            dropout_rate=dropout_rate,
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, config, checkpoint_path, species_to_idx


# =========================================================
# Raw CommuniMap reading / optional merge
# =========================================================
def get_media_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).upper().startswith("MEDIA_")]


def read_raw_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=";")

    raise ValueError(f"Unsupported raw file type: {path}")


def explode_media(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "ID",
        "TREE",
        "TREE_TYPE",
        "OTHER_TREE",
        "LATITUDE",
        "LONGITUDE",
        "TREE_HEIGHT_METHOD",
        "TREE_HEIGHT_IN_METERS",
        "ESTIMATED_TREE_HEIGHT",
        "TREE_CIRCUMFERENCE_METHOD",
        "CIRCUMFERENCE_IN_CM",
        "TREE_TRUNK_SIZE",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    media_cols = get_media_cols(df)

    if "ID" not in df.columns:
        raise ValueError("Raw data is missing ID column.")
    if not media_cols:
        raise ValueError("No MEDIA_* columns found in raw data.")

    rows = []
    for _, row in df[keep_cols + media_cols].iterrows():
        base_id = str(row.get("ID"))

        for media_col in media_cols:
            val = row.get(media_col)
            if pd.isna(val) or str(val).strip() == "":
                continue

            out = {k: row.get(k) for k in keep_cols}
            out["ID"] = base_id
            out["IMAGE_ID"] = f"{base_id}_{media_col}"
            out["MEDIA_COL"] = media_col
            out["MEDIA_SRC"] = str(val)
            rows.append(out)

    return pd.DataFrame(rows)


def merge_manifest_with_raw(manifest_df: pd.DataFrame, raw_path: Path | None) -> pd.DataFrame:
    manifest_df = manifest_df.copy()

    if "ID" not in manifest_df.columns:
        raise ValueError("Manifest is missing ID column.")

    manifest_df["ID"] = manifest_df["ID"].astype(str)

    if raw_path is None:
        return manifest_df

    print(f"Reading raw data for metadata merge: {raw_path}")
    raw_df = read_raw_table(raw_path)
    raw_exploded_df = explode_media(raw_df)

    if raw_exploded_df.empty:
        print("WARNING: raw exploded dataframe is empty; skipping raw merge.")
        return manifest_df

    raw_exploded_df["ID"] = raw_exploded_df["ID"].astype(str)

    raw_keep_cols = [
        "ID",
        "IMAGE_ID",
        "MEDIA_COL",
        "MEDIA_SRC",
        "TREE",
        "TREE_TYPE",
        "OTHER_TREE",
        "LATITUDE",
        "LONGITUDE",
        "TREE_HEIGHT_METHOD",
        "TREE_HEIGHT_IN_METERS",
        "ESTIMATED_TREE_HEIGHT",
        "TREE_CIRCUMFERENCE_METHOD",
        "CIRCUMFERENCE_IN_CM",
        "TREE_TRUNK_SIZE",
    ]
    raw_keep_cols = [c for c in raw_keep_cols if c in raw_exploded_df.columns]

    # Merge by ID + IMAGE_ID where possible. This is the cleanest match.
    if "IMAGE_ID" in manifest_df.columns and "IMAGE_ID" in raw_exploded_df.columns:
        merged = manifest_df.merge(
            raw_exploded_df[raw_keep_cols],
            on=["ID", "IMAGE_ID"],
            how="left",
            suffixes=("", "__RAWMERGE"),
        )
    else:
        # Fallback: ID-level merge. This can duplicate rows if raw has multiple media,
        # so only use one metadata row per ID.
        raw_id_df = raw_exploded_df.drop_duplicates("ID")
        raw_id_keep = [c for c in raw_keep_cols if c != "IMAGE_ID"]
        merged = manifest_df.merge(
            raw_id_df[raw_id_keep],
            on="ID",
            how="left",
            suffixes=("", "__RAWMERGE"),
        )

    # Fill base columns from *_RAW when manifest has missing values.
    for col in [
        "MEDIA_COL",
        "MEDIA_SRC",
        "TREE",
        "TREE_TYPE",
        "OTHER_TREE",
        "LATITUDE",
        "LONGITUDE",
        "TREE_HEIGHT_METHOD",
        "TREE_HEIGHT_IN_METERS",
        "ESTIMATED_TREE_HEIGHT",
        "TREE_CIRCUMFERENCE_METHOD",
        "CIRCUMFERENCE_IN_CM",
        "TREE_TRUNK_SIZE",
    ]:
        raw_col = f"{col}__RAWMERGE"
        if raw_col in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[raw_col]
            else:
                merged[col] = merged[col].where(merged[col].notna(), merged[raw_col])

    # Remove temporary merge columns created only to fill missing metadata.
    temp_cols = [c for c in merged.columns if c.endswith("__RAWMERGE")]
    if temp_cols:
        merged = merged.drop(columns=temp_cols)

    return merged


# =========================================================
# Manifest / missing width preparation
# =========================================================
def load_manifest(dataset_dir: Path) -> pd.DataFrame:
    manifest_path = dataset_dir / "manifests" / "tree_dataset_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    required = ["ID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    optional_cols = [
        "IMAGE_ID",
        "RGB_CROP_PATH",
        "ORIGINAL_RGB_PATH",
        "SAM_LOGITS_PATH",
        "SAM3_MASK_PATH",
        "DEPTH_PATH",
        "TREE_TYPE",
        "OTHER_TREE",
        "CIRCUMFERENCE_IN_CM",
        "TREE_CIRCUMFERENCE_METHOD",
        "TREE_TRUNK_SIZE",
        "DBH_CM",
        "LATITUDE",
        "LONGITUDE",
        "TREE_HEIGHT_METHOD",
        "TREE_HEIGHT_IN_METERS",
        "ESTIMATED_TREE_HEIGHT",
    ]

    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan

    df["ID"] = df["ID"].astype(str)

    return df.reset_index(drop=True)


def add_observed_width_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "DBH_CM" not in df.columns:
        df["DBH_CM"] = np.nan
    if "CIRCUMFERENCE_IN_CM" not in df.columns:
        df["CIRCUMFERENCE_IN_CM"] = np.nan

    dbh = safe_numeric(df["DBH_CM"])
    circ = safe_numeric(df["CIRCUMFERENCE_IN_CM"])

    # Observed DBH: prefer DBH_CM when present, otherwise circumference / pi.
    observed_dbh = dbh.where(dbh > 0, circ / math.pi)
    observed_dbh = observed_dbh.where(observed_dbh > 0, np.nan)

    observed_circ = circ.where(circ > 0, observed_dbh * math.pi)
    observed_circ = observed_circ.where(observed_circ > 0, np.nan)

    df["DBH_CM_OBSERVED"] = observed_dbh
    df["CIRCUMFERENCE_CM_OBSERVED"] = observed_circ
    df["WIDTH_HAS_OBSERVED"] = df["DBH_CM_OBSERVED"].notna()

    return df


def get_required_input_columns(input_mode: str, image_source: str) -> tuple[list[str], str]:
    if image_source == "crop":
        rgb_col = "RGB_CROP_PATH"
    elif image_source == "full":
        rgb_col = "ORIGINAL_RGB_PATH"
    else:
        raise ValueError(f"Unknown image_source: {image_source}")

    required_cols = [rgb_col]

    if input_mode in {"rgb_depth", "rgb_sam_depth", "rgb_sam3_depth"}:
        required_cols.append("DEPTH_PATH")

    if input_mode in {"rgb_sam", "rgb_sam_depth"}:
        required_cols.append("SAM_LOGITS_PATH")

    if input_mode in {"rgb_sam3", "rgb_sam3_depth"}:
        required_cols.append("SAM3_MASK_PATH")

    return required_cols, rgb_col


def prepare_rows_for_inference(
    df: pd.DataFrame,
    input_mode: str,
    image_source: str,
    infer_all: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = add_observed_width_columns(df)
    required_cols, _ = get_required_input_columns(input_mode, image_source)

    if infer_all:
        target_mask = pd.Series(True, index=df.index)
    else:
        target_mask = ~df["WIDTH_HAS_OBSERVED"]

    eligible_mask = target_mask.copy()

    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
        eligible_mask &= df[col].notna() & df[col].map(file_exists)

    # Respect TRAINABLE if present, but do not require it if absent.
    if "TRAINABLE" in df.columns:
        trainable = df["TRAINABLE"].astype(str).str.lower().isin(["true", "1", "yes"])
        eligible_mask &= trainable

    pred_df = df[eligible_mask].copy()
    pred_df = pred_df.reset_index().rename(columns={"index": "MANIFEST_ROW_INDEX"})

    return pred_df, target_mask, required_cols


# =========================================================
# Inference dataset
# =========================================================
class TreeWidthInferenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int,
        input_mode: str,
        image_source: str,
        use_species: bool,
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)
        self.input_mode = input_mode
        self.image_source = image_source
        self.use_species = use_species

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

        if self.use_species and "TREE_SPECIES_IDX" not in self.df.columns:
            raise ValueError("use_species=True, but TREE_SPECIES_IDX is missing.")

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

    def _make_tensor(self, row: pd.Series) -> torch.Tensor:
        rgb = Image.open(row[self.rgb_col]).convert("RGB")

        single_channels = []
        if self.use_sam:
            single_channels.append(self._load_single_channel_image(row["SAM_LOGITS_PATH"]))
        if self.use_sam3:
            single_channels.append(self._load_single_channel_image(row["SAM3_MASK_PATH"]))
        if self.use_depth:
            single_channels.append(self._load_single_channel_image(row["DEPTH_PATH"]))

        # Validation/inference transforms from the training dataset:
        # resize only, no augmentation, same normalisation.
        rgb = TF.resize(rgb, [self.image_size, self.image_size])
        single_channels = [TF.resize(ch, [self.image_size, self.image_size]) for ch in single_channels]

        rgb_tensor = TF.to_tensor(rgb)
        rgb_tensor = TF.normalize(
            rgb_tensor,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        channels = [rgb_tensor]
        for ch in single_channels:
            channels.append(TF.to_tensor(ch).to(dtype=rgb_tensor.dtype))

        return torch.cat(channels, dim=0)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        x = self._make_tensor(row)

        if self.use_species:
            species_idx = torch.tensor(int(row["TREE_SPECIES_IDX"]), dtype=torch.long)
            return x, species_idx, idx

        return x, idx


# =========================================================
# Aggregation and final width feature tables
# =========================================================
def aggregate_tree_width_predictions(
    image_pred_df: pd.DataFrame,
    tree_id_col: str = "ID",
    tree_agg: str = "mean",
    conf_threshold: float | None = None,
) -> pd.DataFrame:
    if image_pred_df.empty:
        return pd.DataFrame()

    rows = []

    for tree_id, group in image_pred_df.groupby(tree_id_col, dropna=False):
        preds = pd.to_numeric(group["DBH_CM_PRED_IMAGE"], errors="coerce").dropna().to_numpy(dtype=float)

        if len(preds) == 0:
            continue

        if tree_agg == "median":
            tree_pred = float(np.median(preds))
            weights = np.ones(len(preds), dtype=float) / len(preds)
        elif tree_agg == "weighted":
            quality = np.ones(len(group), dtype=float)

            for col in ["DINO_SCORE", "SAM_SCORE", "SAM3_SCORE"]:
                if col in group.columns:
                    q = pd.to_numeric(group[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                    quality *= np.clip(q, 0.0, 1.0)

            for col in ["DINO_USED_FULL_IMAGE_FALLBACK", "FULL_IMAGE_FALLBACK"]:
                if col in group.columns:
                    fallback = (
                        group[col]
                        .astype(str)
                        .str.lower()
                        .isin(["true", "1", "yes"])
                        .to_numpy()
                    )
                    quality[fallback] *= 0.5

            if quality.sum() <= 0:
                weights = np.ones(len(group), dtype=float) / len(group)
            else:
                weights = quality / quality.sum()

            tree_pred = float(np.sum(group["DBH_CM_PRED_IMAGE"].to_numpy(dtype=float) * weights))
        else:
            tree_pred = float(np.mean(preds))
            weights = np.ones(len(preds), dtype=float) / len(preds)

        tree_std = float(np.std(preds, ddof=0)) if len(preds) > 1 else 0.0
        tree_min = float(np.min(preds))
        tree_max = float(np.max(preds))
        accepted = True if conf_threshold is None else tree_std <= conf_threshold

        first = group.iloc[0]

        rows.append({
            tree_id_col: tree_id,
            "ID": first.get("ID", tree_id),
            "N_IMAGES_USED": int(len(group)),
            "IMAGE_IDS_USED": json.dumps(group.get("IMAGE_ID", pd.Series(index=group.index, dtype=object)).astype(str).tolist()),
            "MEDIA_COLS_USED": json.dumps(group.get("MEDIA_COL", pd.Series(index=group.index, dtype=object)).astype(str).tolist()) if "MEDIA_COL" in group.columns else np.nan,
            "DBH_CM_TREE_PRED": tree_pred,
            "CIRCUMFERENCE_CM_TREE_PRED": tree_pred * math.pi,
            "DBH_CM_TREE_PRED_STD": tree_std,
            "DBH_CM_TREE_PRED_MIN": tree_min,
            "DBH_CM_TREE_PRED_MAX": tree_max,
            "WIDTH_TREE_PRED_ACCEPTED": bool(accepted),
            "WIDTH_SOURCE_TREE": f"model_inferred_tree_{tree_agg}" if accepted else "model_inferred_tree_not_accepted",
            "TREE_AGG_METHOD": tree_agg,
            "TREE_AGG_WEIGHTS": json.dumps(weights.tolist()),
            "TREE_TYPE": first.get("TREE_TYPE", np.nan),
            "OTHER_TREE": first.get("OTHER_TREE", np.nan),
            "TREE_SPECIES_RAW": first.get("TREE_SPECIES_RAW", np.nan),
            "TREE_SPECIES_LABEL": first.get("TREE_SPECIES_LABEL", np.nan),
            "TREE_SPECIES_IDX": first.get("TREE_SPECIES_IDX", np.nan),
            "LATITUDE": first.get("LATITUDE", np.nan),
            "LONGITUDE": first.get("LONGITUDE", np.nan),
            "MODEL_RUN_NAME": first.get("MODEL_RUN_NAME", np.nan),
            "MODEL_RUN_PATH": first.get("MODEL_RUN_PATH", np.nan),
            "CHECKPOINT_PATH": first.get("CHECKPOINT_PATH", np.nan),
            "INFERENCE_TIMESTAMP": first.get("INFERENCE_TIMESTAMP", np.nan),
            "INPUT_MODE": first.get("INPUT_MODE", np.nan),
            "IMAGE_SOURCE": first.get("IMAGE_SOURCE", np.nan),
            "IMAGE_SIZE": first.get("IMAGE_SIZE", np.nan),
        })

    return pd.DataFrame(rows)


def add_final_width_columns(
    manifest_df: pd.DataFrame,
    image_pred_df: pd.DataFrame,
    tree_pred_df: pd.DataFrame,
    tree_id_col: str,
    tree_agg: str,
) -> pd.DataFrame:
    out = add_observed_width_columns(manifest_df).copy()

    out["DBH_CM_PRED_IMAGE"] = np.nan
    out["DBH_CM_TREE_PRED"] = np.nan
    out["CIRCUMFERENCE_CM_TREE_PRED"] = np.nan
    out["DBH_CM_TREE_PRED_STD"] = np.nan
    out["N_IMAGES_USED_FOR_WIDTH_TREE_PRED"] = np.nan
    out["WIDTH_TREE_PRED_ACCEPTED"] = pd.NA

    # Insert image-level predictions into the original manifest rows.
    if not image_pred_df.empty and "MANIFEST_ROW_INDEX" in image_pred_df.columns:
        for _, row in image_pred_df.iterrows():
            i = int(row["MANIFEST_ROW_INDEX"])
            if 0 <= i < len(out):
                out.loc[i, "DBH_CM_PRED_IMAGE"] = row.get("DBH_CM_PRED_IMAGE", np.nan)

    # Merge tree-level predictions by tree_id_col.
    if not tree_pred_df.empty:
        pred_cols = [
            tree_id_col,
            "DBH_CM_TREE_PRED",
            "CIRCUMFERENCE_CM_TREE_PRED",
            "DBH_CM_TREE_PRED_STD",
            "N_IMAGES_USED",
            "WIDTH_TREE_PRED_ACCEPTED",
        ]
        pred_cols = [c for c in pred_cols if c in tree_pred_df.columns]
        tmp = tree_pred_df[pred_cols].copy()
        tmp = tmp.rename(columns={"N_IMAGES_USED": "N_IMAGES_USED_FOR_WIDTH_TREE_PRED"})

        out[tree_id_col] = out[tree_id_col].astype(str)
        tmp[tree_id_col] = tmp[tree_id_col].astype(str)

        out = out.merge(tmp, on=tree_id_col, how="left", suffixes=("", "_TREE_AGG"))

        # If merge created duplicate-named columns, consolidate them.
        for col in [
            "DBH_CM_TREE_PRED",
            "CIRCUMFERENCE_CM_TREE_PRED",
            "DBH_CM_TREE_PRED_STD",
            "N_IMAGES_USED_FOR_WIDTH_TREE_PRED",
            "WIDTH_TREE_PRED_ACCEPTED",
        ]:
            alt = f"{col}_TREE_AGG"
            if alt in out.columns:
                out[col] = out[col].where(out[col].notna(), out[alt])
                out = out.drop(columns=[alt])

    # Final feature used downstream:
    # observed measurement wins; otherwise accepted tree-level model prediction.
    accepted = out["WIDTH_TREE_PRED_ACCEPTED"].astype(str).str.lower().isin(["true", "1", "yes"])
    has_observed = out["WIDTH_HAS_OBSERVED"].fillna(False).astype(bool)
    has_tree_pred = out["DBH_CM_TREE_PRED"].notna() & accepted

    out["DBH_CM_FINAL"] = np.nan
    out.loc[has_observed, "DBH_CM_FINAL"] = out.loc[has_observed, "DBH_CM_OBSERVED"]
    out.loc[~has_observed & has_tree_pred, "DBH_CM_FINAL"] = out.loc[~has_observed & has_tree_pred, "DBH_CM_TREE_PRED"]

    out["CIRCUMFERENCE_IN_CM_FINAL"] = np.nan
    out.loc[has_observed, "CIRCUMFERENCE_IN_CM_FINAL"] = out.loc[has_observed, "CIRCUMFERENCE_CM_OBSERVED"]
    out.loc[~has_observed & has_tree_pred, "CIRCUMFERENCE_IN_CM_FINAL"] = out.loc[~has_observed & has_tree_pred, "CIRCUMFERENCE_CM_TREE_PRED"]

    out["WIDTH_SOURCE_FINAL"] = "missing_no_prediction"
    out.loc[has_observed, "WIDTH_SOURCE_FINAL"] = "observed"
    out.loc[~has_observed & has_tree_pred, "WIDTH_SOURCE_FINAL"] = f"model_inferred_tree_{tree_agg}"

    out["WIDTH_IS_INFERRED_FINAL"] = out["WIDTH_SOURCE_FINAL"].astype(str).str.startswith("model_inferred")
    out["WIDTH_IS_AVAILABLE_FINAL"] = out["DBH_CM_FINAL"].notna()

    return out


def make_tree_level_final_width_table(
    final_manifest_df: pd.DataFrame,
    tree_id_col: str,
) -> pd.DataFrame:
    rows = []

    for tree_id, group in final_manifest_df.groupby(tree_id_col, dropna=False):
        first = group.iloc[0]
        final_values = pd.to_numeric(group["DBH_CM_FINAL"], errors="coerce").dropna()

        if len(final_values) > 0:
            dbh_final = float(final_values.iloc[0])
            circ_final = dbh_final * math.pi
        else:
            dbh_final = np.nan
            circ_final = np.nan

        rows.append({
            tree_id_col: tree_id,
            "ID": first.get("ID", tree_id),
            "DBH_CM_FINAL": dbh_final,
            "CIRCUMFERENCE_IN_CM_FINAL": circ_final,
            "WIDTH_SOURCE_FINAL": first.get("WIDTH_SOURCE_FINAL", "missing_no_prediction"),
            "WIDTH_IS_INFERRED_FINAL": first.get("WIDTH_IS_INFERRED_FINAL", False),
            "WIDTH_IS_AVAILABLE_FINAL": bool(pd.notna(dbh_final)),
            "DBH_CM_OBSERVED": first.get("DBH_CM_OBSERVED", np.nan),
            "CIRCUMFERENCE_CM_OBSERVED": first.get("CIRCUMFERENCE_CM_OBSERVED", np.nan),
            "DBH_CM_TREE_PRED": first.get("DBH_CM_TREE_PRED", np.nan),
            "CIRCUMFERENCE_CM_TREE_PRED": first.get("CIRCUMFERENCE_CM_TREE_PRED", np.nan),
            "DBH_CM_TREE_PRED_STD": first.get("DBH_CM_TREE_PRED_STD", np.nan),
            "N_IMAGES_IN_MANIFEST": int(len(group)),
            "N_IMAGES_USED_FOR_WIDTH_TREE_PRED": first.get("N_IMAGES_USED_FOR_WIDTH_TREE_PRED", np.nan),
            "TREE_TYPE": first.get("TREE_TYPE", np.nan),
            "OTHER_TREE": first.get("OTHER_TREE", np.nan),
            "TREE_SPECIES_RAW": first.get("TREE_SPECIES_RAW", np.nan),
            "TREE_SPECIES_LABEL": first.get("TREE_SPECIES_LABEL", np.nan),
            "LATITUDE": first.get("LATITUDE", np.nan),
            "LONGITUDE": first.get("LONGITUDE", np.nan),
        })

    return pd.DataFrame(rows)


# =========================================================
# Main inference routine
# =========================================================
def infer_missing_widths(
    dataset_dir: Path,
    run_path: Path,
    raw_data_path: Path | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
    tree_agg: str = "mean",
    tree_id_col: str = "ID",
    conf_threshold: float | None = None,
    infer_all: bool = False,
):
    dataset_dir = Path(dataset_dir)
    run_path = Path(run_path)
    raw_data_path = Path(raw_data_path) if raw_data_path is not None else None

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    if not run_path.exists():
        raise FileNotFoundError(f"Model run path not found: {run_path}")
    if raw_data_path is not None and not raw_data_path.exists():
        raise FileNotFoundError(f"Raw data path not found: {raw_data_path}")

    if tree_agg not in {"mean", "median", "weighted"}:
        raise ValueError("--tree_agg must be one of: mean, median, weighted")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = run_path.name

    manifest_df = load_manifest(dataset_dir)
    manifest_df = merge_manifest_with_raw(manifest_df, raw_data_path)

    if tree_id_col not in manifest_df.columns:
        raise ValueError(f"tree_id_col={tree_id_col!r} not found in manifest columns.")

    model, config, checkpoint_path, species_to_idx = load_model_run(run_path, device=device)

    input_mode = config.get("input_mode", "rgb")
    image_source = config.get("image_source", "full")
    image_size = int(config.get("image_size", 224))
    use_log1p = bool(config.get("use_log1p", False))
    use_species = species_to_idx is not None

    if use_species:
        manifest_df = apply_species_mapping(manifest_df, species_to_idx)
    else:
        if "TREE_TYPE" in manifest_df.columns or "OTHER_TREE" in manifest_df.columns:
            manifest_df["TREE_SPECIES_RAW"] = manifest_df.apply(clean_species_label, axis=1)

    pred_df, target_mask, required_cols = prepare_rows_for_inference(
        manifest_df,
        input_mode=input_mode,
        image_source=image_source,
        infer_all=infer_all,
    )

    print("\nWidth inference setup")
    print("---------------------")
    print(f"Dataset dir: {dataset_dir}")
    print(f"Model run:   {run_path}")
    print(f"Checkpoint:  {checkpoint_path}")
    print(f"Device:      {device}")
    print(f"Input mode:  {input_mode}")
    print(f"Image size:  {image_size}")
    print(f"Image source:{image_source}")
    print(f"Use species: {use_species}")
    print(f"Use log1p:   {use_log1p}")
    print(f"Tree agg:    {tree_agg}")
    print(f"Tree ID col: {tree_id_col}")
    print(f"Required input columns: {required_cols}")

    observed_count = int(add_observed_width_columns(manifest_df)["WIDTH_HAS_OBSERVED"].sum())
    print("\nRows")
    print("----")
    print(f"Manifest rows:                       {len(manifest_df)}")
    print(f"Rows with observed DBH/circumference:{observed_count}")
    print(f"Rows targeted for inference:         {int(target_mask.sum())}")
    print(f"Rows eligible for inference:         {len(pred_df)}")

    if len(pred_df) == 0:
        print("No rows eligible for width inference. Saving final manifest with observed values only.")
        image_pred_df = pd.DataFrame()
        tree_pred_df = pd.DataFrame()
    else:
        infer_ds = TreeWidthInferenceDataset(
            pred_df,
            image_size=image_size,
            input_mode=input_mode,
            image_source=image_source,
            use_species=use_species,
        )

        infer_loader = DataLoader(
            infer_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        all_preds = np.full(len(pred_df), np.nan, dtype=float)

        model.eval()
        with torch.no_grad():
            for batch in infer_loader:
                if use_species:
                    x, species_idx, idx = batch
                    x = x.to(device, non_blocking=True)
                    species_idx = species_idx.to(device, non_blocking=True)
                    raw_pred = model(x, species_idx).view(-1)
                else:
                    x, idx = batch
                    x = x.to(device, non_blocking=True)
                    raw_pred = model(x).view(-1)

                raw_pred_np = raw_pred.detach().cpu().numpy().astype(float)

                if use_log1p:
                    pred_np = np.expm1(raw_pred_np)
                    pred_np = np.clip(pred_np, 0.0, None)
                else:
                    pred_np = raw_pred_np

                all_preds[np.asarray(idx, dtype=int)] = pred_np

        image_pred_df = pred_df.copy()
        image_pred_df["DBH_CM_PRED_IMAGE"] = all_preds
        image_pred_df["CIRCUMFERENCE_CM_PRED_IMAGE"] = image_pred_df["DBH_CM_PRED_IMAGE"] * math.pi
        image_pred_df["WIDTH_PRED_ACCEPTED_IMAGE"] = image_pred_df["DBH_CM_PRED_IMAGE"].notna()
        image_pred_df["MODEL_RUN_NAME"] = run_tag
        image_pred_df["MODEL_RUN_PATH"] = str(run_path)
        image_pred_df["CHECKPOINT_PATH"] = str(checkpoint_path)
        image_pred_df["INFERENCE_TIMESTAMP"] = timestamp
        image_pred_df["INPUT_MODE"] = input_mode
        image_pred_df["IMAGE_SOURCE"] = image_source
        image_pred_df["IMAGE_SIZE"] = image_size

        tree_pred_df = aggregate_tree_width_predictions(
            image_pred_df=image_pred_df,
            tree_id_col=tree_id_col,
            tree_agg=tree_agg,
            conf_threshold=conf_threshold,
        )

    final_manifest_df = add_final_width_columns(
        manifest_df=manifest_df,
        image_pred_df=image_pred_df,
        tree_pred_df=tree_pred_df,
        tree_id_col=tree_id_col,
        tree_agg=tree_agg,
    )

    tree_final_df = make_tree_level_final_width_table(
        final_manifest_df=final_manifest_df,
        tree_id_col=tree_id_col,
    )

    out_dir = dataset_dir / "manifests" / f"width_inference_{run_tag}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_manifest_path = out_dir / f"tree_dataset_manifest_with_final_widths_{run_tag}_{timestamp}.csv"
    image_pred_path = out_dir / f"inferred_width_predictions_image_level_{run_tag}_{timestamp}.csv"
    tree_pred_path = out_dir / f"inferred_width_predictions_tree_level_{tree_agg}_{run_tag}_{timestamp}.csv"
    tree_final_path = out_dir / f"tree_level_width_features_for_height_training_{run_tag}_{timestamp}.csv"
    summary_path = out_dir / f"width_inference_summary_{run_tag}_{timestamp}.json"

    final_manifest_df.to_csv(full_manifest_path, index=False)
    image_pred_df.to_csv(image_pred_path, index=False)
    tree_pred_df.to_csv(tree_pred_path, index=False)
    tree_final_df.to_csv(tree_final_path, index=False)

    summary = {
        "dataset_dir": str(dataset_dir),
        "run_path": str(run_path),
        "checkpoint_path": str(checkpoint_path),
        "timestamp": timestamp,
        "input_mode": input_mode,
        "image_source": image_source,
        "image_size": image_size,
        "use_species": use_species,
        "use_log1p": use_log1p,
        "tree_agg": tree_agg,
        "tree_id_col": tree_id_col,
        "manifest_rows": int(len(manifest_df)),
        "observed_width_rows": int(final_manifest_df["WIDTH_SOURCE_FINAL"].eq("observed").sum()),
        "image_rows_predicted": int(len(image_pred_df)),
        "tree_predictions": int(len(tree_pred_df)),
        "final_available_rows": int(final_manifest_df["WIDTH_IS_AVAILABLE_FINAL"].sum()),
        "final_inferred_rows": int(final_manifest_df["WIDTH_IS_INFERRED_FINAL"].sum()),
        "final_available_trees": int(tree_final_df["WIDTH_IS_AVAILABLE_FINAL"].sum()),
        "final_inferred_trees": int(tree_final_df["WIDTH_IS_INFERRED_FINAL"].sum()),
        "outputs": {
            "full_manifest": str(full_manifest_path),
            "image_predictions": str(image_pred_path),
            "tree_predictions": str(tree_pred_path),
            "tree_final_width_features": str(tree_final_path),
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("\nSaved outputs")
    print("-------------")
    print(f"Full manifest with final widths:     {full_manifest_path}")
    print(f"Image-level width predictions:       {image_pred_path}")
    print(f"Tree-level width predictions:        {tree_pred_path}")
    print(f"Tree-level final width feature file: {tree_final_path}")
    print(f"Summary:                             {summary_path}")

    print("\nFinal width availability")
    print("------------------------")
    print(f"Rows with final width available: {int(final_manifest_df['WIDTH_IS_AVAILABLE_FINAL'].sum())} / {len(final_manifest_df)}")
    print(f"Rows using observed width:       {int(final_manifest_df['WIDTH_SOURCE_FINAL'].eq('observed').sum())}")
    print(f"Rows using inferred width:       {int(final_manifest_df['WIDTH_IS_INFERRED_FINAL'].sum())}")
    print(f"Trees with final width available:{int(tree_final_df['WIDTH_IS_AVAILABLE_FINAL'].sum())} / {len(tree_final_df)}")
    print(f"Trees using inferred width:      {int(tree_final_df['WIDTH_IS_INFERRED_FINAL'].sum())}")

    return full_manifest_path, image_pred_path, tree_pred_path, tree_final_path, summary_path


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Infer missing TreeCo DBH/width values using a trained DBH regression model, "
            "including species-aware models. Produces final measured-or-inferred width "
            "columns for downstream height training."
        )
    )

    # Support both names because older scripts used --raw_data_path and your command used --raw_data.
    parser.add_argument(
        "--raw_data",
        "--raw_data_path",
        dest="raw_data_path",
        default=None,
        help="Optional original CommuniMap XLSX/CSV for metadata merge.",
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Processed TreeCo dataset folder containing manifests/tree_dataset_manifest.csv.",
    )

    parser.add_argument(
        "--run_path",
        required=True,
        help="Trained DBH regression run folder containing config.json, best_model.pth, and species_to_idx.json if species-aware.",
    )

    parser.add_argument(
        "--tree_agg",
        type=str,
        default="mean",
        choices=["mean", "median", "weighted"],
        help="How to aggregate image-level predictions into tree-level DBH predictions.",
    )

    parser.add_argument(
        "--tree_id_col",
        type=str,
        default="ID",
        help="Column used to aggregate multiple images into one tree-level prediction.",
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=None,
        help="Optional max allowed tree-level prediction std in cm for accepting inferred width.",
    )

    parser.add_argument(
        "--infer_all",
        action="store_true",
        help="Infer all eligible rows, including rows that already have observed DBH/circumference. Final columns still prefer observed values.",
    )

    args = parser.parse_args()

    infer_missing_widths(
        dataset_dir=Path(args.dataset_dir),
        run_path=Path(args.run_path),
        raw_data_path=Path(args.raw_data_path) if args.raw_data_path else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        tree_agg=args.tree_agg,
        tree_id_col=args.tree_id_col,
        conf_threshold=args.conf_threshold,
        infer_all=args.infer_all,
    )


if __name__ == "__main__":
    main()
