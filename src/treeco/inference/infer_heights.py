#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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

from treeco.image_models.encoder import TreeCoEncoder


# ============================================================
# Constants
# ============================================================

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

UNKNOWN_STRINGS = {
    "", "nan", "none", "null", "unknown", "unk", "not known",
    "not sure", "unsure", "n/a", "na",
}


# ============================================================
# General helpers
# ============================================================

def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r") as f:
        return json.load(f)


def file_exists(path: Any) -> bool:
    if path is None:
        return False
    try:
        if pd.isna(path):
            return False
    except Exception:
        pass
    return Path(str(path)).expanduser().exists()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return " ".join(str(value).strip().split())


def is_unknownish(value: Any) -> bool:
    return clean_text(value).lower() in UNKNOWN_STRINGS


def text_missing(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.isna() | s.eq("") | s.str.lower().isin(["nan", "none", "null", "na", "n/a"])


def find_column_case_insensitive(df: pd.DataFrame, requested: str) -> str | None:
    requested_lower = str(requested).lower()
    for col in df.columns:
        if str(col).lower() == requested_lower:
            return str(col)
    return None


def safe_class_name(cls: str) -> str:
    return (
        str(cls).replace(">", "gt").replace("<", "lt").replace("+", "plus")
        .replace("-", "_").replace(" ", "")
    )


def get_media_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).upper().startswith("MEDIA_")]


def get_use_dbh(config: dict) -> bool:
    """Current trainer stores use_dbh; use_DBH is retained only as legacy fallback."""
    return bool(config.get("use_dbh", config.get("use_DBH", False)))


# ============================================================
# Species helpers — matches current height trainer
# ============================================================

def _is_missing_text(value: Any) -> bool:
    if value is None:
        return True
    s = str(value).strip()
    return s == "" or s.lower() in UNKNOWN_STRINGS


def clean_species_label(row: pd.Series) -> str:
    """Mirror clean_species_label() used by the current height trainer."""
    direct_cols = [
        "TREE_SPECIES_RAW", "TREE_SPECIES_LABEL", "SPECIES_LABEL",
        "SPECIES", "species", "tree_species",
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


def normalise_species_mapping(mapping: dict) -> dict[str, int]:
    """Accept either label->index or legacy index->label JSON and return label->index."""
    if not mapping:
        raise ValueError("Species mapping is empty.")

    # Current trainer: {"Unknown": 0, "Rare/Other": 1, "Alder": 2, ...}
    if all(isinstance(v, (int, np.integer)) or str(v).isdigit() for v in mapping.values()):
        return {clean_text(label): int(idx) for label, idx in mapping.items()}

    # Legacy possibility: {"0": "Unknown", "1": "Rare/Other", ...}
    if all(str(k).lstrip("-").isdigit() for k in mapping.keys()):
        return {clean_text(label): int(idx) for idx, label in mapping.items()}

    raise ValueError(
        "Could not interpret species mapping. Expected label->index (current trainer) "
        "or index->label (legacy format)."
    )


def load_species_to_idx(run_path: Path, config: dict, checkpoint: dict | None = None) -> dict[str, int] | None:
    if not bool(config.get("use_species", False)):
        return None

    json_path = run_path / "species_to_idx.json"
    if json_path.exists():
        mapping = normalise_species_mapping(load_json(json_path))
        print(f"[SPECIES] Loaded mapping from: {json_path}")
        return mapping

    for source in [config, checkpoint or {}]:
        candidate = source.get("species_to_idx") or source.get("species_mapping")
        if candidate:
            mapping = normalise_species_mapping(candidate)
            print("[SPECIES] Loaded mapping from saved config/checkpoint.")
            return mapping

    raise FileNotFoundError(
        "Model uses species, but no species_to_idx.json or saved species mapping was found."
    )


def apply_saved_species_mapping(df: pd.DataFrame, species_to_idx: dict[str, int] | None, use_species: bool) -> pd.DataFrame:
    df = df.copy()

    if not use_species:
        df["TREE_SPECIES_RAW"] = "Unused"
        df["SPECIES_LABEL"] = "Unused"
        df["SPECIES_INPUT_IDX"] = 0
        return df

    if not species_to_idx:
        raise ValueError("use_species=True but species_to_idx is unavailable.")

    casefold_to_label = {label.casefold(): label for label in species_to_idx}
    rare_idx = species_to_idx.get(SPECIES_RARE)
    unknown_idx = species_to_idx.get(SPECIES_UNKNOWN)

    if rare_idx is None and unknown_idx is None:
        raise ValueError("Saved species mapping contains neither 'Rare/Other' nor 'Unknown'.")

    fallback_idx = rare_idx if rare_idx is not None else unknown_idx
    raw_labels = df.apply(clean_species_label, axis=1)

    mapped_labels, mapped_indices = [], []
    for raw in raw_labels:
        if raw == SPECIES_UNKNOWN and unknown_idx is not None:
            label, idx = SPECIES_UNKNOWN, unknown_idx
        elif raw in species_to_idx:
            label, idx = raw, species_to_idx[raw]
        elif raw.casefold() in casefold_to_label:
            label = casefold_to_label[raw.casefold()]
            idx = species_to_idx[label]
        else:
            idx = fallback_idx
            label = next(name for name, value in species_to_idx.items() if int(value) == int(idx))

        mapped_labels.append(label)
        mapped_indices.append(int(idx))

    df["TREE_SPECIES_RAW"] = raw_labels
    df["SPECIES_LABEL"] = mapped_labels
    df["SPECIES_INPUT_IDX"] = np.asarray(mapped_indices, dtype=int)

    print(f"[SPECIES] Categories in saved vocabulary: {len(species_to_idx)}")
    print(f"[SPECIES] Rows mapped to {SPECIES_RARE}: {(df['SPECIES_LABEL'] == SPECIES_RARE).sum()}")
    print(f"[SPECIES] Rows mapped to {SPECIES_UNKNOWN}: {(df['SPECIES_LABEL'] == SPECIES_UNKNOWN).sum()}")
    return df


# ============================================================
# DBH helpers — matches current height trainer
# ============================================================

def apply_saved_dbh_transform(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Recreate the height trainer's DBH branch exactly:
      saved source column -> optional circumference/pi -> tree mean/median
      -> saved z-score or log1p-z-score.
    """
    df = df.copy()
    use_dbh = get_use_dbh(config)

    if not use_dbh:
        df["DBH_SOURCE_VALID_ROW"] = True
        df["DBH_FEATURE_CM"] = 0.0
        df["DBH_NORM"] = 0.0
        return df

    requested_col = config.get("dbh_source_col")
    if not requested_col:
        raise ValueError("Model was trained with DBH input, but config.json has no dbh_source_col.")

    source_col = find_column_case_insensitive(df, requested_col)
    if source_col is None:
        raise ValueError(
            f"DBH input column '{requested_col}' was not found in the selected manifest.\n"
            f"Available columns:\n{list(df.columns)}"
        )

    values = pd.to_numeric(df[source_col], errors="coerce")
    source_is_circumference = bool(config.get("dbh_source_is_circumference", False))
    if source_is_circumference:
        values = values / math.pi

    valid = values.notna() & np.isfinite(values) & (values > 0)
    df["DBH_SOURCE_VALID_ROW"] = valid.to_numpy(dtype=bool)
    tree_agg = str(config.get("dbh_tree_agg", "median")).lower()
    if tree_agg not in {"mean", "median"}:
        raise ValueError(f"Unknown saved dbh_tree_agg: {tree_agg}")

    tmp = pd.DataFrame({"ID": df["ID"], "DBH_FEATURE_CM_RAW": values})
    tmp = tmp[valid].copy()
    agg_func = "mean" if tree_agg == "mean" else "median"
    tree_dbh = tmp.groupby("ID")["DBH_FEATURE_CM_RAW"].agg(agg_func)
    df["DBH_FEATURE_CM"] = df["ID"].map(tree_dbh)

    transform = str(config.get("dbh_transform", "zscore"))
    mean = config.get("dbh_mean")
    std = config.get("dbh_std")
    if mean is None or std is None:
        raise ValueError("DBH model requires saved dbh_mean and dbh_std in config.json.")

    mean, std = float(mean), float(std)
    if abs(std) < 1e-8:
        std = 1.0

    feature = pd.to_numeric(df["DBH_FEATURE_CM"], errors="coerce").to_numpy(dtype=float)
    valid_feature = np.isfinite(feature) & (feature > 0)
    norm = np.full(len(df), np.nan, dtype=np.float32)

    if transform == "log1p_zscore":
        norm[valid_feature] = ((np.log1p(feature[valid_feature]) - mean) / std).astype(np.float32)
    elif transform == "zscore":
        norm[valid_feature] = ((feature[valid_feature] - mean) / std).astype(np.float32)
    else:
        raise ValueError(f"Unknown saved DBH transform: {transform}")

    df["DBH_NORM"] = norm

    print(f"[DBH] Source column: {source_col}" + (" (circumference / pi)" if source_is_circumference else ""))
    print(f"[DBH] Tree aggregation: {tree_agg}")
    print(f"[DBH] Transform: {transform}")
    print(f"[DBH] Training mean/std: {mean:.6f} / {std:.6f}")
    print(f"[DBH] Trees with valid DBH: {df.loc[np.isfinite(df['DBH_NORM']), 'ID'].nunique()} / {df['ID'].nunique()}")
    return df


# ============================================================
# Exact current height-model architecture
# ============================================================

class TreeCoMultimodalResNetEncoder(nn.Module):
    def __init__(self, backbone: str, input_mode: str):
        super().__init__()
        self.input_mode = input_mode
        self.use_mask = input_mode in {"rgb_sam", "rgb_sam_depth", "rgb_sam3", "rgb_sam3_depth"}
        self.use_depth = input_mode in {"rgb_depth", "rgb_sam_depth", "rgb_sam3_depth"}

        self.rgb_encoder = TreeCoEncoder(in_channels=3, hidden_channels=32, out_channels=32, first_kernel_size=7)
        fusion_channels = 32

        if self.use_mask:
            self.mask_encoder = TreeCoEncoder(in_channels=1, hidden_channels=8, out_channels=8, first_kernel_size=5)
            fusion_channels += 8
        else:
            self.mask_encoder = None

        if self.use_depth:
            self.depth_encoder = TreeCoEncoder(in_channels=1, hidden_channels=16, out_channels=16, first_kernel_size=5)
            fusion_channels += 16
        else:
            self.depth_encoder = None

        self.fusion_encoder = nn.Sequential(
            nn.Conv2d(fusion_channels, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )

        backbone = backbone.lower()
        if backbone not in {"resnet18", "resnet34", "resnet50", "resnet101"}:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # No download is needed at inference; the checkpoint replaces all parameters.
        resnet = getattr(models, backbone)(weights=None)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.image_feat_dim = resnet.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected_channels = 3 + int(self.use_mask) + int(self.use_depth)
        if x.shape[1] != expected_channels:
            raise ValueError(f"{self.input_mode} expects {expected_channels} channels, got {x.shape[1]}")

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


def build_standard_resnet_encoder(backbone: str, in_channels: int) -> tuple[nn.Module, int]:
    backbone = backbone.lower()
    if backbone not in {"resnet18", "resnet34", "resnet50", "resnet101"}:
        raise ValueError(f"Unsupported backbone: {backbone}")

    model = getattr(models, backbone)(weights=None)
    if in_channels != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=old_conv.out_channels, kernel_size=old_conv.kernel_size,
            stride=old_conv.stride, padding=old_conv.padding, bias=False,
        )

    feat_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, feat_dim


class TreeHeightClassifier(nn.Module):
    def __init__(
        self, image_encoder: nn.Module, image_feat_dim: int, num_height_classes: int,
        use_dbh: bool, dbh_hidden_dim: int, use_species: bool, num_species: int,
        species_embedding_dim: int, dropout_rate: float, fusion_hidden_dim: int,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.use_dbh = bool(use_dbh)
        self.use_species = bool(use_species)
        extra_dim = 0

        if self.use_dbh:
            self.dbh_encoder = nn.Sequential(
                nn.Linear(1, dbh_hidden_dim), nn.ReLU(inplace=True), nn.Dropout(p=dropout_rate),
                nn.Linear(dbh_hidden_dim, dbh_hidden_dim), nn.ReLU(inplace=True),
            )
            extra_dim += dbh_hidden_dim
        else:
            self.dbh_encoder = None

        if self.use_species:
            self.species_embedding = nn.Embedding(num_embeddings=num_species, embedding_dim=species_embedding_dim)
            extra_dim += species_embedding_dim
        else:
            self.species_embedding = None

        classifier_in = image_feat_dim + extra_dim
        if fusion_hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate), nn.Linear(classifier_in, fusion_hidden_dim), nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate), nn.Linear(fusion_hidden_dim, num_height_classes),
            )
        else:
            self.classifier = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(classifier_in, num_height_classes))

    def forward(self, x: torch.Tensor, dbh: torch.Tensor | None = None, species: torch.Tensor | None = None) -> torch.Tensor:
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


def build_model_from_config(config: dict, num_species: int, device: torch.device) -> nn.Module:
    backbone = str(config.get("backbone", "resnet50"))
    input_mode = str(config.get("input_mode", "rgb_sam3_depth"))
    use_encoder = bool(config.get("use_encoder", False))
    use_dbh = get_use_dbh(config)
    use_species = bool(config.get("use_species", False))
    num_height_classes = int(config.get("num_height_classes", len(config.get("height_mapping", {})) or 4))
    dbh_hidden_dim = int(config.get("dbh_hidden_dim", 32))
    species_embedding_dim = int(config.get("species_embedding_dim", 16))
    dropout_rate = float(config.get("dropout_rate", 0.1))
    fusion_hidden_dim = int(config.get("fusion_hidden_dim", 0))

    if input_mode not in INPUT_CHANNELS:
        raise ValueError(f"Unsupported input_mode in config: {input_mode}")

    if use_encoder:
        image_encoder = TreeCoMultimodalResNetEncoder(backbone=backbone, input_mode=input_mode)
        image_feat_dim = image_encoder.image_feat_dim
    else:
        image_encoder, image_feat_dim = build_standard_resnet_encoder(backbone, INPUT_CHANNELS[input_mode])

    return TreeHeightClassifier(
        image_encoder=image_encoder, image_feat_dim=image_feat_dim, num_height_classes=num_height_classes,
        use_dbh=use_dbh, dbh_hidden_dim=dbh_hidden_dim, use_species=use_species,
        num_species=num_species if use_species else 1, species_embedding_dim=species_embedding_dim,
        dropout_rate=dropout_rate, fusion_hidden_dim=fusion_hidden_dim,
    ).to(device)


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def load_model_run(run_path: Path, device: torch.device) -> tuple[nn.Module, dict, dict[str, int] | None, dict]:
    run_path = Path(run_path)
    config_path = run_path / "config.json"
    checkpoint_path = run_path / "best_model.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best_model.pth: {checkpoint_path}")

    config = load_json(config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _strip_module_prefix(checkpoint.get("model_state_dict", checkpoint))
    species_to_idx = load_species_to_idx(run_path, config, checkpoint)

    use_species = bool(config.get("use_species", False))
    if use_species:
        num_species_state = state_dict.get("species_embedding.weight")
        num_species = int(num_species_state.shape[0]) if num_species_state is not None else len(species_to_idx)
        if len(species_to_idx) != num_species:
            raise ValueError(
                f"Species mapping has {len(species_to_idx)} categories, but checkpoint embedding has {num_species}."
            )
    else:
        num_species = 1

    model = build_model_from_config(config, num_species=num_species, device=device)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint does not match the architecture reconstructed from config.json. "
            "Check use_encoder, fusion_hidden_dim, input_mode, DBH/species flags and embedding dimensions.\n\n"
            f"Original load error:\n{exc}"
        ) from exc

    model.eval()

    print("\nModel reconstruction")
    print("--------------------")
    print(f"Backbone:              {config.get('backbone')}")
    print(f"Input mode:            {config.get('input_mode')}")
    print(f"Image source:          {config.get('image_source')}")
    print(f"Use modality encoder:  {bool(config.get('use_encoder', False))}")
    print(f"Fusion hidden dim:     {int(config.get('fusion_hidden_dim', 0))}")
    print(f"Height classes:        {int(config.get('num_height_classes', 4))}")
    print(f"Use species:           {use_species}")
    print(f"Species categories:    {num_species}")
    print(f"Use DBH:               {get_use_dbh(config)}")
    print(f"DBH hidden dim:        {config.get('dbh_hidden_dim')}")
    print(f"Loaded checkpoint:     {checkpoint_path}")

    return model, config, species_to_idx, checkpoint


# ============================================================
# Raw CommuniMap loader / merge
# ============================================================

def read_raw_table(path: Path) -> pd.DataFrame:
    """Robust reader for the Sept 2026 CommuniMap export with occasional extra MEDIA fields."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        print(f"[DATA] Reading Excel: {path}")
        return pd.read_excel(path)
    if suffix != ".csv":
        raise ValueError(f"Unsupported raw file type: {suffix}")

    print(f"[DATA] Reading CommuniMap CSV: {path}")
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter=";", quotechar='"')
        header = next(reader)
        rows = list(reader)

    original_ncols = len(header)
    max_ncols = max([len(header)] + [len(row) for row in rows])
    print(f"[DATA] Header columns: {original_ncols}")
    print(f"[DATA] Maximum row columns: {max_ncols}")

    if max_ncols > len(header):
        extra = max_ncols - len(header)
        media_indices = []
        for col in header:
            if str(col).startswith("MEDIA_2635_"):
                try:
                    media_indices.append(int(str(col).split("_")[-1]))
                except ValueError:
                    pass
        next_media_idx = max(media_indices) + 1 if media_indices else 0
        for i in range(extra):
            new_col = f"MEDIA_2635_{next_media_idx + i}"
            print(f"[DATA] Adding column: {new_col}")
            header.append(new_col)

    padded_rows = [row + [""] * (len(header) - len(row)) if len(row) < len(header) else row[:len(header)] for row in rows]
    df = pd.DataFrame(padded_rows, columns=header)
    print(f"[DATA] Loaded CommuniMap CSV: {len(df)} rows × {len(df.columns)} columns")
    return df


def explode_media(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "ID", "TREE", "TREE_TYPE", "OTHER_TREE", "LATITUDE", "LONGITUDE",
        "TREE_HEIGHT_METHOD", "TREE_HEIGHT_IN_METERS", "ESTIMATED_TREE_HEIGHT",
        "TREE_CIRCUMFERENCE_METHOD", "CIRCUMFERENCE_IN_CM", "TREE_TRUNK_SIZE",
    ]
    keep_cols = [col for col in keep_cols if col in df.columns]
    media_cols = get_media_cols(df)

    if "ID" not in df.columns:
        raise ValueError("Raw CommuniMap data is missing ID.")
    if not media_cols:
        raise ValueError("No MEDIA_* columns found in raw data.")

    rows = []
    for _, row in df[keep_cols + media_cols].iterrows():
        base_id = str(row.get("ID"))
        for media_col in media_cols:
            value = row.get(media_col)
            if pd.isna(value) or str(value).strip() == "":
                continue
            out = {col: row.get(col) for col in keep_cols}
            out["ID"] = base_id
            out["IMAGE_ID"] = f"{base_id}_{media_col}"
            out["MEDIA_COL"] = media_col
            out["MEDIA_SRC"] = str(value)
            rows.append(out)

    return pd.DataFrame(rows)


def merge_manifest_with_raw(manifest_df: pd.DataFrame, raw_exploded_df: pd.DataFrame) -> pd.DataFrame:
    manifest_df, raw_exploded_df = manifest_df.copy(), raw_exploded_df.copy()

    if "ID" not in manifest_df.columns:
        raise ValueError("Manifest is missing ID.")
    if "IMAGE_ID" not in manifest_df.columns:
        raise ValueError("Manifest is missing IMAGE_ID. Cannot merge safely.")
    if raw_exploded_df.empty:
        print("[WARNING] Exploded raw data is empty; using manifest metadata only.")
        return manifest_df

    manifest_df["ID"] = manifest_df["ID"].astype(str)
    raw_exploded_df["ID"] = raw_exploded_df["ID"].astype(str)

    raw_keep_cols = [
        "ID", "IMAGE_ID", "MEDIA_COL", "MEDIA_SRC", "TREE", "TREE_TYPE", "OTHER_TREE",
        "LATITUDE", "LONGITUDE", "TREE_HEIGHT_METHOD", "TREE_HEIGHT_IN_METERS",
        "ESTIMATED_TREE_HEIGHT", "TREE_CIRCUMFERENCE_METHOD", "CIRCUMFERENCE_IN_CM",
        "TREE_TRUNK_SIZE",
    ]
    raw_keep_cols = [col for col in raw_keep_cols if col in raw_exploded_df.columns]

    merged = manifest_df.merge(
        raw_exploded_df[raw_keep_cols], on=["ID", "IMAGE_ID"], how="left", suffixes=("", "__RAW"),
    )

    fill_cols = [
        "MEDIA_COL", "MEDIA_SRC", "TREE", "TREE_TYPE", "OTHER_TREE", "LATITUDE", "LONGITUDE",
        "TREE_HEIGHT_METHOD", "TREE_HEIGHT_IN_METERS", "ESTIMATED_TREE_HEIGHT",
        "TREE_CIRCUMFERENCE_METHOD", "CIRCUMFERENCE_IN_CM", "TREE_TRUNK_SIZE",
    ]

    for col in fill_cols:
        raw_col = f"{col}__RAW"
        if raw_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = merged[raw_col]
        else:
            good_existing = merged[col].notna() & (merged[col].astype(str).str.strip() != "")
            merged[col] = merged[col].where(good_existing, merged[raw_col])

    temp_cols = [col for col in merged.columns if col.endswith("__RAW")]
    return merged.drop(columns=temp_cols) if temp_cols else merged


# ============================================================
# Manifest selection
# ============================================================

def resolve_manifest_path(dataset_dir: Path, config: dict) -> Path:
    use_inferred_widths = bool(config.get("use_inferred_widths", False))

    if use_inferred_widths:
        configured = config.get("final_width_manifest_path")
        if not configured:
            raise ValueError("use_inferred_widths=True but config.json has no final_width_manifest_path.")
        manifest_path = Path(configured).expanduser()
        if not manifest_path.is_absolute():
            candidate = dataset_dir / manifest_path
            if candidate.exists():
                manifest_path = candidate
    else:
        manifest_path = dataset_dir / "manifests" / "tree_dataset_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return manifest_path


# ============================================================
# Inference dataset — validation transforms only
# ============================================================

class TreeHeightInferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_size: int, input_mode: str, image_source: str, use_dbh: bool, use_species: bool):
        self.df = df.reset_index(drop=True)
        self.image_size = int(image_size)
        self.input_mode = input_mode
        self.image_source = image_source
        self.use_dbh = bool(use_dbh)
        self.use_species = bool(use_species)

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

    def __len__(self) -> int:
        return len(self.df)

    def _load_single_channel_image(self, path: str) -> Image.Image:
        # Mirrors training dataset behaviour: failed arrays become zeros.
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
            return Image.fromarray(np.zeros((self.image_size, self.image_size), dtype=np.uint8)).convert("L")

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

        rgb = TF.resize(rgb, [self.image_size, self.image_size])
        single_channels = [TF.resize(ch, [self.image_size, self.image_size]) for ch in single_channels]

        rgb_tensor = TF.to_tensor(rgb)
        rgb_tensor = TF.normalize(rgb_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        channels = [rgb_tensor]
        channels.extend(TF.to_tensor(ch).to(dtype=rgb_tensor.dtype) for ch in single_channels)
        x = torch.cat(channels, dim=0)

        dbh = torch.tensor([float(row["DBH_NORM"])] if self.use_dbh else [0.0], dtype=torch.float32)
        species = torch.tensor(int(row["SPECIES_INPUT_IDX"]) if self.use_species else 0, dtype=torch.long)
        return x, dbh, species, idx


# ============================================================
# Missing-height / input selection
# ============================================================

def is_missing_height(df: pd.DataFrame) -> pd.Series:
    """
    Prefer the actual training target columns when available. A tree is inferred only
    when the height class is missing; raw fields are fallback checks for older manifests.
    """
    if "HEIGHT_CLASS_STR" in df.columns:
        return text_missing(df["HEIGHT_CLASS_STR"])

    if "HEIGHT_CLASS_IDX" in df.columns:
        idx = pd.to_numeric(df["HEIGHT_CLASS_IDX"], errors="coerce")
        return idx.isna()

    value_cols = [c for c in ["TREE_HEIGHT_IN_METERS", "ESTIMATED_TREE_HEIGHT"] if c in df.columns]
    if value_cols:
        missing = pd.Series(True, index=df.index)
        for col in value_cols:
            missing &= text_missing(df[col])
        return missing

    if "TREE_HEIGHT_METHOD" in df.columns:
        return text_missing(df["TREE_HEIGHT_METHOD"])

    raise ValueError("Could not determine missing heights: no height target/measurement columns were found.")


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


def prepare_missing_height_rows(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, str, pd.Series, set[str]]:
    df = df.copy()
    input_mode = str(config.get("input_mode", "rgb"))
    image_source = str(config.get("image_source", "full"))
    use_dbh = get_use_dbh(config)
    use_species = bool(config.get("use_species", False))
    required_cols, rgb_col = get_required_input_columns(input_mode, image_source)

    for col in ["ORIGINAL_RGB_PATH", "RGB_CROP_PATH", "SAM_LOGITS_PATH", "SAM3_MASK_PATH", "DEPTH_PATH"]:
        if col not in df.columns:
            df[col] = np.nan

    row_missing = is_missing_height(df)

    # Infer only trees for which every image row lacks an observed height target.
    tree_has_observed = (~row_missing).groupby(df["ID"]).any()
    missing_tree_ids = set(tree_has_observed.index[~tree_has_observed])
    missing_mask = df["ID"].isin(missing_tree_ids)
    pred_df = df[missing_mask].copy()

    for col in required_cols:
        pred_df = pred_df[pred_df[col].notna() & pred_df[col].map(file_exists)].copy()

    if use_dbh:
        dbh_norm = pd.to_numeric(pred_df["DBH_NORM"], errors="coerce")
        source_valid = pred_df.get("DBH_SOURCE_VALID_ROW", pd.Series(True, index=pred_df.index)).astype(bool)
        pred_df = pred_df[source_valid & dbh_norm.notna() & np.isfinite(dbh_norm)].copy()

    if use_species:
        species_idx = pd.to_numeric(pred_df["SPECIES_INPUT_IDX"], errors="coerce")
        pred_df = pred_df[species_idx.notna() & np.isfinite(species_idx)].copy()

    # Training removes duplicate RGB paths; do the same before inference/aggregation.
    pred_df = pred_df.drop_duplicates(subset=[rgb_col], keep="first").copy()
    pred_df = pred_df.reset_index().rename(columns={"index": "orig_index"})
    return pred_df, rgb_col, missing_mask, missing_tree_ids


# ============================================================
# Height classes / aggregation
# ============================================================

def get_class_names(config: dict) -> list[str]:
    mapping = config.get("height_mapping") or config.get("height_class_mapping")
    if mapping:
        return [mapping[key] for key in sorted(mapping.keys(), key=lambda x: int(x))]
    return config.get("class_names", ["0-5", "5-10", "10-15", "15+"])


def aggregate_tree_predictions(pred_df: pd.DataFrame, class_names: list[str], conf_threshold: float | None) -> pd.DataFrame:
    """Match validation exactly: mean image probabilities per tree, then argmax."""
    rows = []

    for tree_id, group in pred_df.groupby("ID", dropna=False, sort=False):
        probs = np.vstack(group["PROBS_TREE_HEIGHT"].apply(json.loads).values).astype(float)
        agg_probs = np.mean(probs, axis=0)
        weights = np.full(len(group), 1.0 / len(group), dtype=float)
        agg_idx = int(np.argmax(agg_probs))
        agg_class = class_names[agg_idx]
        agg_conf = float(agg_probs[agg_idx])
        accepted = True if conf_threshold is None else agg_conf >= conf_threshold
        first = group.iloc[0]

        out = {
            "ID": tree_id,
            "N_IMAGES_USED": int(len(group)),
            "IMAGE_IDS_USED": json.dumps(group["IMAGE_ID"].astype(str).tolist()),
            "MEDIA_COLS_USED": json.dumps(group["MEDIA_COL"].astype(str).tolist()) if "MEDIA_COL" in group.columns else np.nan,
            "TREE_HEIGHT_METHOD_ORIGINAL": first.get("TREE_HEIGHT_METHOD", np.nan),
            "TREE_HEIGHT_IN_METERS_ORIGINAL": first.get("TREE_HEIGHT_IN_METERS", np.nan),
            "ESTIMATED_TREE_HEIGHT_ORIGINAL": first.get("ESTIMATED_TREE_HEIGHT", np.nan),
            "DBH_FEATURE_CM": first.get("DBH_FEATURE_CM", np.nan),
            "DBH_NORM": first.get("DBH_NORM", np.nan),
            "SPECIES_LABEL": first.get("SPECIES_LABEL", np.nan),
            "SPECIES_INPUT_IDX": first.get("SPECIES_INPUT_IDX", np.nan),
            "TREE_TYPE": first.get("TREE_TYPE", np.nan),
            "LATITUDE": first.get("LATITUDE", np.nan),
            "LONGITUDE": first.get("LONGITUDE", np.nan),
            "HEIGHT_CLASS_TREE_PRED_IDX": agg_idx,
            "HEIGHT_CLASS_TREE_PRED_STR": agg_class,
            "HEIGHT_CLASS_TREE_PRED_CONF": agg_conf,
            "HEIGHT_TREE_PRED_ACCEPTED": bool(accepted),
            "TREE_AGG_PROBS": json.dumps(agg_probs.tolist()),
            "TREE_AGG_WEIGHTS": json.dumps(weights.tolist()),
            "HEIGHT_SOURCE_TREE": "model_inferred" if accepted else "not_accepted",
        }

        for i, cls in enumerate(class_names):
            out[f"TREE_HEIGHT_PROB_{safe_class_name(cls)}"] = float(agg_probs[i])
        for col in ["MODEL_RUN_NAME", "MODEL_RUN_PATH", "INFERENCE_TIMESTAMP", "INPUT_MODE", "IMAGE_SOURCE"]:
            out[col] = first.get(col, np.nan)

        rows.append(out)

    return pd.DataFrame(rows)


# ============================================================
# Main inference
# ============================================================

def infer_missing_heights(
    raw_data_path: Path, dataset_dir: Path, run_path: Path,
    batch_size: int = 16, num_workers: int = 4,
    conf_threshold: float | None = None,
    width_manifest: Path | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_data_path, dataset_dir, run_path = Path(raw_data_path), Path(dataset_dir), Path(run_path)

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw CommuniMap file not found: {raw_data_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not run_path.exists():
        raise FileNotFoundError(f"Model run path not found: {run_path}")
    if conf_threshold is not None and not (0.0 <= conf_threshold <= 1.0):
        raise ValueError("--conf_threshold must be between 0 and 1.")

    # Load model/config first so preprocessing comes from the exact saved run.
    model, config, species_to_idx, _ = load_model_run(run_path, device)
    manifest_dir = dataset_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        Path(width_manifest).expanduser()
        if width_manifest is not None
        else resolve_manifest_path(dataset_dir, config)
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    print(f"\nUsing manifest: {manifest_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = run_path.name

    raw_df = read_raw_table(raw_data_path)
    raw_exploded_df = explode_media(raw_df)
    manifest_df = pd.read_csv(manifest_path)
    df = merge_manifest_with_raw(manifest_df, raw_exploded_df)

    # Ensure IDs are consistent before all tree-level mappings.
    df["ID"] = df["ID"].astype(str)

    use_dbh = get_use_dbh(config)
    use_species = bool(config.get("use_species", False))
    df = apply_saved_species_mapping(df, species_to_idx, use_species)
    df = apply_saved_dbh_transform(df, config)

    input_mode = str(config.get("input_mode", "rgb"))
    image_source = str(config.get("image_source", "full"))
    image_size = int(config.get("image_size", 224))
    class_names = get_class_names(config)
    idx_to_class = {i: label for i, label in enumerate(class_names)}

    pred_df, rgb_col, missing_mask, missing_tree_ids = prepare_missing_height_rows(df, config)
    required_cols, _ = get_required_input_columns(input_mode, image_source)

    print("\nHeight inference setup")
    print("----------------------")
    print(f"Raw data:              {raw_data_path}")
    print(f"Dataset:               {dataset_dir}")
    print(f"Model run:             {run_path}")
    print(f"Checkpoint:            {run_path / 'best_model.pth'}")
    print(f"Device:                {device}")
    print(f"Input mode:            {input_mode}")
    print(f"Image source:          {image_source}")
    print(f"Image size:            {image_size}")
    print(f"Use encoder:           {bool(config.get('use_encoder', False))}")
    print(f"Use species:           {use_species}")
    print(f"Use DBH:               {use_dbh}")
    print(f"Tree probability agg:  mean (matches validation)")
    print(f"Required image cols:   {required_cols}")

    print("\nRows / trees")
    print("------------")
    print(f"Manifest rows:                  {len(df)}")
    print(f"Manifest trees:                 {df['ID'].nunique()}")
    print(f"Trees with missing height:      {len(missing_tree_ids)}")
    print(f"Rows belonging to those trees:  {int(missing_mask.sum())}")
    print(f"Rows eligible for inference:    {len(pred_df)}")
    print(f"Trees eligible for inference:   {pred_df['ID'].nunique() if len(pred_df) else 0}")

    if len(pred_df) == 0:
        print("No rows eligible for inference.")
        return None, None, None

    infer_ds = TreeHeightInferenceDataset(
        pred_df, image_size=image_size, input_mode=input_mode, image_source=image_source,
        use_dbh=use_dbh, use_species=use_species,
    )
    infer_loader = DataLoader(
        infer_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_pred_idx, all_conf, all_probs = [], [], []
    model.eval()

    with torch.inference_mode():
        for x, dbh, species, _ in infer_loader:
            x = x.to(device, non_blocking=True)
            dbh = dbh.to(device, non_blocking=True)
            species = species.to(device, non_blocking=True)
            logits = model(x, dbh=dbh if use_dbh else None, species=species if use_species else None)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).cpu().numpy()
            conf = probs.max(dim=1).values.cpu().numpy()
            probs_np = probs.cpu().numpy()

            all_pred_idx.extend(pred_idx.tolist())
            all_conf.extend(conf.tolist())
            all_probs.extend(probs_np.tolist())

    probs_arr = np.asarray(all_probs, dtype=float)
    pred_df["HEIGHT_CLASS_PRED_IDX"] = all_pred_idx
    pred_df["HEIGHT_CLASS_PRED_STR"] = pred_df["HEIGHT_CLASS_PRED_IDX"].map(idx_to_class)
    pred_df["HEIGHT_CLASS_PRED_CONF"] = all_conf
    pred_df["PROBS_TREE_HEIGHT"] = [json.dumps(p) for p in all_probs]

    for i, cls in enumerate(class_names):
        pred_df[f"HEIGHT_PROB_{safe_class_name(cls)}"] = probs_arr[:, i]

    pred_df["HEIGHT_PRED_ACCEPTED"] = True if conf_threshold is None else pred_df["HEIGHT_CLASS_PRED_CONF"] >= conf_threshold
    pred_df["MODEL_RUN_PATH"] = str(run_path)
    pred_df["MODEL_RUN_NAME"] = run_tag
    pred_df["INFERENCE_TIMESTAMP"] = timestamp
    pred_df["INPUT_MODE"] = input_mode
    pred_df["IMAGE_SOURCE"] = image_source

    # Tree-level aggregation exactly as used during validation.
    tree_pred_df = aggregate_tree_predictions(pred_df, class_names, conf_threshold)

    # Full manifest with image-level predictions and tree-level aggregate result repeated on each tree row.
    df_full = df.copy()
    image_new_cols = [
        "HEIGHT_CLASS_PRED_IDX", "HEIGHT_CLASS_PRED_STR", "HEIGHT_CLASS_PRED_CONF", "PROBS_TREE_HEIGHT",
        "HEIGHT_PRED_ACCEPTED", "MODEL_RUN_PATH", "MODEL_RUN_NAME", "INFERENCE_TIMESTAMP", "INPUT_MODE", "IMAGE_SOURCE",
    ]
    for cls in class_names:
        image_new_cols.append(f"HEIGHT_PROB_{safe_class_name(cls)}")

    for col in image_new_cols:
        if col not in df_full.columns:
            df_full[col] = pd.NA if col in {"HEIGHT_CLASS_PRED_STR", "PROBS_TREE_HEIGHT", "MODEL_RUN_PATH", "MODEL_RUN_NAME", "INFERENCE_TIMESTAMP", "INPUT_MODE", "IMAGE_SOURCE", "HEIGHT_PRED_ACCEPTED"} else np.nan

    for _, row in pred_df.iterrows():
        i = int(row["orig_index"])
        for col in image_new_cols:
            df_full.loc[i, col] = row[col]

    df_full["HEIGHT_SOURCE_IMAGE"] = pd.Series(pd.NA, index=df_full.index, dtype="object")
    observed_mask = ~is_missing_height(df_full)
    df_full.loc[observed_mask, "HEIGHT_SOURCE_IMAGE"] = "observed"

    for _, row in pred_df.iterrows():
        i = int(row["orig_index"])
        df_full.loc[i, "HEIGHT_SOURCE_IMAGE"] = "model_inferred" if bool(row["HEIGHT_PRED_ACCEPTED"]) else "not_accepted"

    # Repeat the final tree-level result on all rows belonging to each inferred tree.
    tree_merge_cols = [
        "ID", "HEIGHT_CLASS_TREE_PRED_IDX", "HEIGHT_CLASS_TREE_PRED_STR", "HEIGHT_CLASS_TREE_PRED_CONF",
        "HEIGHT_TREE_PRED_ACCEPTED", "TREE_AGG_PROBS", "HEIGHT_SOURCE_TREE",
    ] + [f"TREE_HEIGHT_PROB_{safe_class_name(cls)}" for cls in class_names]

    tree_for_merge = tree_pred_df[tree_merge_cols].drop_duplicates("ID")
    existing_tree_cols = [c for c in tree_merge_cols if c != "ID" and c in df_full.columns]
    if existing_tree_cols:
        df_full = df_full.drop(columns=existing_tree_cols)
    df_full = df_full.merge(tree_for_merge, on="ID", how="left", validate="many_to_one")

    # Compact image-level table.
    compact_cols = [
        "orig_index", "ID", "IMAGE_ID", "MEDIA_COL", "MEDIA_SRC", rgb_col, "ORIGINAL_RGB_PATH", "RGB_CROP_PATH",
        "SAM_LOGITS_PATH", "SAM3_MASK_PATH", "DEPTH_PATH", "TREE_HEIGHT_METHOD", "TREE_HEIGHT_IN_METERS",
        "ESTIMATED_TREE_HEIGHT", "DBH_FEATURE_CM", "DBH_NORM", "DBH_CM_FINAL", "SPECIES_LABEL", "SPECIES_INPUT_IDX",
        "TREE_TYPE", "OTHER_TREE", "LATITUDE", "LONGITUDE", "HEIGHT_CLASS_PRED_IDX", "HEIGHT_CLASS_PRED_STR",
        "HEIGHT_CLASS_PRED_CONF", "HEIGHT_PRED_ACCEPTED", "PROBS_TREE_HEIGHT", "MODEL_RUN_NAME", "MODEL_RUN_PATH",
        "INFERENCE_TIMESTAMP", "INPUT_MODE", "IMAGE_SOURCE",
    ] + [f"HEIGHT_PROB_{safe_class_name(cls)}" for cls in class_names]
    compact_cols = [col for col in compact_cols if col in pred_df.columns]
    image_pred_df = pred_df[compact_cols].copy().rename(columns={"orig_index": "MANIFEST_ROW_INDEX"})

    # Put all outputs together in their own inference folder.
    inference_dir = manifest_dir / f"height_inference_{run_tag}_{timestamp}"
    inference_dir.mkdir(parents=True, exist_ok=True)

    full_out_path = inference_dir / f"tree_dataset_manifest_with_inferred_heights_{run_tag}_{timestamp}.csv"
    image_out_path = inference_dir / f"inferred_height_predictions_image_level_{run_tag}_{timestamp}.csv"
    tree_out_path = inference_dir / f"inferred_height_predictions_tree_level_mean_{run_tag}_{timestamp}.csv"
    summary_path = inference_dir / f"height_inference_summary_{run_tag}_{timestamp}.json"

    df_full.to_csv(full_out_path, index=False)
    image_pred_df.to_csv(image_out_path, index=False)
    tree_pred_df.to_csv(tree_out_path, index=False)

    summary = {
        "raw_data_path": str(raw_data_path),
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "model_run": str(run_path),
        "checkpoint": str(run_path / "best_model.pth"),
        "device": str(device),
        "input_mode": input_mode,
        "image_source": image_source,
        "image_size": image_size,
        "use_encoder": bool(config.get("use_encoder", False)),
        "use_dbh": use_dbh,
        "use_species": use_species,
        "tree_probability_aggregation": "mean",
        "confidence_threshold": conf_threshold,
        "manifest_rows": int(len(df)),
        "manifest_trees": int(df["ID"].nunique()),
        "trees_with_missing_height": int(len(missing_tree_ids)),
        "eligible_image_rows": int(len(pred_df)),
        "eligible_trees": int(pred_df["ID"].nunique()),
        "accepted_image_rows": int(image_pred_df["HEIGHT_PRED_ACCEPTED"].sum()),
        "accepted_tree_rows": int(tree_pred_df["HEIGHT_TREE_PRED_ACCEPTED"].sum()),
        "class_names": class_names,
        "outputs": {
            "full_manifest": str(full_out_path),
            "image_predictions": str(image_out_path),
            "tree_predictions": str(tree_out_path),
        },
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved outputs")
    print("-------------")
    print(f"Full manifest with inferred heights: {full_out_path}")
    print(f"Image-level predictions:             {image_out_path}")
    print(f"Tree-level mean predictions:         {tree_out_path}")
    print(f"Summary:                             {summary_path}")

    print("\nPrediction totals")
    print("-----------------")
    print(f"Image rows predicted: {len(image_pred_df)}")
    print(f"Tree IDs predicted:   {len(tree_pred_df)}")
    print(f"Accepted image rows:  {int(image_pred_df['HEIGHT_PRED_ACCEPTED'].sum())}")
    print(f"Accepted tree rows:   {int(tree_pred_df['HEIGHT_TREE_PRED_ACCEPTED'].sum())}")

    return full_out_path, image_out_path, tree_out_path


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Infer missing TreeCo height classes using the exact architecture/preprocessing saved by a current height-classifier run."
    )
    parser.add_argument("--raw_data_path", required=True, help="Original CommuniMap CSV/XLSX containing height metadata.")
    parser.add_argument("--dataset_dir", required=True, help="Processed TreeCo dataset directory.")
    parser.add_argument("--run_path", required=True, help="Height-model run containing config.json and best_model.pth.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--conf_threshold", type=float, default=None, help="Optional minimum probability for accepting image/tree predictions.")
    parser.add_argument("--width_manifest",type=Path, default=None, help="Explicit manifest containing the DBH predictions to use.",)
    args = parser.parse_args()

    infer_missing_heights(
        raw_data_path=Path(args.raw_data_path), dataset_dir=Path(args.dataset_dir), run_path=Path(args.run_path),
        batch_size=args.batch_size, num_workers=args.num_workers, conf_threshold=args.conf_threshold, width_manifest=args.width_manifest,
    )


if __name__ == "__main__":
    main()
