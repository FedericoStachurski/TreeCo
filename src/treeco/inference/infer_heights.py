#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
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
# Constants
# =========================================================

INPUT_CHANNELS = {
    "rgb": 3,
    "rgb_depth": 4,
    "rgb_sam": 4,
    "rgb_sam_depth": 5,
    "rgb_sam3": 4,
    "rgb_sam3_depth": 5,
    "gray_sam3_overlay": 3,
}

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


# =========================================================
# General helpers
# =========================================================

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

    return Path(str(path)).exists()


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
    return (
        s.isna()
        | s.eq("")
        | s.str.lower().isin(["nan", "none", "null"])
    )


def find_column_case_insensitive(
    df: pd.DataFrame,
    requested: str,
) -> str | None:
    requested_lower = str(requested).lower()

    for col in df.columns:
        if str(col).lower() == requested_lower:
            return str(col)

    return None


def safe_class_name(cls: str) -> str:
    return (
        str(cls)
        .replace(">", "gt")
        .replace("<", "lt")
        .replace("+", "plus")
        .replace("-", "_")
        .replace(" ", "")
    )


def get_media_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if str(c).upper().startswith("MEDIA_")
    ]


# =========================================================
# Species helpers
# =========================================================

def infer_species_label(row: pd.Series) -> str:
    """
    Mirrors the species-input logic used during height training.
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

    tree_type = (
        clean_text(row["TREE_TYPE"])
        if "TREE_TYPE" in row.index
        else ""
    )

    other_tree = (
        clean_text(row["OTHER_TREE"])
        if "OTHER_TREE" in row.index
        else ""
    )

    if tree_type.lower() in {
        "other",
        "others",
        "other tree",
        "other_tree",
    }:
        if not is_unknownish(other_tree):
            return other_tree

        return "Unknown"

    if not is_unknownish(tree_type):
        return tree_type

    if not is_unknownish(other_tree):
        return other_tree

    return "Unknown"


def apply_saved_species_mapping(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Use the exact species vocabulary stored by training.

    config["species_mapping"] is:
        {"0": "Alder", ..., "15": "Rare/Other", ...}

    We reverse it and map unseen/unknown species to Rare/Other
    when that category exists.
    """

    df = df.copy()

    use_species = bool(config.get("use_species", False))

    if not use_species:
        df["SPECIES_LABEL"] = "Unused"
        df["SPECIES_INPUT_IDX"] = 0
        return df

    saved_mapping = config.get("species_mapping")

    if not saved_mapping:
        raise ValueError(
            "Model was trained with species input, but "
            "config.json does not contain species_mapping."
        )

    idx_to_label = {
        int(idx): clean_text(label)
        for idx, label in saved_mapping.items()
    }

    label_to_idx = {
        label: idx
        for idx, label in idx_to_label.items()
    }

    casefold_to_idx = {
        label.casefold(): idx
        for label, idx in label_to_idx.items()
    }

    rare_idx = casefold_to_idx.get("rare/other")
    unknown_idx = casefold_to_idx.get("unknown")

    if rare_idx is None and unknown_idx is None:
        fallback_idx = min(idx_to_label)
    else:
        fallback_idx = (
            rare_idx
            if rare_idx is not None
            else unknown_idx
        )

    raw_labels = df.apply(
        infer_species_label,
        axis=1,
    ).apply(clean_text)

    mapped_labels = []
    mapped_indices = []

    for label in raw_labels:
        if is_unknownish(label):
            if unknown_idx is not None:
                idx = unknown_idx
            else:
                idx = fallback_idx
        elif label in label_to_idx:
            idx = label_to_idx[label]
        elif label.casefold() in casefold_to_idx:
            idx = casefold_to_idx[label.casefold()]
        else:
            idx = fallback_idx

        mapped_indices.append(int(idx))
        mapped_labels.append(idx_to_label[int(idx)])

    df["SPECIES_LABEL"] = mapped_labels
    df["SPECIES_INPUT_IDX"] = mapped_indices

    return df


# =========================================================
# DBH helpers
# =========================================================

def apply_saved_dbh_transform(
    df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Apply the exact DBH preprocessing saved during training.

    For the current model this means:
      DBH_CM_FINAL -> log1p -> z-score
    using config["dbh_mean"] and config["dbh_std"].
    """

    df = df.copy()

    use_dbh = bool(config.get("use_DBH", False))

    if not use_dbh:
        df["DBH_FEATURE_CM"] = 0.0
        df["DBH_NORM"] = 0.0
        return df

    requested_col = config.get("dbh_source_col")

    if not requested_col:
        raise ValueError(
            "Model was trained with DBH input, but config.json "
            "does not contain dbh_source_col."
        )

    source_col = find_column_case_insensitive(
        df,
        requested_col,
    )

    if source_col is None:
        raise ValueError(
            f"DBH input column '{requested_col}' was not found "
            f"in the selected manifest.\n"
            f"Available columns include:\n{list(df.columns)}"
        )

    values = pd.to_numeric(
        df[source_col],
        errors="coerce",
    )

    source_is_circumference = bool(
        config.get(
            "dbh_source_is_circumference",
            False,
        )
    )

    if source_is_circumference:
        values = values / math.pi

    df["DBH_FEATURE_CM"] = values

    transform = config.get(
        "dbh_transform",
        "zscore",
    )

    mean = config.get("dbh_mean")
    std = config.get("dbh_std")

    if mean is None or std is None:
        raise ValueError(
            "DBH model requires saved dbh_mean and dbh_std "
            "in config.json."
        )

    mean = float(mean)
    std = float(std)

    if abs(std) < 1e-8:
        std = 1.0

    transformed = values.to_numpy(dtype=float)

    valid = np.isfinite(transformed) & (transformed > 0)

    out = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    if transform == "log1p_zscore":
        out[valid] = (
            np.log1p(transformed[valid]) - mean
        ) / std

    elif transform == "zscore":
        out[valid] = (
            transformed[valid] - mean
        ) / std

    else:
        raise ValueError(
            f"Unknown saved DBH transform: {transform}"
        )

    df["DBH_NORM"] = out

    print(
        f"[DBH] Source column: {source_col}"
        + (
            " (circumference converted to DBH)"
            if source_is_circumference
            else ""
        )
    )
    print(f"[DBH] Transform: {transform}")
    print(f"[DBH] Training mean: {mean}")
    print(f"[DBH] Training std:  {std}")
    print(
        f"[DBH] Valid DBH rows: "
        f"{int(np.isfinite(df['DBH_NORM']).sum())} / {len(df)}"
    )

    return df


# =========================================================
# Height model
# Exact architecture used during training
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

        model_builders = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
        }

        if backbone not in model_builders:
            raise ValueError(
                f"Unsupported backbone: {backbone}"
            )

        # weights=None is intentional at inference:
        # best_model.pth contains all learned parameters.
        resnet = model_builders[backbone](
            weights=None
        )

        if (
            in_channels != 3
            or input_mode == "gray_sam3_overlay"
        ):
            old_conv = resnet.conv1

            resnet.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )

        image_feature_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()

        self.image_encoder = resnet
        self.use_dbh = use_dbh
        self.use_species = use_species

        extra_dim = 0

        if use_dbh:
            self.dbh_encoder = nn.Sequential(
                nn.Linear(
                    1,
                    dbh_hidden_dim,
                ),
                nn.ReLU(inplace=True),
                nn.Dropout(
                    p=dropout_rate
                ),
                nn.Linear(
                    dbh_hidden_dim,
                    dbh_hidden_dim,
                ),
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

        classifier_in = (
            image_feature_dim
            + extra_dim
        )

        self.classifier = nn.Sequential(
            nn.Dropout(
                p=dropout_rate
            ),
            nn.Linear(
                classifier_in,
                num_height_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        dbh: torch.Tensor | None = None,
        species: torch.Tensor | None = None,
    ) -> torch.Tensor:

        features = [
            self.image_encoder(x)
        ]

        if self.use_dbh:
            if dbh is None:
                raise ValueError(
                    "use_dbh=True but dbh is None."
                )

            features.append(
                self.dbh_encoder(dbh)
            )

        if self.use_species:
            if species is None:
                raise ValueError(
                    "use_species=True but species is None."
                )

            features.append(
                self.species_embedding(species)
            )

        features = torch.cat(
            features,
            dim=1,
        )

        return self.classifier(
            features
        )


def _strip_module_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:

    if any(
        key.startswith("module.")
        for key in state_dict
    ):
        return {
            key.replace(
                "module.",
                "",
                1,
            ): value
            for key, value in state_dict.items()
        }

    return state_dict


def load_model_run(
    run_path: Path,
    device: torch.device,
    config: dict | None = None,
):
    run_path = Path(run_path)

    config_path = (
        run_path
        / "config.json"
    )

    checkpoint_path = (
        run_path
        / "best_model.pth"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing config.json: {config_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing best_model.pth: "
            f"{checkpoint_path}"
        )

    if config is None:
        config = load_json(
            config_path
        )

    input_mode = config.get(
        "input_mode",
        "rgb",
    )

    if input_mode not in INPUT_CHANNELS:
        raise ValueError(
            f"Unsupported input_mode in config: "
            f"{input_mode}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    state_dict = _strip_module_prefix(
        state_dict
    )

    in_channels = int(
        config.get(
            "in_channels",
            INPUT_CHANNELS[input_mode],
        )
    )

    expected_channels = (
        INPUT_CHANNELS[input_mode]
    )

    if in_channels != expected_channels:
        raise ValueError(
            f"Config mismatch: "
            f"input_mode={input_mode} expects "
            f"{expected_channels} channels, "
            f"but config has "
            f"in_channels={in_channels}"
        )

    use_dbh = bool(
        config.get(
            "use_DBH",
            any(
                key.startswith(
                    "dbh_encoder."
                )
                for key in state_dict
            ),
        )
    )

    use_species = bool(
        config.get(
            "use_species",
            any(
                key.startswith(
                    "species_embedding."
                )
                for key in state_dict
            ),
        )
    )

    num_height_classes = int(
        config.get(
            "num_height_classes",
            checkpoint.get(
                "num_height_classes",
                4,
            ),
        )
    )

    if (
        "classifier.1.weight"
        in state_dict
    ):
        num_height_classes = int(
            state_dict[
                "classifier.1.weight"
            ].shape[0]
        )

    if use_species:
        if (
            "species_embedding.weight"
            in state_dict
        ):
            num_species = int(
                state_dict[
                    "species_embedding.weight"
                ].shape[0]
            )

            species_embedding_dim = int(
                state_dict[
                    "species_embedding.weight"
                ].shape[1]
            )

        else:
            num_species = int(
                config.get(
                    "num_species_inputs",
                    len(
                        config.get(
                            "species_mapping",
                            {},
                        )
                    ),
                )
            )

            species_embedding_dim = int(
                config.get(
                    "species_embedding_dim",
                    16,
                )
            )
    else:
        num_species = 1
        species_embedding_dim = 1

    dbh_hidden_dim = int(
        config.get(
            "dbh_hidden_dim",
            32,
        )
    )

    if (
        use_dbh
        and "dbh_encoder.0.weight"
        in state_dict
    ):
        dbh_hidden_dim = int(
            state_dict[
                "dbh_encoder.0.weight"
            ].shape[0]
        )

    dropout_rate = float(
        config.get(
            "dropout_rate",
            0.1,
        )
    )

    backbone = config.get(
        "backbone",
        checkpoint.get(
            "backbone",
            "resnet18",
        ),
    )

    print("\nModel reconstruction")
    print("--------------------")
    print(
        f"Backbone:              {backbone}"
    )
    print(
        f"Input mode:            {input_mode}"
    )
    print(
        f"Input channels:        {in_channels}"
    )
    print(
        f"Height classes:        {num_height_classes}"
    )
    print(
        f"Use species:           {use_species}"
    )
    print(
        f"Species categories:    {num_species}"
    )
    print(
        f"Species embedding dim: "
        f"{species_embedding_dim}"
    )
    print(
        f"Use DBH:               {use_dbh}"
    )
    print(
        f"DBH hidden dim:        {dbh_hidden_dim}"
    )

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
    ).to(device)

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.eval()

    print(
        f"Loaded checkpoint: "
        f"{checkpoint_path}"
    )

    return model, config


# =========================================================
# Raw CommuniMap loader
# =========================================================

def read_raw_table(
    path: Path,
) -> pd.DataFrame:
    """
    Robust CommuniMap reader.

    The Sept 2026 export has 127 header fields but some
    rows contain 129 MEDIA fields. We preserve them by
    extending the MEDIA header and padding shorter rows.
    """

    path = Path(
        path
    ).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    suffix = (
        path.suffix.lower()
    )

    if suffix in {
        ".xlsx",
        ".xls",
    }:
        print(
            f"[DATA] Reading Excel: "
            f"{path}"
        )

        return pd.read_excel(
            path
        )

    if suffix != ".csv":
        raise ValueError(
            f"Unsupported raw file type: "
            f"{suffix}"
        )

    print(
        f"[DATA] Reading CommuniMap CSV: "
        f"{path}"
    )

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as f:

        reader = csv.reader(
            f,
            delimiter=";",
            quotechar='"',
        )

        header = next(
            reader
        )

        rows = list(
            reader
        )

    original_ncols = len(
        header
    )

    max_ncols = max(
        [len(header)]
        + [
            len(row)
            for row in rows
        ]
    )

    print(
        f"[DATA] Header columns: "
        f"{original_ncols}"
    )
    print(
        f"[DATA] Maximum row columns: "
        f"{max_ncols}"
    )

    if max_ncols > len(header):
        extra = (
            max_ncols
            - len(header)
        )

        print(
            f"[DATA] Found {extra} "
            f"additional media column(s)."
        )

        media_indices = []

        for col in header:
            col_str = str(
                col
            )

            if col_str.startswith(
                "MEDIA_2635_"
            ):
                try:
                    media_indices.append(
                        int(
                            col_str.split(
                                "_"
                            )[-1]
                        )
                    )
                except ValueError:
                    pass

        next_media_idx = (
            max(media_indices) + 1
            if media_indices
            else 0
        )

        for i in range(
            extra
        ):
            new_col = (
                f"MEDIA_2635_"
                f"{next_media_idx + i}"
            )

            print(
                f"[DATA] Adding column: "
                f"{new_col}"
            )

            header.append(
                new_col
            )

    padded_rows = []

    for row in rows:
        if len(row) < len(header):
            row = (
                row
                + [""] * (
                    len(header)
                    - len(row)
                )
            )

        padded_rows.append(
            row[:len(header)]
        )

    df = pd.DataFrame(
        padded_rows,
        columns=header,
    )

    print(
        f"[DATA] Loaded CommuniMap CSV: "
        f"{len(df)} rows × "
        f"{len(df.columns)} columns"
    )

    return df


def explode_media(
    df: pd.DataFrame,
) -> pd.DataFrame:

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

    keep_cols = [
        col
        for col in keep_cols
        if col in df.columns
    ]

    media_cols = get_media_cols(
        df
    )

    if "ID" not in df.columns:
        raise ValueError(
            "Raw CommuniMap data "
            "is missing ID."
        )

    if not media_cols:
        raise ValueError(
            "No MEDIA_* columns "
            "found in raw data."
        )

    rows = []

    for _, row in df[
        keep_cols
        + media_cols
    ].iterrows():

        base_id = str(
            row.get("ID")
        )

        for media_col in media_cols:
            value = row.get(
                media_col
            )

            if (
                pd.isna(value)
                or str(value).strip() == ""
            ):
                continue

            out = {
                col: row.get(col)
                for col in keep_cols
            }

            out["ID"] = base_id
            out["IMAGE_ID"] = (
                f"{base_id}_"
                f"{media_col}"
            )
            out["MEDIA_COL"] = (
                media_col
            )
            out["MEDIA_SRC"] = str(
                value
            )

            rows.append(
                out
            )

    return pd.DataFrame(
        rows
    )


def merge_manifest_with_raw(
    manifest_df: pd.DataFrame,
    raw_exploded_df: pd.DataFrame,
) -> pd.DataFrame:

    manifest_df = (
        manifest_df.copy()
    )

    raw_exploded_df = (
        raw_exploded_df.copy()
    )

    if "ID" not in manifest_df.columns:
        raise ValueError(
            "Manifest is missing ID."
        )

    if "IMAGE_ID" not in manifest_df.columns:
        raise ValueError(
            "Manifest is missing IMAGE_ID. "
            "Cannot merge safely."
        )

    if raw_exploded_df.empty:
        print(
            "WARNING: exploded raw data "
            "is empty."
        )

        return manifest_df

    manifest_df["ID"] = (
        manifest_df["ID"]
        .astype(str)
    )

    raw_exploded_df["ID"] = (
        raw_exploded_df["ID"]
        .astype(str)
    )

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

    raw_keep_cols = [
        col
        for col in raw_keep_cols
        if col in raw_exploded_df.columns
    ]

    merged = manifest_df.merge(
        raw_exploded_df[
            raw_keep_cols
        ],
        on=[
            "ID",
            "IMAGE_ID",
        ],
        how="left",
        suffixes=(
            "",
            "__RAW",
        ),
    )

    fill_cols = [
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

    for col in fill_cols:
        raw_col = (
            f"{col}__RAW"
        )

        if raw_col not in merged.columns:
            continue

        if col not in merged.columns:
            merged[col] = (
                merged[raw_col]
            )
        else:
            merged[col] = (
                merged[col].where(
                    merged[col].notna()
                    & (
                        merged[col]
                        .astype(str)
                        .str.strip()
                        != ""
                    ),
                    merged[raw_col],
                )
            )

    temp_cols = [
        col
        for col in merged.columns
        if col.endswith(
            "__RAW"
        )
    ]

    if temp_cols:
        merged = merged.drop(
            columns=temp_cols
        )

    return merged


# =========================================================
# Manifest selection
# =========================================================

def resolve_manifest_path(
    dataset_dir: Path,
    config: dict,
) -> Path:

    use_inferred_widths = bool(
        config.get(
            "use_inferred_widths",
            False,
        )
    )

    if use_inferred_widths:
        configured = config.get(
            "final_width_manifest_path"
        )

        if not configured:
            raise ValueError(
                "use_inferred_widths=True "
                "but config.json has no "
                "final_width_manifest_path."
            )

        manifest_path = Path(
            configured
        )

        if not manifest_path.is_absolute():
            candidate = (
                dataset_dir
                / manifest_path
            )

            if candidate.exists():
                manifest_path = candidate

    else:
        manifest_path = (
            dataset_dir
            / "manifests"
            / "tree_dataset_manifest.csv"
        )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: "
            f"{manifest_path}"
        )

    return manifest_path


# =========================================================
# Inference dataset
# =========================================================

class TreeHeightInferenceDataset(
    Dataset
):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int,
        input_mode: str,
        image_source: str,
        use_dbh: bool,
        use_species: bool,
    ):
        self.df = (
            df.reset_index(
                drop=True
            )
        )

        self.image_size = int(
            image_size
        )

        self.input_mode = (
            input_mode
        )

        self.image_source = (
            image_source
        )

        self.use_dbh = (
            use_dbh
        )

        self.use_species = (
            use_species
        )

        if input_mode not in INPUT_CHANNELS:
            raise ValueError(
                f"Unknown input_mode: "
                f"{input_mode}"
            )

        self.use_depth = (
            input_mode
            in {
                "rgb_depth",
                "rgb_sam_depth",
                "rgb_sam3_depth",
            }
        )

        self.use_sam = (
            input_mode
            in {
                "rgb_sam",
                "rgb_sam_depth",
            }
        )

        self.use_sam3 = (
            input_mode
            in {
                "rgb_sam3",
                "rgb_sam3_depth",
                "gray_sam3_overlay",
            }
        )

        if image_source == "crop":
            self.rgb_col = (
                "RGB_CROP_PATH"
            )
        elif image_source == "full":
            self.rgb_col = (
                "ORIGINAL_RGB_PATH"
            )
        else:
            raise ValueError(
                f"Unknown image_source: "
                f"{image_source}"
            )

    def __len__(self):
        return len(
            self.df
        )

    def _load_single_channel_image(
        self,
        path: str,
    ) -> Image.Image:

        try:
            arr = np.load(
                path
            ).astype(
                np.float32
            )

            if arr.ndim == 3:
                arr = np.squeeze(
                    arr
                )

            if arr.ndim != 2:
                raise ValueError(
                    f"Expected 2D array, "
                    f"got shape {arr.shape}"
                )

            amin = np.nanmin(
                arr
            )
            amax = np.nanmax(
                arr
            )

            arr = (
                arr - amin
            ) / (
                amax
                - amin
                + 1e-8
            )

            arr = np.nan_to_num(
                arr,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )

            return Image.fromarray(
                (
                    arr * 255
                ).astype(
                    np.uint8
                )
            ).convert("L")

        except Exception:
            return Image.fromarray(
                np.zeros(
                    (
                        self.image_size,
                        self.image_size,
                    ),
                    dtype=np.uint8,
                )
            ).convert("L")

    def __getitem__(
        self,
        idx: int,
    ):
        row = self.df.iloc[
            idx
        ]

        rgb = Image.open(
            row[self.rgb_col]
        ).convert(
            "RGB"
        )

        single_channels = []

        if self.use_sam:
            single_channels.append(
                self._load_single_channel_image(
                    row[
                        "SAM_LOGITS_PATH"
                    ]
                )
            )

        if self.use_sam3:
            single_channels.append(
                self._load_single_channel_image(
                    row[
                        "SAM3_MASK_PATH"
                    ]
                )
            )

        if self.use_depth:
            single_channels.append(
                self._load_single_channel_image(
                    row[
                        "DEPTH_PATH"
                    ]
                )
            )

        # Validation/inference:
        # resize only, no augmentation.
        rgb = TF.resize(
            rgb,
            [
                self.image_size,
                self.image_size,
            ],
        )

        single_channels = [
            TF.resize(
                ch,
                [
                    self.image_size,
                    self.image_size,
                ],
            )
            for ch in single_channels
        ]

        if (
            self.input_mode
            == "gray_sam3_overlay"
        ):
            sam3_img = (
                single_channels[0]
            )

            gray_img = (
                TF.rgb_to_grayscale(
                    rgb,
                    num_output_channels=1,
                )
            )

            gray_t = TF.to_tensor(
                gray_img
            ).float()

            sam3_t = TF.to_tensor(
                sam3_img
            ).float()

            sam3_t = torch.clamp(
                sam3_t,
                0.0,
                1.0,
            )

            overlay_t = (
                gray_t
                * sam3_t
            )

            x = torch.cat(
                [
                    gray_t,
                    sam3_t,
                    overlay_t,
                ],
                dim=0,
            )

            x = TF.normalize(
                x,
                mean=[
                    0.5,
                    0.5,
                    0.5,
                ],
                std=[
                    0.25,
                    0.25,
                    0.25,
                ],
            )

        else:
            rgb_tensor = (
                TF.to_tensor(
                    rgb
                )
            )

            rgb_tensor = (
                TF.normalize(
                    rgb_tensor,
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                )
            )

            channels = [
                rgb_tensor
            ]

            for ch in single_channels:
                channels.append(
                    TF.to_tensor(
                        ch
                    ).to(
                        dtype=(
                            rgb_tensor.dtype
                        )
                    )
                )

            x = torch.cat(
                channels,
                dim=0,
            )

        if self.use_dbh:
            dbh = torch.tensor(
                [
                    float(
                        row[
                            "DBH_NORM"
                        ]
                    )
                ],
                dtype=torch.float32,
            )
        else:
            dbh = torch.zeros(
                1,
                dtype=torch.float32,
            )

        if self.use_species:
            species = torch.tensor(
                int(
                    row[
                        "SPECIES_INPUT_IDX"
                    ]
                ),
                dtype=torch.long,
            )
        else:
            species = torch.tensor(
                0,
                dtype=torch.long,
            )

        return (
            x,
            dbh,
            species,
            idx,
        )


# =========================================================
# Inference row preparation
# =========================================================

def is_missing_height(
    df: pd.DataFrame,
) -> pd.Series:

    if (
        "TREE_HEIGHT_METHOD"
        not in df.columns
    ):
        raise ValueError(
            "Missing TREE_HEIGHT_METHOD "
            "after raw-data merge."
        )

    return text_missing(
        df[
            "TREE_HEIGHT_METHOD"
        ]
    )


def get_required_input_columns(
    input_mode: str,
    image_source: str,
) -> tuple[list[str], str]:

    if image_source == "crop":
        rgb_col = (
            "RGB_CROP_PATH"
        )
    elif image_source == "full":
        rgb_col = (
            "ORIGINAL_RGB_PATH"
        )
    else:
        raise ValueError(
            f"Unknown image_source: "
            f"{image_source}"
        )

    required_cols = [
        rgb_col
    ]

    if input_mode in {
        "rgb_depth",
        "rgb_sam_depth",
        "rgb_sam3_depth",
    }:
        required_cols.append(
            "DEPTH_PATH"
        )

    if input_mode in {
        "rgb_sam",
        "rgb_sam_depth",
    }:
        required_cols.append(
            "SAM_LOGITS_PATH"
        )

    if input_mode in {
        "rgb_sam3",
        "rgb_sam3_depth",
        "gray_sam3_overlay",
    }:
        required_cols.append(
            "SAM3_MASK_PATH"
        )

    return (
        required_cols,
        rgb_col,
    )


def prepare_missing_height_rows(
    df: pd.DataFrame,
    config: dict,
) -> tuple[
    pd.DataFrame,
    str,
    pd.Series,
]:

    df = df.copy()

    input_mode = config.get(
        "input_mode",
        "rgb",
    )

    image_source = config.get(
        "image_source",
        "full",
    )

    use_dbh = bool(
        config.get(
            "use_DBH",
            False,
        )
    )

    use_species = bool(
        config.get(
            "use_species",
            False,
        )
    )

    required_cols, rgb_col = (
        get_required_input_columns(
            input_mode=input_mode,
            image_source=image_source,
        )
    )

    optional_cols = [
        "ORIGINAL_RGB_PATH",
        "RGB_CROP_PATH",
        "SAM_LOGITS_PATH",
        "SAM3_MASK_PATH",
        "DEPTH_PATH",
    ]

    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan

    missing_mask = (
        is_missing_height(
            df
        )
    )

    pred_df = df[
        missing_mask
    ].copy()

    for col in required_cols:
        pred_df = pred_df[
            pred_df[col].notna()
            & pred_df[col].map(
                file_exists
            )
        ].copy()

    if use_dbh:
        dbh_norm = pd.to_numeric(
            pred_df[
                "DBH_NORM"
            ],
            errors="coerce",
        )

        pred_df = pred_df[
            dbh_norm.notna()
            & np.isfinite(
                dbh_norm
            )
        ].copy()

    if use_species:
        pred_df = pred_df[
            pred_df[
                "SPECIES_INPUT_IDX"
            ].notna()
        ].copy()

    pred_df = (
        pred_df
        .reset_index()
        .rename(
            columns={
                "index": "orig_index"
            }
        )
    )

    return (
        pred_df,
        rgb_col,
        missing_mask,
    )


# =========================================================
# Height class helpers
# =========================================================

def get_class_names(
    config: dict,
) -> list[str]:

    mapping = (
        config.get(
            "height_mapping"
        )
        or config.get(
            "height_class_mapping"
        )
    )

    if mapping:
        return [
            mapping[key]
            for key in sorted(
                mapping.keys(),
                key=lambda x: int(x),
            )
        ]

    return config.get(
        "class_names",
        [
            "0-5",
            "5-10",
            "10-15",
            "15+",
        ],
    )


# =========================================================
# Tree-level aggregation
# =========================================================

def aggregate_tree_predictions(
    pred_df: pd.DataFrame,
    class_names: list[str],
    conf_threshold: float | None,
) -> pd.DataFrame:

    rows = []

    for tree_id, group in pred_df.groupby(
        "ID",
        dropna=False,
    ):
        probs = np.vstack(
            group[
                "PROBS_TREE_HEIGHT"
            ]
            .apply(
                json.loads
            )
            .values
        )

        conf = (
            group[
                "HEIGHT_CLASS_PRED_CONF"
            ]
            .to_numpy(
                dtype=float
            )
        )

        quality = np.ones(
            len(group),
            dtype=float,
        )

        for col in [
            "DINO_SCORE",
            "SAM_SCORE",
            "SAM3_SCORE",
        ]:
            if col in group.columns:
                q = pd.to_numeric(
                    group[col],
                    errors="coerce",
                ).fillna(
                    0.0
                ).to_numpy(
                    dtype=float
                )

                quality *= np.clip(
                    q,
                    0.0,
                    1.0,
                )

        for col in [
            "DINO_USED_FULL_IMAGE_FALLBACK",
            "FULL_IMAGE_FALLBACK",
        ]:
            if col in group.columns:
                fallback = (
                    group[col]
                    .astype(str)
                    .str.lower()
                    .isin(
                        [
                            "true",
                            "1",
                            "yes",
                        ]
                    )
                    .to_numpy()
                )

                quality[
                    fallback
                ] *= 0.5

        raw_weights = (
            conf
            * quality
        )

        if (
            raw_weights.sum()
            <= 0
        ):
            weights = (
                np.ones(
                    len(group),
                    dtype=float,
                )
                / len(group)
            )
        else:
            weights = (
                raw_weights
                / raw_weights.sum()
            )

        agg_probs = (
            probs
            * weights[:, None]
        ).sum(
            axis=0
        )

        agg_idx = int(
            np.argmax(
                agg_probs
            )
        )

        agg_class = (
            class_names[
                agg_idx
            ]
        )

        agg_conf = float(
            agg_probs[
                agg_idx
            ]
        )

        accepted = (
            True
            if conf_threshold is None
            else (
                agg_conf
                >= conf_threshold
            )
        )

        first = (
            group.iloc[0]
        )

        out = {
            "ID": tree_id,
            "N_IMAGES_USED": int(
                len(group)
            ),
            "IMAGE_IDS_USED": json.dumps(
                group[
                    "IMAGE_ID"
                ]
                .astype(str)
                .tolist()
            ),
            "MEDIA_COLS_USED": (
                json.dumps(
                    group[
                        "MEDIA_COL"
                    ]
                    .astype(str)
                    .tolist()
                )
                if (
                    "MEDIA_COL"
                    in group.columns
                )
                else np.nan
            ),
            "TREE_HEIGHT_METHOD_ORIGINAL": first.get(
                "TREE_HEIGHT_METHOD",
                np.nan,
            ),
            "TREE_HEIGHT_IN_METERS_ORIGINAL": first.get(
                "TREE_HEIGHT_IN_METERS",
                np.nan,
            ),
            "ESTIMATED_TREE_HEIGHT_ORIGINAL": first.get(
                "ESTIMATED_TREE_HEIGHT",
                np.nan,
            ),
            "DBH_FEATURE_CM": first.get(
                "DBH_FEATURE_CM",
                np.nan,
            ),
            "DBH_NORM": first.get(
                "DBH_NORM",
                np.nan,
            ),
            "SPECIES_LABEL": first.get(
                "SPECIES_LABEL",
                np.nan,
            ),
            "SPECIES_INPUT_IDX": first.get(
                "SPECIES_INPUT_IDX",
                np.nan,
            ),
            "TREE_TYPE": first.get(
                "TREE_TYPE",
                np.nan,
            ),
            "LATITUDE": first.get(
                "LATITUDE",
                np.nan,
            ),
            "LONGITUDE": first.get(
                "LONGITUDE",
                np.nan,
            ),
            "HEIGHT_CLASS_TREE_PRED_IDX": (
                agg_idx
            ),
            "HEIGHT_CLASS_TREE_PRED_STR": (
                agg_class
            ),
            "HEIGHT_CLASS_TREE_PRED_CONF": (
                agg_conf
            ),
            "HEIGHT_TREE_PRED_ACCEPTED": (
                bool(
                    accepted
                )
            ),
            "TREE_AGG_PROBS": json.dumps(
                agg_probs.tolist()
            ),
            "TREE_AGG_WEIGHTS": json.dumps(
                weights.tolist()
            ),
            "HEIGHT_SOURCE_TREE": (
                "model_inferred"
                if accepted
                else "not_accepted"
            ),
        }

        for i, cls in enumerate(
            class_names
        ):
            out[
                f"TREE_HEIGHT_PROB_"
                f"{safe_class_name(cls)}"
            ] = float(
                agg_probs[i]
            )

        for col in [
            "MODEL_RUN_NAME",
            "MODEL_RUN_PATH",
            "INFERENCE_TIMESTAMP",
            "INPUT_MODE",
            "IMAGE_SOURCE",
        ]:
            out[col] = first.get(
                col,
                np.nan,
            )

        rows.append(
            out
        )

    return pd.DataFrame(
        rows
    )


# =========================================================
# Main inference
# =========================================================

def infer_missing_heights(
    raw_data_path: Path,
    dataset_dir: Path,
    run_path: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    conf_threshold: float | None = None,
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    raw_data_path = Path(
        raw_data_path
    )

    dataset_dir = Path(
        dataset_dir
    )

    run_path = Path(
        run_path
    )

    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw CommuniMap file "
            f"not found: "
            f"{raw_data_path}"
        )

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory "
            f"not found: "
            f"{dataset_dir}"
        )

    if not run_path.exists():
        raise FileNotFoundError(
            f"Model run path "
            f"not found: "
            f"{run_path}"
        )

    config_path = (
        run_path
        / "config.json"
    )

    config = load_json(
        config_path
    )

    manifest_dir = (
        dataset_dir
        / "manifests"
    )

    manifest_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = (
        resolve_manifest_path(
            dataset_dir,
            config,
        )
    )

    print(
        f"Using manifest: "
        f"{manifest_path}"
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    run_tag = (
        run_path.name
    )

    # -----------------------------------------------------
    # Read / merge metadata
    # -----------------------------------------------------

    raw_df = read_raw_table(
        raw_data_path
    )

    raw_exploded_df = (
        explode_media(
            raw_df
        )
    )

    manifest_df = pd.read_csv(
        manifest_path
    )

    df = merge_manifest_with_raw(
        manifest_df,
        raw_exploded_df,
    )

    # -----------------------------------------------------
    # Apply the same auxiliary inputs used in training
    # -----------------------------------------------------

    df = apply_saved_species_mapping(
        df,
        config,
    )

    df = apply_saved_dbh_transform(
        df,
        config,
    )

    # -----------------------------------------------------
    # Rebuild exact model and load checkpoint
    # -----------------------------------------------------

    model, config = load_model_run(
        run_path,
        device=device,
        config=config,
    )

    input_mode = config.get(
        "input_mode",
        "rgb",
    )

    image_source = config.get(
        "image_source",
        "full",
    )

    image_size = int(
        config.get(
            "image_size",
            224,
        )
    )

    use_dbh = bool(
        config.get(
            "use_DBH",
            False,
        )
    )

    use_species = bool(
        config.get(
            "use_species",
            False,
        )
    )

    class_names = get_class_names(
        config
    )

    idx_to_class = {
        i: label
        for i, label
        in enumerate(
            class_names
        )
    }

    # -----------------------------------------------------
    # Select rows with missing heights and valid inputs
    # -----------------------------------------------------

    (
        pred_df,
        rgb_col,
        missing_mask,
    ) = prepare_missing_height_rows(
        df,
        config,
    )

    required_cols, _ = (
        get_required_input_columns(
            input_mode,
            image_source,
        )
    )

    print("\nHeight inference setup")
    print("----------------------")
    print(
        f"Raw data:      "
        f"{raw_data_path}"
    )
    print(
        f"Dataset:       "
        f"{dataset_dir}"
    )
    print(
        f"Model run:     "
        f"{run_path}"
    )
    print(
        f"Checkpoint:    "
        f"{run_path / 'best_model.pth'}"
    )
    print(
        f"Device:        "
        f"{device}"
    )
    print(
        f"Input mode:    "
        f"{input_mode}"
    )
    print(
        f"Image source:  "
        f"{image_source}"
    )
    print(
        f"Image size:    "
        f"{image_size}"
    )
    print(
        f"Use species:   "
        f"{use_species}"
    )
    print(
        f"Use DBH:       "
        f"{use_dbh}"
    )
    print(
        f"Required image columns: "
        f"{required_cols}"
    )

    print("\nRows")
    print("----")
    print(
        f"Manifest rows:                  "
        f"{len(df)}"
    )
    print(
        f"Rows with missing height:       "
        f"{int(missing_mask.sum())}"
    )
    print(
        f"Rows eligible for inference:    "
        f"{len(pred_df)}"
    )

    if len(pred_df) == 0:
        print(
            "No rows eligible "
            "for inference."
        )

        return (
            None,
            None,
            None,
        )

    # -----------------------------------------------------
    # DataLoader
    # -----------------------------------------------------

    infer_ds = (
        TreeHeightInferenceDataset(
            pred_df,
            image_size=image_size,
            input_mode=input_mode,
            image_source=image_source,
            use_dbh=use_dbh,
            use_species=use_species,
        )
    )

    infer_loader = DataLoader(
        infer_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    all_pred_idx = []
    all_conf = []
    all_probs = []

    model.eval()

    with torch.no_grad():
        for (
            x,
            dbh,
            species,
            _,
        ) in infer_loader:

            x = x.to(
                device,
                non_blocking=True,
            )

            dbh = dbh.to(
                device,
                non_blocking=True,
            )

            species = species.to(
                device,
                non_blocking=True,
            )

            logits = model(
                x,
                dbh=(
                    dbh
                    if use_dbh
                    else None
                ),
                species=(
                    species
                    if use_species
                    else None
                ),
            )

            probs = torch.softmax(
                logits,
                dim=1,
            )

            pred_idx = (
                probs
                .argmax(
                    dim=1
                )
                .cpu()
                .numpy()
            )

            conf = (
                probs
                .max(
                    dim=1
                )
                .values
                .cpu()
                .numpy()
            )

            probs_np = (
                probs
                .cpu()
                .numpy()
            )

            all_pred_idx.extend(
                pred_idx.tolist()
            )

            all_conf.extend(
                conf.tolist()
            )

            all_probs.extend(
                probs_np.tolist()
            )

    probs_arr = np.asarray(
        all_probs,
        dtype=float,
    )

    pred_df[
        "HEIGHT_CLASS_PRED_IDX"
    ] = all_pred_idx

    pred_df[
        "HEIGHT_CLASS_PRED_STR"
    ] = (
        pred_df[
            "HEIGHT_CLASS_PRED_IDX"
        ]
        .map(
            idx_to_class
        )
    )

    pred_df[
        "HEIGHT_CLASS_PRED_CONF"
    ] = all_conf

    pred_df[
        "PROBS_TREE_HEIGHT"
    ] = [
        json.dumps(
            p
        )
        for p
        in all_probs
    ]

    for i, cls in enumerate(
        class_names
    ):
        pred_df[
            f"HEIGHT_PROB_"
            f"{safe_class_name(cls)}"
        ] = probs_arr[:, i]

    if conf_threshold is None:
        pred_df[
            "HEIGHT_PRED_ACCEPTED"
        ] = True
    else:
        pred_df[
            "HEIGHT_PRED_ACCEPTED"
        ] = (
            pred_df[
                "HEIGHT_CLASS_PRED_CONF"
            ]
            >= conf_threshold
        )

    pred_df[
        "MODEL_RUN_PATH"
    ] = str(
        run_path
    )

    pred_df[
        "MODEL_RUN_NAME"
    ] = run_tag

    pred_df[
        "INFERENCE_TIMESTAMP"
    ] = timestamp

    pred_df[
        "INPUT_MODE"
    ] = input_mode

    pred_df[
        "IMAGE_SOURCE"
    ] = image_source

    # -----------------------------------------------------
    # Full manifest with image-level predictions
    # -----------------------------------------------------

    df_full = df.copy()

    new_cols = [
        "HEIGHT_CLASS_PRED_IDX",
        "HEIGHT_CLASS_PRED_STR",
        "HEIGHT_CLASS_PRED_CONF",
        "PROBS_TREE_HEIGHT",
        "HEIGHT_PRED_ACCEPTED",
        "MODEL_RUN_PATH",
        "MODEL_RUN_NAME",
        "INFERENCE_TIMESTAMP",
        "INPUT_MODE",
        "IMAGE_SOURCE",
    ]

    string_cols = {
        "HEIGHT_CLASS_PRED_STR",
        "PROBS_TREE_HEIGHT",
        "MODEL_RUN_PATH",
        "MODEL_RUN_NAME",
        "INFERENCE_TIMESTAMP",
        "INPUT_MODE",
        "IMAGE_SOURCE",
    }

    bool_cols = {
        "HEIGHT_PRED_ACCEPTED",
    }

    for col in new_cols:
        if col not in df_full.columns:
            if (
                col in string_cols
                or col in bool_cols
            ):
                df_full[col] = pd.Series(
                    pd.NA,
                    index=df_full.index,
                    dtype="object",
                )
            else:
                df_full[col] = np.nan

        elif (
            col in string_cols
            or col in bool_cols
        ):
            df_full[col] = (
                df_full[col]
                .astype(
                    "object"
                )
            )

    for cls in class_names:
        col = (
            f"HEIGHT_PROB_"
            f"{safe_class_name(cls)}"
        )

        if col not in df_full.columns:
            df_full[col] = np.nan

    for _, row in pred_df.iterrows():
        i = int(
            row[
                "orig_index"
            ]
        )

        for col in new_cols:
            df_full.loc[
                i,
                col,
            ] = row[col]

        for cls in class_names:
            col = (
                f"HEIGHT_PROB_"
                f"{safe_class_name(cls)}"
            )

            df_full.loc[
                i,
                col,
            ] = row[col]

    df_full[
        "HEIGHT_SOURCE_IMAGE"
    ] = pd.Series(
        pd.NA,
        index=df_full.index,
        dtype="object",
    )

    observed_mask = (
        ~is_missing_height(
            df_full
        )
    )

    df_full.loc[
        observed_mask,
        "HEIGHT_SOURCE_IMAGE",
    ] = "observed"

    for _, row in pred_df.iterrows():
        i = int(
            row[
                "orig_index"
            ]
        )

        if bool(
            row[
                "HEIGHT_PRED_ACCEPTED"
            ]
        ):
            source = (
                "model_inferred"
            )
        else:
            source = (
                "not_accepted"
            )

        df_full.loc[
            i,
            "HEIGHT_SOURCE_IMAGE",
        ] = source

    # -----------------------------------------------------
    # Compact image-level prediction table
    # -----------------------------------------------------

    compact_cols = [
        "orig_index",
        "ID",
        "IMAGE_ID",
        "MEDIA_COL",
        "MEDIA_SRC",
        rgb_col,
        "ORIGINAL_RGB_PATH",
        "RGB_CROP_PATH",
        "SAM_LOGITS_PATH",
        "SAM3_MASK_PATH",
        "DEPTH_PATH",
        "TREE_HEIGHT_METHOD",
        "TREE_HEIGHT_IN_METERS",
        "ESTIMATED_TREE_HEIGHT",
        "DBH_FEATURE_CM",
        "DBH_NORM",
        "DBH_CM_FINAL",
        "SPECIES_LABEL",
        "SPECIES_INPUT_IDX",
        "TREE_TYPE",
        "OTHER_TREE",
        "LATITUDE",
        "LONGITUDE",
        "HEIGHT_CLASS_PRED_IDX",
        "HEIGHT_CLASS_PRED_STR",
        "HEIGHT_CLASS_PRED_CONF",
        "HEIGHT_PRED_ACCEPTED",
        "PROBS_TREE_HEIGHT",
        "MODEL_RUN_NAME",
        "MODEL_RUN_PATH",
        "INFERENCE_TIMESTAMP",
        "INPUT_MODE",
        "IMAGE_SOURCE",
    ]

    compact_cols = [
        col
        for col in compact_cols
        if col in pred_df.columns
    ]

    image_pred_df = (
        pred_df[
            compact_cols
        ]
        .copy()
        .rename(
            columns={
                "orig_index":
                    "MANIFEST_ROW_INDEX"
            }
        )
    )

    # -----------------------------------------------------
    # Tree-level aggregation
    # -----------------------------------------------------

    tree_pred_df = (
        aggregate_tree_predictions(
            pred_df=pred_df,
            class_names=class_names,
            conf_threshold=(
                conf_threshold
            ),
        )
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    full_out_path = (
        manifest_dir
        / (
            "tree_dataset_manifest_"
            "with_image_level_"
            "inferred_heights_"
            f"{run_tag}_"
            f"{timestamp}.csv"
        )
    )

    image_out_path = (
        manifest_dir
        / (
            "inferred_height_"
            "predictions_image_level_"
            f"{run_tag}_"
            f"{timestamp}.csv"
        )
    )

    tree_out_path = (
        manifest_dir
        / (
            "inferred_height_"
            "predictions_tree_level_"
            "weighted_"
            f"{run_tag}_"
            f"{timestamp}.csv"
        )
    )

    df_full.to_csv(
        full_out_path,
        index=False,
    )

    image_pred_df.to_csv(
        image_out_path,
        index=False,
    )

    tree_pred_df.to_csv(
        tree_out_path,
        index=False,
    )

    print("\nSaved outputs")
    print("-------------")
    print(
        f"Full image-level manifest:       "
        f"{full_out_path}"
    )
    print(
        f"Compact image predictions:       "
        f"{image_out_path}"
    )
    print(
        f"Tree-level weighted predictions: "
        f"{tree_out_path}"
    )

    print("\nPrediction totals")
    print("-----------------")
    print(
        f"Image rows predicted: "
        f"{len(image_pred_df)}"
    )
    print(
        f"Tree IDs predicted:   "
        f"{len(tree_pred_df)}"
    )
    print(
        f"Accepted image rows:  "
        f"{int(image_pred_df['HEIGHT_PRED_ACCEPTED'].sum())}"
    )
    print(
        f"Accepted tree rows:   "
        f"{int(tree_pred_df['HEIGHT_TREE_PRED_ACCEPTED'].sum())}"
    )

    return (
        full_out_path,
        image_out_path,
        tree_out_path,
    )


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Infer missing TreeCo height classes using the "
            "same image + species + DBH architecture and "
            "preprocessing saved by the training run."
        )
    )

    parser.add_argument(
        "--raw_data_path",
        required=True,
        help=(
            "Original CommuniMap XLSX/CSV containing "
            "TREE_HEIGHT_METHOD and measurements."
        ),
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
        help=(
            "Processed TreeCo dataset directory."
        ),
    )

    parser.add_argument(
        "--run_path",
        required=True,
        help=(
            "Height-model run containing config.json "
            "and best_model.pth."
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=None,
        help=(
            "Optional minimum prediction confidence "
            "for accepting inferred height classes."
        ),
    )

    args = parser.parse_args()

    infer_missing_heights(
        raw_data_path=Path(
            args.raw_data_path
        ),
        dataset_dir=Path(
            args.dataset_dir
        ),
        run_path=Path(
            args.run_path
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conf_threshold=(
            args.conf_threshold
        ),
    )


if __name__ == "__main__":
    main()