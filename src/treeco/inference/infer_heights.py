#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


INPUT_CHANNELS = {
    "rgb": 3,
    "rgb_depth": 4,
    "rgb_sam": 4,
    "rgb_sam_depth": 5,
}

def get_media_cols(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if str(c).upper().startswith("MEDIA_")
    ]


# =========================================================
# Model
# =========================================================
def build_resnet_model(
    backbone: str,
    num_classes: int,
    in_channels: int,
    dropout_rate: float,
    device: torch.device,
) -> nn.Module:
    backbone = backbone.lower()

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

    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(model.fc.in_features, num_classes),
    )

    return model.to(device)


def load_model_run(run_path: Path, device: torch.device):
    run_path = Path(run_path)

    config_path = run_path / "config.json"
    checkpoint_path = run_path / "best_model.pth"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in model run: {config_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best_model.pth in model run: {checkpoint_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    input_mode = config.get("input_mode", "rgb")
    if input_mode not in INPUT_CHANNELS:
        raise ValueError(f"Unsupported input_mode in config: {input_mode}")

    in_channels = int(config.get("in_channels", INPUT_CHANNELS[input_mode]))
    expected_channels = INPUT_CHANNELS[input_mode]

    if in_channels != expected_channels:
        raise ValueError(
            f"Config mismatch: input_mode={input_mode} expects "
            f"{expected_channels} channels, but config has in_channels={in_channels}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = build_resnet_model(
        backbone=config.get("backbone", "resnet18"),
        num_classes=int(config["num_classes"]),
        in_channels=in_channels,
        dropout_rate=float(config.get("dropout_rate", 0.0)),
        device=device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, config


# =========================================================
# Raw data loading / media explosion
# =========================================================
def read_raw_table(path: Path) -> pd.DataFrame:
    path = Path(path)

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
        "LATITUDE",
        "LONGITUDE",
        "OTHER_TREE",
        "TREE_HEIGHT_METHOD",
        "TREE_HEIGHT_IN_METERS",
        "ESTIMATED_TREE_HEIGHT",
        "TREE_CIRCUMFERENCE_METHOD",
        "CIRCUMFERENCE_IN_CM",
        "TREE_TRUNK_SIZE",
    ]

    keep_cols = [c for c in keep_cols if c in df.columns]
    media_cols = get_media_cols(df)

    if not media_cols:
        raise ValueError(
            "No MEDIA_* columns found in raw data. "
            "Check the CommuniMap export column names."
        )

    base = df[keep_cols + media_cols].copy()
    rows = []

    for _, row in base.iterrows():
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

    out_df = pd.DataFrame(rows)

    if out_df.empty:
        return out_df

    tree_type = (
        out_df.get("TREE_TYPE", "")
        .fillna("")
        .astype(str)
        .str.strip()
    )

    if "OTHER_TREE" in out_df.columns:
        other_tree = (
            out_df["OTHER_TREE"]
            .fillna("")
            .astype(str)
            .str.strip()
        )
    else:
        other_tree = pd.Series("", index=out_df.index)

    out_df["TREE_TYPE_RAW"] = tree_type
    out_df["OTHER_TREE_RAW"] = other_tree

    out_df["TREE_SPECIES_LABEL"] = np.where(
        tree_type.str.lower().eq("other") & other_tree.ne(""),
        other_tree,
        tree_type,
    )

    out_df["TREE_SPECIES_LABEL"] = (
        out_df["TREE_SPECIES_LABEL"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    return out_df


def merge_manifest_with_raw(manifest_df: pd.DataFrame, raw_exploded_df: pd.DataFrame) -> pd.DataFrame:
    manifest_df = manifest_df.copy()
    raw_exploded_df = raw_exploded_df.copy()

    manifest_df["ID"] = manifest_df["ID"].astype(str)
    raw_exploded_df["ID"] = raw_exploded_df["ID"].astype(str)

    if "IMAGE_ID" not in manifest_df.columns:
        raise ValueError("Manifest is missing IMAGE_ID. Cannot merge safely.")

    if "IMAGE_ID" not in raw_exploded_df.columns:
        raise ValueError("Exploded raw table is missing IMAGE_ID. Cannot merge safely.")

    raw_keep_cols = [
        "ID",
        "IMAGE_ID",
        "TREE",
        "TREE_TYPE",
        "TREE_TYPE_RAW",
        "OTHER_TREE_RAW",
        "TREE_SPECIES_LABEL",
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

    merged = manifest_df.merge(
        raw_exploded_df[raw_keep_cols],
        on=["ID", "IMAGE_ID"],
        how="left",
        suffixes=("", "_RAW"),
    )

    return merged


# =========================================================
# Dataset
# =========================================================
class TreeHeightInferenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_size: int,
        input_mode: str,
        image_source: str,
    ):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.input_mode = input_mode
        self.image_source = image_source

        self.use_depth = input_mode in {"rgb_depth", "rgb_sam_depth"}
        self.use_sam = input_mode in {"rgb_sam", "rgb_sam_depth"}

        if image_source == "crop":
            self.rgb_col = "RGB_CROP_PATH"
        elif image_source == "full":
            self.rgb_col = "ORIGINAL_RGB_PATH"
        else:
            raise ValueError(f"Unknown image_source: {image_source}")

        self.rgb_tfms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.single_resize = transforms.Resize((image_size, image_size))
        self.single_to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.df)

    def _load_single_channel_npy(self, path: str, dtype: torch.dtype) -> torch.Tensor:
        try:
            arr = np.load(path).astype(np.float32)

            if arr.ndim == 3:
                arr = np.squeeze(arr)

            if arr.ndim != 2:
                raise ValueError(f"Expected 2D array, got {arr.shape}")

            amin = np.nanmin(arr)
            amax = np.nanmax(arr)

            arr = (arr - amin) / (amax - amin + 1e-8)
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)

            img = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")
            img = self.single_resize(img)

            tensor = self.single_to_tensor(img).to(dtype=dtype)

        except Exception:
            tensor = torch.zeros(
                1,
                self.image_size,
                self.image_size,
                dtype=dtype,
            )

        return tensor

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        rgb = Image.open(row[self.rgb_col]).convert("RGB")
        rgb_tensor = self.rgb_tfms(rgb)

        channels = [rgb_tensor]

        # Channel order is RGB + SAM + DEPTH.
        if self.use_sam:
            channels.append(
                self._load_single_channel_npy(
                    row["SAM_LOGITS_PATH"],
                    dtype=rgb_tensor.dtype,
                )
            )

        if self.use_depth:
            channels.append(
                self._load_single_channel_npy(
                    row["DEPTH_PATH"],
                    dtype=rgb_tensor.dtype,
                )
            )

        x = torch.cat(channels, dim=0)

        return x, idx


# =========================================================
# Helpers
# =========================================================
def get_class_names(config: dict) -> list[str]:
    mapping = config.get("height_class_mapping")

    if mapping is not None:
        return [mapping[k] for k in sorted(mapping.keys(), key=lambda x: int(x))]

    return config.get("class_names", ["0-5", "5-10", "10-15", "15+"])


def safe_class_name(cls: str) -> str:
    return (
        str(cls)
        .replace(">", "gt")
        .replace("<", "lt")
        .replace("+", "plus")
        .replace("-", "_")
        .replace(" ", "")
    )


def file_exists(path) -> bool:
    if pd.isna(path):
        return False
    return Path(str(path)).exists()


def text_missing(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    return s.isna() | s.eq("") | s.str.lower().isin(["nan", "none", "null"])


def is_missing_height(df: pd.DataFrame) -> pd.Series:
    if "TREE_HEIGHT_METHOD" not in df.columns:
        raise ValueError("Missing required column after merge: TREE_HEIGHT_METHOD")

    return text_missing(df["TREE_HEIGHT_METHOD"])


def get_required_input_columns(input_mode: str, image_source: str) -> tuple[list[str], str]:
    if image_source == "crop":
        rgb_col = "RGB_CROP_PATH"
    elif image_source == "full":
        rgb_col = "ORIGINAL_RGB_PATH"
    else:
        raise ValueError(f"Unknown image_source: {image_source}")

    required_cols = [rgb_col]

    if input_mode in {"rgb_depth", "rgb_sam_depth"}:
        required_cols.append("DEPTH_PATH")

    if input_mode in {"rgb_sam", "rgb_sam_depth"}:
        required_cols.append("SAM_LOGITS_PATH")

    return required_cols, rgb_col


def prepare_missing_height_rows(
    df: pd.DataFrame,
    input_mode: str,
    image_source: str,
) -> tuple[pd.DataFrame, str, pd.Series]:
    df = df.copy()

    required_cols, rgb_col = get_required_input_columns(
        input_mode=input_mode,
        image_source=image_source,
    )

    optional_cols = [
        "ORIGINAL_RGB_PATH",
        "RGB_CROP_PATH",
        "SAM_LOGITS_PATH",
        "DEPTH_PATH",
    ]

    for col in optional_cols:
        if col not in df.columns:
            df[col] = np.nan

    missing_mask = is_missing_height(df)

    pred_df = df[missing_mask].copy()

    for col in required_cols:
        pred_df = pred_df[
            pred_df[col].notna()
            & pred_df[col].map(file_exists)
        ].copy()

    pred_df = pred_df.reset_index().rename(columns={"index": "orig_index"})

    return pred_df, rgb_col, missing_mask


def aggregate_tree_predictions(
    pred_df: pd.DataFrame,
    class_names: list[str],
    conf_threshold: float | None,
) -> pd.DataFrame:
    rows = []

    for tree_id, group in pred_df.groupby("ID", dropna=False):
        probs = np.vstack(group["PROBS_TREE_HEIGHT"].apply(json.loads).values)
        conf = group["HEIGHT_CLASS_PRED_CONF"].to_numpy(dtype=float)

        quality = np.ones(len(group), dtype=float)

        if "DINO_SCORE" in group.columns:
            dino = pd.to_numeric(
                group["DINO_SCORE"],
                errors="coerce",
            ).fillna(0.0).to_numpy()
            quality *= np.clip(dino, 0.0, 1.0)

        if "SAM_SCORE" in group.columns:
            sam = pd.to_numeric(
                group["SAM_SCORE"],
                errors="coerce",
            ).fillna(0.0).to_numpy()
            quality *= np.clip(sam, 0.0, 1.0)

        if "DINO_USED_FULL_IMAGE_FALLBACK" in group.columns:
            fallback = (
                group["DINO_USED_FULL_IMAGE_FALLBACK"]
                .astype(str)
                .str.lower()
                .isin(["true", "1", "yes"])
                .to_numpy()
            )
            quality[fallback] *= 0.5

        raw_weights = conf * quality

        if raw_weights.sum() <= 0:
            weights = np.ones(len(group), dtype=float) / len(group)
        else:
            weights = raw_weights / raw_weights.sum()

        agg_probs = (probs * weights[:, None]).sum(axis=0)

        agg_idx = int(np.argmax(agg_probs))
        agg_class = class_names[agg_idx]
        agg_conf = float(agg_probs[agg_idx])

        accepted = True if conf_threshold is None else agg_conf >= conf_threshold
        first = group.iloc[0]

        out = {
            "ID": tree_id,
            "N_IMAGES_USED": int(len(group)),
            "IMAGE_IDS_USED": json.dumps(group["IMAGE_ID"].astype(str).tolist()),
            "MEDIA_COLS_USED": json.dumps(group["MEDIA_COL"].astype(str).tolist())
            if "MEDIA_COL" in group.columns else np.nan,
            "TREE_HEIGHT_METHOD_ORIGINAL": first.get("TREE_HEIGHT_METHOD", np.nan),
            "TREE_HEIGHT_IN_METERS_ORIGINAL": first.get("TREE_HEIGHT_IN_METERS", np.nan),
            "ESTIMATED_TREE_HEIGHT_ORIGINAL": first.get("ESTIMATED_TREE_HEIGHT", np.nan),
            "TREE_CIRCUMFERENCE_METHOD": first.get("TREE_CIRCUMFERENCE_METHOD", np.nan),
            "CIRCUMFERENCE_IN_CM": first.get("CIRCUMFERENCE_IN_CM", np.nan),
            "TREE_TRUNK_SIZE": first.get("TREE_TRUNK_SIZE", np.nan),
            "TREE_TYPE": first.get("TREE_TYPE", np.nan),
            "TREE_SPECIES_LABEL": first.get("TREE_SPECIES_LABEL", np.nan),
            "LATITUDE": first.get("LATITUDE", np.nan),
            "LONGITUDE": first.get("LONGITUDE", np.nan),
            "HEIGHT_CLASS_TREE_PRED_IDX": agg_idx,
            "HEIGHT_CLASS_TREE_PRED_STR": agg_class,
            "HEIGHT_CLASS_TREE_PRED_CONF": agg_conf,
            "HEIGHT_TREE_PRED_ACCEPTED": accepted,
            "TREE_AGG_PROBS": json.dumps(agg_probs.tolist()),
            "TREE_AGG_WEIGHTS": json.dumps(weights.tolist()),
            "HEIGHT_SOURCE_TREE": "model_inferred" if accepted else "not_accepted",
        }

        for i, cls in enumerate(class_names):
            out[f"TREE_HEIGHT_PROB_{safe_class_name(cls)}"] = float(agg_probs[i])

        for col in [
            "MODEL_RUN_NAME",
            "MODEL_RUN_PATH",
            "INFERENCE_TIMESTAMP",
            "INPUT_MODE",
            "IMAGE_SOURCE",
        ]:
            out[col] = first.get(col, np.nan)

        rows.append(out)

    return pd.DataFrame(rows)


# =========================================================
# Inference
# =========================================================
def infer_missing_heights(
    raw_data_path: Path,
    dataset_dir: Path,
    run_path: Path,
    batch_size: int = 16,
    num_workers: int = 4,
    conf_threshold: float | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw_data_path = Path(raw_data_path)
    dataset_dir = Path(dataset_dir)
    run_path = Path(run_path)

    manifest_dir = dataset_dir / "manifests"
    manifest_path = manifest_dir / "tree_dataset_manifest.csv"

    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw CommuniMap file not found: {raw_data_path}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    if not run_path.exists():
        raise FileNotFoundError(f"Model run path not found: {run_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = run_path.name

    raw_df = read_raw_table(raw_data_path)
    raw_exploded_df = explode_media(raw_df)

    manifest_df = pd.read_csv(manifest_path)
    df = merge_manifest_with_raw(manifest_df, raw_exploded_df)

    model, config = load_model_run(run_path, device=device)

    input_mode = config.get("input_mode", "rgb")
    image_source = config.get("image_source", "crop")
    image_size = int(config.get("image_size", 224))
    class_names = get_class_names(config)
    idx_to_class = {i: c for i, c in enumerate(class_names)}

    pred_df, rgb_col, missing_mask = prepare_missing_height_rows(
        df,
        input_mode=input_mode,
        image_source=image_source,
    )

    required_cols, _ = get_required_input_columns(input_mode, image_source)

    print(f"Raw data: {raw_data_path}")
    print(f"Dataset: {dataset_dir}")
    print(f"Model run: {run_path}")
    print(f"Device: {device}")
    print(f"Input mode: {input_mode}")
    print(f"Image source: {image_source}")
    print(f"Required input columns: {required_cols}")
    print(f"Rows in manifest: {len(df)}")
    print(f"Rows with missing TREE_HEIGHT_METHOD: {int(missing_mask.sum())}")
    print(f"Rows eligible for inference: {len(pred_df)}")

    if len(pred_df) == 0:
        print("No rows eligible for inference.")
        return None, None, None

    infer_ds = TreeHeightInferenceDataset(
        pred_df,
        image_size=image_size,
        input_mode=input_mode,
        image_source=image_source,
    )

    infer_loader = DataLoader(
        infer_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    all_pred_idx = []
    all_conf = []
    all_probs = []

    model.eval()

    with torch.no_grad():
        for x, _ in infer_loader:
            x = x.to(device, non_blocking=True)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            pred_idx = probs.argmax(dim=1).cpu().numpy()
            conf = probs.max(dim=1).values.cpu().numpy()
            probs_np = probs.cpu().numpy()

            all_pred_idx.extend(pred_idx.tolist())
            all_conf.extend(conf.tolist())
            all_probs.extend(probs_np.tolist())

    probs_arr = np.array(all_probs)

    pred_df["HEIGHT_CLASS_PRED_IDX"] = all_pred_idx
    pred_df["HEIGHT_CLASS_PRED_STR"] = pred_df["HEIGHT_CLASS_PRED_IDX"].map(idx_to_class)
    pred_df["HEIGHT_CLASS_PRED_CONF"] = all_conf
    pred_df["PROBS_TREE_HEIGHT"] = [json.dumps(p) for p in all_probs]

    for i, cls in enumerate(class_names):
        pred_df[f"HEIGHT_PROB_{safe_class_name(cls)}"] = probs_arr[:, i]

    if conf_threshold is None:
        pred_df["HEIGHT_PRED_ACCEPTED"] = True
    else:
        pred_df["HEIGHT_PRED_ACCEPTED"] = (
            pred_df["HEIGHT_CLASS_PRED_CONF"] >= conf_threshold
        )

    pred_df["MODEL_RUN_PATH"] = str(run_path)
    pred_df["MODEL_RUN_NAME"] = run_tag
    pred_df["INFERENCE_TIMESTAMP"] = timestamp
    pred_df["INPUT_MODE"] = input_mode
    pred_df["IMAGE_SOURCE"] = image_source

    # -----------------------------------------------------
    # Full image-level manifest with inferred values inserted
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
            if col in string_cols:
                df_full[col] = pd.Series(pd.NA, index=df_full.index, dtype="object")
            elif col in bool_cols:
                df_full[col] = pd.Series(pd.NA, index=df_full.index, dtype="object")
            else:
                df_full[col] = np.nan
        elif col in string_cols or col in bool_cols:
            df_full[col] = df_full[col].astype("object")

    for cls in class_names:
        col = f"HEIGHT_PROB_{safe_class_name(cls)}"
        if col not in df_full.columns:
            df_full[col] = np.nan

    for _, row in pred_df.iterrows():
        i = int(row["orig_index"])

        for col in new_cols:
            df_full.loc[i, col] = row[col]

        for cls in class_names:
            col = f"HEIGHT_PROB_{safe_class_name(cls)}"
            df_full.loc[i, col] = row[col]

    df_full["HEIGHT_SOURCE_IMAGE"] = pd.Series(pd.NA, index=df_full.index, dtype="object")
    df_full.loc[~is_missing_height(df_full), "HEIGHT_SOURCE_IMAGE"] = "observed"

    for _, row in pred_df.iterrows():
        i = int(row["orig_index"])
        if bool(row["HEIGHT_PRED_ACCEPTED"]):
            df_full.loc[i, "HEIGHT_SOURCE_IMAGE"] = "model_inferred"
        else:
            df_full.loc[i, "HEIGHT_SOURCE_IMAGE"] = "not_accepted"

    # -----------------------------------------------------
    # Compact image-level table
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
        "DEPTH_PATH",
        "TREE_HEIGHT_METHOD",
        "TREE_HEIGHT_IN_METERS",
        "ESTIMATED_TREE_HEIGHT",
        "TREE_CIRCUMFERENCE_METHOD",
        "CIRCUMFERENCE_IN_CM",
        "TREE_TRUNK_SIZE",
        "TREE_TYPE",
        "TREE_SPECIES_LABEL",
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

    compact_cols = [c for c in compact_cols if c in pred_df.columns]

    image_pred_df = pred_df[compact_cols].copy()
    image_pred_df = image_pred_df.rename(columns={"orig_index": "MANIFEST_ROW_INDEX"})

    # -----------------------------------------------------
    # Tree-level aggregation
    # -----------------------------------------------------
    tree_pred_df = aggregate_tree_predictions(
        pred_df=pred_df,
        class_names=class_names,
        conf_threshold=conf_threshold,
    )

    full_out_path = manifest_dir / (
        f"tree_dataset_manifest_with_image_level_inferred_heights_{run_tag}_{timestamp}.csv"
    )

    image_out_path = manifest_dir / (
        f"inferred_height_predictions_image_level_{run_tag}_{timestamp}.csv"
    )

    tree_out_path = manifest_dir / (
        f"inferred_height_predictions_tree_level_weighted_{run_tag}_{timestamp}.csv"
    )

    df_full.to_csv(full_out_path, index=False)
    image_pred_df.to_csv(image_out_path, index=False)
    tree_pred_df.to_csv(tree_out_path, index=False)

    print("\nSaved outputs:")
    print(f"Full image-level manifest:      {full_out_path}")
    print(f"Compact image-level predictions:{image_out_path}")
    print(f"Tree-level weighted predictions:{tree_out_path}")
    print(f"Image rows predicted: {len(image_pred_df)}")
    print(f"Tree IDs predicted:   {len(tree_pred_df)}")
    print(f"Accepted image rows:  {int(image_pred_df['HEIGHT_PRED_ACCEPTED'].sum())}")
    print(f"Accepted tree rows:   {int(tree_pred_df['HEIGHT_TREE_PRED_ACCEPTED'].sum())}")

    return full_out_path, image_out_path, tree_out_path


# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Infer missing TreeCo height classes using a trained model run. "
            "Uses original CommuniMap file + processed TreeCo manifest."
        )
    )

    parser.add_argument(
        "--raw_data_path",
        required=True,
        help="Original CommuniMap XLSX/CSV file containing TREE_HEIGHT_METHOD and measurements.",
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Processed TreeCo dataset folder containing manifests/tree_dataset_manifest.csv.",
    )

    parser.add_argument(
        "--run_path",
        required=True,
        help="Trained model run folder containing config.json and best_model.pth.",
    )

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=None,
        help="Optional confidence threshold for accepting predictions.",
    )

    args = parser.parse_args()

    infer_missing_heights(
        raw_data_path=Path(args.raw_data_path),
        dataset_dir=Path(args.dataset_dir),
        run_path=Path(args.run_path),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    main()