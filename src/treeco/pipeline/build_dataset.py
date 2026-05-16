#!/usr/bin/env python3
"""
Build a local TreeCo training dataset from a CommuniMap export.

DINO is always used to find/crop the tree.
RGB crops, SAM logits, and Depth Anything maps are saved only if requested.

Outputs:
- manifests/tree_dataset_manifest.csv
    Final kept/trainable image-level manifest.

- manifests/tree_dataset_all_attempted_rows.csv
    Audit log of all image rows attempted, including failures and CLIP rejects.

- manifests/tree_level_summary.csv
    Tree-level summary grouped by original CommuniMap ID.

- manifests/summary.json
    Dataset-level metadata and counts.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from treeco.image_models.download import get_model_path
from treeco.image_models.sam import infer_sam_mask, load_sam
from treeco.utils import CLIP_filtering_system
from treeco.utils import depth_anything
from treeco.utils import groundingdino_box_cropping


# ---------------------------------------------------------------------
# Basic IO / cleaning helpers
# ---------------------------------------------------------------------


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type: {path}")


def clean_missing(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, str):
        x = x.strip()
        if x.lower() in {"", "nan", "none", "null", "n/a", "na"}:
            return np.nan

    return x


def clean_id(x) -> str:
    """Keep CommuniMap entry IDs stable, avoiding Excel-style 123.0 IDs."""
    if pd.isna(x):
        return ""

    if isinstance(x, (int, np.integer)):
        return str(x)

    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return str(int(x))

    return str(x).strip()


def extract_numeric(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    m = re.search(r"[-+]?\d*\.?\d+", str(x).replace(",", "."))
    return float(m.group()) if m else np.nan


def make_image_uuid(id_val, media_src: str) -> str:
    """Stable image identifier across rebuilds, based on original ID + URL."""
    s = f"{clean_id(id_val)}::{str(media_src).strip()}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def get_media_columns(df: pd.DataFrame) -> list[str]:
    """Detect CommuniMap media columns dynamically instead of hard-coding MEDIA_2635_* only."""
    return sorted([c for c in df.columns if str(c).startswith("MEDIA_")])


# ---------------------------------------------------------------------
# Raw CommuniMap image explosion and labels
# ---------------------------------------------------------------------


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
    media_cols = get_media_columns(df)

    if not media_cols:
        return pd.DataFrame()

    base = df[keep_cols + media_cols].copy()
    rows = []

    for _, row in base.iterrows():
        base_id = clean_id(row.get("ID"))

        for media_col in media_cols:
            val = row.get(media_col)

            if pd.isna(val) or str(val).strip() == "":
                continue

            media_src = str(val).strip()

            out = {k: row.get(k) for k in keep_cols}
            out["ID"] = base_id
            out["IMAGE_ID"] = f"{base_id}_{media_col}"
            out["IMAGE_UUID"] = make_image_uuid(base_id, media_src)
            out["MEDIA_COL"] = media_col
            out["MEDIA_SRC"] = media_src

            rows.append(out)

    out_df = pd.DataFrame(rows)

    if not out_df.empty:
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


def add_height_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "TREE_HEIGHT_IN_METERS" in df.columns:
        df["TREE_HEIGHT_IN_METERS_CLEAN"] = df["TREE_HEIGHT_IN_METERS"].map(extract_numeric)
    else:
        df["TREE_HEIGHT_IN_METERS_CLEAN"] = np.nan

    if "ESTIMATED_TREE_HEIGHT" in df.columns:
        df["ESTIMATED_TREE_HEIGHT_CLEAN"] = df["ESTIMATED_TREE_HEIGHT"].map(extract_numeric)
    else:
        df["ESTIMATED_TREE_HEIGHT_CLEAN"] = np.nan

    df["HEIGHT_VALUE_M"] = df["TREE_HEIGHT_IN_METERS_CLEAN"].fillna(
        df["ESTIMATED_TREE_HEIGHT_CLEAN"]
    )

    bins = [-np.inf, 5, 10, 15, np.inf]
    labels = ["0-5", "5-10", "10-15", "15+"]

    df["HEIGHT_CLASS_STR"] = pd.cut(
        df["HEIGHT_VALUE_M"],
        bins=bins,
        labels=labels,
        right=False,
    ).astype("object")

    label_map = {label: i for i, label in enumerate(labels)}
    df["HEIGHT_CLASS_IDX"] = df["HEIGHT_CLASS_STR"].map(label_map)

    return df


def trunk_size_to_continuous_cm(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip().lower()
    s = s.replace("–", "-").replace("—", "-")

    # Range, e.g. "30-60 cm"
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", s)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2))
        return (lo + hi) / 2.0

    # Open-ended, e.g. "100+ cm"
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+", s)
    if m:
        return float(m.group(1))

    return np.nan


def trunk_size_to_continuous_cm(value):
    if pd.isna(value):
        return np.nan

    s = str(value).strip().lower()
    s = s.replace("–", "-").replace("—", "-")

    # "60–100 cm" or "60-100 cm"
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", s)
    if m:
        lo = float(m.group(1))
        hi = float(m.group(2))
        return (lo + hi) / 2.0

    # "100+ cm"
    m = re.search(r"(\d+(?:\.\d+)?)\s*\+", s)
    if m:
        return float(m.group(1))

    return np.nan


def add_diameter_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    measured_cm = df["CIRCUMFERENCE_IN_CM"].map(extract_numeric)

    estimated_cm = df["TREE_TRUNK_SIZE"].map(
        trunk_size_to_continuous_cm
    )

    method = (
        df["TREE_CIRCUMFERENCE_METHOD"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df["CIRCUMFERENCE_CM_CLEAN"] = np.where(
        method.str.contains("tape|measure"),
        measured_cm,
        estimated_cm,
    )

    # Continuous diameter target
    df["DBH_CM"] = df["CIRCUMFERENCE_CM_CLEAN"] / np.pi

    return df


# ---------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------


def make_local_filename(row) -> str:
    # IMAGE_ID is readable; IMAGE_UUID is stable if URLs/media ordering ever change.
    safe_image_id = str(row["IMAGE_ID"]).replace("/", "_").replace("\\", "_")
    safe_uuid = str(row.get("IMAGE_UUID", ""))[:8]
    return f"{safe_image_id}_{safe_uuid}.jpg"


def load_remote_image(url: str, timeout: int = 20):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


def assert_model_exists(path: str | Path, name: str) -> None:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"{name} model not found at:\n"
            f"  {path}\n\n"
            f"Run:\n"
            f"  treeco-download-models"
        )


def box_area(box) -> float:
    if box is None:
        return 0.0

    x1, y1, x2, y2 = [float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_area_ratio(box, image_size) -> float:
    w, h = image_size
    image_area = float(w * h)

    if image_area <= 0:
        return 1.0

    return box_area(box) / image_area


def expand_box_for_context(box, image_size, scale_x=1.25, scale_y=1.45):
    return groundingdino_box_cropping.expand_box(
        box,
        image_size=image_size,
        scale_x=scale_x,
        scale_y=scale_y,
    )


# ---------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------


def get_manifest_columns(df_img: pd.DataFrame) -> list[str]:
    keep_cols = [
        "ID",
        "IMAGE_ID",
        "IMAGE_UUID",
        "MEDIA_COL",
        "MEDIA_SRC",
        "LATITUDE",
        "LONGITUDE",
        "ORIGINAL_RGB_PATH",
        "RGB_CROP_PATH",
        "SAM_LOGITS_PATH",
        "SAM_SCORE",
        "DEPTH_PATH",
        "TREE_BOX",
        "DINO_SCORE",
        "ENTRY_MAX_DINO_SCORE",
        "DINO_LABEL",
        "DINO_AREA_RATIO",
        "DINO_USED_FULL_IMAGE_FALLBACK",
        "IMAGE_WIDTH",
        "IMAGE_HEIGHT",
        "CROP_WIDTH",
        "CROP_HEIGHT",
        "PROCESS_STATUS",
        "FAILURE_REASON",
        "TRAINABLE",
        "TREE_TYPE",
        "OTHER_TREE",
        "TREE_TYPE_RAW",
        "OTHER_TREE_RAW",
        "TREE_SPECIES_LABEL",
        "HEIGHT_VALUE_M",
        "HEIGHT_CLASS_STR",
        "HEIGHT_CLASS_IDX",
        "TREE_HEIGHT_IN_METERS_CLEAN",
        "ESTIMATED_TREE_HEIGHT_CLEAN",
        "CIRCUMFERENCE_CM_CLEAN",
        "DBH_CM",
        "TREE",
        "TREE_HEIGHT_METHOD",
        "TREE_HEIGHT_IN_METERS",
        "ESTIMATED_TREE_HEIGHT",
        "TREE_CIRCUMFERENCE_METHOD",
        "CIRCUMFERENCE_IN_CM",
        "TREE_TRUNK_SIZE",
        "tree_score",
        "top_prompt",
    ]

    return [c for c in keep_cols if c in df_img.columns]


def build_tree_level_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty or "ID" not in df_all.columns:
        return pd.DataFrame()

    def n_true(x):
        return int(pd.Series(x).fillna(False).astype(bool).sum())

    agg_spec = {
        "IMAGE_ID": "count",
        "TRAINABLE": n_true,
        "HEIGHT_CLASS_IDX": lambda x: bool(pd.Series(x).notna().any()),
        "DBH_CM": lambda x: bool(pd.Series(x).notna().any()),
        "DINO_SCORE": "max",
        "tree_score": "max",
        "DINO_USED_FULL_IMAGE_FALLBACK": n_true,
    }

    existing_agg_spec = {k: v for k, v in agg_spec.items() if k in df_all.columns}

    out = df_all.groupby("ID").agg(existing_agg_spec).reset_index()

    rename_map = {
        "IMAGE_ID": "N_IMAGES_ATTEMPTED",
        "TRAINABLE": "N_TRAINABLE_IMAGES",
        "HEIGHT_CLASS_IDX": "HAS_HEIGHT_LABEL",
        "DBH_CM": "HAS_DBH_LABEL",
        "DINO_SCORE": "BEST_DINO_SCORE",
        "tree_score": "BEST_CLIP_TREE_SCORE",
        "DINO_USED_FULL_IMAGE_FALLBACK": "N_FULL_IMAGE_DINO_FALLBACKS",
    }

    out = out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns})
    out["HAS_ANY_TRAINABLE_IMAGE"] = out.get("N_TRAINABLE_IMAGES", 0).astype(int) > 0

    return out


def safe_float_mean(series: pd.Series):
    if series is None or len(series) == 0:
        return None
    value = series.mean()
    return None if pd.isna(value) else float(value)


def safe_float_min(series: pd.Series):
    if series is None or len(series) == 0:
        return None
    value = series.min()
    return None if pd.isna(value) else float(value)


def safe_float_max(series: pd.Series):
    if series is None or len(series) == 0:
        return None
    value = series.max()
    return None if pd.isna(value) else float(value)


# ---------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------


def build_tree_dataset(
    input_path: str | Path,
    out_root: str | Path,
    dataset_name: str,
    keep_rgb: bool = True,
    keep_sam: bool = False,
    keep_depth: bool = False,
    clip_model_path: str | Path | None = None,
    clip_threshold: float = 0.5,
    dino_model_path: str | Path | None = None,
    dino_threshold: float = 0.25,
    dino_text_threshold: float = 0.30,
    dino_score_min: float = 0.25,
    depth_ckpt: str | Path | None = None,
    sam_ckpt: str | Path | None = None,
) -> Path:
    input_path = Path(input_path)
    out_root = Path(out_root)

    if clip_model_path is None:
        clip_model_path = get_model_path("clip")
    if dino_model_path is None:
        dino_model_path = get_model_path("grounding_dino")
    if keep_depth and depth_ckpt is None:
        depth_ckpt = get_model_path("depth_anything")
    if keep_sam and sam_ckpt is None:
        sam_ckpt = get_model_path("sam")

    assert_model_exists(clip_model_path, "CLIP")
    assert_model_exists(dino_model_path, "GroundingDINO")
    if keep_depth:
        assert_model_exists(depth_ckpt, "Depth Anything")
    if keep_sam:
        assert_model_exists(sam_ckpt, "SAM")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:6]
    dataset_dir = out_root / f"{dataset_name}_{run_id}_{timestamp}"

    manifest_dir = dataset_dir / "manifests"
    original_rgb_dir = dataset_dir / "original_rgb"
    rgb_crop_dir = dataset_dir / "rgb_crops"
    sam_dir = dataset_dir / "sam_full_logits"
    depth_dir = dataset_dir / "depth_maps"

    dirs = [manifest_dir]
    if keep_rgb:
        dirs.extend([original_rgb_dir, rgb_crop_dir])
    if keep_sam:
        dirs.append(sam_dir)
    if keep_depth:
        dirs.append(depth_dir)

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Building dataset in: {dataset_dir}")
    print(f"keep_rgb={keep_rgb}, keep_sam_logits={keep_sam}, keep_depth={keep_depth}")
    print(f"CLIP model: {clip_model_path}")
    print(f"DINO model: {dino_model_path}")
    if keep_sam:
        print(f"SAM checkpoint: {sam_ckpt}")
    if keep_depth:
        print(f"Depth checkpoint: {depth_ckpt}")

    df = load_table(input_path)
    n_raw_rows = len(df)

    try:
        df = df.map(clean_missing)
    except AttributeError:
        df = df.applymap(clean_missing)

    if "ID" in df.columns:
        df["ID"] = df["ID"].map(clean_id)

    n_raw_tree_yes_rows = None
    if "TREE" in df.columns:
        tree_mask = df["TREE"].astype(str).str.lower().str.strip().eq("yes")
        n_raw_tree_yes_rows = int(tree_mask.sum())
        df = df[tree_mask].copy()

    df_img = explode_media(df)
    if df_img.empty:
        raise RuntimeError("No media rows found.")

    df_img = add_height_labels(df_img)
    df_img = add_diameter_labels(df_img)

    n_exploded_image_rows = len(df_img)
    n_exploded_unique_ids = int(df_img["ID"].nunique()) if "ID" in df_img.columns else 0

    clip_model, clip_processor, clip_device = CLIP_filtering_system.load_clip_model(
        model_path=str(clip_model_path)
    )

    dino_processor, dino_model, dino_device = groundingdino_box_cropping.load_grounding_dino(
        model_path=str(dino_model_path)
    )

    sam_predictor = None
    if keep_sam:
        sam_predictor, sam_device = load_sam(sam_ckpt)
        print(f"Loaded SAM model from {sam_ckpt} on {sam_device}")

    depth_model = None
    depth_device = None
    if keep_depth:
        depth_model, depth_device = depth_anything.load_depth_anything_v2(
            ckpt_path=str(depth_ckpt)
        )

    tree_scores = []
    top_prompts = []
    keep_flags = []
    process_statuses = []
    failure_reasons = []

    original_rgb_paths = []
    rgb_crop_paths = []
    sam_logits_paths = []
    sam_scores = []
    depth_paths = []

    box_jsons = []
    dino_scores = []
    dino_labels = []
    dino_area_ratios = []
    dino_full_fallbacks = []

    image_widths = []
    image_heights = []
    crop_widths = []
    crop_heights = []

    for _, row in tqdm(
        df_img.iterrows(),
        total=len(df_img),
        desc="Filter + DINO crop + optional outputs",
    ):
        img = load_remote_image(row["MEDIA_SRC"])

        def append_failed(reason: str, status: str = "FAILED"):
            tree_scores.append(None)
            top_prompts.append(reason)
            keep_flags.append(False)
            process_statuses.append(status)
            failure_reasons.append(reason)

            original_rgb_paths.append(None)
            rgb_crop_paths.append(None)
            sam_logits_paths.append(None)
            sam_scores.append(None)
            depth_paths.append(None)

            box_jsons.append(None)
            dino_scores.append(None)
            dino_labels.append(None)
            dino_area_ratios.append(None)
            dino_full_fallbacks.append(False)

            image_widths.append(None)
            image_heights.append(None)
            crop_widths.append(None)
            crop_heights.append(None)

        if img is None:
            append_failed("LOAD_ERROR")
            continue

        try:
            clip_res = CLIP_filtering_system.score_images_batch(
                image_paths=[img],
                model=clip_model,
                processor=clip_processor,
                device=clip_device,
                batch_size=1,
                threshold=clip_threshold,
            )[0]
        except Exception:
            append_failed("CLIP_ERROR")
            continue

        tree_score = clip_res.get("tree_score")
        top_prompt = clip_res.get("top_prompt")
        is_tree = bool(clip_res.get("is_tree", False))

        tree_scores.append(tree_score)
        top_prompts.append(top_prompt)
        keep_flags.append(is_tree)

        if not is_tree:
            process_statuses.append("CLIP_REJECT")
            failure_reasons.append("CLIP_REJECT")

            original_rgb_paths.append(None)
            rgb_crop_paths.append(None)
            sam_logits_paths.append(None)
            sam_scores.append(None)
            depth_paths.append(None)

            box_jsons.append(None)
            dino_scores.append(None)
            dino_labels.append(None)
            dino_area_ratios.append(None)
            dino_full_fallbacks.append(False)

            image_widths.append(img.width)
            image_heights.append(img.height)
            crop_widths.append(None)
            crop_heights.append(None)
            continue

        try:
            result = groundingdino_box_cropping.detect_tree_box(
                img,
                processor=dino_processor,
                model=dino_model,
                device=dino_device,
                threshold=dino_threshold,
                text_threshold=dino_text_threshold,
                score_min=dino_score_min,
            )

            raw_box = result.get("best_box")
            dino_score = result.get("best_score")
            dino_label = result.get("best_label")
            used_full_fallback = result.get("used_full_image_fallback", False)

            if raw_box is None:
                expanded_box = [0, 0, img.width, img.height]
                crop = img
                area_ratio = 1.0
                used_full_fallback = True
                box_data = {
                    "raw_box": None,
                    "expanded_box": [float(v) for v in expanded_box],
                }
            else:
                area_ratio = box_area_ratio(raw_box, img.size)
                expanded_box = expand_box_for_context(
                    raw_box,
                    img.size,
                    scale_x=1.08,
                    scale_y=1.12,
                )
                crop = img.crop(expanded_box)
                box_data = {
                    "raw_box": [float(v) for v in raw_box],
                    "expanded_box": [float(v) for v in expanded_box],
                }

            crop_name = make_local_filename(row)

            original_rgb_path = None
            rgb_crop_path = None
            sam_logits_path = None
            sam_score = None
            depth_path = None

            if keep_rgb:
                original_rgb_path = original_rgb_dir / crop_name
                rgb_crop_path = rgb_crop_dir / crop_name

                img.save(original_rgb_path, format="JPEG", quality=95)
                crop.save(rgb_crop_path, format="JPEG", quality=95)

            if keep_sam:
                # Full original-image SAM logits. No DINO box prompt, no crop.
                sam_result = infer_sam_mask(img, sam_predictor)

                sam_score = sam_result["score"]
                sam_logits_path = sam_dir / crop_name.replace(".jpg", "_sam_full_logits.npy")

                np.save(
                    sam_logits_path,
                    sam_result["logits"].astype(np.float32),
                )

            if keep_depth:
                # Depth is inferred on the original full RGB image.
                depth = depth_anything.infer_depth(img, depth_model, depth_device)

                depth_path = depth_dir / crop_name.replace(".jpg", "_depth_full.npy")

                np.save(
                    depth_path,
                    depth.astype(np.float32),
                )

            original_rgb_paths.append(str(original_rgb_path) if original_rgb_path is not None else None)
            rgb_crop_paths.append(str(rgb_crop_path) if rgb_crop_path is not None else None)
            sam_logits_paths.append(str(sam_logits_path) if sam_logits_path is not None else None)
            sam_scores.append(float(sam_score) if sam_score is not None else None)
            depth_paths.append(str(depth_path) if depth_path is not None else None)

            box_jsons.append(json.dumps(box_data))
            dino_scores.append(float(dino_score) if dino_score is not None else None)
            dino_labels.append(dino_label)
            dino_area_ratios.append(float(area_ratio))
            dino_full_fallbacks.append(bool(used_full_fallback))

            image_widths.append(img.width)
            image_heights.append(img.height)
            crop_widths.append(crop.width)
            crop_heights.append(crop.height)

            process_statuses.append("OK")
            failure_reasons.append(None)

        except Exception as e:
            print(f"Processing failed for {row.get('IMAGE_ID')}: {e}")
            append_failed(f"PROCESSING_ERROR: {e}")

    df_img["tree_score"] = tree_scores
    df_img["top_prompt"] = top_prompts
    df_img["is_tree"] = keep_flags
    df_img["PROCESS_STATUS"] = process_statuses
    df_img["FAILURE_REASON"] = failure_reasons

    df_img["ORIGINAL_RGB_PATH"] = original_rgb_paths
    df_img["RGB_CROP_PATH"] = rgb_crop_paths
    df_img["SAM_LOGITS_PATH"] = sam_logits_paths
    df_img["SAM_SCORE"] = sam_scores
    df_img["DEPTH_PATH"] = depth_paths

    df_img["TREE_BOX"] = box_jsons
    df_img["DINO_SCORE"] = dino_scores
    df_img["DINO_LABEL"] = dino_labels
    df_img["DINO_AREA_RATIO"] = dino_area_ratios
    df_img["DINO_USED_FULL_IMAGE_FALLBACK"] = dino_full_fallbacks

    df_img["IMAGE_WIDTH"] = image_widths
    df_img["IMAGE_HEIGHT"] = image_heights
    df_img["CROP_WIDTH"] = crop_widths
    df_img["CROP_HEIGHT"] = crop_heights

    valid = df_img["is_tree"].fillna(False).astype(bool)

    if keep_rgb:
        valid = (
            valid
            & df_img["ORIGINAL_RGB_PATH"].notna()
            & df_img["RGB_CROP_PATH"].notna()
        )

    if keep_sam:
        valid = valid & df_img["SAM_LOGITS_PATH"].notna()

    if keep_depth:
        valid = valid & df_img["DEPTH_PATH"].notna()

    df_img["TRAINABLE"] = valid.astype(bool)

    # ENTRY_MAX_DINO_SCORE should be computed over all rows that reached DINO,
    # before filtering to only trainable rows.
    if len(df_img) > 0 and "ID" in df_img.columns:
        df_img["ENTRY_MAX_DINO_SCORE"] = df_img.groupby("ID")["DINO_SCORE"].transform("max")
    else:
        df_img["ENTRY_MAX_DINO_SCORE"] = np.nan

    df_final = df_img[df_img["TRAINABLE"]].reset_index(drop=True)

    print("After full pipeline:", len(df_final))

    manifest_cols_all = get_manifest_columns(df_img)
    manifest_cols_final = get_manifest_columns(df_final)

    attempted_manifest_path = manifest_dir / "tree_dataset_all_attempted_rows.csv"
    final_manifest_path = manifest_dir / "tree_dataset_manifest.csv"
    tree_summary_path = manifest_dir / "tree_level_summary.csv"

    df_img[manifest_cols_all].to_csv(attempted_manifest_path, index=False)
    df_final[manifest_cols_final].to_csv(final_manifest_path, index=False)

    tree_summary = build_tree_level_summary(df_img)
    tree_summary.to_csv(tree_summary_path, index=False)

    dupes = df_final.duplicated(["ID", "MEDIA_COL", "MEDIA_SRC"], keep=False)

    if dupes.any():
        print("WARNING: duplicate ID/MEDIA_COL/MEDIA_SRC rows found in final manifest:")
        print(df_final.loc[dupes, ["ID", "IMAGE_ID", "MEDIA_COL", "MEDIA_SRC"]])

    n_measured_attempted = int(
        (
            df_img["HEIGHT_CLASS_IDX"].notna()
            | df_img["DBH_CM"].notna()
        ).sum()
    )
    n_measured_kept = int(
        (
            df_final["HEIGHT_CLASS_IDX"].notna()
            | df_final["DBH_CM"].notna()
        ).sum()
    )

    measured_attempted_ids = set(
        df_img.loc[
            df_img["HEIGHT_CLASS_IDX"].notna() | df_img["DBH_CM"].notna(),
            "ID",
        ].astype(str)
    )
    measured_kept_ids = set(
        df_final.loc[
            df_final["HEIGHT_CLASS_IDX"].notna() | df_final["DBH_CM"].notna(),
            "ID",
        ].astype(str)
    )

    summary = {
        "created": timestamp,
        "run_id": run_id,
        "dataset_dir": str(dataset_dir),
        "input_file": str(input_path),
        "keep_rgb": keep_rgb,
        "keep_sam_logits": keep_sam,
        "keep_depth": keep_depth,
        "clip_threshold": clip_threshold,
        "dino_threshold": dino_threshold,
        "dino_text_threshold": dino_text_threshold,
        "dino_score_min": dino_score_min,
        "dino_expand_scale_x": 1.08,
        "dino_expand_scale_y": 1.12,
        "height_bins": ["0-5", "5-10", "10-15", "15+"],
        "diameter_target": "DBH_CM continuous regression target",
        "clip_model_path": str(clip_model_path),
        "dino_model_path": str(dino_model_path),
        "sam_ckpt": str(sam_ckpt) if sam_ckpt is not None else None,
        "depth_ckpt": str(depth_ckpt) if depth_ckpt is not None else None,
        "n_raw_rows": int(n_raw_rows),
        "n_raw_tree_yes_rows": n_raw_tree_yes_rows,
        "n_exploded_image_rows": int(n_exploded_image_rows),
        "n_exploded_unique_ids": int(n_exploded_unique_ids),
        "n_rows_final": int(len(df_final)),
        "n_unique_ids_final": int(df_final["ID"].nunique()) if "ID" in df_final.columns else 0,
        "n_unique_images_final": int(df_final["IMAGE_ID"].nunique()) if "IMAGE_ID" in df_final.columns else 0,
        "n_height_labeled_final": int(df_final["HEIGHT_CLASS_IDX"].notna().sum())
        if "HEIGHT_CLASS_IDX" in df_final.columns
        else 0,
        "n_diameter_labeled_final": int(df_final["DBH_CM"].notna().sum())
        if "DBH_CM" in df_final.columns
        else 0,
        "n_measured_attempted_image_rows": n_measured_attempted,
        "n_measured_kept_image_rows": n_measured_kept,
        "n_measured_lost_image_rows": int(n_measured_attempted - n_measured_kept),
        "n_measured_attempted_trees": int(len(measured_attempted_ids)),
        "n_measured_kept_trees": int(len(measured_kept_ids)),
        "n_measured_completely_lost_trees": int(len(measured_attempted_ids - measured_kept_ids)),
        "n_clip_rejects": int((df_img["PROCESS_STATUS"] == "CLIP_REJECT").sum()),
        "n_load_errors": int((df_img["PROCESS_STATUS"] == "FAILED").sum()),
        "n_processing_ok": int((df_img["PROCESS_STATUS"] == "OK").sum()),
        "n_full_image_dino_fallbacks_final": int(df_final["DINO_USED_FULL_IMAGE_FALLBACK"].sum())
        if "DINO_USED_FULL_IMAGE_FALLBACK" in df_final.columns
        else 0,
        "n_duplicate_image_links_final": int(dupes.sum()),
        "mean_images_per_tree_final": float(df_final.groupby("ID")["IMAGE_ID"].count().mean())
        if len(df_final) and "ID" in df_final.columns and "IMAGE_ID" in df_final.columns
        else 0,
        "max_images_per_tree_final": int(df_final.groupby("ID")["IMAGE_ID"].count().max())
        if len(df_final) and "ID" in df_final.columns and "IMAGE_ID" in df_final.columns
        else 0,
        "sam_score_mean_final": safe_float_mean(df_final["SAM_SCORE"]) if keep_sam and len(df_final) else None,
        "sam_score_min_final": safe_float_min(df_final["SAM_SCORE"]) if keep_sam and len(df_final) else None,
        "sam_score_max_final": safe_float_max(df_final["SAM_SCORE"]) if keep_sam and len(df_final) else None,
        "dino_score_mean_final": safe_float_mean(df_final["DINO_SCORE"]) if len(df_final) else None,
        "dino_score_min_final": safe_float_min(df_final["DINO_SCORE"]) if len(df_final) else None,
        "dino_score_max_final": safe_float_max(df_final["DINO_SCORE"]) if len(df_final) else None,
        "attempted_manifest": str(attempted_manifest_path),
        "final_manifest": str(final_manifest_path),
        "tree_level_summary": str(tree_summary_path),
    }

    summary_path = manifest_dir / "summary.json"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)

    print("Saved attempted manifest:", attempted_manifest_path)
    print("Saved final manifest:", final_manifest_path)
    print("Saved tree-level summary:", tree_summary_path)
    print(json.dumps(summary, indent=4))

    return final_manifest_path
