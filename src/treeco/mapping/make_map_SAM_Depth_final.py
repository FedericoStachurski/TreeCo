#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import branca.colormap as cm
from branca.element import Element
import folium
import numpy as np
import pandas as pd
from html import escape
import hashlib
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from urllib.parse import parse_qs, quote, urlsplit
from PIL import Image
from matplotlib import colormaps

# =========================================================
# Assumptions
# =========================================================

SEQ_RATE_MEAN = 0.02
SEQ_RATE_UNC = 0.01

KG_CO2E_PER_KETTLE_BOIL = 0.018

HEIGHT_CLASS_MAP = {
    "<5": (2.5, 2.5),
    "0-5": (2.5, 2.5),
    "5-10": (7.5, 2.5),
    "10-15": (12.5, 2.5),
    "15-20": (17.5, 2.5),
    "20+": (22.5, 2.5),
    "15+": (17.5, 2.5),
}


# =========================================================
# Helpers
# =========================================================

def parse_original_height(value):
    """
    Interpret original CommuniMap height values.

    Examples:
        2-5       -> 3.5
        5-10      -> 7.5
        10-15     -> 12.5
        12        -> 12.0
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip().lower()
    s = s.replace("–", "-").replace("—", "-")
    s = s.replace(",", ".")

    # Match the original dataset's open-ended categories
    if s == "10+ metres":
        return 12.5

    if s in HEIGHT_CLASS_MAP:
        return HEIGHT_CLASS_MAP[s][0]

    # Handle numerical ranges
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)",
        s
    )

    if match:
        lo = float(match.group(1))
        hi = float(match.group(2))
        return (lo + hi) / 2

    # Single numerical value
    match = re.search(r"\d+(?:\.\d+)?", s)

    return float(match.group()) if match else np.nan

def clean_species_name(x):
    if pd.isna(x):
        return "unknown"

    x = str(x).strip().lower()
    x = x.encode("ascii", "ignore").decode("ascii")
    x = re.sub(r"\(.*?\)", "", x)
    x = re.sub(r"\s+", " ", x).strip()

    if x in {"", "nan", "none", "unknown"}:
        return "unknown"

    return x


def canonical_species_group(x):
    x = clean_species_name(x)

    if "beech" in x:
        return "beech"
    if "oak" in x:
        return "oak"
    if "ash" in x:
        return "ash"
    if "lime" in x:
        return "lime"
    if "sycamore" in x:
        return "sycamore"
    if "maple" in x:
        return "maple"
    if "birch" in x:
        return "birch"
    if "pine" in x:
        return "pine"
    if any(k in x for k in ["spruce", "fir", "cedar", "cypress", "yew"]):
        return "conifer_other"

    return "unknown"


def first_non_null(series):
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return s.iloc[0]


def list_non_null(series):
    return [
        v for v in series.dropna().astype(str).tolist()
        if v.strip() and v.strip().lower() not in {"nan", "none", "null"}
    ]


def height_class_to_midpoint(label):
    if pd.isna(label):
        return np.nan, np.nan

    label = str(label).strip()

    if label in HEIGHT_CLASS_MAP:
        return HEIGHT_CLASS_MAP[label]

    return np.nan, np.nan


def carbon_stock_from_height_diameter(height_m, diameter_cm):
    """
    Carbon stock formula:
      IF(height < 28, 0.0577*height^2*diameter, 0.0346*height^2*diameter) * 0.25

    height_m in metres
    diameter_cm in cm
    output kg C
    """
    if pd.isna(height_m) or pd.isna(diameter_cm):
        return np.nan

    if height_m <= 0 or diameter_cm < 0:
        return np.nan

    if height_m < 28:
        return 0.0577 * height_m**2 * diameter_cm * 0.25

    return 0.0346 * height_m**2 * diameter_cm * 0.25


def carbon_stock_uncertainty(
    height_m,
    height_unc_m,
    diameter_cm,
    diameter_unc_cm,
    carbon_stock_kg,
):
    if any(pd.isna(x) for x in [
        height_m,
        height_unc_m,
        diameter_cm,
        diameter_unc_cm,
        carbon_stock_kg,
    ]):
        return np.nan

    if height_m <= 0 or diameter_cm <= 0 or carbon_stock_kg <= 0:
        return np.nan

    rel2 = (
        (2.0 * height_unc_m / height_m) ** 2
        + (diameter_unc_cm / diameter_cm) ** 2
    )

    return carbon_stock_kg * np.sqrt(rel2)


def pick_height(row):
    """
    Select tree height using the following priority:

    1. Citizen-measured height
    2. Citizen-estimated height category
    3. Model-inferred height
    """

    # =====================================================
    # 1. Citizen-measured height
    # =====================================================

    measured = parse_original_height(
        row.get("TREE_HEIGHT_IN_METERS")
    )

    if pd.notna(measured) and measured > 0:
        return measured, 1.0, "citizen_measured"

    # =====================================================
    # 2. Citizen-estimated height
    # =====================================================

    estimated = parse_original_height(
        row.get("ESTIMATED_TREE_HEIGHT")
    )

    if pd.notna(estimated) and estimated > 0:
        return estimated, 2.5, "citizen_estimated"

    # =====================================================
    # 3. Model-inferred height
    # =====================================================

    inferred_class = row.get("HEIGHT_CLASS_PRED_STR")

    h, u = height_class_to_midpoint(inferred_class)

    if pd.notna(h):
        return h, u, "model_inferred"

    return np.nan, np.nan, "missing"


def pick_diameter(row, pred_is_circumference=False):
    """
    Select the best available DBH.

    Priority:
      1. DBH_CM_FINAL
         - observed DBH where available
         - otherwise tree-level inferred DBH

      2. Legacy observed DBH_CM

      3. Legacy cleaned circumference converted to DBH

      4. Legacy DBH_CM_PRED

    DBH_CM_FINAL is already diameter in cm and must NOT
    be divided by pi.
    """

    # =====================================================
    # Prefer original citizen-provided circumference
    # =====================================================

    raw_circ = pd.to_numeric(
        row.get("CIRCUMFERENCE_IN_CM"),
        errors="coerce"
    )

    if pd.notna(raw_circ) and raw_circ > 0:

        diameter = float(raw_circ) / np.pi

        method = str(
            row.get("TREE_CIRCUMFERENCE_METHOD", "")
        ).lower()

        source = (
            "citizen_measured"
            if "tape" in method or "measure" in method
            else "citizen_reported"
        )

        return (
            diameter,
            max(0.10 * diameter, 2.0),
            source
        )

    # =====================================================
    # New TreeCo pipeline: preferred final DBH
    # =====================================================
    final_dbh = pd.to_numeric(
        row.get("DBH_CM_FINAL"),
        errors="coerce",
    )

    if pd.notna(final_dbh) and final_dbh > 0:

        # Work out whether DBH_CM_FINAL came from an
        # observation or from the width model.
        observed_final = pd.to_numeric(
            row.get("DBH_CM_OBSERVED"),
            errors="coerce",
        )

        if pd.notna(observed_final) and observed_final > 0:

            # This value may originate from an estimated
            # trunk-size category rather than a tape measurement.

            return (
                float(final_dbh),
                max(0.10 * float(final_dbh), 2.0),
                "citizen_estimated_dbh",
            )

        return (
            float(final_dbh),
            max(0.15 * float(final_dbh), 3.0),
            "model_inferred_dbh_final",
        )

    # =====================================================
    # Legacy fallback: directly observed DBH
    # =====================================================
    observed_dbh = pd.to_numeric(
        row.get("DBH_CM"),
        errors="coerce",
    )

    if pd.notna(observed_dbh) and observed_dbh > 0:
        return (
            float(observed_dbh),
            max(0.10 * float(observed_dbh), 2.0),
            "observed_dbh",
        )

    # =====================================================
    # Legacy fallback: observed circumference
    # =====================================================
    circumference_clean = pd.to_numeric(
        row.get("CIRCUMFERENCE_CM_CLEAN"),
        errors="coerce",
    )

    if (
        pd.notna(circumference_clean)
        and circumference_clean > 0
    ):
        diameter = float(circumference_clean) / np.pi

        return (
            diameter,
            max(0.10 * diameter, 2.0),
            "observed_circumference_converted",
        )

    # =====================================================
    # Legacy predicted DBH
    # =====================================================
    pred = pd.to_numeric(
        row.get("DBH_CM_PRED"),
        errors="coerce",
    )

    if pd.notna(pred) and pred > 0:

        pred = float(pred)

        if pred_is_circumference:
            diameter = pred / np.pi

            return (
                diameter,
                max(0.15 * diameter, 3.0),
                "model_inferred_circumference_converted",
            )

        return (
            pred,
            max(0.15 * pred, 3.0),
            "model_inferred_dbh",
        )

    return np.nan, np.nan, "missing"


# =========================================================
# Core pipeline
# =========================================================

def aggregate_measurement_sources(series):
    sources = set(series.dropna().astype(str))

    # Remove missing values
    sources -= {"missing", "nan", "None", ""}

    if not sources:
        return "missing"

    if len(sources) == 1:
        return next(iter(sources))

    # Multiple sources contributed to the aggregated measurement
    return "mixed: " + ", ".join(sorted(sources))

def existing_file(value: object, dataset_dir: Path) -> Path | None:
    """Resolve only files already saved in the chosen TreeCo dataset."""
    if not isinstance(value, str) or not value.strip() or value.strip().lower() in {"nan", "none", "null"}:
        return None
    path = Path(value.strip()).expanduser()
    candidates = [path] if path.is_absolute() else [dataset_dir / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def match_saved_image_outputs(df: pd.DataFrame, dataset_dir: Path) -> pd.DataFrame:
    """Match existing RGB/SAM3/depth paths to the combined manifest by IMAGE_ID.

    No inference, mask conversion, or image writing is performed here.
    """
    df = df.copy()
    if "IMAGE_ID" not in df.columns:
        raise ValueError("Combined manifest lacks IMAGE_ID; saved images cannot be matched.")
    keys = df["IMAGE_ID"].astype("string").str.strip()
    reference_path = dataset_dir / "manifests" / "tree_dataset_manifest.csv"
    if not reference_path.is_file():
        raise FileNotFoundError(f"Original dataset manifest not found: {reference_path}")

    ref = pd.read_csv(reference_path, dtype={"IMAGE_ID": "string"}, low_memory=False)
    if "IMAGE_ID" not in ref:
        raise ValueError(f"Reference dataset manifest lacks IMAGE_ID: {reference_path}")
    ref["IMAGE_ID"] = ref["IMAGE_ID"].astype("string").str.strip()
    duplicate_count = int(ref["IMAGE_ID"].duplicated().sum())
    if duplicate_count:
        print(f"[POPUP] {duplicate_count} duplicate IMAGE_ID rows; using first matching entry")
    ref = ref.drop_duplicates("IMAGE_ID").set_index("IMAGE_ID")
    print(f"[POPUP] Matching saved images by IMAGE_ID from: {reference_path}")

    for column in ("MEDIA_SRC", "ORIGINAL_RGB_PATH", "SAM3_MASK_PATH", "DEPTH_PATH"):
        if column not in df:
            df[column] = pd.NA
        for suffix in ("_x", "_y", "__height", "__width"):
            alt = f"{column}{suffix}"
            if alt in df:
                blank = df[column].isna() | df[column].astype("string").str.strip().isin(["", "nan", "None", "null"])
                df[column] = df[column].where(~blank.fillna(True), df[alt])

        if column in ref:
            matched = keys.map(ref[column])
            if column == "MEDIA_SRC":
                blank = df[column].isna() | df[column].astype("string").str.strip().isin(["", "nan", "None", "null"])
            else:
                # Stale absolute paths are replaced with the dataset reference path.
                blank = ~df[column].map(lambda v: existing_file(v, dataset_dir) is not None)
            df[column] = df[column].where(~blank.fillna(True), matched)

    for column in ("ORIGINAL_RGB_PATH", "SAM3_MASK_PATH", "DEPTH_PATH"):
        df[column] = df[column].map(lambda v: str(path) if (path := existing_file(v, dataset_dir)) else None)
        print(f"[POPUP] {column}: {df[column].notna().sum():,}/{len(df):,} image rows with saved files")
    return df


def read_splitcp_half_width(path: Path) -> float:
    """Recover the calibrated, tree-level SplitCP radius from saved test intervals.

    Use CP_UPPER_CM - PUNCC_PRED_DBH_CM because some lower bounds are
    truncated at zero. These held-out predictions are NOT merged by ID.
    """
    results = pd.read_csv(path)
    required = {"PUNCC_PRED_DBH_CM", "CP_UPPER_CM"}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"SplitCP file is missing columns: {sorted(missing)}")

    predicted = pd.to_numeric(results["PUNCC_PRED_DBH_CM"], errors="coerce")
    upper = pd.to_numeric(results["CP_UPPER_CM"], errors="coerce")
    radii = (upper - predicted).replace([np.inf, -np.inf], np.nan).dropna()
    if radii.empty or (radii <= 0).any():
        raise ValueError("SplitCP file contains no valid positive interval half-widths")

    half_width = float(radii.median())
    largest_difference = float((radii - half_width).abs().max())
    if largest_difference > 0.05:
        raise ValueError(
            "SplitCP half-width varies between rows; this is not a fixed-width "
            "SplitCP result. Do not reuse these intervals as a single calibrated radius."
        )
    print(f"[DBH SplitCP] Calibrated half-width: {half_width:.4f} cm "
          f"(from {len(radii)} saved predictions; nominal coverage 90%)")
    return half_width


def attach_dbh_conformal_intervals(agg: pd.DataFrame, half_width: float | None) -> pd.DataFrame:
    """Apply tree-level intervals only to DBH from the matching trained regressor."""
    agg = agg.copy()
    agg["DBH_CP_LOWER_CM"] = np.nan
    agg["DBH_CP_UPPER_CM"] = np.nan
    agg["DBH_CP_HALF_WIDTH_CM"] = np.nan
    agg["DBH_CP_TARGET_COVERAGE"] = np.nan

    if half_width is None:
        return agg

    # Avoid applying model residual calibration to citizen-derived DBH or
    # mixed-source tree groups, where its coverage meaning would be invalid.
    use = (
        agg["DIAMETER_SOURCE"].eq("model_inferred_dbh_final")
        & pd.to_numeric(agg["DIAMETER_CM"], errors="coerce").gt(0)
    )
    agg.loc[use, "DBH_CP_LOWER_CM"] = (
        agg.loc[use, "DIAMETER_CM"] - half_width
    ).clip(lower=0)
    agg.loc[use, "DBH_CP_UPPER_CM"] = agg.loc[use, "DIAMETER_CM"] + half_width
    agg.loc[use, "DBH_CP_HALF_WIDTH_CM"] = half_width
    agg.loc[use, "DBH_CP_TARGET_COVERAGE"] = 0.90
    print(f"[DBH SplitCP] Intervals added to {int(use.sum())} model-inferred trees "
          "(citizen-derived/mixed-source DBH excluded).")
    return agg


def build_carbon_table(
    manifest_path: Path,
    pred_is_circumference: bool = False,
    dataset_dir: Path | None = None,
    image_registry: dict | None = None,
    dbh_cp_half_width: float | None = None,
) -> pd.DataFrame:
    if dataset_dir is None:
        dataset_dir = Path(manifest_path).resolve().parent.parent
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    df = pd.read_csv(manifest_path, dtype={"IMAGE_ID": "string"}, low_memory=False)
    df = match_saved_image_outputs(df, dataset_dir)

    # The HTTP endpoint uses only these vetted, IMAGE_ID-indexed saved paths.
    # Keep the registry in memory. Nothing is written to a preview folder.
    if image_registry is not None:
        for image_id, rgb, sam3, depth in zip(
            df["IMAGE_ID"], df["ORIGINAL_RGB_PATH"],
            df["SAM3_MASK_PATH"], df["DEPTH_PATH"]
        ):
            if pd.notna(image_id):
                image_registry[str(image_id).strip()] = {
                    "rgb": rgb, "sam3": sam3, "depth": depth,
                }

    print(f"Loaded manifest: {manifest_path}")
    print(f"Image-level rows: {len(df)}")
    print(f"Unique tree IDs / entries: {df['ID'].nunique()}")

    height_rows = df.apply(lambda r: pick_height(r), axis=1)
    df[[
        "HEIGHT_M_IMAGE",
        "HEIGHT_UNC_M_IMAGE",
        "HEIGHT_SOURCE_FINAL_IMAGE",
    ]] = pd.DataFrame(height_rows.tolist(), index=df.index)

    diameter_rows = df.apply(
        lambda r: pick_diameter(
            r,
            pred_is_circumference=pred_is_circumference,
        ),
        axis=1,
    )

    df[[
        "DIAMETER_CM_IMAGE",
        "DIAMETER_UNC_CM_IMAGE",
        "DIAMETER_SOURCE_FINAL_IMAGE",
    ]] = pd.DataFrame(diameter_rows.tolist(), index=df.index)

    valid_img = (
        df["HEIGHT_M_IMAGE"].notna()
        & df["DIAMETER_CM_IMAGE"].notna()
    )

    df["CARBON_STOCK_KG_IMAGE"] = np.nan
    df.loc[valid_img, "CARBON_STOCK_KG_IMAGE"] = df.loc[valid_img].apply(
        lambda r: carbon_stock_from_height_diameter(
            r["HEIGHT_M_IMAGE"],
            r["DIAMETER_CM_IMAGE"],
        ),
        axis=1,
    )

    df["CARBON_STOCK_UNC_KG_IMAGE"] = np.nan
    df.loc[valid_img, "CARBON_STOCK_UNC_KG_IMAGE"] = df.loc[valid_img].apply(
        lambda r: carbon_stock_uncertainty(
            r["HEIGHT_M_IMAGE"],
            r["HEIGHT_UNC_M_IMAGE"],
            r["DIAMETER_CM_IMAGE"],
            r["DIAMETER_UNC_CM_IMAGE"],
            r["CARBON_STOCK_KG_IMAGE"],
        ),
        axis=1,
    )

    # Identify whether height or DBH was inferred
    df["HEIGHT_MODEL_INFERRED_IMAGE"] = (
        df["HEIGHT_SOURCE_FINAL_IMAGE"]
        .astype(str)
        .str.startswith("model_inferred")
    )

    df["DIAMETER_MODEL_INFERRED_IMAGE"] = (
        df["DIAMETER_SOURCE_FINAL_IMAGE"]
        .astype(str)
        .str.startswith("model_inferred")
    )

    df["CARBON_SEQUESTERED_KG_YR_IMAGE"] = (
        df["CARBON_STOCK_KG_IMAGE"] * SEQ_RATE_MEAN
    )

    df["CARBON_SEQUESTERED_UNC_KG_YR_IMAGE"] = np.sqrt(
        (df["CARBON_STOCK_KG_IMAGE"] * SEQ_RATE_UNC) ** 2
        + (SEQ_RATE_MEAN * df["CARBON_STOCK_UNC_KG_IMAGE"]) ** 2
    )

    co2_factor = 44.0 / 12.0

    df["CO2_SEQUESTERED_KG_YR_IMAGE"] = (
        df["CARBON_SEQUESTERED_KG_YR_IMAGE"] * co2_factor
    )

    df["CO2_SEQUESTERED_UNC_KG_YR_IMAGE"] = (
        df["CARBON_SEQUESTERED_UNC_KG_YR_IMAGE"] * co2_factor
    )

    df["KETTLE_BOILS_PER_YEAR_IMAGE"] = (
        df["CO2_SEQUESTERED_KG_YR_IMAGE"] / KG_CO2E_PER_KETTLE_BOIL
    )

    df["KETTLE_BOILS_UNC_PER_YEAR_IMAGE"] = (
        df["CO2_SEQUESTERED_UNC_KG_YR_IMAGE"] / KG_CO2E_PER_KETTLE_BOIL
    )

    # =========================================================
    # Preserve image-level information for popup visualisations
    # =========================================================

    # Each dict carries an original IMAGE_ID and its saved model outputs;
    # aggregation keeps images paired with the correct masks/depth maps.
    df["POPUP_IMAGE"] = [
        {
            "image_id": str(image_id),
            "url": url if isinstance(url, str) else None,
            "has_rgb": isinstance(rgb, str),
            "has_sam3": isinstance(sam3, str) and isinstance(rgb, str),
            "has_depth": isinstance(depth, str),
        }
        for image_id, url, rgb, sam3, depth in zip(
            df["IMAGE_ID"], df["MEDIA_SRC"],
            df["ORIGINAL_RGB_PATH"], df["SAM3_MASK_PATH"], df["DEPTH_PATH"]
        )
    ]

    agg = (
        df.groupby("ID", dropna=False)
        .agg(
            LATITUDE=("LATITUDE", "first"),
            LONGITUDE=("LONGITUDE", "first"),
            TREE_TYPE=("TREE_TYPE", first_non_null),
            TREE_SPECIES_LABEL=("TREE_SPECIES_LABEL", first_non_null),

            HEIGHT_M=("HEIGHT_M_IMAGE", "mean"),
            HEIGHT_UNC_M=("HEIGHT_UNC_M_IMAGE", "mean"),
            DIAMETER_CM=("DIAMETER_CM_IMAGE", "mean"),
            DIAMETER_UNC_CM=("DIAMETER_UNC_CM_IMAGE", "mean"),

            CARBON_STOCK_KG=("CARBON_STOCK_KG_IMAGE", "mean"),
            CARBON_STOCK_UNC_KG=("CARBON_STOCK_UNC_KG_IMAGE", "mean"),

            CARBON_SEQUESTERED_KG_YR=("CARBON_SEQUESTERED_KG_YR_IMAGE", "mean"),
            CARBON_SEQUESTERED_UNC_KG_YR=("CARBON_SEQUESTERED_UNC_KG_YR_IMAGE", "mean"),

            CO2_SEQUESTERED_KG_YR=("CO2_SEQUESTERED_KG_YR_IMAGE", "mean"),
            CO2_SEQUESTERED_UNC_KG_YR=("CO2_SEQUESTERED_UNC_KG_YR_IMAGE", "mean"),

            KETTLE_BOILS_PER_YEAR=("KETTLE_BOILS_PER_YEAR_IMAGE", "mean"),
            KETTLE_BOILS_UNC_PER_YEAR=("KETTLE_BOILS_UNC_PER_YEAR_IMAGE", "mean"),
            N_IMAGES=("IMAGE_ID", "nunique"),
            IMAGE_IDS=("IMAGE_ID", list_non_null),
            IMAGE_URLS=("MEDIA_SRC", list_non_null),
            POPUP_IMAGES=("POPUP_IMAGE", list),

            HEIGHT_SOURCE=(
                "HEIGHT_SOURCE_FINAL_IMAGE",
                aggregate_measurement_sources
            ),

            DIAMETER_SOURCE=(
                "DIAMETER_SOURCE_FINAL_IMAGE",
                aggregate_measurement_sources
            ),

            HEIGHT_MODEL_INFERRED=("HEIGHT_MODEL_INFERRED_IMAGE", "any"),
            DIAMETER_MODEL_INFERRED=("DIAMETER_MODEL_INFERRED_IMAGE", "any"),
        )
        .reset_index()
    )

    agg["SEQ_RATE_YR"] = SEQ_RATE_MEAN
    agg["SEQ_RATE_UNC_YR"] = SEQ_RATE_UNC

    agg["SPECIES_CLEAN"] = agg["TREE_SPECIES_LABEL"].apply(clean_species_name)
    agg["SPECIES_GROUP"] = agg["SPECIES_CLEAN"].apply(canonical_species_group)

    agg["HAS_MODEL_INFERENCE"] = (agg["HEIGHT_MODEL_INFERRED"]
                                   |agg["DIAMETER_MODEL_INFERRED"])

    # SplitCP is calibrated at tree level, AFTER aggregation across images.
    # Neither saved .npy files nor the popup image server are changed.
    agg = attach_dbh_conformal_intervals(agg, dbh_cp_half_width)

    # =========================================================
    # Propagate DBH conformal bounds into carbon calculations
    # =========================================================

    for bound in ["LOWER", "UPPER"]:

        # Carbon stock (kg C)
        # Carbon is linear in DBH for fixed height. Scale the actual
        # aggregated carbon estimate, rather than recomputing it from
        # (mean height)**2, which could shift the central estimate.
        agg[f"CARBON_STOCK_CP_{bound}_KG"] = (
            agg["CARBON_STOCK_KG"]
            * agg[f"DBH_CP_{bound}_CM"]
            / agg["DIAMETER_CM"].where(agg["DIAMETER_CM"].gt(0))
        )

        # Annual carbon sequestration (kg C/year)
        agg[f"CARBON_SEQUESTERED_CP_{bound}_KG_YR"] = (
            agg[f"CARBON_STOCK_CP_{bound}_KG"] * SEQ_RATE_MEAN
        )

        # Annual CO2 equivalent (kg CO2/year)
        agg[f"CO2_SEQUESTERED_CP_{bound}_KG_YR"] = (
            agg[f"CARBON_SEQUESTERED_CP_{bound}_KG_YR"] * (44.0 / 12.0)
        )

        # Kettle-boil equivalent
        agg[f"KETTLE_BOILS_CP_{bound}_YR"] = (
            agg[f"CO2_SEQUESTERED_CP_{bound}_KG_YR"]
            / KG_CO2E_PER_KETTLE_BOIL
        )

    # =========================================================
    # Final DBH-based bounds: SplitCP for AI DBH, +/-20% for
    # citizen-reported, estimated, or measured DBH.
    # This does NOT reuse the former placeholder 10-15% errors.
    # =========================================================
    citizen_dbh_sources = {
        "citizen_measured",
        "citizen_reported",
        "citizen_estimated_dbh",
        "observed_dbh",
        "observed_circumference_converted",
    }

    def citizen_only(source):
        # A tree may have multiple citizen measurement categories across
        # photographs. Mixed citizen+model sources are deliberately excluded.
        if not isinstance(source, str) or source in {"", "missing"}:
            return False
        if source.startswith("mixed: "):
            members = [part.strip() for part in source[len("mixed: "):].split(",")]
            return bool(members) and all(m in citizen_dbh_sources for m in members)
        return source in citizen_dbh_sources

    citizen_mask = (
        agg["DIAMETER_SOURCE"].map(citizen_only)
        & agg["DIAMETER_CM"].gt(0)
    )
    splitcp_mask = agg["DBH_CP_HALF_WIDTH_CM"].notna()

    agg["DBH_UNC_METHOD"] = "unavailable"
    agg["DBH_UNC_CM_DISPLAY"] = np.nan
    agg["DBH_UNC_LOWER_CM"] = np.nan
    agg["DBH_UNC_UPPER_CM"] = np.nan

    agg.loc[splitcp_mask, "DBH_UNC_METHOD"] = "SplitCP (90% DBH target)"
    agg.loc[splitcp_mask, "DBH_UNC_CM_DISPLAY"] = agg.loc[
        splitcp_mask, "DBH_CP_HALF_WIDTH_CM"
    ]
    agg.loc[splitcp_mask, "DBH_UNC_LOWER_CM"] = agg.loc[
        splitcp_mask, "DBH_CP_LOWER_CM"
    ]
    agg.loc[splitcp_mask, "DBH_UNC_UPPER_CM"] = agg.loc[
        splitcp_mask, "DBH_CP_UPPER_CM"
    ]

    agg.loc[citizen_mask, "DBH_UNC_METHOD"] = "Citizen DBH (assumed 20%)"
    agg.loc[citizen_mask, "DBH_UNC_CM_DISPLAY"] = (
        agg.loc[citizen_mask, "DIAMETER_CM"] * 0.20
    )
    agg.loc[citizen_mask, "DBH_UNC_LOWER_CM"] = (
        agg.loc[citizen_mask, "DIAMETER_CM"] * 0.80
    )
    agg.loc[citizen_mask, "DBH_UNC_UPPER_CM"] = (
        agg.loc[citizen_mask, "DIAMETER_CM"] * 1.20
    )
    # Reflect citizen uncertainty in the exported diameter uncertainty column.
    agg.loc[citizen_mask, "DIAMETER_UNC_CM"] = agg.loc[
        citizen_mask, "DBH_UNC_CM_DISPLAY"
    ]

    # Height and sequestration rate are held fixed here. The existing
    # central carbon estimate is preserved; uncertainty only varies DBH.
    for bound in ("LOWER", "UPPER"):
        scale = (
            agg[f"DBH_UNC_{bound}_CM"]
            / agg["DIAMETER_CM"].where(agg["DIAMETER_CM"].gt(0))
        )
        agg[f"CARBON_STOCK_DBH_{bound}_KG"] = agg["CARBON_STOCK_KG"] * scale
        agg[f"CARBON_SEQUESTERED_DBH_{bound}_KG_YR"] = (
            agg[f"CARBON_STOCK_DBH_{bound}_KG"] * SEQ_RATE_MEAN
        )
        agg[f"CO2_SEQUESTERED_DBH_{bound}_KG_YR"] = (
            agg[f"CARBON_SEQUESTERED_DBH_{bound}_KG_YR"] * (44.0 / 12.0)
        )
        agg[f"KETTLE_BOILS_DBH_{bound}_YR"] = (
            agg[f"CO2_SEQUESTERED_DBH_{bound}_KG_YR"]
            / KG_CO2E_PER_KETTLE_BOIL
        )

    print(f"[DBH uncertainty] SplitCP: {int(splitcp_mask.sum())} trees; "
          f"citizen +/-20%: {int(citizen_mask.sum())} trees; "
          f"other/no bounds: {int((~(splitcp_mask | citizen_mask)).sum())} trees")

    print("\n" + "=" * 60)
    print("TREE MEASUREMENT SOURCES")
    print("=" * 60)

    valid = agg[
        agg["CARBON_SEQUESTERED_KG_YR"].notna()
    ]

    print("\nHeight sources:")
    print(valid["HEIGHT_SOURCE"].value_counts())

    print("\nDiameter sources:")
    print(valid["DIAMETER_SOURCE"].value_counts())

    print("\nSource combinations:")
    print(
        valid.groupby(
            ["HEIGHT_SOURCE", "DIAMETER_SOURCE"]
        ).size().sort_values(ascending=False)
    )

    print("\nTotal valid trees:", len(valid))
    print(
        "Trees with AI inference:",
        valid["HAS_MODEL_INFERENCE"].sum()
    )

    print(f"Entry-level rows: {len(agg)}")
    print("Valid carbon rows:", int(agg["CARBON_STOCK_KG"].notna().sum()))
    print("Valid sequestration rows:", int(agg["CARBON_SEQUESTERED_KG_YR"].notna().sum()))

    return agg

def render_saved_image(image_id: str, kind: str, registry: dict) -> tuple[bytes, str] | None:
    """Render ONE selected photograph/mask/depth into RAM on HTTP request.

    This function never writes files and never executes an ML model.
    """
    item = registry.get(image_id)
    if item is None or kind not in {"original", "sam3", "depth"}:
        return None
    rgb_file = item.get("rgb")

    if kind == "original":
        if not rgb_file:
            return None
        path = Path(rgb_file)
        # The original JPG or PNG can be streamed directly, without conversions.
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return path.read_bytes(), mime
        with Image.open(path) as src:
            with BytesIO() as result:
                src.convert("RGB").save(result, "JPEG", quality=85)
                return result.getvalue(), "image/jpeg"

    source = item.get(kind)
    if not source:
        return None
    values = np.squeeze(np.load(source, allow_pickle=False))
    if values.ndim != 2:
        raise ValueError(f"Expected 2D saved {kind} map, got shape {values.shape}")

    if kind == "sam3":
        if not rgb_file:
            return None
        with Image.open(rgb_file) as rgb:
            image = rgb.convert("RGB")
        image.thumbnail((840, 840), Image.Resampling.LANCZOS)
        mask = Image.fromarray((np.nan_to_num(values) > 0).astype("uint8") * 255)
        mask = mask.resize(image.size, Image.Resampling.NEAREST)
        green = Image.new("RGB", image.size, (0, 255, 100))
        result_image = Image.composite(Image.blend(image, green, 0.5), image, mask)
    else:
        # Resize the 2D float field BEFORE applying a colourmap, for speed.
        depth_float = np.asarray(values, dtype=np.float32)
        finite = depth_float[np.isfinite(depth_float)]
        if finite.size == 0:
            return None
        low, high = np.percentile(finite, (2, 98))
        depth_float = np.nan_to_num(depth_float, nan=low, posinf=high, neginf=low)
        depth_image = Image.fromarray(depth_float, mode="F")
        depth_image.thumbnail((840, 840), Image.Resampling.BILINEAR)
        depth_small = np.asarray(depth_image, dtype=np.float32)
        depth_small = np.clip((depth_small - low) / max(high - low, 1e-6), 0, 1)
        colours = (colormaps["turbo"](depth_small)[..., :3] * 255).astype(np.uint8)
        result_image = Image.fromarray(colours, mode="RGB")

    with BytesIO() as out:
        result_image.save(out, format="JPEG", quality=80)
        return out.getvalue(), "image/jpeg"


def serve_map(out_map: Path, registry: dict, host: str = "127.0.0.1", port: int = 8765) -> None:
    """Serve saved map and on-demand layers via localhost; no preview files."""
    @lru_cache(maxsize=64)
    def get_cached_image(image_id: str, kind: str):
        return render_saved_image(image_id, kind, registry)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            if parsed.path in {"/", "/map", f"/{out_map.name}"}:
                contents = out_map.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif parsed.path == "/image":
                query = parse_qs(parsed.query)
                image_id = query.get("image_id", [""])[0]
                kind = query.get("kind", [""])[0]
                if image_id not in registry or kind not in {"original", "sam3", "depth"}:
                    self.send_error(404, "Image not found in current dataset")
                    return
                try:
                    rendered = (
                        render_saved_image(image_id, kind, registry)
                        if kind == "original" else get_cached_image(image_id, kind)
                    )
                except Exception as exc:
                    print(f"[POPUP] Failed to render {kind} for {image_id}: {exc}")
                    self.send_error(500, "Could not read saved image")
                    return
                if rendered is None:
                    self.send_error(404, f"No saved {kind} result for {image_id}")
                    return
                contents, mime = rendered
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "private, max-age=3600")
            else:
                self.send_error(404)
                return
            self.send_header("Content-Length", str(len(contents)))
            self.end_headers()
            try:
                self.wfile.write(contents)
            except (BrokenPipeError, ConnectionResetError):
                pass

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        raise RuntimeError(f"Cannot start localhost image server on {host}:{port}: {exc}") from exc
    print(f"\n[MAP] Open in your browser: http://{host}:{port}/", flush=True)
    print("[MAP] SAM3 overlays / depth maps are loaded ONLY when clicked.", flush=True)
    print("[MAP] No preview folders or extra image files. Press Ctrl+C to stop the server.", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\n[MAP] Server stopped.")
    finally:
        server.server_close()


def make_image_scroller(image_items, color, server_base_url):

    # =========================================================
    # Prepare images and associated SAM3 / Depth outputs
    # =========================================================

    if not isinstance(image_items, list):
        return ""

    images = [
        item for item in image_items
        if isinstance(item, dict)
        and item.get("image_id") not in {None, "", "<NA>"}
        and (item.get("has_rgb") or isinstance(item.get("url"), str))
    ]

    if not images:
        return ""

    uid = hashlib.md5(
        "|".join(item["image_id"] for item in images).encode()
    ).hexdigest()[:10]

    slides = []
    n = len(images)

    for i, item in enumerate(images):

        image_id = quote(item["image_id"], safe="")
        original_url = (
            f"{server_base_url}/image?image_id={image_id}&kind=original"
            if item.get("has_rgb") else item.get("url", "")
        )
        url_html = escape(original_url, quote=True)

        # Only store URLs in the HTML: .npy arrays are never loaded at map-build time.
        sources = [("Original", original_url)]
        if item.get("has_sam3"):
            sources.append(("SAM3", f"{server_base_url}/image?image_id={image_id}&kind=sam3"))
        if item.get("has_depth"):
            sources.append(("Depth Anything", f"{server_base_url}/image?image_id={image_id}&kind=depth"))

        # =====================================================
        # Buttons to switch between visualisations
        # =====================================================

        buttons_html = "".join(
            f'<button type="button" '
            f'data-src="{escape(src, quote=True)}" '
            'onclick="var slide=this.closest(\'.tree-slide\');'
            'slide.querySelector(\'.tree-photo\').src=this.dataset.src;'
            'slide.querySelector(\'.tree-photo-link\').href=this.dataset.src;" '
            'style="'
            'padding:8px 12px;'
            'border:1px solid #555;'
            'border-radius:7px;'
            'background:#f0f0f0;'
            'font-weight:bold;'
            'cursor:pointer;'
            '">'
            f'{escape(label)}</button>'
            for label, src in sources
        )

        prev_i = (i - 1) % n
        next_i = (i + 1) % n

        slide_id = f"tree-slider-{uid}-{i}"
        slide = f"""
        <div
            id="{slide_id}"
            class="tree-slide"
            style="
                min-width:100%;
                width:100%;
                flex:0 0 100%;
                scroll-snap-align:start;
                position:relative;
                box-sizing:border-box;
            "
        >

            <a
                class="tree-photo-link"
                href="{url_html}"
                target="_blank"
                style="display:block;"
            >
                <img
                    class="tree-photo"
                    loading="lazy"
                    src="{url_html}"
                    style="
                        width:100%;
                        height:470px;
                        object-fit:cover;
                        display:block;
                        border-radius:12px;
                        border:3px solid {color};
                        box-sizing:border-box;
                    "
                >
            </a>

            <div
                style="
                    position:absolute;
                    top:12px;
                    right:12px;
                    background:rgba(0,0,0,0.65);
                    color:white;
                    border-radius:14px;
                    padding:4px 9px;
                    font-size:12px;
                    font-weight:700;
                "
            >
                {i + 1} / {n}
            </div>

            <a
                href="#tree-slider-{uid}-{prev_i}"
                style="
                    position:absolute;
                    left:10px;
                    top:50%;
                    transform:translateY(-50%);
                    width:34px;
                    height:34px;
                    line-height:31px;
                    border-radius:50%;
                    background:rgba(0,0,0,0.55);
                    color:white;
                    font-size:25px;
                    font-weight:bold;
                    text-decoration:none;
                    text-align:center;
                "
            >
                &#8249;
            </a>

            <a
                href="#tree-slider-{uid}-{next_i}"
                style="
                    position:absolute;
                    right:10px;
                    top:50%;
                    transform:translateY(-50%);
                    width:34px;
                    height:34px;
                    line-height:31px;
                    border-radius:50%;
                    background:rgba(0,0,0,0.55);
                    color:white;
                    font-size:25px;
                    font-weight:bold;
                    text-decoration:none;
                    text-align:center;
                "
            >
                &#8250;
            </a>

            <!-- ===================================== -->
            <!-- SAM3 / DEPTH VISUALISATION BUTTONS   -->
            <!-- ===================================== -->

            <div style="
                display:flex;
                justify-content:center;
                gap:8px;
                margin-top:12px;
                margin-bottom:12px;
                flex-wrap:wrap;
            ">
                {buttons_html}
            </div>

        </div>
        """

        slides.append(slide)

    return f"""
    <div
        style="
            margin-top:12px;
            width:100%;
        "
    >

        <div
            style="
                width:100%;
                display:flex;
                overflow-x:auto;
                scroll-snap-type:x mandatory;
                scroll-behavior:smooth;
                border-radius:12px;
            "
        >
            {''.join(slides)}
        </div>

        <div
            style="
                text-align:center;
                margin-top:6px;
                font-size:12px;
                color:#666;
            "
        >
            Scroll / swipe to view images
        </div>

    </div>
    """

def make_popup_html(row, colormap, server_base_url):
    c_seq = row["CARBON_SEQUESTERED_KG_YR"]

    color = colormap(c_seq)

    img_html = make_image_scroller(
        row.get("POPUP_IMAGES", []),
        color,
        server_base_url,
    )

    # Display the calibrated interval where DBH comes from the model.
    cp_lower = row.get("DBH_CP_LOWER_CM", np.nan)
    cp_upper = row.get("DBH_CP_UPPER_CM", np.nan)
    dbh = row.get("DIAMETER_CM", np.nan)
    dbh_method = str(row.get("DBH_UNC_METHOD", "unavailable"))
    dbh_display_unc = row.get("DBH_UNC_CM_DISPLAY", np.nan)

    if pd.notna(cp_lower) and pd.notna(cp_upper):
        diameter_html = f'<div><b>Model DBH:</b> {dbh:.1f} cm</div>'
        dbh_cp_html = (
            f'<div style="margin:7px 0;padding:9px;background:#edf7ed;'
            f'border-left:4px solid #389b59;border-radius:4px;">'
            f'<b>DBH SplitCP interval (90% target):</b><br>'
            f'{cp_lower:.1f}–{cp_upper:.1f} cm'
            f'</div>'
        )
    elif dbh_method.startswith("Citizen") and pd.notna(dbh_display_unc):
        diameter_html = f'<div><b>Diameter:</b> {dbh:.1f} ± {dbh_display_unc:.1f} cm (20%)</div>'
        dbh_cp_html = ""
    else:
        diameter_html = f'<div><b>Diameter:</b> {dbh:.1f} cm</div>'
        dbh_cp_html = ""

    # =========================================================
    # One displayed DBH-derived carbon uncertainty per tree.
    # SplitCP is symmetric before the physical zero clipping;
    # the upper deviation retains that symmetric half-width.
    # Citizen DBH has an explicitly assumed +/-20% uncertainty.
    # =========================================================
    carbon_upper = row.get("CARBON_SEQUESTERED_DBH_UPPER_KG_YR", np.nan)
    carbon_lower = row.get("CARBON_SEQUESTERED_DBH_LOWER_KG_YR", np.nan)
    if pd.notna(carbon_upper) and pd.notna(carbon_lower):
        carbon_unc = max(0.0, float(carbon_upper) - float(c_seq))
        carbon_display = f"{c_seq:.1f} ± {carbon_unc:.1f} kg C / yr"
        if dbh_method.startswith("SplitCP"):
            carbon_label = "DBH SplitCP-derived uncertainty (90% DBH target)"
        else:
            carbon_label = "Citizen DBH uncertainty (assumed 20%)"
    else:
        carbon_display = f"{c_seq:.1f} kg C / yr"
        carbon_label = "DBH uncertainty unavailable"

    carbon_cp_html = f"""
    <div style="text-align:center; margin-bottom:12px;">
        <div style="font-size:26px; font-weight:900; color:{color};">
            {carbon_display}
        </div>
        <div style="font-size:13px; color:#666;">
            {carbon_label}
        </div>
    </div>
    """

    return f"""
    <div style="width:375px; font-family:Arial, sans-serif;">

        {carbon_cp_html}


        <div style="font-size:14px; text-align:center; margin-bottom:10px;">
            Average across {int(row.get("N_IMAGES", 1))} image(s)
        </div>


        <div><b>ID:</b> {row.get("ID")}</div>
        <div><b>Species:</b> {row.get("SPECIES_GROUP", "unknown")}</div>
        <div><b>Height:</b> {row.get("HEIGHT_M", np.nan):.1f} ± {row.get("HEIGHT_UNC_M", np.nan):.1f} m</div>
        {diameter_html}
        {dbh_cp_html}
        <div><b>Height source:</b> {row.get("HEIGHT_SOURCE")}</div>
        <div><b>Diameter source:</b> {row.get("DIAMETER_SOURCE")}</div>

        {img_html}
    </div>
    """

def marker_radius(value, vmin, vmax):
    if pd.isna(value):
        return 5.0

    return max(
        5.0,
        min(
            14.0,
            5.0 + 9.0 * ((value - vmin) / (vmax - vmin + 1e-9)),
        ),
    )


def build_map(df: pd.DataFrame, out_html: Path, server_base_url: str):
    df_map = df[
        df["LATITUDE"].notna()
        & df["LONGITUDE"].notna()
        & df["CARBON_SEQUESTERED_KG_YR"].notna()
    ].copy()

    print(f"Entry-level rows going into map: {len(df_map)}")

    if df_map.empty:
        print("No valid rows for map.")
        return

    m = folium.Map(
        location=[55.8642, -4.2518],
        zoom_start=11,
        tiles="Esri.WorldGrayCanvas",
    )

    # Make only the basemap grayscale
    gray_map_css = """
    <style>
    .leaflet-tile-pane {
        filter: grayscale(100%) brightness(105%) contrast(90%);
    }
    </style>
    """

    m.get_root().header.add_child(
        Element(gray_map_css)
)

    vmin = float(df_map["CARBON_SEQUESTERED_KG_YR"].min())
    vmax = float(df_map["CARBON_SEQUESTERED_KG_YR"].max())

    if vmin == vmax:
        vmax = vmin + 1e-6

    colormap = cm.LinearColormap(
        colors=["green", "yellow", "orange", "red"],
        vmin=vmin,
        vmax=vmax,
    )

    # colormap.caption = "Carbon sequestered (kg C / year)"
    # colormap.add_to(m)



    # =========================================================
    # CUSTOM MAP LEGEND — Carbon + AI inference
    # =========================================================

    n_inferred = int(df_map["HAS_MODEL_INFERENCE"].sum())
    n_not_inferred = len(df_map) - n_inferred

    midpoint = (vmin + vmax) / 2

    map_legend_html = f"""
    <div style="
        position:fixed;
        top:20px;
        right:20px;
        width:450px;
        max-width:calc(100vw - 40px);
        box-sizing:border-box;
        z-index:99999;
        background:rgba(255,255,255,0.97);
        padding:20px;
        border-radius:14px;
        border-right: 6px solid green;
        border:2px solid #333;
        box-shadow:0 4px 18px rgba(0,0,0,0.3);
        font-family:Arial,sans-serif;
    ">

        <!-- CARBON COLOUR BAR -->

        <div style="
            font-size:22px;
            font-weight:900;
            margin-bottom:5px;
        ">
            Estimated Carbon Sequestration
        </div>

        <div style="
            font-size:17px;
            font-weight:bold;
            margin-bottom:12px;
        ">
            kg C / year
        </div>

        <div style="
            width:100%;
            height:30px;
            border-radius:5px;
            border:2px solid #333;
            box-sizing:border-box;
            background:linear-gradient(
                to right,
                green 0%,
                yellow 33.33%,
                orange 66.67%,
                red 100%
            );
        "></div>

        <div style="
            display:flex;
            justify-content:space-between;
            font-size:18px;
            font-weight:900;
            margin-top:7px;
        ">
            <span>{vmin:.1f}</span>
            <span>{midpoint:.1f}</span>
            <span>{vmax:.1f}</span>
        </div>

        <!-- SEPARATOR -->

        <hr style="
            border:0;
            border-top:2px solid #ddd;
            margin:16px 0;
        ">

        <!-- AI INFERENCE LEGEND -->

        <div style="
            font-size:20px;
            font-weight:900;
            margin-bottom:14px;
        ">
            Tree Measurement Sources
        </div>

        <div style="
            display:flex;
            align-items:center;
            gap:12px;
            margin-bottom:12px;
            font-size:17px;
            font-weight:bold;
        ">
            <span style="
                display:inline-block;
                width:19px;
                height:19px;
                flex-shrink:0;
                border-radius:50%;
                background:#61b84b;
                border:4px solid black;
            "></span>

            Inferred height and/or DBH ({n_inferred})
        </div>

        <div style="
            display:flex;
            align-items:center;
            gap:12px;
            font-size:17px;
            font-weight:bold;
        ">
            <span style="
                display:inline-block;
                width:19px;
                height:19px;
                flex-shrink:0;
                border-radius:50%;
                background:#61b84b;
                border:2px solid white;
                outline:1px solid #aaa;
            "></span>

            Observed (est./mea.) dimensions ({n_not_inferred})
        </div>

        <div style="
            font-size:14px;
            margin-top:15px;
            color:#555;
        ">
            Circle colour and size indicate estimated carbon sequestration.
        </div>

    </div>
    """

    m.get_root().html.add_child(Element(map_legend_html))

    for _, row in df_map.iterrows():
        seq = float(row["CARBON_SEQUESTERED_KG_YR"])
        color = colormap(seq)
        inferred = bool(row["HAS_MODEL_INFERENCE"])

        folium.CircleMarker(
            location=[row["LATITUDE"], row["LONGITUDE"]],
            radius=marker_radius(seq, vmin, vmax),
            # Highlight model-inferred measurements
            color="#000000" if inferred else "#ffffff",
            weight=3.5 if inferred else 1.0,
            fill=True,
            fill_color=color,
            fill_opacity=0.85 if inferred else 0.55,
            popup=folium.Popup(
            make_popup_html(row, colormap, server_base_url),
            max_width=420,
        ),
        ).add_to(m)

    # =========================================================
    # TOTAL CARBON SEQUESTRATION WITH DBH UNCERTAINTY BOUNDS
    # =========================================================

    # Central carbon sequestration
    central = df_map["CARBON_SEQUESTERED_KG_YR"]

    # DBH-derived bounds include SplitCP for model DBH and +/-20% for
    # citizen-derived DBH. Trees without an applicable method retain
    # their central value in the total.
    lower = df_map["CARBON_SEQUESTERED_DBH_LOWER_KG_YR"].fillna(central)
    upper = df_map["CARBON_SEQUESTERED_DBH_UPPER_KG_YR"].fillna(central)

    # Total carbon sequestration (kg C/year)
    total_c = central.sum()
    total_c_lower = lower.sum()
    total_c_upper = upper.sum()

    # Total CO2 equivalent (kg CO2/year)
    co2_factor = 44.0 / 12.0

    total_co2 = total_c * co2_factor
    total_co2_lower = total_c_lower * co2_factor
    total_co2_upper = total_c_upper * co2_factor

    # Total kettle-boil equivalent
    total_kettles = total_co2 / KG_CO2E_PER_KETTLE_BOIL
    total_kettles_lower = total_co2_lower / KG_CO2E_PER_KETTLE_BOIL
    total_kettles_upper = total_co2_upper / KG_CO2E_PER_KETTLE_BOIL

    summary_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        width: 330px;
        z-index: 9999;
        font-size: 15px;
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 0 15px rgba(0,0,0,0.25);
        border: 2px solid #333;
        border-left: 6px solid green;
        font-family: Arial, sans-serif;
    ">
        <div style="font-size:18px; font-weight:800; margin-bottom:10px;">
            Carbon estimates from {len(df_map)} tree entries
        </div>
        <div style="font-size:12px;color:#666;margin-bottom:8px;">
            DBH bounds: SplitCP for model estimates; ±20% for citizen DBH.
        </div>

        <b>Total C sequestration:</b><br>
        {total_c:,.2f} kg C / year<br>
        <span style="color:#278b46;">
            DBH-derived range: {total_c_lower:,.2f}–{total_c_upper:,.2f} kg C / year
        </span><br><br>

        <b>Total CO₂e sequestration:</b><br>
        {total_co2:,.2f} kg CO₂e / year<br>
        <span style="color:#278b46;">
            DBH-derived range: {total_co2_lower:,.2f}–{total_co2_upper:,.2f} kg CO₂e / year
        </span><br><br>

        <b>Kettle boils equivalent:</b><br>
        {total_kettles:,.0f} kettle boils / year<br>
        <span style="color:#278b46;">
            DBH-derived range: {total_kettles_lower:,.0f}–{total_kettles_upper:,.0f}
        </span>
    </div>
    """

    m.get_root().html.add_child(Element(summary_html))
    m.save(str(out_html))

    print(f"Saved map to: {out_html}")


# =========================================================
# CLI
# =========================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Compute TreeCo carbon stock and sequestration from a combined "
            "image-level inferred height + width manifest, aggregated to entry/tree level."
        )
    )

    ap.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Combined manifest with inferred heights and widths.",
    )

    ap.add_argument(
        "--dataset-dir", "--dataset_dir",
        dest="dataset_dir",
        required=True,
        type=Path,
        help="Built TreeCo dataset folder containing manifests/, original_rgb/, sam3_largest_tree_masks/, depth_maps/.",
    )

    ap.add_argument(
        "--serve",
        action="store_true",
        help="After building the map, run a localhost server for on-demand SAM3/depth layers.",
    )

    ap.add_argument("--port", type=int, default=8765, help="Localhost map port (default: 8765).")

    ap.add_argument(
        "--dbh-conformal-csv", "--dbh_conformal_csv",
        dest="dbh_conformal_csv", type=Path, default=None,
        help="Saved tree-level puncc_split_tree_predictions.csv for the SAME "
             "DBH model used in width inference. Apply fixed-width 90%% SplitCP "
             "intervals to model-inferred DBH only.",
    )

    ap.add_argument(
        "--outdir",
        default=None,
        type=Path,
        help="Output directory. Default: manifest parent.",
    )

    ap.add_argument(
        "--csv-name",
        default="tree_carbon_estimates_entry_level.csv",
    )

    ap.add_argument(
        "--map-name",
        default="tree_carbon_sequestration_map.html",
    )

    ap.add_argument(
        "--pred-is-circumference",
        action="store_true",
        help=(
            "Use this if DBH_CM_PRED is actually predicted circumference in cm. "
            "It will convert to diameter by dividing by pi."
        ),
    )

    return ap.parse_args()


def main():
    args = parse_args()

    outdir = args.outdir if args.outdir is not None else args.manifest.parent
    outdir.mkdir(parents=True, exist_ok=True)

    dbh_cp_half_width = (
        read_splitcp_half_width(args.dbh_conformal_csv)
        if args.dbh_conformal_csv is not None else None
    )

    image_registry = {}
    df = build_carbon_table(
        manifest_path=args.manifest,
        pred_is_circumference=args.pred_is_circumference,
        dataset_dir=args.dataset_dir,
        image_registry=image_registry,
        dbh_cp_half_width=dbh_cp_half_width,
    )

    out_csv = outdir / args.csv_name
    out_map = outdir / args.map_name

    df.to_csv(out_csv, index=False)
    print(f"Saved carbon table to: {out_csv}")

    build_map(df, out_map, f"http://127.0.0.1:{args.port}")

    preview_cols = [
        "ID",
        "SPECIES_GROUP",
        "N_IMAGES",
        "HEIGHT_M",
        "DIAMETER_CM",
        "DBH_CP_LOWER_CM",
        "DBH_CP_UPPER_CM",
        "CARBON_STOCK_KG",
        "CARBON_SEQUESTERED_KG_YR",
        "CO2_SEQUESTERED_KG_YR",
        "KETTLE_BOILS_PER_YEAR",
        "HEIGHT_SOURCE",
        "DIAMETER_SOURCE",
    ]

    preview_cols = [c for c in preview_cols if c in df.columns]

    print("\nPreview:")
    print(df[preview_cols].head(10).to_string(index=False))

    if args.serve:
        serve_map(out_map, image_registry, port=args.port)
    else:
        print("\n[MAP] Layer images require the localhost server. Use --serve to enable SAM3/depth buttons.")


if __name__ == "__main__":
    main()