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
    "10+ metres": (12.5, 2.5),
    "5-10 metres": (7.5, 2.5),
    "0-5 metres": (2.5, 2.5),
}


# =========================================================
# Helpers
# =========================================================

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


def find_media_cols(df):
    return [c for c in df.columns if str(c).upper().startswith("MEDIA_")]


def parse_numeric(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip().lower()

    if s in {"", "nan", "none", "null"}:
        return np.nan

    nums = re.findall(r"\d+(?:\.\d+)?", s)

    if not nums:
        return np.nan

    vals = [float(v) for v in nums]

    if "+" in s:
        return vals[0]

    if len(vals) >= 2:
        return float(np.mean(vals))

    return vals[0]


def parse_height(row):
    """Use ALL citizen-provided heights; numeric measurements before categories.

    Raw height ranges are interpreted using their midpoints, consistently
    with the updated TreeCo mapping pipeline (e.g. 2-5 -> 3.5 metres).
    """
    for col in ("TREE_HEIGHT_IN_METERS", "TREE_HEIGHT_IN_METERS_CLEAN"):
        value = row.get(col)
        if pd.notna(value) and str(value).strip():
            height = parse_numeric(value)
            if pd.notna(height) and height > 0:
                return float(height), 1.0, "citizen_measured"

    for col in ("ESTIMATED_TREE_HEIGHT", "ESTIMATED_TREE_HEIGHT_CLEAN", "HEIGHT_CLASS_STR"):
        value = row.get(col)
        if pd.isna(value) or not str(value).strip():
            continue

        label = str(value).strip().lower().replace("–", "-").replace("—", "-")
        if label in HEIGHT_CLASS_MAP:
            height, uncertainty = HEIGHT_CLASS_MAP[label]
        else:
            height, uncertainty = parse_numeric(label), 2.5

        if pd.notna(height) and height > 0:
            return float(height), float(uncertainty), "citizen_estimated"

    # HEIGHT_VALUE_M may exist in processed data but is not a reliable
    # measurement-provenance indicator. Never use it to claim 'measured'.
    return np.nan, np.nan, "missing"


def trunk_size_to_circumference_cm(value):
    """Turn a citizen trunk-size category into an estimated circumference.

    Mirrors the TreeCo dataset-builder convention; the assumption that
    TREE_TRUNK_SIZE denotes *circumference*, not diameter, must be verified.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower().replace("–", "-").replace("—", "-")

    match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", s)
    if match:
        return (float(match.group(1)) + float(match.group(2))) / 2.0

    match = re.search(r"(\d+(?:\.\d+)?)\s*\+", s)
    return float(match.group(1)) if match else np.nan


def parse_circumference(row):
    """Keep measured circumference AND citizen-estimated trunk-size bins.

    Consistent priority with the revised TreeCo inference map:
    measured raw circumference > estimated trunk-size category > raw
    numeric fallback > cleaned numeric fallback. No AI values are used.
    """
    method = str(row.get("TREE_CIRCUMFERENCE_METHOD", "") or "").strip().lower()
    is_measured = "tape" in method or "measure" in method
    raw = parse_numeric(row.get("CIRCUMFERENCE_IN_CM"))
    estimated = trunk_size_to_circumference_cm(row.get("TREE_TRUNK_SIZE"))

    if is_measured and pd.notna(raw) and raw > 0:
        return float(raw), max(0.10 * float(raw), 2.0 * np.pi), "citizen_measured"

    if pd.notna(estimated) and estimated > 0:
        return float(estimated), max(0.10 * float(estimated), 2.0 * np.pi), "citizen_estimated_trunk_size"

    # Retain numerical citizen contributions even if their method wasn't
    # recorded. These are reported, not confirmed tape-measured.
    if pd.notna(raw) and raw > 0:
        return float(raw), max(0.10 * float(raw), 2.0 * np.pi), "citizen_reported_numeric"

    clean = parse_numeric(row.get("CIRCUMFERENCE_CM_CLEAN"))
    if pd.notna(clean) and clean > 0:
        return float(clean), max(0.10 * float(clean), 2.0 * np.pi), "citizen_reported_clean"

    return np.nan, np.nan, "missing"


def carbon_stock_from_height_diameter(height_m, diameter_cm):
    if pd.isna(height_m) or pd.isna(diameter_cm):
        return np.nan

    if height_m <= 0 or diameter_cm <= 0:
        return np.nan

    if height_m < 28:
        return 0.0577 * height_m**2 * diameter_cm * 0.25

    return 0.0346 * height_m**2 * diameter_cm * 0.25


def carbon_stock_uncertainty(height_m, height_unc_m, diameter_cm, diameter_unc_cm, carbon_stock_kg):
    if any(pd.isna(x) for x in [height_m, height_unc_m, diameter_cm, diameter_unc_cm, carbon_stock_kg]):
        return np.nan

    if height_m <= 0 or diameter_cm <= 0 or carbon_stock_kg <= 0:
        return np.nan

    rel2 = (2.0 * height_unc_m / height_m) ** 2 + (diameter_unc_cm / diameter_cm) ** 2
    return carbon_stock_kg * np.sqrt(rel2)


def load_table(path):
    import csv

    path = Path(path).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    suffix = path.suffix.lower()

    # =====================================================
    # Excel
    # =====================================================
    if suffix in {".xlsx", ".xls"}:
        print(f"[DATA] Reading Excel: {path}")
        return pd.read_excel(path)

    # =====================================================
    # CommuniMap CSV
    # =====================================================
    if suffix == ".csv":

        print(f"[DATA] Reading CommuniMap CSV: {path}")

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

            header = next(reader)
            rows = list(reader)

        original_ncols = len(header)

        max_ncols = max(
            [len(header)]
            + [len(row) for row in rows]
        )

        print(
            f"[DATA] Header columns: "
            f"{original_ncols}"
        )

        print(
            f"[DATA] Maximum row columns: "
            f"{max_ncols}"
        )

        # -------------------------------------------------
        # Handle extra MEDIA columns
        # -------------------------------------------------
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

                col_str = str(col)

                if col_str.startswith(
                    "MEDIA_2635_"
                ):

                    try:
                        media_indices.append(
                            int(
                                col_str.split("_")[-1]
                            )
                        )

                    except ValueError:
                        pass

            next_media_idx = (
                max(media_indices) + 1
                if media_indices
                else 0
            )

            for i in range(extra):

                new_col = (
                    f"MEDIA_2635_"
                    f"{next_media_idx + i}"
                )

                print(
                    f"[DATA] Adding column: "
                    f"{new_col}"
                )

                header.append(new_col)

        # -------------------------------------------------
        # Pad shorter rows
        # -------------------------------------------------
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

    raise ValueError(
        f"Unsupported file type: {path}"
    )


def list_images_for_group(group):
    """Keep ALL original image URLs, including rows lacking dimensions."""
    media_cols = find_media_cols(group)
    urls = []
    for _, row in group.iterrows():
        for col in media_cols:
            value = row.get(col)
            if pd.notna(value) and str(value).strip().startswith("http"):
                urls.append(str(value).strip())
    return list(dict.fromkeys(urls))


def first_non_missing_source(series):
    present = [str(s) for s in series.dropna() if str(s).strip().lower() not in {"", "missing", "nan"}]
    if not present:
        return "missing"
    distinct = list(dict.fromkeys(present))
    return distinct[0] if len(distinct) == 1 else "mixed: " + ", ".join(distinct)


def aggregate_to_entry_level(df):
    """Do not discard an entry or its photographs if dimensions are missing."""
    rows = []
    numeric_avg_cols = [
        "HEIGHT_M", "HEIGHT_UNC_M", "CIRCUMFERENCE_CM", "CIRCUMFERENCE_UNC_CM",
        "DIAMETER_CM", "DIAMETER_UNC_CM", "CARBON_STOCK_KG", "CARBON_STOCK_UNC_KG",
        "CARBON_SEQUESTERED_KG_YR", "CARBON_SEQUESTERED_UNC_KG_YR",
        "CO2_SEQUESTERED_KG_YR", "CO2_SEQUESTERED_UNC_KG_YR",
        "KETTLE_BOILS_PER_YEAR", "KETTLE_BOILS_UNC_PER_YEAR",
    ]

    for tree_id, group in df.groupby("ID", dropna=False, sort=False):
        out = group.iloc[0].to_dict()
        out["N_ROWS_FOR_ID"] = len(group)
        out["IMAGE_URLS"] = list_images_for_group(group)
        out["N_ORIGINAL_IMAGES"] = len(out["IMAGE_URLS"])

        # Use the first *valid pair* of coordinates rather than blindly
        # taking coordinates from the first record for that ID.
        coords = group[
            group["LATITUDE"].between(-90, 90)
            & group["LONGITUDE"].between(-180, 180)
        ]
        if not coords.empty:
            out["LATITUDE"] = float(coords.iloc[0]["LATITUDE"])
            out["LONGITUDE"] = float(coords.iloc[0]["LONGITUDE"])
        else:
            out["LATITUDE"] = np.nan
            out["LONGITUDE"] = np.nan

        # Keep dimensions used for the carbon estimate consistent with its
        # contributing rows. Other citizen records still contribute images
        # and remain accessible in the raw-row audit CSV.
        paired = group.loc[group["ROW_HAS_BOTH_DIMENSIONS"]]
        dimension_rows = paired if not paired.empty else group
        for col in numeric_avg_cols:
            rows_for_col = dimension_rows if col in {
                "HEIGHT_M", "HEIGHT_UNC_M", "CIRCUMFERENCE_CM",
                "CIRCUMFERENCE_UNC_CM", "DIAMETER_CM", "DIAMETER_UNC_CM"
            } else group
            out[col] = pd.to_numeric(rows_for_col[col], errors="coerce").mean()

        out["HEIGHT_SOURCE"] = first_non_missing_source(dimension_rows["HEIGHT_SOURCE"])
        out["CIRCUMFERENCE_SOURCE"] = first_non_missing_source(dimension_rows["CIRCUMFERENCE_SOURCE"])
        out["HAS_ANY_MEASURED_OR_ESTIMATED_HEIGHT"] = bool(group["HEIGHT_M"].notna().any())
        out["HAS_ANY_MEASURED_OR_ESTIMATED_CIRCUMFERENCE"] = bool(group["CIRCUMFERENCE_CM"].notna().any())
        out["N_ROWS_WITH_BOTH_DIMENSIONS"] = int(group["ROW_HAS_BOTH_DIMENSIONS"].sum())
        out["COMBINED_ACROSS_ROWS"] = False
        rows.append(out)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        return out_df

    # If one record has a height, another record of the SAME entry has
    # circumference, we have enough citizen information even if neither
    # individual record had both. Only fill estimates previously missing.
    combine = (
        out_df["CARBON_STOCK_KG"].isna()
        & out_df["HEIGHT_M"].gt(0)
        & out_df["DIAMETER_CM"].gt(0)
    )
    out_df.loc[combine, "COMBINED_ACROSS_ROWS"] = True
    if combine.any():
        out_df.loc[combine, "CARBON_STOCK_KG"] = out_df.loc[combine].apply(
            lambda r: carbon_stock_from_height_diameter(r["HEIGHT_M"], r["DIAMETER_CM"]), axis=1
        )
        out_df.loc[combine, "CARBON_STOCK_UNC_KG"] = out_df.loc[combine].apply(
            lambda r: carbon_stock_uncertainty(
                r["HEIGHT_M"], r["HEIGHT_UNC_M"], r["DIAMETER_CM"],
                r["DIAMETER_UNC_CM"], r["CARBON_STOCK_KG"]
            ), axis=1
        )
        carbon = out_df.loc[combine, "CARBON_STOCK_KG"]
        carbon_unc = out_df.loc[combine, "CARBON_STOCK_UNC_KG"]
        out_df.loc[combine, "CARBON_SEQUESTERED_KG_YR"] = carbon * SEQ_RATE_MEAN
        out_df.loc[combine, "CARBON_SEQUESTERED_UNC_KG_YR"] = np.sqrt(
            (carbon * SEQ_RATE_UNC) ** 2 + (SEQ_RATE_MEAN * carbon_unc) ** 2
        )
        out_df.loc[combine, "CO2_SEQUESTERED_KG_YR"] = (
            out_df.loc[combine, "CARBON_SEQUESTERED_KG_YR"] * (44.0 / 12.0)
        )
        out_df.loc[combine, "CO2_SEQUESTERED_UNC_KG_YR"] = (
            out_df.loc[combine, "CARBON_SEQUESTERED_UNC_KG_YR"] * (44.0 / 12.0)
        )
        out_df.loc[combine, "KETTLE_BOILS_PER_YEAR"] = (
            out_df.loc[combine, "CO2_SEQUESTERED_KG_YR"] / KG_CO2E_PER_KETTLE_BOIL
        )
        out_df.loc[combine, "KETTLE_BOILS_UNC_PER_YEAR"] = (
            out_df.loc[combine, "CO2_SEQUESTERED_UNC_KG_YR"] / KG_CO2E_PER_KETTLE_BOIL
        )

    out_df["HAS_CARBON_ESTIMATE"] = out_df["CARBON_SEQUESTERED_KG_YR"].notna()
    out_df["HAS_COORDINATES"] = (
        out_df["LATITUDE"].between(-90, 90)
        & out_df["LONGITUDE"].between(-180, 180)
    )
    return out_df


# =========================================================
# Carbon table
# =========================================================

def build_raw_carbon_table(input_path, raw_rows_output=None):
    df = load_table(input_path)
    print(f"Loaded: {input_path}")
    print(f"Rows before filtering: {len(df)}")

    if "TREE" in df.columns:
        df = df[df["TREE"].astype(str).str.strip().str.lower().eq("yes")].copy()

    print(f"Tree contribution rows (all retained): {len(df)}")
    if df.empty:
        return pd.DataFrame()

    df["ID"] = df["ID"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
    df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")

    height_rows = df.apply(parse_height, axis=1)
    df[["HEIGHT_M", "HEIGHT_UNC_M", "HEIGHT_SOURCE"]] = pd.DataFrame(
        height_rows.tolist(), index=df.index
    )

    circ_rows = df.apply(parse_circumference, axis=1)
    df[["CIRCUMFERENCE_CM", "CIRCUMFERENCE_UNC_CM", "CIRCUMFERENCE_SOURCE"]] = pd.DataFrame(
        circ_rows.tolist(), index=df.index
    )

    df["DIAMETER_CM"] = df["CIRCUMFERENCE_CM"] / np.pi
    df["DIAMETER_UNC_CM"] = df["CIRCUMFERENCE_UNC_CM"] / np.pi

    df["ROW_HAS_BOTH_DIMENSIONS"] = df["HEIGHT_M"].gt(0) & df["DIAMETER_CM"].gt(0)
    print("Rows with both citizen dimensions:", int(df["ROW_HAS_BOTH_DIMENSIONS"].sum()))

    if "TREE_TYPE" in df.columns:
        species = df["TREE_TYPE"].copy()
        if "OTHER_TREE" in df.columns:
            other_mask = species.astype(str).str.lower().eq("other")
            species.loc[other_mask] = df.loc[other_mask, "OTHER_TREE"]
        df["SPECIES"] = species
    else:
        df["SPECIES"] = "unknown"

    df["SPECIES_CLEAN"] = df["SPECIES"].apply(clean_species_name)
    df["SPECIES_GROUP"] = df["SPECIES_CLEAN"].apply(canonical_species_group)

    # Incomplete rows are NOT filtered out. They naturally receive NaN
    # carbon, while their labels/images remain available for aggregation.
    df["CARBON_STOCK_KG"] = df.apply(
        lambda r: carbon_stock_from_height_diameter(r["HEIGHT_M"], r["DIAMETER_CM"]), axis=1
    )
    df["CARBON_STOCK_UNC_KG"] = df.apply(
        lambda r: carbon_stock_uncertainty(
            r["HEIGHT_M"], r["HEIGHT_UNC_M"], r["DIAMETER_CM"],
            r["DIAMETER_UNC_CM"], r["CARBON_STOCK_KG"]
        ), axis=1
    )

    df["SEQ_RATE_YR"] = SEQ_RATE_MEAN
    df["SEQ_RATE_UNC_YR"] = SEQ_RATE_UNC
    df["CARBON_SEQUESTERED_KG_YR"] = df["CARBON_STOCK_KG"] * SEQ_RATE_MEAN
    df["CARBON_SEQUESTERED_UNC_KG_YR"] = np.sqrt(
        (df["CARBON_STOCK_KG"] * SEQ_RATE_UNC) ** 2
        + (SEQ_RATE_MEAN * df["CARBON_STOCK_UNC_KG"]) ** 2
    )
    df["CO2_SEQUESTERED_KG_YR"] = df["CARBON_SEQUESTERED_KG_YR"] * (44.0 / 12.0)
    df["CO2_SEQUESTERED_UNC_KG_YR"] = df["CARBON_SEQUESTERED_UNC_KG_YR"] * (44.0 / 12.0)
    df["KETTLE_BOILS_PER_YEAR"] = df["CO2_SEQUESTERED_KG_YR"] / KG_CO2E_PER_KETTLE_BOIL
    df["KETTLE_BOILS_UNC_PER_YEAR"] = df["CO2_SEQUESTERED_UNC_KG_YR"] / KG_CO2E_PER_KETTLE_BOIL

    # Save every original TREE=yes contribution row, including rows with
    # no size measurements, before collapsing multiple rows to each tree ID.
    if raw_rows_output is not None:
        df.to_csv(raw_rows_output, index=False)
        print(f"Saved every raw citizen tree row to: {raw_rows_output}")

    df_entry = aggregate_to_entry_level(df)
    print("All unique citizen tree entries:", len(df_entry))
    print("Entries with both usable citizen dimensions:", int(df_entry["HAS_CARBON_ESTIMATE"].sum()))
    print("Entries completing dimensions across rows:", int(df_entry["COMBINED_ACROSS_ROWS"].sum()))
    print("Entries with location and carbon:", int((df_entry["HAS_CARBON_ESTIMATE"] & df_entry["HAS_COORDINATES"]).sum()))
    print("Entries preserved without carbon estimate:", int((~df_entry["HAS_CARBON_ESTIMATE"]).sum()))
    print("Citizen height sources:\n", df_entry["HEIGHT_SOURCE"].value_counts().to_string())
    print("Citizen circumference sources:\n", df_entry["CIRCUMFERENCE_SOURCE"].value_counts().to_string())
    return df_entry


# =========================================================
# Map
# =========================================================

def make_image_scroller(image_urls, color):
    import re
    import hashlib

    if image_urls is None:
        return ""

    if isinstance(image_urls, list):
        urls = image_urls
    else:
        urls = re.findall(
            r'https?://[^\s,"\']+',
            str(image_urls),
        )

    urls = [
        u.strip()
        for u in urls
        if str(u).startswith("http")
    ]

    if not urls:
        return ""

    # Unique ID so sliders from different tree popups
    # do not interfere with each other.
    uid = hashlib.md5(
        "|".join(urls).encode("utf-8")
    ).hexdigest()[:10]

    slides = []

    n = len(urls)

    for i, url in enumerate(urls):

        prev_i = (i - 1) % n
        next_i = (i + 1) % n

        slide_id = f"tree-slider-{uid}-{i}"

        slide = f"""
        <div
            id="{slide_id}"
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
                href="{url}"
                target="_blank"
                style="display:block;"
            >
                <img
                    src="{url}"
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


def fmt(value, digits=1):
    return "Not provided" if pd.isna(value) else f"{float(value):,.{digits}f}"


def make_popup_html(row, colormap):
    seq = row.get("CARBON_SEQUESTERED_KG_YR", np.nan)
    has_carbon = pd.notna(seq)
    color = colormap(float(seq)) if has_carbon else "#888888"
    images = make_image_scroller(row.get("IMAGE_URLS", []), color)

    if has_carbon:
        carbon_info = f"""
            <div style="font-size:24px;font-weight:900;color:{color};text-align:center;">
                {fmt(seq, 3)} ± {fmt(row.get('CARBON_SEQUESTERED_UNC_KG_YR'), 3)} kg C / yr
            </div>
            <div style="font-size:14px;text-align:center;">
                {fmt(row.get('CO2_SEQUESTERED_KG_YR'), 3)} kg CO₂e / yr
                · ≈ {fmt(row.get('KETTLE_BOILS_PER_YEAR'), 1)} kettle boils / yr
            </div>"""
    else:
        carbon_info = """
            <div style="font-size:19px;font-weight:800;color:#666;text-align:center;">
                No carbon estimate (missing citizen dimension)
            </div>"""

    return f"""
    <div style="width:375px;font-family:Arial,sans-serif;">
        {carbon_info}
        <div style="font-size:13px;text-align:center;margin:10px 0;">
            {int(row.get('N_ROWS_FOR_ID', 1))} original record(s) ·
            {int(row.get('N_ORIGINAL_IMAGES', 0))} original photograph(s)
        </div>
        <div><b>ID:</b> {row.get('ID', 'NA')}</div>
        <div><b>Species:</b> {row.get('SPECIES_GROUP', 'unknown')}</div>
        <div><b>Height:</b> {fmt(row.get('HEIGHT_M'))} ± {fmt(row.get('HEIGHT_UNC_M'))} m</div>
        <div><b>Circumference:</b> {fmt(row.get('CIRCUMFERENCE_CM'))} cm</div>
        <div><b>Diameter:</b> {fmt(row.get('DIAMETER_CM'))} ± {fmt(row.get('DIAMETER_UNC_CM'))} cm</div>
        <div><b>Height source:</b> {row.get('HEIGHT_SOURCE', 'missing')}</div>
        <div><b>Circumference source:</b> {row.get('CIRCUMFERENCE_SOURCE', 'missing')}</div>
        {images}
    </div>
    """


def marker_radius(value, vmin, vmax):
    if pd.isna(value):
        return 5.0

    return max(
        5.0,
        min(14.0, 5.0 + 9.0 * ((value - vmin) / (vmax - vmin + 1e-9))),
    )


def build_map(df, out_html):
    # Every raw tree entry is preserved in the all-entries CSV. Only valid
    # geographical positions can be mapped; non-estimable trees are shown
    # as a separate, toggleable grey layer rather than being discarded.
    geo = df[df["HAS_COORDINATES"]].copy()
    with_carbon = geo[geo["CARBON_SEQUESTERED_KG_YR"].notna()].copy()
    without_carbon = geo[geo["CARBON_SEQUESTERED_KG_YR"].isna()].copy()

    print("Map: citizen-only carbon estimates:", len(with_carbon))
    print("Map: other citizen tree observations (grey):", len(without_carbon))
    print("Not mappable (missing/invalid coordinates):", len(df) - len(geo))

    if geo.empty:
        print("No tree entries with valid coordinates to plot.")
        return

    m = folium.Map(location=[55.8642, -4.2518], zoom_start=11, tiles="Esri.WorldGrayCanvas")
    m.get_root().header.add_child(Element("""
    <style>.leaflet-tile-pane { filter: grayscale(100%) brightness(105%) contrast(90%); }</style>
    """))

    if with_carbon.empty:
        vmin, vmax = 0.0, 1.0
    else:
        vmin = float(with_carbon["CARBON_SEQUESTERED_KG_YR"].min())
        vmax = float(with_carbon["CARBON_SEQUESTERED_KG_YR"].max())
        if vmin == vmax:
            vmax = vmin + 1e-6

    colormap = cm.LinearColormap(colors=["green", "yellow", "orange", "red"], vmin=vmin, vmax=vmax)
    colormap.caption = "Carbon sequestration from citizen-provided measurements and estimates (kg C/year)"
    if not with_carbon.empty:
        colormap.add_to(m)

    carbon_layer = folium.FeatureGroup(name=f"Citizen-only carbon estimates ({len(with_carbon)})", show=True)
    other_layer = folium.FeatureGroup(name=f"Other citizen tree observations ({len(without_carbon)})", show=False)

    for _, row in with_carbon.iterrows():
        seq = float(row["CARBON_SEQUESTERED_KG_YR"])
        color = colormap(seq)
        folium.CircleMarker(
            location=[float(row["LATITUDE"]), float(row["LONGITUDE"])],
            radius=marker_radius(seq, vmin, vmax), color=color, fill=True,
            fill_color=color, fill_opacity=0.85, weight=2,
            popup=folium.Popup(make_popup_html(row, colormap), max_width=420),
        ).add_to(carbon_layer)

    for _, row in without_carbon.iterrows():
        folium.CircleMarker(
            location=[float(row["LATITUDE"]), float(row["LONGITUDE"])],
            radius=4, color="#666666", fill=True,
            fill_color="#999999", fill_opacity=0.5, weight=1,
            popup=folium.Popup(make_popup_html(row, colormap), max_width=420),
        ).add_to(other_layer)

    carbon_layer.add_to(m)
    other_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    total_c = with_carbon["CARBON_SEQUESTERED_KG_YR"].sum()
    total_c_unc = np.sqrt(np.square(with_carbon["CARBON_SEQUESTERED_UNC_KG_YR"].fillna(0)).sum())
    total_co2 = with_carbon["CO2_SEQUESTERED_KG_YR"].sum()
    total_co2_unc = np.sqrt(np.square(with_carbon["CO2_SEQUESTERED_UNC_KG_YR"].fillna(0)).sum())
    total_kettles = with_carbon["KETTLE_BOILS_PER_YEAR"].sum()
    total_kettles_unc = np.sqrt(np.square(with_carbon["KETTLE_BOILS_UNC_PER_YEAR"].fillna(0)).sum())

    summary_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;width:330px;z-index:9999;
                font-size:15px;background:white;border:2px solid #333;border-radius:12px;
                padding:15px;box-shadow:0 0 15px rgba(0,0,0,0.25);font-family:Arial,sans-serif;">
        <div style="font-size:18px;font-weight:800;margin-bottom:10px;">
            Citizen contributions: {len(with_carbon)} carbon-estimable tree entries
        </div>
        <div style="font-size:12px;color:#666;margin-bottom:10px;">
            {len(without_carbon)} more mapped entries lack a dimension (toggle grey layer);
            {len(df) - len(geo)} entries lack valid coordinates.
        </div>
        <b>Total C sequestration:</b><br>
        {total_c:,.2f} ± {total_c_unc:,.2f} kg C / year<br><br>
        <b>Total CO₂e sequestration:</b><br>
        {total_co2:,.2f} ± {total_co2_unc:,.2f} kg CO₂e / year<br><br>
        <b>Kettle boils equivalent:</b><br>
        {total_kettles:,.0f} ± {total_kettles_unc:,.0f} kettle boils / year
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
        description="Map ALL raw CommuniMap tree contributions; estimate carbon from available citizen measurements or size-category estimates (no AI)."
    )

    ap.add_argument(
        "--input-xlsx",
        "--input",
        required=True,
        type=Path,
        help="Raw CommuniMap XLSX/CSV file.",
    )

    ap.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Output directory.",
    )

    ap.add_argument(
        "--csv-name",
        default="raw_observed_tree_carbon_estimates_entry_level.csv",  # retained for old comparison scripts
    )

    ap.add_argument(
        "--raw-rows-csv-name",
        default="raw_citizen_all_tree_rows.csv",
        help="All TREE=yes raw records with parsed measurements, unaggregated.",
    )

    ap.add_argument(
        "--all-csv-name",
        default="raw_citizen_all_tree_entries.csv",
        help="Audit CSV retaining every tree entry, including missing dimensions or coordinates.",
    )

    ap.add_argument(
        "--map-name",
        default="raw_observed_tree_sequestration_map.html",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = build_raw_carbon_table(
        args.input_xlsx,
        raw_rows_output=args.outdir / args.raw_rows_csv_name,
    )

    out_csv = args.outdir / args.csv_name
    out_map = args.outdir / args.map_name

    # Preserve EVERY tree entry in an audit CSV. Carbon CSV stays compatible
    # with earlier workflows by containing only entries having mapped carbon.
    all_csv = args.outdir / args.all_csv_name
    df.to_csv(all_csv, index=False)
    print(f"Saved all citizen tree entries to: {all_csv}")

    carbon_df = df.loc[df["HAS_CARBON_ESTIMATE"] & df["HAS_COORDINATES"]].copy()
    carbon_df.to_csv(out_csv, index=False)
    print(f"Saved carbon-estimable entries to: {out_csv}")

    build_map(df, out_map)

    preview_cols = [
        "ID",
        "SPECIES_GROUP",
        "N_ROWS_FOR_ID",
        "HEIGHT_M",
        "CIRCUMFERENCE_CM",
        "DIAMETER_CM",
        "CARBON_STOCK_KG",
        "CARBON_SEQUESTERED_KG_YR",
        "CO2_SEQUESTERED_KG_YR",
        "KETTLE_BOILS_PER_YEAR",
    ]

    preview_cols = [c for c in preview_cols if c in df.columns]

    print("\nPreview:")
    print(carbon_df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()