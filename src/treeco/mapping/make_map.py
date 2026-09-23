#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
import ast

import branca.colormap as cm
from branca.element import Element
import folium
import numpy as np
import pandas as pd
from html import escape

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

    if height_m <= 0 or diameter_cm <= 0:
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
    Prefer observed height if present.
    Otherwise use inferred height class.
    """
    observed = pd.to_numeric(row.get("HEIGHT_VALUE_M"), errors="coerce")

    if pd.notna(observed):
        return observed, 1.0, "observed"

    observed_clean = pd.to_numeric(row.get("TREE_HEIGHT_IN_METERS_CLEAN"), errors="coerce")

    if pd.notna(observed_clean):
        return observed_clean, 1.0, "observed_clean"

    inferred_class = row.get("HEIGHT_CLASS_PRED_STR")

    h, u = height_class_to_midpoint(inferred_class)

    if pd.notna(h):
        return h, u, "model_inferred"

    fallback_class = row.get("HEIGHT_CLASS_STR")

    h, u = height_class_to_midpoint(fallback_class)

    if pd.notna(h):
        return h, u, "class_midpoint"

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
            return (
                float(final_dbh),
                max(0.10 * float(final_dbh), 2.0),
                "observed_dbh_final",
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

def build_carbon_table(
    manifest_path: Path,
    pred_is_circumference: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)

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

    agg = (
        df.groupby("ID", dropna=False)
        .agg(
            LATITUDE=("LATITUDE", "first"),
            LONGITUDE=("LONGITUDE", "first"),
            TREE_TYPE=("TREE_TYPE", first_non_null),
            TREE_SPECIES_LABEL=("TREE_SPECIES_LABEL", first_non_null),

            N_IMAGES=("IMAGE_ID", "nunique"),
            IMAGE_IDS=("IMAGE_ID", list_non_null),
            IMAGE_URLS=("MEDIA_SRC", list_non_null),

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

            HEIGHT_SOURCE=("HEIGHT_SOURCE_FINAL_IMAGE", first_non_null),
            DIAMETER_SOURCE=("DIAMETER_SOURCE_FINAL_IMAGE", first_non_null),
        )
        .reset_index()
    )

    agg["SEQ_RATE_YR"] = SEQ_RATE_MEAN
    agg["SEQ_RATE_UNC_YR"] = SEQ_RATE_UNC

    agg["SPECIES_CLEAN"] = agg["TREE_SPECIES_LABEL"].apply(clean_species_name)
    agg["SPECIES_GROUP"] = agg["SPECIES_CLEAN"].apply(canonical_species_group)

    print(f"Entry-level rows: {len(agg)}")
    print("Valid carbon rows:", int(agg["CARBON_STOCK_KG"].notna().sum()))
    print("Valid sequestration rows:", int(agg["CARBON_SEQUESTERED_KG_YR"].notna().sum()))

    return agg


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

def make_popup_html(row, colormap):
    c_seq = row["CARBON_SEQUESTERED_KG_YR"]
    c_seq_unc = row["CARBON_SEQUESTERED_UNC_KG_YR"]
    co2_seq = row["CO2_SEQUESTERED_KG_YR"]
    co2_seq_unc = row["CO2_SEQUESTERED_UNC_KG_YR"]
    kettles = row["KETTLE_BOILS_PER_YEAR"]
    kettles_unc = row["KETTLE_BOILS_UNC_PER_YEAR"]

    color = colormap(c_seq)

    img_html = make_image_scroller(row.get("IMAGE_URLS", []), color)

    return f"""
    <div style="width:375px; font-family:Arial, sans-serif;">
        <div style="font-size:24px; font-weight:900; color:{color}; text-align:center;">
            {c_seq:.1f} ± {c_seq_unc:.1f} kg C / yr
        </div>

        <div style="font-size:14px; text-align:center; margin-bottom:10px;">
            Average across {int(row.get("N_IMAGES", 1))} image(s)
        </div>

        <div style="font-size:15px; text-align:center; margin-bottom:10px;">
            {co2_seq:.1f} ± {co2_seq_unc:.1f} kg CO₂e / yr
        </div>

        <div style="font-size:15px; text-align:center; margin-bottom:10px;">
            ≈ {kettles:,.1f} ± {kettles_unc:,.1f} kettle boils / yr
        </div>

        <div><b>ID:</b> {row.get("ID")}</div>
        <div><b>Species:</b> {row.get("SPECIES_GROUP", "unknown")}</div>
        <div><b>Height:</b> {row.get("HEIGHT_M", np.nan):.1f} ± {row.get("HEIGHT_UNC_M", np.nan):.1f} m</div>
        <div><b>Diameter:</b> {row.get("DIAMETER_CM", np.nan):.1f} ± {row.get("DIAMETER_UNC_CM", np.nan):.1f} cm</div>
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


def build_map(df: pd.DataFrame, out_html: Path):
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
        tiles="OpenStreetMap",
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

    colormap.caption = "Carbon sequestered (kg C / year)"
    colormap.add_to(m)

    for _, row in df_map.iterrows():
        seq = float(row["CARBON_SEQUESTERED_KG_YR"])
        color = colormap(seq)

        folium.CircleMarker(
            location=[row["LATITUDE"], row["LONGITUDE"]],
            radius=marker_radius(seq, vmin, vmax),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=2,
            popup=folium.Popup(
                make_popup_html(row, colormap),
                max_width=420,
            ),
        ).add_to(m)

    total_c = df_map["CARBON_SEQUESTERED_KG_YR"].sum()
    total_c_unc = np.sqrt(
        np.square(df_map["CARBON_SEQUESTERED_UNC_KG_YR"].fillna(0)).sum()
    )

    total_co2 = df_map["CO2_SEQUESTERED_KG_YR"].sum()
    total_co2_unc = np.sqrt(
        np.square(df_map["CO2_SEQUESTERED_UNC_KG_YR"].fillna(0)).sum()
    )

    total_kettles = df_map["KETTLE_BOILS_PER_YEAR"].sum()
    total_kettles_unc = np.sqrt(
        np.square(df_map["KETTLE_BOILS_UNC_PER_YEAR"].fillna(0)).sum()
    )

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
        border-left: 6px solid green;
        font-family: Arial, sans-serif;
    ">
        <div style="font-size:18px; font-weight:800; margin-bottom:10px;">
            Carbon estimates from {len(df_map)} tree entries
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

    df = build_carbon_table(
        manifest_path=args.manifest,
        pred_is_circumference=args.pred_is_circumference,
    )

    out_csv = outdir / args.csv_name
    out_map = outdir / args.map_name

    df.to_csv(out_csv, index=False)
    print(f"Saved carbon table to: {out_csv}")

    build_map(df, out_map)

    preview_cols = [
        "ID",
        "SPECIES_GROUP",
        "N_IMAGES",
        "HEIGHT_M",
        "DIAMETER_CM",
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


if __name__ == "__main__":
    main()