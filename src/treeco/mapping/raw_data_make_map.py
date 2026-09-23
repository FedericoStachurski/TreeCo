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
    for col in ["TREE_HEIGHT_IN_METERS", "TREE_HEIGHT_IN_METERS_CLEAN", "HEIGHT_VALUE_M"]:
        if col in row.index:
            h = parse_numeric(row.get(col))
            if pd.notna(h):
                return h, 1.0, col

    for col in ["ESTIMATED_TREE_HEIGHT", "ESTIMATED_TREE_HEIGHT_CLEAN", "HEIGHT_CLASS_STR"]:
        if col in row.index:
            label = row.get(col)

            if pd.notna(label):
                label = str(label).strip()

                if label in HEIGHT_CLASS_MAP:
                    h, u = HEIGHT_CLASS_MAP[label]
                    return h, u, col

                h = parse_numeric(label)
                if pd.notna(h):
                    return h, 2.5, col

    return np.nan, np.nan, "missing"


def parse_circumference(row):
    for col in ["CIRCUMFERENCE_IN_CM", "CIRCUMFERENCE_CM_CLEAN"]:
        if col in row.index:
            c = parse_numeric(row.get(col))
            if pd.notna(c):
                return c, max(0.1 * c, 5.0), col

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
    urls = []
    media_cols = find_media_cols(group)

    for _, row in group.iterrows():
        for col in media_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).startswith("http"):
                urls.append(str(val))

    return list(dict.fromkeys(urls))


def first_non_null(series):
    s = series.dropna()
    if len(s) == 0:
        return np.nan
    return s.iloc[0]


def aggregate_to_entry_level(df):
    rows = []

    numeric_avg_cols = [
        "HEIGHT_M",
        "HEIGHT_UNC_M",
        "CIRCUMFERENCE_CM",
        "CIRCUMFERENCE_UNC_CM",
        "DIAMETER_CM",
        "DIAMETER_UNC_CM",
        "CARBON_STOCK_KG",
        "CARBON_STOCK_UNC_KG",
        "CARBON_SEQUESTERED_KG_YR",
        "CARBON_SEQUESTERED_UNC_KG_YR",
        "CO2_SEQUESTERED_KG_YR",
        "CO2_SEQUESTERED_UNC_KG_YR",
        "KETTLE_BOILS_PER_YEAR",
        "KETTLE_BOILS_UNC_PER_YEAR",
    ]

    for tree_id, group in df.groupby("ID", dropna=False):
        first = group.iloc[0].copy()
        out = first.to_dict()

        out["N_ROWS_FOR_ID"] = int(len(group))
        out["IMAGE_URLS"] = list_images_for_group(group)

        for col in numeric_avg_cols:
            if col in group.columns:
                out[col] = pd.to_numeric(group[col], errors="coerce").mean()

        out["HEIGHT_SOURCE"] = first_non_null(group["HEIGHT_SOURCE"]) if "HEIGHT_SOURCE" in group.columns else np.nan
        out["CIRCUMFERENCE_SOURCE"] = first_non_null(group["CIRCUMFERENCE_SOURCE"]) if "CIRCUMFERENCE_SOURCE" in group.columns else np.nan

        rows.append(out)

    return pd.DataFrame(rows)


# =========================================================
# Carbon table
# =========================================================

def build_raw_carbon_table(input_path):
    df = load_table(input_path)

    print(f"Loaded: {input_path}")
    print(f"Rows before filtering: {len(df)}")

    if "TREE" in df.columns:
        df = df[df["TREE"].astype(str).str.lower().eq("yes")].copy()

    print(f"Tree rows: {len(df)}")

    height_rows = df.apply(parse_height, axis=1)
    df[["HEIGHT_M", "HEIGHT_UNC_M", "HEIGHT_SOURCE"]] = pd.DataFrame(
        height_rows.tolist(),
        index=df.index,
    )

    circ_rows = df.apply(parse_circumference, axis=1)
    df[["CIRCUMFERENCE_CM", "CIRCUMFERENCE_UNC_CM", "CIRCUMFERENCE_SOURCE"]] = pd.DataFrame(
        circ_rows.tolist(),
        index=df.index,
    )

    df["DIAMETER_CM"] = df["CIRCUMFERENCE_CM"] / np.pi
    df["DIAMETER_UNC_CM"] = df["CIRCUMFERENCE_UNC_CM"] / np.pi

    valid = (
        df["LATITUDE"].notna()
        & df["LONGITUDE"].notna()
        & df["HEIGHT_M"].notna()
        & df["DIAMETER_CM"].notna()
    )

    df = df[valid].copy()

    print(f"Rows with observed height + circumference + coordinates: {len(df)}")

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

    df["CARBON_STOCK_KG"] = df.apply(
        lambda r: carbon_stock_from_height_diameter(
            r["HEIGHT_M"],
            r["DIAMETER_CM"],
        ),
        axis=1,
    )

    df["CARBON_STOCK_UNC_KG"] = df.apply(
        lambda r: carbon_stock_uncertainty(
            r["HEIGHT_M"],
            r["HEIGHT_UNC_M"],
            r["DIAMETER_CM"],
            r["DIAMETER_UNC_CM"],
            r["CARBON_STOCK_KG"],
        ),
        axis=1,
    )

    df["SEQ_RATE_YR"] = SEQ_RATE_MEAN
    df["SEQ_RATE_UNC_YR"] = SEQ_RATE_UNC

    df["CARBON_SEQUESTERED_KG_YR"] = df["CARBON_STOCK_KG"] * df["SEQ_RATE_YR"]

    df["CARBON_SEQUESTERED_UNC_KG_YR"] = np.sqrt(
        (df["CARBON_STOCK_KG"] * df["SEQ_RATE_UNC_YR"]) ** 2
        + (df["SEQ_RATE_YR"] * df["CARBON_STOCK_UNC_KG"]) ** 2
    )

    co2_factor = 44.0 / 12.0

    df["CO2_SEQUESTERED_KG_YR"] = df["CARBON_SEQUESTERED_KG_YR"] * co2_factor
    df["CO2_SEQUESTERED_UNC_KG_YR"] = df["CARBON_SEQUESTERED_UNC_KG_YR"] * co2_factor

    df["KETTLE_BOILS_PER_YEAR"] = df["CO2_SEQUESTERED_KG_YR"] / KG_CO2E_PER_KETTLE_BOIL
    df["KETTLE_BOILS_UNC_PER_YEAR"] = df["CO2_SEQUESTERED_UNC_KG_YR"] / KG_CO2E_PER_KETTLE_BOIL

    df_entry = aggregate_to_entry_level(df)

    print(f"Entry-level rows after ID aggregation: {len(df_entry)}")
    print("Non-null carbon rows:", int(df_entry["CARBON_STOCK_KG"].notna().sum()))

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
            {c_seq:.3f} ± {c_seq_unc:.3f} kg C / yr
        </div>

        <div style="font-size:14px; text-align:center; margin-bottom:10px;">
            Average across {int(row.get("N_ROWS_FOR_ID", 1))} row(s) for this ID
        </div>

        <div style="font-size:15px; text-align:center; margin-bottom:10px;">
            {co2_seq:.3f} ± {co2_seq_unc:.3f} kg CO₂e / yr
        </div>

        <div style="font-size:15px; text-align:center; margin-bottom:10px;">
            ≈ {kettles:,.1f} ± {kettles_unc:,.1f} kettle boils / yr
        </div>

        <div><b>ID:</b> {row.get("ID", "NA")}</div>
        <div><b>Species:</b> {row.get("SPECIES_GROUP", "unknown")}</div>
        <div><b>Height:</b> {row.get("HEIGHT_M", np.nan):.1f} ± {row.get("HEIGHT_UNC_M", np.nan):.1f} m</div>
        <div><b>Circumference:</b> {row.get("CIRCUMFERENCE_CM", np.nan):.1f} cm</div>
        <div><b>Diameter:</b> {row.get("DIAMETER_CM", np.nan):.1f} ± {row.get("DIAMETER_UNC_CM", np.nan):.1f} cm</div>
        <div><b>Height source:</b> {row.get("HEIGHT_SOURCE")}</div>
        <div><b>Circumference source:</b> {row.get("CIRCUMFERENCE_SOURCE")}</div>

        {img_html}
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

    colormap.caption = "Carbon sequestered from observed data only (kg C / year)"
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
            popup=folium.Popup(make_popup_html(row, colormap), max_width=420),
        ).add_to(m)

    total_c = df_map["CARBON_SEQUESTERED_KG_YR"].sum()
    total_c_unc = np.sqrt(np.square(df_map["CARBON_SEQUESTERED_UNC_KG_YR"].fillna(0)).sum())

    total_co2 = df_map["CO2_SEQUESTERED_KG_YR"].sum()
    total_co2_unc = np.sqrt(np.square(df_map["CO2_SEQUESTERED_UNC_KG_YR"].fillna(0)).sum())

    total_kettles = df_map["KETTLE_BOILS_PER_YEAR"].sum()
    total_kettles_unc = np.sqrt(np.square(df_map["KETTLE_BOILS_UNC_PER_YEAR"].fillna(0)).sum())

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
            Observed-only carbon estimates from {len(df_map)} tree entries
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
        description="Make an observed-only tree carbon/sequestration map from raw CommuniMap XLSX/CSV."
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
        default="raw_observed_tree_carbon_estimates_entry_level.csv",
    )

    ap.add_argument(
        "--map-name",
        default="raw_observed_tree_sequestration_map.html",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = build_raw_carbon_table(args.input_xlsx)

    out_csv = args.outdir / args.csv_name
    out_map = args.outdir / args.map_name

    df.to_csv(out_csv, index=False)
    print(f"Saved CSV to: {out_csv}")

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
    print(df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()