#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def latest_file(manifest_dir: Path, pattern: str) -> Path:
    files = sorted(
        manifest_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not files:
        raise FileNotFoundError(
            f"No files found in {manifest_dir} matching pattern: {pattern}"
        )

    return files[0]


def main():
    ap = argparse.ArgumentParser(
        description="Combine inferred TreeCo height and width manifests."
    )

    ap.add_argument(
        "--dataset_dir",
        required=True,
        help="TreeCo dataset directory containing manifests/.",
    )

    ap.add_argument(
        "--height_manifest",
        default=None,
        help="Optional explicit height manifest path.",
    )

    ap.add_argument(
        "--width_manifest",
        default=None,
        help="Optional explicit width manifest path.",
    )

    ap.add_argument(
        "--out_name",
        default="tree_dataset_manifest_with_inferred_heights_and_widths.csv",
        help="Output CSV name inside manifests/.",
    )

    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    manifest_dir = dataset_dir / "manifests"

    if not manifest_dir.exists():
        raise FileNotFoundError(f"Missing manifests directory: {manifest_dir}")

    height_path = (
        Path(args.height_manifest)
        if args.height_manifest is not None
        else latest_file(
            manifest_dir,
            "tree_dataset_manifest_with_image_level_inferred_heights_*.csv",
        )
    )

    width_path = (
        Path(args.width_manifest)
        if args.width_manifest is not None
        else latest_file(
            manifest_dir,
            "tree_dataset_manifest_with_image_level_inferred_widths_*.csv",
        )
    )

    print(f"Using height manifest: {height_path}")
    print(f"Using width manifest:  {width_path}")

    height_df = pd.read_csv(height_path)
    width_df = pd.read_csv(width_path)

    merge_keys = ["ID", "IMAGE_ID"]

    for key in merge_keys:
        if key not in height_df.columns:
            raise ValueError(f"Height manifest missing key column: {key}")
        if key not in width_df.columns:
            raise ValueError(f"Width manifest missing key column: {key}")

    height_cols = [
        "ID",
        "IMAGE_ID",
        "HEIGHT_CLASS_PRED_IDX",
        "HEIGHT_CLASS_PRED_STR",
        "HEIGHT_CLASS_PRED_CONF",
        "HEIGHT_PRED_ACCEPTED",
        "HEIGHT_SOURCE_IMAGE",
        "PROBS_TREE_HEIGHT",
    ]

    height_cols += [
        c for c in height_df.columns
        if c.startswith("HEIGHT_PROB_")
    ]

    height_cols = [c for c in height_cols if c in height_df.columns]

    combined = width_df.merge(
        height_df[height_cols],
        on=merge_keys,
        how="left",
        suffixes=("", "_HEIGHT"),
    )

    out_path = manifest_dir / args.out_name
    combined.to_csv(out_path, index=False)

    print(f"\nSaved combined manifest to:")
    print(out_path)
    print(f"Rows: {len(combined)}")
    print(f"Columns: {len(combined.columns)}")


if __name__ == "__main__":
    main()