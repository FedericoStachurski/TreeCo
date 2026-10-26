#!/usr/bin/env python3
"""Build an inclusive TreeCo carbon map from ALL citizen trees + processed AI results.

Citizen measurements and image URLs are authoritative. AI is used only to fill
missing height/DBH. Original tree entries are never lost to image filtering.

Example:
  python treeco_merge_citizen_and_ai_map.py \
    --raw-entries ~/TreeCo/outputs/communimap_sept2026_citizen_full/raw_citizen_all_tree_entries.csv \
    --treeco-entries ~/TreeCo/outputs/communimap_sept2026_carbon_v4/tree_carbon_estimates_entry_level.csv \
    --outdir ~/TreeCo/outputs/communimap_sept2026_merged
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

SEQ_RATE_MEAN = 0.02
SEQ_RATE_UNC = 0.01
KG_CO2E_PER_KETTLE_BOIL = 0.018
CARBON_FIELDS = (
    'CARBON_STOCK_KG', 'CARBON_STOCK_UNC_KG',
    'CARBON_SEQUESTERED_KG_YR', 'CARBON_SEQUESTERED_UNC_KG_YR',
    'CO2_SEQUESTERED_KG_YR', 'CO2_SEQUESTERED_UNC_KG_YR',
    'KETTLE_BOILS_PER_YEAR', 'KETTLE_BOILS_UNC_PER_YEAR'
)


def numeric(value):
    return pd.to_numeric(value, errors='coerce')


def positive(value):
    v = numeric(value)
    return bool(pd.notna(v) and np.isfinite(v) and v > 0)


def parse_image_urls(value):
    if isinstance(value, list):
        values = value
    elif pd.isna(value):
        values = []
    else:
        value = str(value).strip()
        try:
            values = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            values = [value] if value.startswith('http') else []
    if isinstance(values, str):
        values = [values]
    return [str(v).strip() for v in values if str(v).strip().startswith(('http://', 'https://'))]


def read_entries(path):
    df = pd.read_csv(path, dtype={'ID': 'string'}, low_memory=False)
    df['ID'] = df['ID'].astype('string').str.strip().str.replace(r'\.0$', '', regex=True)
    df = df.dropna(subset=['ID']).copy()
    if df['ID'].duplicated().any():
        sample = df.loc[df['ID'].duplicated(), 'ID'].head(6).tolist()
        raise ValueError(f'Expected one row per ID in {path}; duplicates include {sample}')
    return df.set_index('ID', drop=False)


def row_value(row, key, default=np.nan):
    return row.get(key, default) if row is not None else default


def choose_measurement(raw_row, ai_row, name, uncertainty_name, source_name):
    raw_value = row_value(raw_row, name)
    if positive(raw_value):
        raw_source = row_value(raw_row, source_name)
        if source_name == 'DIAMETER_SOURCE':
            raw_source = row_value(raw_row, 'CIRCUMFERENCE_SOURCE')
            if pd.notna(raw_source) and str(raw_source).lower() != 'missing':
                raw_source = str(raw_source).replace('trunk_size', 'dbh_from_trunk_size')
        source = str(raw_source) if pd.notna(raw_source) and str(raw_source).lower() != 'missing' else 'citizen_provided'
        return float(raw_value), numeric(row_value(raw_row, uncertainty_name)), source, False, 'raw'

    ai_value = row_value(ai_row, name)
    if positive(ai_value):
        ai_source = str(row_value(ai_row, source_name, 'processed')).strip()
        from_ai = ('model_inferred' in ai_source)
        return float(ai_value), numeric(row_value(ai_row, uncertainty_name)), ai_source, from_ai, 'processed'

    return np.nan, np.nan, 'missing', False, 'missing'


def carbon_estimates(height, height_unc, diameter, diameter_unc):
    result = dict.fromkeys(CARBON_FIELDS, np.nan)
    if not (positive(height) and positive(diameter)):
        return result

    rate = 0.0577 if height < 28 else 0.0346
    stock = rate * height**2 * diameter * 0.25
    result['CARBON_STOCK_KG'] = stock
    result['CARBON_SEQUESTERED_KG_YR'] = stock * SEQ_RATE_MEAN
    result['CO2_SEQUESTERED_KG_YR'] = stock * SEQ_RATE_MEAN * (44 / 12)
    result['KETTLE_BOILS_PER_YEAR'] = result['CO2_SEQUESTERED_KG_YR'] / KG_CO2E_PER_KETTLE_BOIL

    if pd.notna(height_unc) and pd.notna(diameter_unc):
        stock_unc = stock * np.sqrt((2 * height_unc / height)**2 + (diameter_unc / diameter)**2)
        seq_unc = np.sqrt((stock * SEQ_RATE_UNC)**2 + (SEQ_RATE_MEAN * stock_unc)**2)
        result.update({
            'CARBON_STOCK_UNC_KG': stock_unc,
            'CARBON_SEQUESTERED_UNC_KG_YR': seq_unc,
            'CO2_SEQUESTERED_UNC_KG_YR': seq_unc * (44 / 12),
            'KETTLE_BOILS_UNC_PER_YEAR': seq_unc * (44 / 12) / KG_CO2E_PER_KETTLE_BOIL
        })
    return result


def coordinate_pair(row):
    lat, lon = numeric(row_value(row, 'LATITUDE')), numeric(row_value(row, 'LONGITUDE'))
    if pd.notna(lat) and pd.notna(lon) and -90 <= lat <= 90 and -180 <= lon <= 180:
        return float(lat), float(lon)
    return np.nan, np.nan


def first_label(raw_row, ai_row, key, fallback='unknown'):
    for row in (raw_row, ai_row):
        val = row_value(row, key)
        if pd.notna(val) and str(val).strip().lower() not in {'', 'nan', 'none', 'missing'}:
            return str(val)
    return fallback


def merge_entries(raw, processed):
    entries = []
    for tree_id in raw.index.union(processed.index, sort=False):
        r = raw.loc[tree_id] if tree_id in raw.index else None
        p = processed.loc[tree_id] if tree_id in processed.index else None

        h, hu, hs, h_ai, h_origin = choose_measurement(r, p, 'HEIGHT_M', 'HEIGHT_UNC_M', 'HEIGHT_SOURCE')
        d, du, ds, d_ai, d_origin = choose_measurement(r, p, 'DIAMETER_CM', 'DIAMETER_UNC_CM', 'DIAMETER_SOURCE')
        raw_lat, raw_lon = coordinate_pair(r)
        ai_lat, ai_lon = coordinate_pair(p)
        lat, lon = (raw_lat, raw_lon) if pd.notna(raw_lat) else (ai_lat, ai_lon)

        raw_images = parse_image_urls(row_value(r, 'IMAGE_URLS'))
        processed_images = parse_image_urls(row_value(p, 'IMAGE_URLS'))
        all_images = list(dict.fromkeys(raw_images + processed_images))
        n_processed = row_value(p, 'N_IMAGES', len(processed_images))
        n_processed = int(numeric(n_processed)) if pd.notna(numeric(n_processed)) else len(processed_images)

        estimate = carbon_estimates(h, hu, d, du)
        # If both citizen dimensions are available, preserve the exact raw
        # sequestration and uncertainty: don't silently change the baseline.
        if h_origin == d_origin == 'raw' and positive(row_value(r, 'CARBON_SEQUESTERED_KG_YR')):
            estimate = {key: numeric(row_value(r, key)) for key in CARBON_FIELDS}

        row = {
            'ID': tree_id, 'LATITUDE': lat, 'LONGITUDE': lon,
            'TREE_TYPE': first_label(r, p, 'TREE_TYPE'),
            'SPECIES_GROUP': first_label(r, p, 'SPECIES_GROUP'),
            'TREE_SPECIES_LABEL': first_label(r, p, 'TREE_SPECIES_LABEL', first_label(r, p, 'SPECIES')),
            'HEIGHT_M': h, 'HEIGHT_UNC_M': hu, 'HEIGHT_SOURCE': hs,
            'DIAMETER_CM': d, 'DIAMETER_UNC_CM': du, 'DIAMETER_SOURCE': ds,
            'HEIGHT_MODEL_INFERRED': bool(h_ai), 'DIAMETER_MODEL_INFERRED': bool(d_ai),
            'HAS_MODEL_INFERENCE': bool(h_ai or d_ai),
            'N_IMAGES': len(all_images), 'N_ORIGINAL_IMAGES': len(raw_images),
            'N_PROCESSED_IMAGES': n_processed, 'IMAGE_URLS': all_images,
            'HAS_RAW_ENTRY': r is not None, 'HAS_PROCESSED_ENTRY': p is not None,
            'HEIGHT_SELECTED_FROM': h_origin, 'DBH_SELECTED_FROM': d_origin,
            'SEQ_RATE_YR': SEQ_RATE_MEAN, 'SEQ_RATE_UNC_YR': SEQ_RATE_UNC,
            **estimate
        }
        entries.append(row)

    out = pd.DataFrame(entries)
    out['HAS_COORDINATES'] = (out['LATITUDE'].between(-90, 90) & out['LONGITUDE'].between(-180, 180))
    out['HAS_CARBON_ESTIMATE'] = out['CARBON_SEQUESTERED_KG_YR'].notna()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-entries', type=Path, required=True, help='Raw map raw_citizen_all_tree_entries.csv')
    ap.add_argument('--treeco-entries', type=Path, required=True, help='TreeCo v4 tree_carbon_estimates_entry_level.csv')
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--skip-map', action='store_true', help='For validation without Folium/TreeCo installed')
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw, processed = read_entries(args.raw_entries), read_entries(args.treeco_entries)
    merged = merge_entries(raw, processed)
    final = merged.loc[merged['HAS_COORDINATES'] & merged['HAS_CARBON_ESTIMATE']].copy()
    baseline_ids = set(raw.loc[
        raw['CARBON_SEQUESTERED_KG_YR'].notna()
        & raw['LATITUDE'].between(-90, 90)
        & raw['LONGITUDE'].between(-180, 180)
    ].index)
    final_ids = set(final['ID'])
    retained = baseline_ids & final_ids
    lost = baseline_ids - final_ids
    new_ai = final.loc[final['HAS_MODEL_INFERENCE'] & ~final['ID'].isin(baseline_ids)]
    new_other = final.loc[~final['HAS_MODEL_INFERENCE'] & ~final['ID'].isin(baseline_ids)]

    print(f'Original citizen trees:                     {len(raw)}')
    print(f'Raw citizen carbon entries (before geo):    {len(baseline_ids)}')
    print(f'Original carbon entries retained:           {len(retained)}')
    print(f'Original carbon entries missing:            {len(lost)}')
    print(f'Newly carbon-estimable via AI:              {len(new_ai)}')
    print(f'Newly carbon-estimable without AI:          {len(new_other)}')
    print(f'Final carbon-estimable mapped tree entries: {len(final)}')
    if lost:
        raise AssertionError(f'Cannot drop original citizen carbon entries: {sorted(lost)[:20]}')

    merged.to_csv(args.outdir / 'tree_carbon_all_entries_including_incomplete.csv', index=False)
    final.to_csv(args.outdir / 'tree_carbon_estimates_entry_level.csv', index=False)
    new_ai.to_csv(args.outdir / 'additional_trees_enabled_by_ai.csv', index=False)
    print('Saved merged CSVs to:', args.outdir)

    if not args.skip_map:
        try:
            import importlib
            treeco_map = importlib.import_module("treeco.mapping.make_map")
        except ImportError as exc:
            raise RuntimeError('Cannot import treeco.mapping.make_map. Run inside your (treeco) environment.') from exc
        # Existing popup says "Average across ... image(s)"; these images
        # now include all original photos, not just those used for inference.
        original_popup = treeco_map.make_popup_html
        def corrected_popup(row, colormap):
            html = original_popup(row, colormap)
            import re
            html = re.sub(
                r'Average across\s+\d+\s+image\(s\)',
                f"{int(row['N_ORIGINAL_IMAGES'])} original photograph(s) · "
                f"{int(row['N_PROCESSED_IMAGES'])} processed image(s)",
                html
            )
            return html
        treeco_map.make_popup_html = corrected_popup
        html_path = args.outdir / 'tree_carbon_sequestration_map.html'
        treeco_map.build_map(final, html_path)
        print('Saved map:', html_path)


if __name__ == '__main__':
    main()