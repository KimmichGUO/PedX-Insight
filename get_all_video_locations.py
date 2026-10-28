import re

import pandas as pd
from pathlib import Path
from modules.paths import RESULTS_ROOT, SUMMARY_ROOT
from modules.mapping_utils import find_city_in_mapping
from modules.pipeline import read_manifest
from modules.localization.localize import localization_succeeded

LOCATION_FIELDS = ['lat', 'lon', 'confidence_level', 'confidence_spread_m', 'street_names',
                   'candidates', 'route_latlon', 'route_length_m', 'trajectory_source']


def _source_sha(value):
    text = str(value).strip().lower() if value is not None else ''
    return text if re.fullmatch(r'[0-9a-f]{64}', text) else ''


def bind_localization_to_analysis(row, folder):
    """Retain raw history while failing closed on an unverified/different clip."""
    out = dict(row)
    manifest = read_manifest(folder) or {}
    if not isinstance(manifest, dict):
        manifest = {}
    configuration = manifest.get('configuration')
    expected = _source_sha(configuration.get('source_sha256') if isinstance(configuration, dict) else None)
    localized = _source_sha(row.get('source_video_sha256'))
    out['source_video_sha256'] = localized or None
    out['localization_origin_run_id'] = row.get('analysis_run_id')
    out['analysis_run_id'] = manifest.get('run_id') if manifest.get('status') == 'complete' else None
    out['analysis_source_video_sha256'] = expected or None
    status = str(row.get('status') or '').strip().lower()
    if status == 'ok':
        if manifest.get('status') != 'complete' or not manifest.get('run_id') or not expected or not localized:
            status = 'source_unverified'
        elif localized != expected:
            status = 'source_mismatch'
        elif not localization_succeeded(row):
            status = 'invalid_position'
    out['status'] = status
    if status != 'ok':
        for column in LOCATION_FIELDS:
            out[column] = None
    return out


def main(results_root=None, mapping_path=None, output_root=None):
    """Aggregate per-video [L1]localization.csv files into summary_data/all_video_locations.csv.

    Mirrors get_all_video_info.py: iterates analysis_results/<city>_<link>/ folders
    (folder name split at the FIRST underscore — YouTube ids may contain underscores),
    reads each [L1]localization.csv (written by `main.py --mode localize`), joins the
    city-centroid lat/lon from mapping.csv for comparison, and writes one row per video.

    The output is the input contract of PedX-Visualizer's
    scripts/import-video-coordinates.js (columns: city, link, video_name, lat, lon,
    confidence_level, confidence_spread_m, street_names, status, source, city_lat, city_lon,
    route_latlon, route_length_m, trajectory_source).
    """
    final_results_path = Path(results_root or RESULTS_ROOT)
    mapping_csv_path = Path(mapping_path or './mapping.csv')
    output_csv_path = Path(output_root or SUMMARY_ROOT) / 'all_video_locations.csv'

    if not final_results_path.exists():
        raise FileNotFoundError(final_results_path)

    mapping_df = None
    if mapping_csv_path.exists():
        try:
            mapping_df = pd.read_csv(mapping_csv_path)
            print(f"Successfully loaded mapping.csv with {len(mapping_df)} records")
        except Exception as e:
            print(f"Warning: could not read mapping.csv: {e}")
    else:
        print("Warning: mapping.csv does not exist — city_lat/city_lon will be empty")

    def city_centroid(city, link):
        """Centroid of the mapping row whose videos list contains this link.

        City names are ambiguous (e.g. London UK vs London, Ontario — mapping.csv has
        one row per city INSTANCE), so a name-only lookup can return the wrong city;
        the link check disambiguates, falling back to the first name match.
        """
        if mapping_df is None:
            return (None, None)
        matched = find_city_in_mapping(city, link, mapping_df)
        return (matched.get('lat'), matched.get('lon')) if matched is not None else (None, None)

    # Copied verbatim from each [L1] row. Deliberately excludes 'city' and 'video_name':
    # those are derived from the folder name (first-underscore split), matching
    # get_all_video_info.py, so the two summary CSVs stay join-consistent. The [L1] row's
    # own 'city' is the "City, Country" query string sent to the localizer, not the slug.
    location_columns = ['lat', 'lon', 'confidence_level',
                        'confidence_spread_m', 'street_names', 'status', 'source',
                        'result_json', 'candidates',
                        # Camera route through the city (visual odometry snapped to OSM).
                        # Older [L1] files predate these columns, hence .get(col, None).
                        'route_latlon', 'route_length_m', 'trajectory_source',
                        'source_video_sha256', 'analysis_run_id', 'localized_at']

    aggregated_rows = []

    for folder_path in sorted(final_results_path.iterdir()):
        if not folder_path.is_dir():
            continue

        folder_name = folder_path.name
        if '_' not in folder_name:
            print(f"Warning: Folder name {folder_name} does not contain an underscore, skipped")
            continue

        first_underscore_index = folder_name.index('_')
        # Folder prefix is the underscore-free name slug (e.g. 'London1' = city + running
        # index from the crawler bridge); strip the numeric suffix to recover the city.
        city = re.sub(r'\d+$', '', folder_name[:first_underscore_index])
        link = folder_name[first_underscore_index + 1:]

        loc_csv = folder_path / '[L1]localization.csv'
        if not loc_csv.exists():
            print(f"  Warning: {loc_csv} does not exist, skipped")
            continue

        try:
            loc_df = pd.read_csv(loc_csv)
        except Exception as e:
            print(f"  Error reading {loc_csv}: {e}")
            continue
        if loc_df.empty:
            print(f"  Warning: {loc_csv} is empty, skipped")
            continue

        loc = loc_df.iloc[0]
        row_data = {'city': city, 'link': link, 'video_name': folder_name}
        for col in location_columns:
            row_data[col] = loc.get(col, None)
        row_data = bind_localization_to_analysis(row_data, folder_path)

        centroid = city_centroid(city, link)
        row_data['city_lat'] = centroid[0]
        row_data['city_lon'] = centroid[1]

        aggregated_rows.append(row_data)
        print(f"Processed {folder_name}: status={row_data.get('status')}, "
              f"lat={row_data.get('lat')}, lon={row_data.get('lon')}")

    columns_order = ['city', 'link', 'video_name', 'lat', 'lon', 'confidence_level',
                     'confidence_spread_m', 'street_names', 'status', 'source',
                     'city_lat', 'city_lon', 'result_json', 'candidates',
                     'route_latlon', 'route_length_m', 'trajectory_source',
                     'source_video_sha256', 'analysis_run_id', 'localized_at',
                     'analysis_source_video_sha256', 'localization_origin_run_id']

    result_df = pd.DataFrame(aggregated_rows).reindex(columns=columns_order) if aggregated_rows \
        else pd.DataFrame(columns=columns_order)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    ok_count = int((result_df['status'] == 'ok').sum()) if not result_df.empty else 0
    print(f"\nAggregated {len(result_df)} localization rows ({ok_count} with status=ok)")
    print(f"Results saved to {output_csv_path}")


if __name__ == "__main__":
    main()
