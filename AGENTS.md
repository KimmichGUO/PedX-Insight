# AGENTS.md — working guide for PedX-Insight

Guidance for AI agents (and humans) working in this repository. Read this before making changes.

## What this project is

PedX-Insight is a computer-vision **toolkit for analyzing global pedestrian crossing behavior**
from street/dashcam videos. It runs YOLO-based detectors and CSV-driven logic over a video and
writes per-video results, then aggregates them across a dataset of cities for statistical study.

## How it runs

- **`main.py`** — single entry point. `--mode <name>` dispatches to one analysis module.
  ```bash
  python main.py --mode <mode> --source_video_path PATH [--analysis_interval 1.0] [--weights_yolo yolo11n.pt]
  ```
  `--mode single_all` runs the whole per-video pipeline; `--mode mul_all` does so for every video
  in a folder. See `README.md` for the full mode list.
- **`run.py`** — batch driver over a mapping CSV: per row it downloads (yt-dlp), analyzes
  (`main.py --mode single_all`), then deletes the video. `--start_row`, `--start_step`.
- **Aggregation** (run after per-video analysis):
  `python get_all_pede_info.py` and `python get_all_video_info.py` → `summary_data/*.csv`;
  `python statistics_with_pdf_save.py` → stats CSVs + PDFs.

## Repository layout

```
main.py                     mode dispatch
run.py                      batch download→analyze→delete
new_track_id_with_imgs.py   pedestrian tracking (produces [B1]tracked_pedestrians.csv) — upstream of most modules
dataset*.py, update_dataset.py   dataset/mapping helpers
get_all_*.py, statistics_with_pdf_save.py   cross-video aggregation + stats
modules/<name>/<name>.py    one analysis module each; exposes a run_*/detect_*/… function
external/Monocular-OSM-Localization/   git submodule (video→lat/lon), see "Localization"
analysis_results/<video_name>/   per-video outputs (NOT tracked)
summary_data/               aggregated outputs (NOT tracked; created by get_all_*.py)
```

## Pipeline data-flow conventions

- Each module resolves its own paths as `analysis_results/<video_name>/[CODE]name.csv`, where
  `<video_name>` is the video filename without extension. Output codes: `[B1]` tracking,
  `[P*]` pedestrian attrs, `[V*]` vehicle, `[E*]` environment, `[C*]` crossing/advanced,
  `[A1]`/`[A2]` video/pedestrian summaries, `[L1]` localization.
- Modules read the CSVs of upstream modules (e.g. crossing/risky/summary all consume
  `[B1]tracked_pedestrians.csv` and `[C3]crossing_judge.csv`). **When you change a producer's
  columns, grep for every consumer.** Producer/consumer column mismatches are the most common bug
  class here (e.g. `'Total'` vs `'total'`, `id` vs `track_id`).
- **`frame_id` base differs across modules.** Tracking is 1-indexed (`frame_id` starts at 1);
  several detectors are 0-indexed. They still align at the sampling stride `N = ceil(fps*interval)`
  because lookups happen at multiples of `N`. Be careful when adding cross-module frame lookups.
- **Producer density differs.** Some detectors write one row per *analyzed* frame (traffic light,
  traffic sign, accident, sidewalk, vehicle type); others write one row per *frame* with
  carry-forward (crosswalk, weather, daytime, road condition). Summary code (`modules/summary/
  video_info.py`) divides each metric by that source's own `len(df)`, **not** total video frames —
  keep that when adding metrics.
- **Sidewalk is special:** `modules/sidewalk/sidewalk_detect.py` runs as its own **subprocess**
  (`python modules/sidewalk/sidewalk_detect.py …`) because of SGDepth's relative imports; it needs
  external weights (see README link).

## Dependency policy (IMPORTANT — decided July 2026)

`requirements.txt` is pinned to the **newest** releases, **numpy 2.5.x + OpenCV 5.x majors
included**. To make that installable, **`paddlex` was removed** (Option B):

- paddlex 3.7.2 has a **core** cap `numpy<2.4` and its extras pin
  `opencv-contrib-python==4.10.0.84`. With the newest numpy/OpenCV, `pip install -r
  requirements.txt` is `ResolutionImpossible`. Removing paddlex resolves it cleanly.
- **Consequence:** `modules/age_gender` (`--mode ag`, age/gender/race) needs paddlex and is
  **disabled**. The paddlex import is **lazy** (inside `run_age_gender`), so `main.py` and all
  other modes still import/run; invoking `--mode ag` raises a clear, actionable `ImportError`.
- **To re-enable age/gender:** install paddlex in a *separate* environment (`numpy<2.4`,
  `opencv-contrib-python==4.10.0.84`) and run that mode there, or reimplement the module on
  another model. Do **not** add paddlex back to `requirements.txt` without re-capping numpy/OpenCV
  (it will break the install).
- When bumping deps, re-run a resolver check: `python -m pip install --dry-run -r requirements.txt`
  (or a focused subset) and watch for `ResolutionImpossible`. `paddlex` and `ultralytics` export
  extras are the usual sources of `numpy<2` caps.

## Localization (`--mode localize`)

Wraps the [Monocular-OSM-Localization](https://github.com/M-Colley/Monocular-OSM-Localization)
submodule (video + city → lat/lon on OpenStreetMap), pinned to its `0.1.0` release. Code:
`modules/localization/localize.py`.

- The tool ships no package metadata (no `setup.py`/`pyproject.toml`, not on PyPI), so it is
  **not** a pip dependency even at 0.1.0 — it stays vendored as a git submodule and is **never
  imported**, only run as a **subprocess** with a configurable interpreter: `--osm_python` →
  `$OSM_LOCALIZATION_PYTHON` → the submodule `.venv` → PedX's own interpreter (used only if the
  tool's deps are importable there, probed via `osmnx`). 0.1.0's pins are compatible with PedX's
  env, so `pip install -r external/Monocular-OSM-Localization/requirements.txt` into PedX's venv
  makes localize run with no separate interpreter.
- Emits per-video `[L1]localization.csv` (`lat`, `lon`, `confidence_level`,
  `confidence_spread_m`, `street_names`, `status`, `candidates`). `--city` is inferred from
  `mapping.csv` if omitted — the video id is everything after the FIRST underscore of the
  filename stem (YouTube ids can contain underscores; never rsplit).
- `run.py --localize` runs it per video after analysis and BEFORE deletion (Step 2.5,
  `check=False` so failures never block the batch). Not part of `single_all`/`mul_all`.
- `get_all_video_locations.py` aggregates all `[L1]` files →
  `summary_data/all_video_locations.csv` (city/video_name derived from the folder name, NOT
  from the [L1] row — keeps join-consistency with all_video_info.csv). That CSV is the input
  contract of PedX-Visualizer's `scripts/import-video-coordinates.js` (joins `videos.link`,
  writes real latitude/longitude + street_name + localization_confidence, replacing mock data).
- Setup is documented in `README.md` → "Video Geolocation".

## Cross-repo pipeline (Crawler → Insight → Visualizer)

- PedX-Crawler (`../PedX-Crawler`) discovers videos → `data/outputs/discovery.csv`;
  its `scripts/discovery_to_mapping.py` converts that to this repo's `mapping_one_each.csv`
  (+ merges city rows into `mapping.csv`). Names are underscore-free (`London1`) by design.
- This repo analyzes (`run.py`) → `analysis_results/<name>_<videoid>/…` → aggregators →
  `summary_data/all_video_info.csv`, `all_pedestrian_info.csv`, `all_time_info.csv`
  (produced from run.py's `[A3]analysis_time.csv`, keyed by link), `all_video_locations.csv`.
- PedX-Visualizer (`../PedX-Visualizer`) ingests those CSVs (`scripts/aggregate-csv-data.js`,
  UPSERT key = `videos.link` = bare YouTube id) and imports real coordinates
  (`scripts/import-video-coordinates.js`). `all_video_info.csv` also carries
  `data_collected_date` (analysis date) for the Visualizer's temporal features.

## Testing / verification conventions

There is no formal test suite. Verify changes by:

1. **Byte-compile** touched files: `python -m compileall -q <files>`.
2. **Exercise CSV-driven logic end-to-end** with synthetic inputs — build a tiny video with
   `cv2.VideoWriter` + synthetic `[B1]`/`[E*]`/`[V*]` CSVs under `analysis_results/<name>/`, then
   call the module functions and assert on the output CSV. (The CV *detector* modules need YOLO
   weights/GPU and can't be run headless; the crossing/summary/localization logic can.)
3. Prefer the dedicated tools (Read/Grep/Edit) over shell text utilities.
4. Do temp work under the scratchpad dir, never in the project tree; clean up `__pycache__`.

## Gotchas / known rough edges

- `summary_data/` and `analysis_results/` are created at runtime and are not tracked. Aggregators
  now `mkdir` them; keep that if you add outputs.
- `summary_data/all_time_info.csv` (used by `statistics_with_pdf_save.py` section 1 and the
  Visualizer's `analysis_seconds`) is produced by `get_all_video_info.py` from the per-video
  `[A3]analysis_time.csv` files that `run.py` writes — videos analyzed via bare `main.py`
  (not `run.py`) have no `[A3]` and are simply absent from it.
- CSV encoding is UTF-8 across the aggregation/stats path (city/country names are non-ASCII); keep
  reads and writes consistent.
- On Windows, avoid non-ASCII glyphs in `print()` (legacy code-page `UnicodeEncodeError`).
- Keep module output CSVs header-safe on empty input (write with explicit `columns=…`) so downstream
  `pd.read_csv` never hits `EmptyDataError`.
