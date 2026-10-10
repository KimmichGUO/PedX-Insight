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
modules/localization/localize.py   video→lat/lon via the `monocular_osm` pip pkg, see "Localization"
modules/scene_seg/scene_seg.py     [E10] Mask2Former/Vistas segmentation (candidate, not in single_all)
modules/paths.py            RESULTS_ROOT (env PEDX_RESULTS_ROOT, default analysis_results)
modules/provenance.py       [A0] per-mode provenance (git sha, weights digest, packages)
modules/completeness.py     shared "is this folder fully analysed" gate for both aggregators
modules/mapping_utils.py    shared city join to mapping.csv (normalised names, 0-as-missing)
modules/stats_utils.py      cluster bootstrap CIs, BH-FDR, city-level collapse, age bands
statistics_inferential.py   variance partition + adjusted associations (statsmodels)
validation/                 JAAD ground-truth benchmark of the detectors (see validation/README.md)
tests/                      pytest suite (plain asserts; also runnable as scripts)
analysis_results/<video_name>/   per-video outputs (NOT tracked)
summary_data/               aggregated outputs (TRACKED: the published dataset's history)
```

## Pipeline data-flow conventions

- Each module resolves its own paths as `<RESULTS_ROOT>/<video_name>/[CODE]name.csv`, where
  `<video_name>` is the video filename without extension and `RESULTS_ROOT` comes from
  `modules/paths.py` (env `PEDX_RESULTS_ROOT`, default `analysis_results`). **New modules must
  use `RESULTS_ROOT`, never a literal `"analysis_results"`** -- benchmark runs rely on the
  redirect to stay out of the published dataset. It is bound at import time: tests must patch
  the module constant, not set the env var mid-process. Output codes: `[A0]` provenance,
  `[B1]` tracking, `[P*]` pedestrian attrs, `[V*]` vehicle, `[E*]` environment (`[E10]` scene
  segmentation), `[C*]` crossing/advanced, `[A1]`/`[A2]` video/pedestrian summaries,
  `[L1]` localization.
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
- **Age/gender was reimplemented without paddlex (2026-07-16):** OpenCV YuNet face detection
  + InsightFace `genderage.onnx` via `cv2.dnn` (weights auto-download, <2 MB). Do **not** add
  paddlex back to `requirements.txt`: it would re-cap numpy/OpenCV and break the install.
- `statsmodels` (stats scripts), `transformers` (`scene_seg`) and `scikit-learn` (JAAD parser)
  are core requirements. Known local-env damage (not a requirements problem): statsmodels'
  pip metadata reports version `None`, torchaudio is built for a different CUDA than torch
  (scene_seg hides it before importing transformers), and site-packages holds leftover
  `~orch`/`~ransformers` partial installs. Verify installs in a fresh venv.
- When bumping deps, re-run a resolver check: `python -m pip install --dry-run -r requirements.txt`
  (or a focused subset) and watch for `ResolutionImpossible`. `paddlex` and `ultralytics` export
  extras are the usual sources of `numpy<2` caps.

## Localization (`--mode localize`)

Wraps the [Monocular-OSM-Localization](https://github.com/M-Colley/Monocular-OSM-Localization)
tool (video + city → lat/lon on OpenStreetMap), now a pip package (distribution
`monocular-osm-localization`, import name `monocular_osm`). Code: `modules/localization/localize.py`.

- The tool is an **optional** dependency, kept out of the core `requirements.txt` (heavy
  osmnx/open3d/pycolmap/… stack); install via `pip install -r requirements-localize.txt`. It is
  **never imported** — always run as a **subprocess** via `python -m monocular_osm.cli` (≡ the
  `osm-localize` console script) so its heavy stack stays out of PedX's process. The interpreter
  is resolved: `--osm_python` → `$OSM_LOCALIZATION_PYTHON` → PedX's own interpreter when
  `monocular_osm` is importable there (probed via `importlib.util.find_spec`). So installing the
  package into PedX's venv makes localize run with no separate interpreter. The subprocess runs
  in a gitignored `osm_localization_cache/` (its `--data-dir`) so its caches never litter the repo.
- **`MLY_TOKEN` must be set for usable accuracy.** The tool's default VPR source is `mapillary`;
  without the env var it silently falls back to sparse KartaView and the GPS-free result degrades
  from ~30–156 m to kilometre-scale (which is why the current `all_video_locations.csv` shows
  `confidence_level=low` / 4–14 km spreads). `subprocess.run` passes no `env=`, so the token is
  inherited from the parent process — nothing in PedX needs changing beyond setting it.
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

`python -m pytest -q tests` runs the suite (plain asserts; each file also runs as a script).
`tests/test_aggregation_integrity.py` pins bugs that reached published data (risky-crossing
evidence rule, completeness gate, `'1.0'` flag tokens, city join, NA cluster ids). Also:

1. **Byte-compile** touched files: `python -m compileall -q <files>`.
2. **Exercise CSV-driven logic end-to-end** with synthetic inputs — build a tiny video with
   `cv2.VideoWriter` + synthetic `[B1]`/`[E*]`/`[V*]` CSVs under `analysis_results/<name>/`, then
   call the module functions and assert on the output CSV. (The CV *detector* modules need YOLO
   weights/GPU and can't be run headless; the crossing/summary/localization logic can.)
3. Prefer the dedicated tools (Read/Grep/Edit) over shell text utilities.
4. Do temp work under the scratchpad dir, never in the project tree; clean up `__pycache__`.

## Gotchas / known rough edges

- `analysis_results/` is runtime-only and untracked. `summary_data/` IS tracked: its history is
  the history of the published numbers. Aggregators `mkdir` both; keep that if you add outputs.
- **A scheduled task ("PedX Nightly Publish", 03:00, `scripts/nightly_publish.ps1`) runs the
  aggregators + stats on the CURRENT WORKING TREE and publishes to the live Visualizer DB.**
  Whatever branch is checked out at 03:00 goes live, so never leave the tree in a broken state
  overnight. The Visualizer importer keys on the exact column names of the `summary_data`
  CSVs; adding columns is safe, renaming is not (coordinate with whoever owns PedX-Visualizer).
- **Statistics rules** (see the docstrings of `statistics_with_pdf_save.py` and
  `modules/stats_utils.py`): rows are clustered in cities, so publish n, number of cities and
  a cluster-bootstrap CI with every rate and blank under-powered cells; NA is not 0 (never
  listwise-delete on one metric to compute another); weight cities equally across cities;
  never mix measured values with the imported city constants (`crossing_time`/`crossing_speed`
  come from an external study); correlate city-level covariates on one row per city.
- pandas 3: `str + NA` is NA (fillna before building ids), `.values` can be read-only under
  copy-on-write, and an int column gains NA -> float -> `'1.0'` in CSVs (use `Int64`).
- **City attribution is unverified.** The crawler files a video under the city it searched
  for; titles show some do not depict that city (e.g. Paris2 "How does a blind girl cross the
  road in the UK?", NewYork3 a campaign video). Localization (`--mode localize`) or a manual
  eligibility check is needed before city-level claims.
- `summary_data/all_time_info.csv` (used by `statistics_with_pdf_save.py` section 1 and the
  Visualizer's `analysis_seconds`) is produced by `get_all_video_info.py` from the per-video
  `[A3]analysis_time.csv` files that `run.py` writes — videos analyzed via bare `main.py`
  (not `run.py`) have no `[A3]` and are simply absent from it.
- CSV encoding is UTF-8 across the aggregation/stats path (city/country names are non-ASCII); keep
  reads and writes consistent.
- On Windows, avoid non-ASCII glyphs in `print()` (legacy code-page `UnicodeEncodeError`).
- Keep module output CSVs header-safe on empty input (write with explicit `columns=…`) so downstream
  `pd.read_csv` never hits `EmptyDataError`.
