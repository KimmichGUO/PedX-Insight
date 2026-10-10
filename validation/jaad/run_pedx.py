"""Run PedX over JAAD clips: resumable, failure-tolerant, ISOLATED from analysis_results/.

Every PedX invocation is `python main.py --mode <m> --source_video_path <clip>` with
cwd = repo root (model weight paths are repo-relative) and env
PEDX_RESULTS_ROOT=<results_root>, so outputs land in <results_root>/<clip>/ and can never
be picked up by the get_all_*.py aggregators / nightly publish. The runner refuses to
start if the results root resolves inside the repo.

Mode order and dependencies are taken from main.py's `single_all` (only the subset the
benchmark scores, plus what those modes read):

    id_img                -> [B0] [B1] [B2] pedestrian_img/   (everything downstream)
    ego      <- id_img    -> [B3]  (B2 boxes mask people out of the flow estimate)
    ag       <- id_img    -> [P6]  (reads pedestrian_img crops)
    weather, daytime, light, traffic_sign, crosswalk, sidewalk -> [E1] [E6] [E2] [E3] [E7] [E9]
    cross_pede <- id_img, sidewalk      -> [C3]  (crossing_5part reads [B1] + [E9] only;
                                                  lane/width are NOT inputs)
    scale    <- crosswalk               -> [S2]  (stripe calibration, feeds speed)
    speed    <- id_img, ego, cross_pede, scale -> [S1]
    pose     <- id_img, cross_pede      -> [P12]
    risky    <- cross_pede, light, crosswalk, traffic_sign -> [C1]
    run_red  <- cross_pede, light       -> [C5]
    scene_seg                           -> [E10] (standalone candidate module; skipped
                                                  with status 'unavailable' if absent)

A clip/mode is DONE when all its output CSVs exist and [A0]provenance.csv has a row for
the mode (main.py appends it only after the mode returned normally; this also catches
sidewalk, which writes its CSV incrementally and would leave a partial file on a crash).
Done modes are skipped. A mode whose dependency failed is not run (status
'skipped_upstream'), so a later re-run can fill it in. Failures are appended to
<results_root>/_harness/failures.csv and the batch continues.

Usage:
    python -m validation.jaad.run_pedx --sample 40                # writes sample_clips.txt
    python -m validation.jaad.run_pedx --clips_file validation/jaad/sample_clips.txt
    python -m validation.jaad.run_pedx --clips all --workers 3
    python -m validation.jaad.run_pedx --clips video_0001 video_0002 --modes weather light
"""

import argparse
import csv
import datetime as _dt
import hashlib
import os
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = os.environ.get("PEDX_JAAD_RUNS", r"E:/PedX-validation-data/pedx_runs")
DEFAULT_JAAD_ROOT = os.environ.get("JAAD_ROOT", r"E:/PedX-validation-data/JAAD")
SAMPLE_FILE = Path(__file__).with_name("sample_clips.txt")
SAMPLE_SEED = 20261010

# mode -> (output files that must exist, upstream modes it reads)
MODE_SPECS = OrderedDict([
    ("id_img", (["[B1]tracked_pedestrians.csv", "[B2]dense_tracks.csv"], [])),
    ("ego", (["[B3]ego_motion.csv"], ["id_img"])),
    ("ag", (["[P6]age_gender.csv"], ["id_img"])),
    ("weather", (["[E1]weather.csv"], [])),
    ("daytime", (["[E6]daytime.csv"], [])),
    ("light", (["[E2]traffic_light.csv"], [])),
    ("traffic_sign", (["[E3]traffic_sign.csv"], [])),
    ("crosswalk", (["[E7]crosswalk_detection.csv"], [])),
    ("sidewalk", (["[E9]sidewalk_detection.csv"], [])),
    ("cross_pede", (["[C3]crossing_judge.csv"], ["id_img", "sidewalk"])),
    ("scale", (["[S2]scale_calibration.csv"], ["crosswalk"])),
    ("speed", (["[S1]pedestrian_speed.csv"], ["id_img", "ego", "cross_pede", "scale"])),
    ("pose", (["[P12]pose_behavior.csv"], ["id_img", "cross_pede"])),
    ("risky", (["[C1]risky_crossing.csv"], ["cross_pede", "light", "crosswalk", "traffic_sign"])),
    ("run_red", (["[C5]red_light_runner.csv"], ["cross_pede", "light"])),
    ("scene_seg", (["[E10]scene_segmentation.csv"], [])),
])
DEFAULT_MODES = list(MODE_SPECS)
# Modes implemented by a file that may not exist yet in this checkout.
MODE_IMPL = {"scene_seg": REPO / "modules" / "scene_seg" / "scene_seg.py"}
# Source file(s) whose content decides each mode's output (hashed per run, so outputs made
# by different module versions can be told apart while another session edits modules/).
MODE_SRC = {
    "id_img": ["new_track_id_with_imgs.py", "mybotsort.yaml"],
    "ego": ["modules/speed/ego_motion.py"],
    "ag": ["modules/age_gender/age_gender_detect.py"],
    "weather": ["modules/weather/weather.py"],
    "daytime": ["modules/daynight/daytime.py"],
    "light": ["modules/traffic_light/traffic_light.py"],
    "traffic_sign": ["modules/traffic_sign/traffic_sign.py"],
    "crosswalk": ["modules/crosswalk/crosswalk.py"],
    "sidewalk": ["modules/sidewalk/sidewalk_detect.py"],
    "cross_pede": ["modules/crossing_judge/crossing_5part.py"],
    "scale": ["modules/speed/scale_calibration.py"],
    "speed": ["modules/speed/speed_estimation.py"],
    "pose": ["modules/insights/pose_behavior.py"],
    "risky": ["modules/risky_crossing/risky_crossing.py"],
    "run_red": ["modules/run_redlight/run_redlight.py"],
    "scene_seg": ["modules/scene_seg/scene_seg.py"],
}
# Benchmark-only sitecustomize that neutralises cv2.destroyAllWindows() when OpenCV has no
# GUI backend (see cv2_gui_shim/sitecustomize.py for the production bug it works around).
CV2_SHIM_DIR = Path(__file__).with_name("cv2_gui_shim")

LOG_COLUMNS = ["clip", "mode", "status", "returncode", "seconds", "started_utc",
               "code_digest", "mode_src_digest", "cv2_shim", "log_path"]


def utc_now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------------- sampling

def stratified_sample(clips_df, n, seed=SAMPLE_SEED, strata=("weather", "day_night")):
    """Deterministic stratified sample of clip ids.

    Every non-empty stratum (weather x day_night, NaN -> 'n/a') gets at least one clip,
    the remainder is allocated proportionally by largest remainder, and clips are drawn
    without replacement inside each stratum with a fixed-seed RandomState. Returns a
    sorted list of clip ids.
    """
    df = clips_df.copy()
    for s in strata:
        df[s] = df[s].fillna("n/a").astype(str)
    if n >= len(df):
        return sorted(df["clip"].tolist())
    groups = {k: sorted(g["clip"].tolist()) for k, g in df.groupby(list(strata))}
    keys = sorted(groups)
    sizes = np.array([len(groups[k]) for k in keys], dtype=float)
    alloc = np.minimum(1, sizes).astype(int)
    if alloc.sum() > n:            # more strata than slots: keep the largest strata
        order = np.argsort(-sizes, kind="stable")
        alloc[:] = 0
        alloc[order[:n]] = 1
    rest = n - alloc.sum()
    if rest > 0:
        cap = sizes - alloc
        quota = cap / cap.sum() * rest
        add = np.minimum(np.floor(quota).astype(int), cap.astype(int))
        alloc += add
        left = rest - add.sum()
        rema = np.where(cap - add > 0, quota - add, -1.0)
        for i in np.argsort(-rema, kind="stable")[:left]:
            alloc[i] += 1
    rng = np.random.RandomState(seed)
    out = []
    for k, a in zip(keys, alloc):
        if a > 0:
            out.extend(rng.choice(groups[k], size=int(a), replace=False).tolist())
    return sorted(out)


def write_sample_file(clip_ids, n, seed, path=SAMPLE_FILE, clips_df=None):
    lines = [f"# JAAD stratified sample: n={n}, seed={seed}, strata=weather x day_night",
             "# generated by validation/jaad/run_pedx.py --sample (deterministic)"]
    if clips_df is not None:
        sub = clips_df[clips_df["clip"].isin(clip_ids)].copy()
        sub["weather"] = sub["weather"].fillna("n/a")
        sub["day_night"] = sub["day_night"].fillna("n/a")
        for (w, d), g in sub.groupby(["weather", "day_night"]):
            lines.append(f"# stratum weather={w} day_night={d}: {len(g)}")
    lines.extend(clip_ids)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_clips_file(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


# ---------------------------------------------------------------------------- isolation

def assert_isolated(results_root):
    """Refuse a results root inside the repo (in particular analysis_results/)."""
    rr = Path(results_root).resolve()
    repo = REPO.resolve()
    if rr == repo or repo in rr.parents:
        raise SystemExit(f"[run_pedx] refusing results root inside the repo: {rr}")
    return rr


def leaked_folders(prefix="video_"):
    """JAAD-named folders in the repo's analysis_results/ (must always be empty)."""
    ar = REPO / "analysis_results"
    if not ar.is_dir():
        return []
    return sorted(p.name for p in ar.iterdir() if p.is_dir() and p.name.startswith(prefix))


def code_digest():
    """Short sha256 over the PedX code the runs execute (main.py, tracker, modules/**.py).

    Recorded per mode run: another session may edit modules while a long batch runs, and
    git_sha/git_dirty in [A0] cannot distinguish two different dirty states.
    """
    h = hashlib.sha256()
    files = [REPO / "main.py", REPO / "new_track_id_with_imgs.py"]
    files += sorted((REPO / "modules").rglob("*.py"))
    for f in files:
        if "__pycache__" in f.parts or "SGDepth" in f.parts:
            continue
        try:
            h.update(f.relative_to(REPO).as_posix().encode())
            h.update(f.read_bytes())
        except OSError:
            continue
    return h.hexdigest()[:12]


def mode_src_digest(mode):
    h = hashlib.sha256()
    for rel in MODE_SRC.get(mode, []):
        f = REPO / rel
        h.update(rel.encode())
        h.update(f.read_bytes() if f.exists() else b"<missing>")
    return h.hexdigest()[:12]


# ------------------------------------------------------------------------------- state

def provenance_modes(clip_dir):
    p = Path(clip_dir) / "[A0]provenance.csv"
    if not p.exists():
        return set()
    try:
        with open(p, newline="", encoding="utf-8") as f:
            return {r.get("mode") for r in csv.DictReader(f)}
    except Exception:
        return set()


def mode_done(clip_dir, mode):
    outs, _ = MODE_SPECS[mode]
    if not all((Path(clip_dir) / o).exists() for o in outs):
        return False
    return mode in provenance_modes(clip_dir)


def gpu_free_mb():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return min(int(x) for x in r.stdout.split())
    except Exception:
        return None


def wait_for_gpu(min_free_mb, poll_s=30, max_wait_s=3600):
    if not min_free_mb:
        return True
    t0 = time.time()
    while True:
        free = gpu_free_mb()
        if free is None or free >= min_free_mb:
            return True
        if time.time() - t0 > max_wait_s:
            return False
        print(f"[run_pedx] GPU free {free} MB < {min_free_mb} MB; waiting...", flush=True)
        time.sleep(poll_s)


class _Logger:
    def __init__(self, harness_dir):
        self.dir = Path(harness_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_log = self.dir / "run_log.csv"
        self.fail_log = self.dir / "failures.csv"
        self.lock = threading.Lock()

    def _append(self, path, row):
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)

    def log(self, row):
        with self.lock:
            self._append(self.run_log, row)
            if row["status"] in ("failed", "timeout"):
                self._append(self.fail_log, row)


# --------------------------------------------------------------------------------- run

def run_clip(clip, video_path, modes, results_root, interval, logger, timeout_s,
             min_free_mb, digest, cv2_shim=True, force_modes=()):
    clip_dir = Path(results_root) / clip
    log_dir = Path(results_root) / "_harness" / "logs" / clip
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PEDX_RESULTS_ROOT"] = str(results_root)
    env["PYTHONIOENCODING"] = "utf-8"
    if cv2_shim:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(CV2_SHIM_DIR)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    status = {}
    for mode in modes:
        _, deps = MODE_SPECS[mode]
        row = {"clip": clip, "mode": mode, "returncode": "", "seconds": "",
               "started_utc": utc_now(), "code_digest": digest,
               "mode_src_digest": mode_src_digest(mode), "cv2_shim": bool(cv2_shim),
               "log_path": ""}
        if mode not in force_modes and mode_done(clip_dir, mode):
            status[mode] = "done"
            continue                         # resumable: not re-logged
        impl = MODE_IMPL.get(mode)
        if impl is not None and not impl.exists():
            status[mode] = "unavailable"
            logger.log({**row, "status": "unavailable"})
            continue
        bad = [d for d in deps if d in modes and status.get(d) not in ("done", "ok")]
        bad += [d for d in deps if d not in modes and not mode_done(clip_dir, d)]
        if bad:
            status[mode] = "skipped_upstream"
            logger.log({**row, "status": "skipped_upstream", "log_path": ";".join(bad)})
            continue
        if not wait_for_gpu(min_free_mb):
            status[mode] = "failed"
            logger.log({**row, "status": "failed", "returncode": "gpu_busy"})
            continue
        log_path = log_dir / f"{mode}.log"
        cmd = [sys.executable, "main.py", "--mode", mode, "--source_video_path", str(video_path),
               "--analysis_interval", str(interval)]
        t0 = time.time()
        try:
            with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
                res = subprocess.run(cmd, cwd=str(REPO), env=env, stdout=lf,
                                     stderr=subprocess.STDOUT, timeout=timeout_s)
            rc = res.returncode
            st = "ok" if rc == 0 and mode_done(clip_dir, mode) else "failed"
        except subprocess.TimeoutExpired:
            rc, st = "timeout", "timeout"
        secs = round(time.time() - t0, 1)
        status[mode] = st
        logger.log({**row, "status": st, "returncode": rc, "seconds": secs,
                    "log_path": str(log_path)})
        print(f"[run_pedx] {clip} {mode:12s} {st:8s} {secs:7.1f}s", flush=True)
    return clip, status


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run PedX over JAAD clips (isolated, resumable).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--clips", nargs="+", help="clip ids (video_0001 ...) or 'all'")
    g.add_argument("--clips_file", help="file with one clip id per line (# comments)")
    g.add_argument("--sample", type=int, help="stratified sample size (writes sample_clips.txt)")
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES, choices=list(MODE_SPECS))
    ap.add_argument("--analysis_interval", type=float, default=1.0)
    ap.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--jaad_root", default=DEFAULT_JAAD_ROOT)
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--workers", type=int, default=1, help="clips processed in parallel")
    ap.add_argument("--timeout", type=int, default=3600, help="per-mode timeout (s)")
    ap.add_argument("--min_free_vram_mb", type=int, default=3000,
                    help="wait before each mode until this much VRAM is free (0 = off)")
    ap.add_argument("--force_modes", nargs="+", default=[], choices=list(MODE_SPECS),
                    help="re-run these modes even if their outputs exist (e.g. after a module edit)")
    ap.add_argument("--no_cv2_shim", action="store_true",
                    help="run without the destroyAllWindows shim (reproduces the OpenCV crash)")
    ap.add_argument("--dry_run", action="store_true", help="select clips, print plan, exit")
    args = ap.parse_args(argv)

    results_root = assert_isolated(args.results_root)
    clips_dir = Path(args.jaad_root) / "JAAD_clips"
    # Respect main.py's canonical order regardless of the order given on the CLI.
    modes = [m for m in MODE_SPECS if m in set(args.modes)]

    if args.sample is not None:
        from validation.jaad.gt import build_tables
        clips_df = build_tables(args.jaad_root, with_fps=False)["clips"]
        clip_ids = stratified_sample(clips_df, args.sample, seed=args.seed)
        write_sample_file(clip_ids, args.sample, args.seed, clips_df=clips_df)
        print(f"[run_pedx] stratified sample of {len(clip_ids)} written to {SAMPLE_FILE}")
    elif args.clips_file:
        clip_ids = read_clips_file(args.clips_file)
    elif args.clips == ["all"]:
        clip_ids = sorted(p.stem for p in clips_dir.glob("video_*.mp4"))
    else:
        clip_ids = sorted(set(args.clips))

    missing = [c for c in clip_ids if not (clips_dir / f"{c}.mp4").exists()]
    if missing:
        print(f"[run_pedx] WARNING: {len(missing)} clips missing on disk, skipped: {missing[:5]}")
    clip_ids = [c for c in clip_ids if c not in missing]
    print(f"[run_pedx] {len(clip_ids)} clips x {len(modes)} modes -> {results_root} "
          f"(interval {args.analysis_interval}s, workers {args.workers})", flush=True)
    if args.dry_run:
        print("\n".join(clip_ids))
        return 0

    leaked_before = leaked_folders()
    if leaked_before:
        print(f"[run_pedx] WARNING: analysis_results/ already holds JAAD folders: {leaked_before}")
    results_root.mkdir(parents=True, exist_ok=True)
    logger = _Logger(results_root / "_harness")
    digest = code_digest()
    t0 = time.time()
    n_done = 0
    progress = results_root / "_harness" / "progress.txt"
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(run_clip, c, clips_dir / f"{c}.mp4", modes, results_root,
                          args.analysis_interval, logger, args.timeout,
                          args.min_free_vram_mb, digest, not args.no_cv2_shim,
                          tuple(args.force_modes)) for c in clip_ids]
        for fut in as_completed(futs):
            clip, status = fut.result()
            n_done += 1
            el = time.time() - t0
            eta = el / n_done * (len(clip_ids) - n_done)
            bad = {m: s for m, s in status.items() if s not in ("done", "ok", "unavailable")}
            msg = (f"{utc_now()} clips {n_done}/{len(clip_ids)} elapsed {el / 60:.1f} min "
                   f"eta {eta / 60:.1f} min | last {clip} {bad if bad else 'ok'}")
            print(f"[run_pedx] {msg}", flush=True)
            with open(progress, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
    leaked_after = leaked_folders()
    if leaked_after:
        print(f"[run_pedx] ERROR: JAAD folders found in analysis_results/: {leaked_after}")
        return 2
    print(f"[run_pedx] finished in {(time.time() - t0) / 60:.1f} min; "
          f"analysis_results/ holds no video_* folders.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
