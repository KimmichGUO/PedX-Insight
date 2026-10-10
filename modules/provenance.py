"""Per-video provenance: which code and which model weights produced each output.

Every published number should be traceable to the exact code revision and weight files
that produced it. Until now nothing recorded either: a folder analysed in July and one
analysed last night were indistinguishable, even though the risky-crossing construct, the
tracker and several detectors changed in between (`pipeline_version` only separates the
1 Hz era from the dense one).

`record_mode_run` is called by main.py after every successful single-mode run and appends
one row to `<RESULTS_ROOT>/<video>/[A0]provenance.csv`:

    mode, started_utc, finished_utc, git_sha, git_dirty, weights_digest, packages

`weights_digest` is a short hash over the sha256 of every weight file in the repo; the full
path -> sha256 map for each digest is kept in `[A0]weights_manifest.json`. Hashing ~1 GB of
weights per mode would be wasteful, so file hashes are cached by (path, size, mtime) in a
gitignored `.weights_sha256_cache.json`.

Provenance must never break an analysis: every failure here is caught and reported.
"""

import csv
import datetime as _dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from modules.paths import RESULTS_ROOT

REPO = Path(__file__).resolve().parents[1]
WEIGHT_EXTS = (".pt", ".pth", ".onnx", ".safetensors", ".ckpt")
# Directories that hold data or third-party checkouts, never PedX's own weights.
_SKIP_DIRS = {".git", "analysis_results", "videos", "external", "summary_data",
              "osm_localization_cache", "__pycache__", ".venv", "venv"}
_HASH_CACHE = REPO / ".weights_sha256_cache.json"
_PACKAGES = ("numpy", "pandas", "opencv-python", "torch", "ultralytics", "transformers",
             "supervision", "scipy")

PROVENANCE_COLUMNS = ["mode", "started_utc", "finished_utc", "git_sha", "git_dirty",
                      "weights_digest", "packages"]


def utc_now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_state():
    """(sha, dirty) of the repo, or (None, None) when git is unavailable.

    `dirty` considers tracked files only: untracked scratch files do not change what the
    pipeline computes, modified tracked files do.
    """
    try:
        sha = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=15).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        return (sha or None), bool(status)
    except Exception:
        return None, None


def _iter_weight_files():
    for dirpath, dirnames, filenames in os.walk(REPO):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.lower().endswith(WEIGHT_EXTS):
                yield Path(dirpath) / fn


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def weights_manifest():
    """{repo-relative path: sha256} for every weight file, using the (size, mtime) cache."""
    try:
        cache = json.loads(_HASH_CACHE.read_text(encoding="utf-8"))
    except Exception:
        cache = {}
    manifest, dirty_cache = {}, False
    for p in _iter_weight_files():
        rel = p.relative_to(REPO).as_posix()
        st = p.stat()
        key = f"{st.st_size}:{st.st_mtime_ns}"
        hit = cache.get(rel)
        if hit and hit.get("key") == key:
            digest = hit["sha256"]
        else:
            digest = _sha256(p)
            cache[rel] = {"key": key, "sha256": digest}
            dirty_cache = True
        manifest[rel] = digest
    if dirty_cache:
        try:
            _HASH_CACHE.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
        except Exception:
            pass
    return manifest


def manifest_digest(manifest):
    blob = json.dumps(sorted(manifest.items())).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def package_versions():
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": sys.version.split()[0]}
    for name in _PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
    return out


def record_mode_run(video_path, mode, started_utc):
    """Append a provenance row for a mode that just completed successfully."""
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        out_dir = os.path.join(RESULTS_ROOT, video_name)
        if not os.path.isdir(out_dir):
            return  # the mode wrote nothing for this video; nothing to attribute
        sha, dirty = git_state()
        manifest = weights_manifest()
        digest = manifest_digest(manifest)

        man_path = os.path.join(out_dir, "[A0]weights_manifest.json")
        try:
            with open(man_path, encoding="utf-8") as f:
                manifests = json.load(f)
        except Exception:
            manifests = {}
        if digest not in manifests:
            manifests[digest] = manifest
            with open(man_path, "w", encoding="utf-8") as f:
                json.dump(manifests, f, indent=1, sort_keys=True)

        prov_path = os.path.join(out_dir, "[A0]provenance.csv")
        new_file = not os.path.exists(prov_path)
        with open(prov_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=PROVENANCE_COLUMNS)
            if new_file:
                w.writeheader()
            w.writerow({
                "mode": mode,
                "started_utc": started_utc,
                "finished_utc": utc_now(),
                "git_sha": sha,
                "git_dirty": dirty,
                "weights_digest": digest,
                "packages": json.dumps(package_versions(), sort_keys=True),
            })
    except Exception as e:  # provenance must never fail an analysis
        print(f"[warn] provenance not recorded for mode {mode}: {e}", flush=True)


def summarize(folder_path):
    """Video-level provenance for the aggregators.

    Returns (code_version, code_version_mixed): the git sha of the most recent mode run
    (suffixed '+dirty' when it ran on modified tracked files) and whether the folder's
    outputs come from more than one revision. (None, None) when no [A0] exists -- every
    folder analysed before provenance was introduced.
    """
    path = Path(folder_path) / "[A0]provenance.csv"
    if not path.exists():
        return None, None
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception:
        return None, None
    if df.empty or "git_sha" not in df.columns:
        return None, None
    df = df.sort_values("finished_utc")
    last = df.iloc[-1]
    sha = str(last["git_sha"]) if pd.notna(last["git_sha"]) else None
    if sha and str(last.get("git_dirty")).strip().lower() == "true":
        sha += "+dirty"
    mixed = df["git_sha"].dropna().nunique() > 1
    return sha, bool(mixed)
