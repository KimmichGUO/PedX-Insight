# Reproducible development and CI

Use CPython 3.12 in a fresh virtual environment. `requirements.txt` contains minimum
versions under the NumPy 2.5 / OpenCV 5 policy; it is not an exact dependency lock.
Only `opencv-python` is requested: PedX uses core OpenCV APIs (including YuNet and
DNN), and Ultralytics explicitly depends on that distribution. Installing contrib
or headless distributions alongside it makes several packages overwrite the same
`cv2` namespace. Do not repair an existing shared environment by uninstalling a
provider in place; recreate a clean environment instead.

```powershell
py -3.12 -m venv "$env:TEMP\pedx-dev"
& "$env:TEMP\pedx-dev\Scripts\python.exe" -m pip install --upgrade pip
& "$env:TEMP\pedx-dev\Scripts\python.exe" -m pip install --dry-run -r requirements-dev.txt -c constraints-py312-observed.txt
& "$env:TEMP\pedx-dev\Scripts\python.exe" -m pip install -r requirements-dev.txt -c constraints-py312-observed.txt
& "$env:TEMP\pedx-dev\Scripts\python.exe" -m pip check
& "$env:TEMP\pedx-dev\Scripts\python.exe" -B scripts/check_environment.py --imports --strict
```

For CUDA, install a matching Torch/Torchvision pair from the appropriate official
PyTorch index before the remaining requirements. The public versions in the
constraints allow a matched CPU or CUDA build. CI explicitly installs CPU wheels
first, avoiding NVIDIA runtime downloads and checking that the resulting runtime is
CPU-only. FFmpeg and Node remain external command-line tools used by video downloads;
the optional localization stack remains in `requirements-localize.txt` and runs as a
subprocess. `yt-dlp` and the age-model ONNX Runtime fallback are declared core tools.

## What the snapshot establishes

`environment-observed-py312.json` records observed direct distributions on October 27,
2026. The production Python 3.12 runtime passed the offline synthetic suite, but it is
not a clean installation of the current requirements:

- Pillow was **10.4.0**, below the required **12.3.0**. Its incompatible observed value
  is deliberately omitted from `constraints-py312-observed.txt`; clean installs must
  satisfy the existing newer floor and CI must verify that combination.
- Statsmodels distribution metadata returned `None`; the imported module reported
  **0.15.0**. The constraint uses the module version, never `statsmodels==None`.
- Plain, contrib, and headless OpenCV providers were installed. Only the plain provider is retained in the
  corrected requirement profile; clean CI verifies the required core APIs.
- Torch and Torchvision were **2.13.0+cu126 / 0.28.0+cu126**. Public-version pins do not
  reproduce the precise platform/CUDA binaries.

The constraints pin observed, compliant direct dependencies. They are **not a complete
transitive lock**, have no artifact hashes, and do not pin model weights. A successful
CI run is evidence for its fresh CPU environment, not proof of GPU inference or global
detector validity. CI retains its resolver report, installed freeze, and import diagnostics
as an artifact for 14 days. Preserve those when producing a release, and use the pipeline's
model/source provenance separately.

## Offline tests

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:CUDA_VISIBLE_DEVICES = ''
& "$env:TEMP\pedx-dev\Scripts\python.exe" -B -m pytest -q -p no:cacheprovider tests --basetemp "$env:TEMP\pedx-pytest" -k 'not real_archived_crops and not empty_and_non_png_folders and not unreadable_png_is_skipped'
```

The three excluded age/gender tests initialize real models or analyze archived crops.
Scene segmentation and crosswalk tests use fake model loaders and synthetic videos.
Aggregation tests write only synthetic fixtures in temporary directories. This command
does not run production aggregators or publish data. The strict environment check is
expected to flag the known drift in the old shared installation until it is recreated.

## Verified Windows CPU lock (October 27, 2026)

`constraints-py312-cpu-lock.txt` is a complete, artifact-hashed **requirements lock**
for Windows x86_64 and CPython 3.12. It was installed with hash verification into a
new temporary venv, using the official PyTorch CPU wheels and PyPI artifacts. Use
it with **`-r`**, rather than `-c`: every runtime/development dependency is listed,
including transitive dependencies and the selected artifact's SHA256.

```powershell
py -3.12 -m venv "$env:TEMP\pedx-cpu-locked"
& "$env:TEMP\pedx-cpu-locked\Scripts\python.exe" -m pip install --require-hashes --index-url https://pypi.org/simple -r constraints-py312-cpu-lock.txt
& "$env:TEMP\pedx-cpu-locked\Scripts\python.exe" -m pip check
& "$env:TEMP\pedx-cpu-locked\Scripts\python.exe" -B scripts/check_environment.py --imports --strict
```

The tested environment used Python **3.12.9**, Torch **2.13.0+cpu**, Torchvision
**0.28.0+cpu**, NumPy **2.5.3**, the sole OpenCV provider **5.0.0.93**, and Pillow
**13.0.0**. `pip check` and all eleven strict scientific/runtime imports passed.
Torch reported `torch.version.cuda is None`. The final offline suite passed **242 tests**
in **34.42 seconds**, with the three documented real-model tests deselected. The older observed profile
above remains a record of the shared environment; this clean CPU lock establishes
a separately tested package environment.

[`environment-py312-cpu/`](environment-py312-cpu/) retains the complete installed
freeze (including the bootstrap pip version), original resolver and install
reports, strict environment check, CPU assertion, and JUnit/text test results.
`verification.json` records the lock checksum and verification scope. The lock
contains 87 selected packages; bootstrap pip itself is recorded in the freeze.
FilterPy is distributed as a source archive: its source hash is pinned, while its
wheel build is not a hermetic compiler/toolchain build.

These artifact hashes target this Windows/Python ABI; they are not a Linux/CUDA
lock. The Linux CI profile still resolves its matching CPU artifacts independently.
This verifies dependency installation and offline logic, not detector inference,
optional localization, model weights, or the scientific validity of measurements.
