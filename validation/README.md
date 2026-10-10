# Validation of PedX detectors against ground truth

PedX's published numbers come from detectors that had never been checked against labelled
data. `validation/jaad/` benchmarks them on **JAAD** (Joint Attention in Autonomous Driving,
York University; 346 short dashcam clips with per-frame labels) and writes
`validation/results/JAAD_REPORT.md`.

What is scored (each against JAAD's own labels, with 95% CIs from a cluster bootstrap over
clips):

| task | PedX output | JAAD label |
|---|---|---|
| weather | `[E1]` | clip weather |
| day / night | `[E6]` | clip time of day |
| traffic light | `[E2]` | per-frame traffic light state |
| crosswalk | `[E7]` shipped, `[E10]` Vistas segmentation, `[X7]` counterfactual | per-frame `ped_crossing` |
| pedestrian detection | `[B1]` 1 Hz, `[B2]` dense | pedestrian boxes (by occlusion, height) |
| crossing | `[C3]` | `crossing` attribute (matched tracks) |
| ego-motion gate | `[B3]` | ego-vehicle stopped / moving |
| looking | `[P12]` | `look` behaviour |
| age / gender | `[P6]` | pedestrian age group / gender |
| signs, risky coverage | `[E3]`, `[C1]` | `ped_sign`, `stop_sign`; coverage only |

JAAD has no "risky crossing" label, so `[C1]` is validated for **coverage** (how often a
signal state is observed) only; the risk classification itself cannot be validated here.

## Reproduce

Data (about 3.1 GB of clips, outside the repo; Mask2Former weights ~0.9 GB for `[E10]`):

```bash
git clone --depth 1 https://github.com/ykotseruba/JAAD.git   E:/PedX-validation-data/JAAD     # MIT
curl -L -o E:/PedX-validation-data/downloads/JAAD_clips.zip \
     http://data.nvision2.eecs.yorku.ca/JAAD_dataset/data/JAAD_clips.zip
# unzip so that E:/PedX-validation-data/JAAD/JAAD_clips/video_0001.mp4 exists
python -c "from huggingface_hub import snapshot_download as s; s('facebook/mask2former-swin-large-mapillary-vistas-semantic', cache_dir='E:/PedX-validation-data/hf_cache', allow_patterns=['*.json','*.safetensors'])"
```

Paths default to `E:/PedX-validation-data/...`; override with `JAAD_ROOT`, `PEDX_JAAD_RUNS`,
`PEDX_HF_CACHE`.

```bash
python -m validation.jaad.gt                                  # ground-truth tables (uses JAAD's own parser)
python -m validation.jaad.run_pedx --clips all --workers 3    # PedX over every clip; resumable
python -m validation.jaad.probes --clips all                  # [X7] counterfactual crosswalk class probe
python -m validation.jaad.report                              # scores -> validation/results/
python -m pytest -q tests/test_jaad_scoring.py                # harness unit tests (synthetic data)
```

Runtime on one RTX 4090: about 20 s per clip effective with 3 workers, so roughly 2 hours for
all 346. `run_pedx` skips every clip/mode that is already done, so an interrupted run resumes.

## Isolation (important)

`run_pedx` sets `PEDX_RESULTS_ROOT` to a directory **outside the repo** and refuses to start if
it resolves inside it. Benchmark outputs must never land in `analysis_results/`: the nightly job
aggregates everything there into the published dataset.

## Verdict rules

Each metric's 95% CI is tested against a bar: *pass* when the whole CI is above it, *fail* only
when the whole CI is below it, otherwise *inconclusive* (a lower bound just under the bar is
"cannot tell yet", never "fail"). Bars and the rules are in `validation/jaad/report.py`
(`verdicts`), and are printed in the report.

## Caveats

- JAAD is dashcam footage from a few countries and is mostly daytime and dry. A pass here is
  necessary, not sufficient, for PedX's global dataset; a fail is informative everywhere.
- JAAD does not box every pedestrian, so detection precision is a lower bound.
- `cv2_gui_shim/` is a benchmark-only `sitecustomize` that made `cv2.destroyAllWindows()` a no-op
  while the production modules still crashed on it; the modules now guard the call themselves
  (commit "Guard cv2.destroyAllWindows ..."), so the shim is only needed to re-run old checkouts
  (`run_pedx --no_cv2_shim` disables it).
