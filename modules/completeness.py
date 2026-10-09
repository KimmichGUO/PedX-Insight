"""Shared completeness gate for a per-video analysis folder.

A run that dies mid-pipeline still leaves an `analysis_results/<video>/` directory, and
the summary modes will happily write an [A1]/[A2] over its partial inputs -- so the
presence of a summary file is NOT evidence that the video was analysed.

The live example is `analysis_results/Luanda_z3Gx2hp3Vo8/`: the 2026-07-27 batch died at
`lane_detection` with roughly 29 of 40 modes never run, leaving person crops and a
[P6]age_gender.csv of inferred ages and genders for a city that has no behavioural data
at all. Admitting that folder would put a privacy liability into the published dataset
with no scientific counterpart.

Both aggregators (`get_all_video_info.py` and `get_all_pede_info.py`) must apply the SAME
gate, or a folder rejected by one still reaches the dataset through the other.
"""

# The inputs every published column ultimately depends on. Deliberately a floor, not the
# full 40-module inventory: a folder missing any of these cannot produce a usable row.
REQUIRED_OUTPUTS = [
    '[B1]tracked_pedestrians.csv',   # pedestrian counts, the denominator of everything
    '[C3]crossing_judge.csv',        # crossing windows -> every crossing metric
    '[E1]weather.csv',               # environment rollup
    '[E2]traffic_light.csv',         # risky-crossing signal evidence
    '[E7]crosswalk_detection.csv',   # crosswalk_prob / crosswalk usage
    '[V6]vehicle_count.csv',         # vehicle columns
]


def missing_outputs(folder_path):
    """Return the required outputs absent from `folder_path` (a pathlib.Path)."""
    return [name for name in REQUIRED_OUTPUTS if not (folder_path / name).exists()]
