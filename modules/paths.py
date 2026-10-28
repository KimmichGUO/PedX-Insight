"""Root directory for per-video analysis outputs.

Every module writes to `<RESULTS_ROOT>/<video_name>/[CODE]name.csv`. The root defaults to
the relative `analysis_results` (production, consumed by the get_all_*.py aggregators and
the nightly publish). Setting PEDX_RESULTS_ROOT redirects a run elsewhere -- the JAAD
validation harness uses this so benchmark clips can never be picked up by the aggregators
and published as city data.
"""

import os

RESULTS_ROOT = os.environ.get("PEDX_RESULTS_ROOT") or "analysis_results"
SUMMARY_ROOT = os.environ.get("PEDX_SUMMARY_ROOT") or "summary_data"
