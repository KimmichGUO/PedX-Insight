"""Validation harnesses that score PedX modules against labelled ground truth.

Nothing here is imported by the production pipeline. Benchmark runs write to a
results root OUTSIDE the repo's analysis_results/ (via PEDX_RESULTS_ROOT) so they
can never be aggregated into the published city dataset.
"""
