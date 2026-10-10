"""JAAD benchmark for PedX (Joint Attention in Autonomous Driving, York University).

    gt.py        tidy ground-truth tables built with JAAD's own parser (jaad_data.py)
    run_pedx.py  resumable runner: PedX modes over JAAD clips, isolated results root
    probes.py    counterfactual probes that quantify module bugs (never production)
    score.py     metric computation (alignment, matching, cluster bootstrap)
    report.py    writes validation/results/{jaad_metrics.csv, cm_*.csv, JAAD_REPORT.md}
"""
