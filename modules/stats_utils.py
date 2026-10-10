"""Shared statistics helpers for the aggregation/stats scripts.

The pedestrian-level rows in `summary_data/all_pedestrian_info.csv` are NOT independent:
pedestrians are nested in videos, and videos in cities (one camera, one detector run, one
street, one culture). Treating them as i.i.d. understates every uncertainty and lets a
single busy video dominate a group rate -- Manila alone contributes 358 of the 1371 rows.

The helpers here make the clustering explicit:

  * `rate_with_ci`      -- a pooled rate with a CLUSTER bootstrap CI (resample whole
                           cities, not pedestrians; Field & Welsh 2007), plus its n and
                           number of clusters, blanked below a minimum support.
  * `cluster_bootstrap_ci` -- the same resampling for an arbitrary statistic.
  * `bh_fdr`            -- Benjamini-Hochberg q-values (statsmodels).
  * `city_level`        -- collapse video rows to one row per city before any city-level
                           correlation, so a city with three videos is not counted thrice
                           against socio-economic covariates that are constant within it.

Everything is seeded so the published tables are reproducible run-to-run.
"""

import numpy as np
import pandas as pd

# Minimum support before a rate is published. A rate below this is written as blank (the
# Visualizer importer skips blank cells) with its n still shown, so a thin cell is visible
# rather than plotted as a confident percentage.
MIN_N = 10
MIN_CLUSTERS = 2
# Fewer clusters than this and a cluster bootstrap CI is not meaningful (the resampling
# distribution has too few distinct support points); the CI is left blank.
MIN_CLUSTERS_FOR_CI = 5
N_BOOT = 2000
SEED = 20261009


def _as_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def rate_with_ci(values, clusters, *, scale=1.0, n_boot=N_BOOT, seed=SEED, alpha=0.05,
                 min_n=MIN_N, min_clusters=MIN_CLUSTERS,
                 min_clusters_for_ci=MIN_CLUSTERS_FOR_CI):
    """Pooled mean of a 0/1 (or numeric) series with a cluster-bootstrap percentile CI.

    Returns a dict: estimate, ci_low, ci_high, n, n_clusters. NA observations are dropped
    (an undetermined observation is not a 0). `scale=100` reports percentages.
    """
    v = _as_numeric(pd.Series(values)).reset_index(drop=True)
    c = pd.Series(clusters).reset_index(drop=True)
    keep = v.notna() & c.notna()
    v, c = v[keep], c[keep]
    n, k = int(len(v)), int(c.nunique())
    out = {"estimate": None, "ci_low": None, "ci_high": None, "n": n, "n_clusters": k}
    if n < min_n or k < min_clusters:
        return out
    out["estimate"] = float(v.mean()) * scale
    if k < min_clusters_for_ci:
        return out
    # Ratio estimator over resampled clusters: sum of values / count of observations.
    g = pd.DataFrame({"v": v.values, "c": c.values}).groupby("c")["v"].agg(["sum", "count"])
    sums, counts = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    boot = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    out["ci_low"], out["ci_high"] = float(lo) * scale, float(hi) * scale
    return out


def rate_columns(frame, metrics, cluster_col, *, scale=1.0, names=None, **kw):
    """One output row's worth of columns for several metrics over `frame`.

    For each metric m (published as names[m] if given) emits m, m_n, m_n_clusters,
    m_ci_low, m_ci_high. The bare metric column keeps the name downstream importers key on.
    """
    names = names or {}
    row = {}
    for m in metrics:
        col = names.get(m, m)
        r = rate_with_ci(frame[m], frame[cluster_col], scale=scale, **kw)
        row[col] = r["estimate"]
        row[f"{col}_n"] = r["n"]
        row[f"{col}_n_clusters"] = r["n_clusters"]
        row[f"{col}_ci_low"] = r["ci_low"]
        row[f"{col}_ci_high"] = r["ci_high"]
    return row


def cluster_bootstrap_ci(frame, cluster_col, stat_fn, *, n_boot=N_BOOT, seed=SEED,
                         alpha=0.05, min_clusters_for_ci=MIN_CLUSTERS_FOR_CI):
    """Percentile CI for `stat_fn(frame)` by resampling whole clusters with replacement.

    Resampled duplicates of a cluster get distinct ids so stat_fn may itself group by
    cluster. Iterations where stat_fn returns NaN (e.g. a constant column) are dropped.
    """
    clusters = frame[cluster_col].dropna().unique()
    if len(clusters) < min_clusters_for_ci:
        return None, None
    parts = {cid: grp for cid, grp in frame.groupby(cluster_col)}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat(
            [parts[cid].assign(**{cluster_col: f"{cid}#{i}"}) for i, cid in enumerate(pick)],
            ignore_index=True)
        s = stat_fn(sample)
        if s is not None and np.isfinite(s):
            stats.append(s)
    if len(stats) < n_boot // 2:
        return None, None
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values; NA p-values stay NA and do not count toward m."""
    from statsmodels.stats.multitest import multipletests
    p = pd.Series(pvals, dtype="float64")
    q = pd.Series(np.nan, index=p.index)
    ok = p.notna()
    if ok.any():
        q[ok] = multipletests(p[ok].to_numpy(), method="fdr_bh")[1]
    return q


def city_level(video_df, value_cols, *, city_col="city", constant_cols=()):
    """Collapse video rows to one row per city.

    `value_cols` (per-video measurements) are averaged across a city's videos;
    `constant_cols` (city/country covariates such as gini or population) are taken once.
    `n_videos` records how many videos stand behind each city row.
    """
    df = video_df.copy()
    for col in list(value_cols) + list(constant_cols):
        if col in df.columns:
            df[col] = _as_numeric(df[col])
    agg = {c: "mean" for c in value_cols if c in df.columns}
    agg.update({c: "first" for c in constant_cols if c in df.columns})
    out = df.groupby(city_col).agg(agg)
    out["n_videos"] = df.groupby(city_col).size()
    return out.reset_index()


def city_cluster_id(df, city_col="city", country_col="country"):
    """Add a `cluster` column identifying the city (city names collide across countries).

    fillna BEFORE concatenating: under pandas 3, str + NA is NA, and a NA cluster id
    silently drops the row from every clustered computation.
    """
    df = df.copy()
    country = df[country_col].fillna("").astype(str) if country_col in df.columns else ""
    df["cluster"] = df[city_col].fillna("").astype(str) + "|" + country
    return df


AGE_BINS = [0, 18, 30, 45, 65, 200]
AGE_LABELS = ["<18", "18-29", "30-44", "45-64", "65+"]


def age_band(age):
    """Bin a numeric (model-inferred) age into life-stage bands.

    Grouping by the raw integer age produced ~40 one- or two-person 'groups'; inferred ages
    are also only accurate to several years, so bands are the honest resolution.
    """
    return pd.cut(_as_numeric(age), bins=AGE_BINS, labels=AGE_LABELS, right=False)
