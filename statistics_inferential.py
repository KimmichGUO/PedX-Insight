"""Inferential statistics over summary_data/ -> inferential_*.csv + forest plot.

statistics_with_pdf_save.py describes; this script tests. It answers two questions that
group means cannot, because pedestrians are clustered in videos and videos in cities:

1. VARIANCE PARTITION -- how much of the variation in each behaviour lies BETWEEN
   cities (the cross-cultural signal the project exists to measure) versus between
   videos of one city and between pedestrians. For crossing speed this is a linear mixed
   model (random intercepts for city, and for video within city), with a likelihood-
   ratio test for the city component. For the 0/1 outcomes it is a Bayesian logistic
   mixed model (statsmodels BinomialBayesMixedGLM, variational Bayes) and the
   latent-scale ICC  s2_city / (s2_city + s2_video + pi^2/3).

2. ADJUSTED ASSOCIATIONS -- does a pedestrian/scene attribute (gender, age band, phone
   use, crowd size at crossing start, daytime, weather, traffic volume) predict red-light
   running, risky crossing or crossing speed, holding the design covariates fixed?
   * 0/1 outcomes: GEE logistic regression, exchangeable working correlation within
     city, bias-reduced (Mancl-DeRouen) sandwich covariance; Wald tests use a t
     distribution with (cities - 1) df, because ~20 clusters is few for a z-test.
   * crossing speed: linear mixed model with a random intercept per city.
   One model per predictor FAMILY (focal predictor + design covariates) rather than one
   all-in model: gender/age are only inferred for ~40% of crossers, and a joint model
   would discard everyone else. q-values are Benjamini-Hochberg across all terms of an
   outcome.
   Design covariates, always included: pipeline_version (dense vs legacy 1 Hz tracking)
   and the video's traffic_light_prob -- the detector-recall proxy that the 2026-09
   audit found confounded with the risky-crossing metric. For risky_crossing also
   risky_crossing_coverage.

Measurement caveats travel with the results: weather and daytime come from classifiers
that have not yet been validated against ground truth (see validation/), and crossing
speed is restricted to the plausible human range SPEED_RANGE_MPS -- values outside it
are dominated by ID switches and waiting pedestrians (documented in modules/speed).
"""

import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from modules.stats_utils import SEED, age_band, bh_fdr, city_cluster_id
from modules.paths import SUMMARY_ROOT
from modules.quality import apply_metric_policy

OUT = SUMMARY_ROOT
# No CreationDate: identical results give byte-identical PDFs, so the tracked summary_data/
# history only changes when the numbers do.
PDF_METADATA = {"CreationDate": None}
SPEED_RANGE_MPS = (0.3, 3.0)
MIN_OBS = 50          # per model
MIN_EVENTS = 10       # per 0/1 model (fewer and the logit is not estimable)
MIN_CLUSTERS = 8      # cluster-robust inference needs a handful of clusters at least
MIN_MULTI_VIDEO_CITIES = 5   # city and video variance are split only with this many multi-video cities
SPLIT_NOTE = ("city and video variance are not separable ({k} cities have more than one video; "
              "{m} needed): var_city_or_video / icc_city_or_video is the between-video variance, "
              "which includes between-city variance")
MIN_LEVEL_N = 20      # categorical levels with fewer rows are dropped from that model
MAX_LOGIT_SE = 5.0    # a logit SE beyond this signals (quasi-)separation, not an estimate

BINARY_OUTCOMES = ["run_red_light", "risky_crossing"]
SPEED_OUTCOME = "crossing_speed_mps"

# focal predictor families: name -> (formula terms, required columns, caveat)
FAMILIES = {
    "gender": ("female", ["female"], "gender inferred from face crops (YuNet + genderage.onnx)"),
    "age_band": ("C(age_band, Treatment('30-44'))", ["age_band"], "age inferred from face crops"),
    "phone": ("phone", ["phone"], ""),
    "crowd": ("nearby_start", ["nearby_start"], "pedestrians nearby at crossing start"),
    "daytime": ("C(daytime, Treatment('Day'))", ["daytime"], "daytime classifier not yet validated"),
    "weather": ("C(weather, Treatment('shine'))", ["weather"], "weather classifier not yet validated"),
    "traffic": ("traffic_z", ["traffic_z"], "avg vehicles in view, standardised"),
}


def load():
    include_experimental = os.environ.get("PEDX_INCLUDE_EXPERIMENTAL") == "1"
    ped = apply_metric_policy(pd.read_csv(f"{OUT}/all_pedestrian_info.csv", encoding="utf-8"), include_experimental=include_experimental)
    vid = apply_metric_policy(pd.read_csv(f"{OUT}/all_video_info.csv", encoding="utf-8"), include_experimental=include_experimental)
    nuis = vid[["link", "pipeline_version", "traffic_light_prob", "risky_crossing_coverage"]]
    d = ped.merge(nuis, on="link", how="left")
    d = city_cluster_id(d)
    num = lambda c: pd.to_numeric(d[c], errors="coerce")
    for c in BINARY_OUTCOMES + [SPEED_OUTCOME, "traffic_light_prob", "risky_crossing_coverage"]:
        d[c] = num(c)
    d["female"] = d["gender"].map({"female": 1.0, "male": 0.0})
    d["age_band"] = age_band(d["age"]).astype(object)
    d["phone"] = num("phone_using")
    d["nearby_start"] = num("nearby_count_beginning")
    tv = num("avg_vehicle_total")
    d["traffic_z"] = (tv - tv.mean()) / tv.std()
    d["dense"] = (d["pipeline_version"] == "dense_v2").astype(float)
    lo, hi = SPEED_RANGE_MPS
    d["speed_ok"] = d[SPEED_OUTCOME].where(d[SPEED_OUTCOME].between(lo, hi))
    return d


def _nuisance(sub, outcome):
    terms = []
    if sub["dense"].nunique() > 1:
        terms.append("dense")
    if outcome != SPEED_OUTCOME and sub["traffic_light_prob"].nunique() > 1:
        terms.append("traffic_light_prob")
    if outcome == "risky_crossing" and sub["risky_crossing_coverage"].nunique() > 1:
        terms.append("risky_crossing_coverage")
    return terms


# ----------------------------------------------------------------- variance partition

def variance_partition(d):
    import statsmodels.formula.api as smf
    rows = []
    rng = np.random.default_rng(SEED)

    # crossing speed: LMM with city RE + video variance component
    sub = d.dropna(subset=["speed_ok"])
    base = {"outcome": "crossing_speed_mps (0.3-3.0 m/s)", "adjusted_for": "none", "n_obs": len(sub),
            "n_videos": sub["link"].nunique(), "n_cities": sub["cluster"].nunique(),
            "n_multi_video_cities": _multi_video_cities(sub)}
    # City and video variance are separable only when some city has several videos.
    # (re_formula="1" is required: with vc_formula, statsmodels otherwise drops the
    # per-group random intercept.)
    separable = base["n_videos"] > base["n_cities"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if separable:
                m = smf.mixedlm("speed_ok ~ 1", sub, groups=sub["cluster"], re_formula="1",
                                vc_formula={"video": "0 + C(link)"}).fit(reml=True)
                v_video = float(m.vcomp[0])
            else:
                m = smf.mixedlm("speed_ok ~ 1", sub, groups=sub["cluster"]).fit(reml=True)
                v_video = 0.0
            v_city = float(m.cov_re.iloc[0, 0])
            v_res = float(m.scale)
            # LRT for the city component: ML fits with and without it (boundary test,
            # 50:50 chi-bar-square mixture).
            m1 = smf.mixedlm("speed_ok ~ 1", sub, groups=sub["cluster"]).fit(reml=False)
            m0 = smf.ols("speed_ok ~ 1", sub).fit()
            lr = max(0.0, 2 * (m1.llf - m0.llf))
            p = 0.5 * stats.chi2.sf(lr, 1)
        tot = v_city + v_video + v_res
        split = separable and base["n_multi_video_cities"] >= MIN_MULTI_VIDEO_CITIES
        rows.append({**base,
                     "model": "LMM (REML), city RE" + (" + video VC" if separable else ""),
                     "var_city": v_city if split else None, "var_video": v_video if split else None,
                     "var_city_or_video": v_city + v_video, "var_residual": v_res,
                     "icc_city": v_city / tot if split else None,
                     "icc_video": v_video / tot if split else None,
                     "icc_city_or_video": (v_city + v_video) / tot,
                     "city_component_p": p,
                     "note": "" if split else SPLIT_NOTE.format(k=base["n_multi_video_cities"],
                                                                 m=MIN_MULTI_VIDEO_CITIES)})
    except Exception as e:
        rows.append({**base, "model": "LMM", "note": f"failed: {e}"})

    # 0/1 outcomes: Bayesian logistic mixed model, latent-scale ICC. Fitted twice:
    # unadjusted, and adjusted for the design covariates. If between-city variance were
    # just detector recall (the 2026-09 risky-crossing finding), it would shrink when
    # traffic_light_prob enters as a fixed effect.
    for y in BINARY_OUTCOMES:
        sub0 = d.dropna(subset=[y])
        for adjusted in (False, True):
            nuis = _nuisance(sub0, y) if adjusted else []
            sub = sub0.dropna(subset=nuis)
            base = {"outcome": y, "adjusted_for": "+".join(nuis) or "none", "n_obs": len(sub),
                    "n_videos": sub["link"].nunique(), "n_cities": sub["cluster"].nunique(),
                    "n_multi_video_cities": _multi_video_cities(sub)}
            if sub[y].sum() < MIN_EVENTS:
                rows.append({**base, "model": "GLMM", "note": "too few events"})
                continue
            if adjusted:
                # standardise continuous covariates so the VB prior scale is sensible
                for c in nuis:
                    if c != "dense":
                        sub = sub.assign(**{c: (sub[c] - sub[c].mean()) / sub[c].std()})
            formula = f"{y} ~ 1" + "".join(f" + {c}" for c in nuis)
            latent = np.pi ** 2 / 3
            try:
                # ONE random intercept per video (between-video variance, which INCLUDES
                # between-city variance). A two-component city+video fit is not identified
                # when most cities have one video: the variational fit then returns the two
                # components in an arbitrary order from run to run (observed on identical
                # data: var_city 0.36/var_video 0.02 one night, 0.02/0.38 the next).
                v_cv, lo_cv, hi_cv = _vb_icc(formula, {"video": "0 + C(link)"}, sub, rng, latent)
                row = {**base, "model": "Bayesian logistic GLMM (VB), video RE (includes city)",
                       "var_city_or_video": v_cv["video"], "var_residual": latent,
                       "icc_city_or_video": v_cv["video"] / (v_cv["video"] + latent),
                       "icc_city_or_video_ci_low": lo_cv, "icc_city_or_video_ci_high": hi_cv,
                       "city_component_p": None}
                split = base["n_multi_video_cities"] >= MIN_MULTI_VIDEO_CITIES
                if split:
                    v2, _, _ = _vb_icc(formula, {"city": "0 + C(cluster)", "video": "0 + C(link)"},
                                           sub, rng, latent)
                    tot = v2["city"] + v2["video"] + latent
                    # guard against label switching: the two components must sum to the
                    # single-component total, otherwise the split is not reported
                    split = abs((v2["city"] + v2["video"]) - v_cv["video"]) <= 0.5 * v_cv["video"]
                if split:
                    row.update({"var_city": v2["city"], "var_video": v2["video"],
                                "icc_city": v2["city"] / tot, "icc_video": v2["video"] / tot})
                    row["note"] = "latent-scale ICC; variational Bayes understates uncertainty"
                else:
                    row["note"] = ("latent-scale ICC; variational Bayes understates uncertainty; "
                                   + SPLIT_NOTE.format(k=base["n_multi_video_cities"],
                                                       m=MIN_MULTI_VIDEO_CITIES))
                rows.append(row)
            except Exception as e:
                rows.append({**base, "model": "GLMM", "note": f"failed: {e}"})
    out = pd.DataFrame(rows)
    # Rounded so identical data gives byte-identical CSVs (the nightly job rewrites this tracked
    # file; unrounded variational estimates differ in the 6th decimal between runs).
    num = out.select_dtypes("number").columns
    out[num] = out[num].round(4)
    return out


def _multi_video_cities(sub):
    """Number of cities with at least two videos in `sub`."""
    return int((sub.groupby("cluster")["link"].nunique() >= 2).sum())


def _vb_icc(formula, vc, sub, rng, latent, n_draw=4000):
    """Variational-Bayes logistic mixed model; returns ({component: variance}, ci_low, ci_high)
    for the latent-scale ICC of the SUM of the components (the identified quantity)."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = BinomialBayesMixedGLM.from_formula(formula, vc, sub)
        fit = model.fit_vb()
    names = list(model.vcp_names)
    mean = dict(zip(names, fit.vcp_mean))
    sd = dict(zip(names, fit.vcp_sd))
    # vcp are log standard deviations; propagate their (approximate) posterior.
    draws = {k: np.exp(rng.normal(mean[k], sd[k], n_draw)) ** 2 for k in names}
    total = sum(draws.values())
    icc = total / (total + latent)
    var = {k: float(np.exp(mean[k]) ** 2) for k in names}
    return var, float(np.quantile(icc, 0.025)), float(np.quantile(icc, 0.975))


# --------------------------------------------------------------- adjusted associations

def _t_inference(b, se, n_clusters):
    dof = max(n_clusters - 1, 1)
    t = b / se
    p = 2 * stats.t.sf(abs(t), dof)
    half = stats.t.ppf(0.975, dof) * se
    return p, b - half, b + half


def associations(d):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    rows = []
    outcomes = BINARY_OUTCOMES + [SPEED_OUTCOME]
    for y in outcomes:
        ycol = "speed_ok" if y == SPEED_OUTCOME else y
        for fam, (terms, needed, caveat) in FAMILIES.items():
            sub = d.dropna(subset=[ycol] + needed).copy()
            dropped = []
            if terms.startswith("C("):
                # A level with a handful of rows gives a separated, uninformative estimate
                # (e.g. risky crossing at night: OR CI 0..inf). Drop such levels.
                col = needed[0]
                counts = sub[col].value_counts()
                rare = counts[counts < MIN_LEVEL_N].index.tolist()
                if rare:
                    dropped = [f"{lvl} (n={counts[lvl]})" for lvl in rare]
                    sub = sub[~sub[col].isin(rare)]
                ref = terms.split("Treatment('")[1].split("'")[0]
                if ref not in set(sub[col]) or sub[col].nunique() < 2:
                    rows.append({"outcome": y, "family": fam, "term": fam,
                                 "n_obs": len(sub),
                                 "note": "fewer than two levels with >= MIN_LEVEL_N rows"})
                    continue
            nuis = _nuisance(sub, y)
            sub = sub.dropna(subset=nuis)
            k = sub["cluster"].nunique()
            base = {"outcome": y, "family": fam, "n_obs": len(sub), "n_clusters": k,
                    "adjusted_for": "+".join(nuis),
                    "caveat": "; ".join(x for x in [caveat, ("dropped sparse levels: "
                                                             + ", ".join(dropped)) if dropped else ""] if x)}
            if len(sub) < MIN_OBS or k < MIN_CLUSTERS:
                rows.append({**base, "term": fam, "note": "insufficient data"})
                continue
            if y != SPEED_OUTCOME and sub[ycol].sum() < MIN_EVENTS:
                rows.append({**base, "term": fam, "note": "too few events"})
                continue
            formula = f"{ycol} ~ {terms}" + "".join(f" + {t}" for t in nuis)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if y == SPEED_OUTCOME:
                        res = smf.mixedlm(formula, sub, groups=sub["cluster"]).fit(reml=True)
                        method, scale = "LMM, random intercept per city", "difference_m_per_s"
                        params, ses = res.fe_params, res.bse_fe
                    else:
                        res = smf.gee(formula, groups="cluster", data=sub,
                                      family=sm.families.Binomial(),
                                      cov_struct=sm.cov_struct.Exchangeable()
                                      ).fit(cov_type="bias_reduced")
                        method = "GEE logistic, exchangeable/city, bias-reduced SE, t(k-1)"
                        scale = "odds_ratio"
                        params, ses = res.params, res.bse
            except Exception as e:
                rows.append({**base, "term": fam, "note": f"failed: {e}"})
                continue
            for term in params.index:
                if term == "Intercept" or term in nuis:
                    continue
                b, se = float(params[term]), float(ses[term])
                if not np.isfinite(se) or se == 0 or (scale == "odds_ratio" and se > MAX_LOGIT_SE):
                    rows.append({**base, "term": term, "note": "not estimable (separation?)"})
                    continue
                p, lo, hi = _t_inference(b, se, k)
                if scale == "odds_ratio":
                    est, lo, hi = np.exp(b), np.exp(lo), np.exp(hi)
                else:
                    est = b
                rows.append({**base, "term": term, "estimate": est, "ci_low": lo, "ci_high": hi,
                             "scale": scale, "p_value": p, "method": method, "note": ""})
    out = pd.DataFrame(rows)
    out["q_value_bh"] = np.nan
    for column in ("p_value", "estimate", "ci_low", "ci_high", "n_clusters"):
        if column not in out:
            out[column] = np.nan
    for y in out["outcome"].unique():
        m = out["outcome"] == y
        out.loc[m, "q_value_bh"] = bh_fdr(out.loc[m, "p_value"]).to_numpy()
    out["term_label"] = out["term"].map(pretty_term)
    cols = ["outcome", "family", "term", "term_label", "estimate", "ci_low", "ci_high", "scale", "p_value",
            "q_value_bh", "n_obs", "n_clusters", "adjusted_for", "method", "caveat", "note"]
    return out.reindex(columns=cols)


def pretty_term(term):
    """"C(age_band, Treatment('30-44'))[T.18-29]" -> "18-29 vs 30-44"."""
    m = re.match(r"C\((\w+), Treatment\('([^']+)'\)\)\[T\.(.+)\]", str(term))
    if m:
        return f"{m.group(3)} vs {m.group(2)}"
    return {"female": "female vs male", "phone": "phone use vs none",
            "nearby_start": "per extra pedestrian nearby",
            "traffic_z": "per SD of traffic volume"}.get(term, term)


def forest_plot(assoc, path):
    est = assoc.dropna(subset=["estimate"])
    if est.empty:
        print(f"PDF skipped (no estimates): {path}")
        return
    with PdfPages(path, metadata=PDF_METADATA) as pdf:
        for y, grp in est.groupby("outcome", sort=False):
            grp = grp.iloc[::-1]
            fig, ax = plt.subplots(figsize=(9, 0.42 * len(grp) + 1.8))
            ypos = np.arange(len(grp))
            ratio = grp["scale"].iloc[0] == "odds_ratio"
            ax.errorbar(grp["estimate"], ypos,
                        xerr=[grp["estimate"] - grp["ci_low"], grp["ci_high"] - grp["estimate"]],
                        fmt="o", capsize=3, color="#1f4e79")
            ax.axvline(1.0 if ratio else 0.0, color="grey", lw=1, ls="--")
            if ratio:
                ax.set_xscale("log")
                ticks = [t for t in (0.1, 0.25, 0.5, 1, 2, 4, 10)
                         if grp["ci_low"].min() / 1.5 <= t <= grp["ci_high"].max() * 1.5]
                ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
                ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            labels = [f"{r.family}: {pretty_term(r.term)}  (n={int(r.n_obs)}, k={int(r.n_clusters)})"
                      + ("  *" if pd.notna(r.q_value_bh) and r.q_value_bh < 0.05 else "")
                      for r in grp.itertuples()]
            ax.set_yticks(ypos)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel("odds ratio (log scale)" if ratio else "difference in m/s")
            ax.set_title(f"{y}: adjusted associations, 95% CI (t, k-1 df)\n"
                         "* = BH-FDR q < 0.05 within outcome; k = cities", fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"PDF saved: {path}")


def main():
    d = load()
    vp = variance_partition(d)
    vp.to_csv(f"{OUT}/inferential_variance_partition.csv", index=False, encoding="utf-8")
    print(vp.to_string(index=False))
    assoc = associations(d)
    assoc.to_csv(f"{OUT}/inferential_associations.csv", index=False, encoding="utf-8")
    shown = assoc.dropna(subset=["estimate"])[["outcome", "term", "estimate", "ci_low", "ci_high",
                                               "p_value", "q_value_bh", "n_obs", "n_clusters"]]
    print(shown.to_string(index=False))
    forest_plot(assoc, f"{OUT}/inferential_forest.pdf")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
