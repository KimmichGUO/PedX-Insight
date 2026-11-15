"""Regression tests for city identity, identifiable models, and seeded inference."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from modules.stats_utils import SEED, city_cluster_id
import statistics_inferential as si


def test_stable_filming_city_ids_define_statistical_clusters():
    frame = pd.DataFrame({
        "city": ["Springfield", "Springfield", "Springfield, Illinois", "Search alias"],
        "country": ["United States"] * 4,
        "city_id": ["geonames:4250542", "geonames:4951788", "geonames:4250542", "searched:wrong"],
        "verified_city_id": [None, None, None, "geonames:4250542"],
    })
    clusters = city_cluster_id(frame)["cluster"]
    assert clusters.iloc[0] == clusters.iloc[2] == clusters.iloc[3]
    assert clusters.iloc[0] != clusters.iloc[1]


def test_legacy_cluster_fallback_disambiguates_states_and_keeps_missing_country():
    frame = pd.DataFrame({"city": ["Springfield", "Springfield", "Ulm"],
                          "country": ["USA", "USA", None], "admin1_code": ["IL", "MA", None]})
    clusters = city_cluster_id(frame)["cluster"]
    assert clusters.nunique() == 3
    assert clusters.iloc[2] == "Ulm|"


def _clustered_frame():
    n = 80
    return pd.DataFrame({"cluster": np.repeat([f"city{i}" for i in range(8)], 10),
                         "link": np.repeat([f"video{i}" for i in range(8)], 10),
                         "speed_ok": np.linspace(0.8, 1.8, n), "run_red_light": [0, 1] * (n // 2),
                         "risky_crossing": [0, 1] * (n // 2), "dense": 1,
                         "traffic_light_prob": 0.5, "risky_crossing_coverage": 0.7,
                         "phone": [0, 1] * (n // 2)})


def test_binary_models_require_events_and_non_events(monkeypatch):
    frame = _clustered_frame()
    frame["speed_ok"] = np.nan
    frame["run_red_light"] = 1
    frame["risky_crossing"] = [0] * 9 + [1] * 71
    monkeypatch.setattr(si, "FAMILIES", {"phone": ("phone", ["phone"], "")})
    # No model fitting is needed to discover these outcomes are too one-sided.
    monkeypatch.setattr(si, "_vb_icc", lambda *a, **kw: pytest.fail("unsupported binary fit"))
    vp = si.variance_partition(frame)
    assert vp.loc[vp["outcome"].isin(si.BINARY_OUTCOMES), "note"].eq("too few events or non-events").all()
    assoc = si.associations(frame)
    assert assoc.loc[assoc["outcome"].isin(si.BINARY_OUTCOMES), "note"].eq("too few events or non-events").all()
    assert assoc["estimate"].isna().all()


def test_variance_models_require_the_same_observation_and_city_support(monkeypatch):
    frame = _clustered_frame().iloc[:40]
    monkeypatch.setattr(si, "_vb_icc", lambda *a, **kw: pytest.fail("unsupported variance fit"))
    monkeypatch.setattr(si, "_speed_variance", lambda *a, **kw: pytest.fail("unsupported variance fit"))
    result = si.variance_partition(frame)
    assert result["note"].str.contains("insufficient data").all()
    assert not any(column.startswith("icc_") for column in result)


def test_city_likelihood_test_retains_video_variance_in_null_and_alternative(monkeypatch):
    import statsmodels.formula.api as smf
    calls = []

    def mixedlm(formula, data, **kwargs):
        call = {"formula": formula, **kwargs, "fit_reml": []}
        calls.append(call)

        def fit(*, reml):
            call["fit_reml"].append(reml)
            return SimpleNamespace(converged=True, cov_re=pd.DataFrame([[0.2]]),
                                   vcomp=np.array([0.3]), scale=0.1,
                                   llf=-10.0 if kwargs["re_formula"] == "1" else -12.0)
        return SimpleNamespace(fit=fit)

    monkeypatch.setattr(smf, "mixedlm", mixedlm)
    city, video, residual, p = si._speed_variance(_clustered_frame(), split=True)
    assert (city, video, residual) == (0.2, 0.3, 0.1)
    assert len(calls) == 2
    full, null = calls
    assert full["vc_formula"] == null["vc_formula"] == {"video": "0 + C(link)"}
    assert full["re_formula"] == "1" and null["re_formula"] == "0"
    assert full["fit_reml"] == [True, False] and null["fit_reml"] == [False]
    assert full["groups"].nunique() == 8 and len(np.unique(null["groups"])) == 1
    assert p == pytest.approx(0.5 * si.stats.chi2.sf(4.0, 1))


def test_unidentified_city_component_uses_video_groups_and_withholds_city_test(monkeypatch):
    import statsmodels.formula.api as smf
    frame = _clustered_frame()

    def mixedlm(formula, data, **kwargs):
        pd.testing.assert_series_equal(kwargs["groups"], frame["link"])
        return SimpleNamespace(fit=lambda **kw: SimpleNamespace(
            converged=True, cov_re=pd.DataFrame([[0.4]]), scale=0.1))

    monkeypatch.setattr(smf, "mixedlm", mixedlm)
    assert si._speed_variance(frame, split=False) == (0.0, 0.4, 0.1, None)


def test_nested_speed_likelihood_test_runs_with_real_statsmodels():
    rng = np.random.default_rng(3)
    rows = []
    for city in range(8):
        for video in range(2):
            for _ in range(15):
                rows.append({"cluster": f"c{city}", "link": f"v{city}_{video}",
                             "speed_ok": 1.4 + (city - 3.5) * 0.13 + (video - 0.5) * 0.2 + rng.normal(0, 0.1)})
    city, video, residual, p = si._speed_variance(pd.DataFrame(rows), split=True)
    assert np.isfinite([city, video, residual, p]).all()
    assert city > 0 and video > 0 and residual > 0
    assert 0 <= p < 0.05  # Deliberately strong, known synthetic city intercepts.


def test_nonconverged_models_do_not_publish_inference():
    with pytest.raises(RuntimeError, match="did not converge"):
        si._require_converged(SimpleNamespace(converged=False))


@pytest.mark.parametrize("success", [True, False])
def test_variational_fit_uses_fixed_seed_and_rejects_failed_optimizer(monkeypatch, success):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    seen = []

    def fit_vb(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(optim_retvals={"success": success},
                               vcp_mean=np.array([-1.0]), vcp_sd=np.array([0.1]))

    monkeypatch.setattr(BinomialBayesMixedGLM, "from_formula", lambda *a, **kw: SimpleNamespace(
        fit_vb=fit_vb, vcp_names=["video"]))
    if success:
        a = si._vb_icc("run_red_light ~ 1", {}, _clustered_frame(), np.random.default_rng(SEED), np.pi**2 / 3)
        b = si._vb_icc("run_red_light ~ 1", {}, _clustered_frame(), np.random.default_rng(SEED), np.pi**2 / 3)
        assert a == b
    else:
        with pytest.raises(RuntimeError, match="did not converge"):
            si._vb_icc("run_red_light ~ 1", {}, _clustered_frame(), np.random.default_rng(SEED), np.pi**2 / 3)
    assert all(kwargs == {"rng": SEED} for kwargs in seen)
