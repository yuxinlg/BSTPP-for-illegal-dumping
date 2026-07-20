"""SBC harness smoke test (R=2, short chains) so scripts/sbc.py cannot rot.

Per the runbook's mechanics section: the harness lives in the repo with a tiny
smoke test in the suite; real runs happen on the analysis machine. This test
asserts the MECHANICS -- prior draw -> simulate -> NUTS fit -> ranks ->
resumable storage -> report -- not calibration (R=2 says nothing about
uniformity, deliberately). Priors here are smoke-specific low-count overrides
for speed; they flow through the same matched-by-construction path as the
pre-registered stage-1 priors, so nothing config-shaped is stubbed out.

Marked slow: per-stage smokes with real NUTS fits (20 warmup / 44 raw draws
-> L=11) for stages 1-3, plus a separate adaptive-length smoke that forces
one ESS retry.
Deselect with -m "not slow".
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root -> import bstpp
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ["NUMPYRO_SPHINXBUILD"] = "1"   # suppress per-fit progress bars (fit path's own switch)
import warnings
warnings.filterwarnings("ignore")

import json

import numpy as np
import numpyro.distributions as dist
import pytest

import sbc


def smoke_priors():
    # Low-count variant of the stage-1 priors: exp(0)*50 ~ 50 background
    # events, keeping two NUTS fits fast. Same families, same code path.
    return dict(
        a_0=dist.Normal(0.0, 0.2),
        alpha=dist.Beta(2.0, 6.0),
        beta=dist.LogNormal(0.0, 0.5),
        sigmax_2=dist.LogNormal(float(np.log(0.005)), 0.5),
    )


@pytest.mark.slow
def test_sbc_harness_smoke(tmp_path, monkeypatch):
    out_dir = str(tmp_path / "sbc_smoke")
    cfg = sbc.SBCConfig(replicates=2, master_seed=0, num_warmup=20,
                        num_samples=44, rank_draws=11, min_ess_ratio=0.0,
                        out_dir=out_dir)
    priors = smoke_priors()

    # -- run: two replicates end-to-end -----------------------------------
    records, n_new = sbc.run_sbc(cfg, priors=priors)
    assert n_new == 2 and len(records) == 2
    for rec in records:
        assert rec["raw_draws"] == 44
        assert rec["n_fit_attempts"] == 1
        assert rec["L"] == 11 and rec["thin_stride"] == 4
        assert rec["n_events"] >= 2
        assert rec["diverging"] >= 0
        for name in sbc.PRIMARY + ["exc_share"]:
            assert 0 <= rec["ranks"][name] <= rec["L"], f"rank out of range for {name}"
            assert np.isfinite(rec["min_quantile_ess"][name])
        # stage-1 identity: constant background => log_background is a_0 plus
        # a constant, so their ranks MUST coincide (a_0 fully identified here)
        assert rec["ranks"]["log_background"] == rec["ranks"]["a_0"]
        assert np.isfinite(rec["truth_log_background"])
        assert 0.0 <= rec["truth_exc_share"] <= 1.0

    # storage: config + one JSON line per replicate
    cfg_path, jsonl_path = sbc._paths(out_dir)
    assert os.path.exists(cfg_path)
    with open(jsonl_path, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    assert len(lines) == 2
    stored = json.load(open(cfg_path, encoding="utf-8"))
    assert stored["config_hash"] == cfg.config_hash(priors)
    assert "unit_annotation" in stored and "sigmax_2" in stored["unit_annotation"]
    assert stored["identity"]["max_num_samples"] == cfg.max_num_samples

    # -- resume: identical config performs zero new fits -------------------
    records2, n_new2 = sbc.run_sbc(cfg, priors=priors)
    assert n_new2 == 0 and len(records2) == 2

    # -- config guard: a changed chain setting must refuse to pool ---------
    cfg_bad = sbc.SBCConfig(replicates=2, master_seed=0, num_warmup=21,
                            num_samples=44, rank_draws=11, min_ess_ratio=0.0,
                            out_dir=out_dir)
    with pytest.raises(RuntimeError, match="config mismatch"):
        sbc.run_sbc(cfg_bad, priors=priors)

    # -- implementation guard: code/provenance drift must also refuse pooling
    original_impl = sbc.implementation_identity()
    changed_impl = {**original_impl, "sbc_script_sha256": "0" * 64}
    monkeypatch.setattr(sbc, "implementation_identity", lambda: changed_impl)
    with pytest.raises(RuntimeError, match="config mismatch"):
        sbc.run_sbc(cfg, priors=priors)

    # -- report: mechanics only (R=2 says nothing about uniformity) --------
    out = sbc.report(out_dir, mc_draws=500)
    assert out["R"] == 2 and out["L"] == 11
    for name in sbc.PRIMARY:
        entry = out["functionals"][name]
        assert 0.0 <= entry["mc_p_value"] <= 1.0
        assert np.isfinite(entry["ecdf_sup_stat"])
        assert sum(entry["rank_hist_bins"]) == 2
    assert "exc_share" in out["supplementary"]
    assert "sampling_diagnostics" in out
    assert out["sampling_diagnostics"]["replicates_with_ess_retry"] == 0
    assert os.path.exists(os.path.join(out_dir, "report.json"))


@pytest.mark.slow
def test_sbc_adaptive_chain_retries_only_weak_replicate(tmp_path, monkeypatch):
    """Force the first ESS evaluation to fail; only that replicate is refit."""
    out_dir = str(tmp_path / "sbc_adaptive")
    # threshold = 0.95 * 11 = 10.45; fake E=6 => ceil(11/6)=2 => 44->88
    cfg = sbc.SBCConfig(replicates=2, master_seed=0, num_warmup=20,
                        num_samples=44, max_num_samples=176, rank_draws=11,
                        min_ess_ratio=0.95, out_dir=out_dir)
    priors = smoke_priors()

    real_ess = sbc.min_quantile_ess
    n_ess_calls = {"n": 0}

    def flaky_ess(draws, n_quantiles=19):
        # First attempt scores 5 diagnostic functionals; force all below threshold.
        n_ess_calls["n"] += 1
        if n_ess_calls["n"] <= 5:
            return 6.0
        return real_ess(draws, n_quantiles)

    fit_calls = {"n": 0}
    real_run_mcmc = sbc.Hawkes_Model.run_mcmc

    def counting_run_mcmc(self, *args, **kwargs):
        fit_calls["n"] += 1
        return real_run_mcmc(self, *args, **kwargs)

    monkeypatch.setattr(sbc, "min_quantile_ess", flaky_ess)
    monkeypatch.setattr(sbc.Hawkes_Model, "run_mcmc", counting_run_mcmc)

    records, n_new = sbc.run_sbc(cfg, priors=priors)
    assert n_new == 2 and len(records) == 2
    # Three fits total: r0 attempt0 (fail) + r0 attempt1 (pass) + r1 attempt0
    assert fit_calls["n"] == 3

    r0, r1 = records[0], records[1]
    assert r0["r"] == 0 and r1["r"] == 1
    assert r0["n_fit_attempts"] == 2
    assert r0["fit_attempts"][0]["num_samples"] == 44
    assert r0["fit_attempts"][1]["num_samples"] == 88
    assert r0["raw_draws"] == 88
    assert r0["fit_attempts"][0]["min_primary_ess"] == 6.0
    assert min(r0["min_quantile_ess"][name] for name in sbc.PRIMARY) >= 0.95 * 11

    assert r1["n_fit_attempts"] == 1
    assert r1["raw_draws"] == 44
    assert len(r1["fit_attempts"]) == 1


def test_replicate_mcmc_keys_are_distinct_and_reproducible():
    s0a = sbc.replicate_seeds(0, 0)
    s0b = sbc.replicate_seeds(0, 0)
    s1 = sbc.replicate_seeds(0, 1)
    assert np.array_equal(np.asarray(s0a["mcmc_key"]), np.asarray(s0b["mcmc_key"]))
    assert not np.array_equal(np.asarray(s0a["mcmc_key"]), np.asarray(s1["mcmc_key"]))
    k0 = sbc.attempt_mcmc_key(s0a["mcmc_key"], 0)
    k1 = sbc.attempt_mcmc_key(s0a["mcmc_key"], 1)
    assert not np.array_equal(np.asarray(k0), np.asarray(k1))
    assert np.array_equal(
        np.asarray(sbc.attempt_mcmc_key(s0a["mcmc_key"], 1)),
        np.asarray(sbc.attempt_mcmc_key(s0b["mcmc_key"], 1)))


def test_uniform_thinning_and_ess_helpers():
    thinned, indices, stride = sbc.uniformly_thin(np.arange(508), 127)
    assert stride == 4
    assert len(thinned) == 127
    assert np.array_equal(thinned, indices)
    rng = np.random.default_rng(0)
    assert sbc.min_quantile_ess(rng.normal(size=508)) > 50
    # Replicate-44 style scaling: 508 * ceil(127/91.7) = 1016
    assert sbc.next_num_samples(508, 91.7, 127, 4064) == 1016
    assert sbc.next_num_samples(508, 91.7, 127, 800) == 800


def test_stage2_config_and_truth_plumbing():
    """Fast stage-2 plumbing: matched truth-draw dimensions, pre-registered
    grid indices within range, identity fields, jsonable truth."""
    priors = sbc.stage2_priors()
    gen = sbc.build_model_stage2(sbc.make_placeholder(), priors)
    assert gen.args["sp_var_mu"] == sbc.UNIT_GAIN_SP_VAR_MU == 0.0
    seeds = sbc.replicate_seeds(0, 0)
    truth = sbc.draw_truth_stage2(seeds["prior_key"], priors, gen)
    assert truth["z_temporal"].shape == (gen.args["z_dim_temporal"],)
    assert truth["z_seasonal"].shape == (gen.args["z_dim_seasonal"],)
    assert truth["z_spatial"].shape == (gen.args["z_dim_spatial"],)
    js = sbc._jsonable_truth(truth)
    json.dumps(js)  # must serialize
    assert isinstance(js["a_0"], float) and len(js["z_spatial"]) == 20
    n_xy = int(gen.args["n_xy"])
    for t_i, ix, iy in sbc.STAGE2_GRID_POINTS:
        assert 0 <= t_i < gen.args["n_t"]
        assert 0 <= iy * n_xy + ix < n_xy ** 2
    assert sbc.SBCConfig(stage=1).out_dir == os.path.join("results", "sbc_stage1")
    cfg = sbc.SBCConfig(stage=2)
    assert cfg.out_dir == os.path.join("results", "sbc_stage2")
    ident = cfg.identity(priors)
    assert ident["stage"] == 2 and ident["model"] == "lgcp"
    assert ident["sp_var_mu"] == sbc.UNIT_GAIN_SP_VAR_MU == 0.0
    assert ident["p_threshold"] == 0.005
    assert ident["rank_targets"]["primaries"] == sbc.PRIMARY_STAGE2
    assert len(sbc.PRIMARY_STAGE2) == 9


def test_stage2p_config_domain_and_plumbing():
    """Fast stage-2p plumbing (pre-registered in docs/sbc_runbook.md 'Stage
    2p pre-registration' BEFORE this implementation): the polygon identity,
    the analytic area, probe-cell interiority, in-domain placeholder events,
    and stage-2-verbatim statistical constants."""
    from shapely.geometry import box, Point
    poly = sbc.stage2p_domain().geometry.iloc[0]
    # analytic |A| = 0.928950 (four corner-cut triangles off the unit square)
    assert poly.area == pytest.approx(sbc.STAGE2P_DOMAIN_AREA, abs=1e-12)
    # all five archived pointwise grid CELLS strictly interior (comparability
    # with the rectangular stages; registered rationale)
    for _, ix, iy in sbc.STAGE2_GRID_POINTS:
        cell = box(ix / 25, iy / 25, (ix + 1) / 25, (iy + 1) / 25)
        assert poly.contains(cell), f"probe cell ({ix},{iy}) not interior"
    # placeholder events lie inside the polygon (3a reject contracts)
    ph = sbc.make_placeholder_stage2p()
    assert len(ph) == 50
    assert all(poly.covers(Point(x, y)) for x, y in zip(ph["X"], ph["Y"]))
    assert sbc.stage_placeholder("2p") is sbc.make_placeholder_stage2p
    assert sbc.stage_placeholder(2) is sbc.make_placeholder
    # stage-2-verbatim design constants and identity
    cfg = sbc.SBCConfig(stage="2p")
    assert cfg.out_dir == os.path.join("results", "sbc_stage2p")
    priors = sbc.stage_priors("2p")
    assert sbc.prior_spec(priors) == sbc.prior_spec(sbc.stage2_priors())
    ident = cfg.identity(priors)
    assert ident["stage"] == "2p" and ident["model"] == "lgcp"
    assert ident["domain"] == [list(v) for v in sbc.STAGE2P_DOMAIN_VERTICES]
    assert ident["sp_var_mu"] == 0.0
    assert ident["p_threshold"] == 0.005
    assert ident["rank_targets"]["primaries"] == sbc.PRIMARY_STAGE2
    assert sbc.BUDGET_BAND["2p"] == sbc.BUDGET_BAND[2]
    # the generator model constructs on the polygon (clipped support live)
    gen = sbc.build_model_stage2p(ph, priors)
    assert len(np.asarray(gen.args["spatial_grid_cells"])) < 625
    areas = np.asarray(gen.args["integration_areas"], dtype=np.float64)
    assert areas.sum() == pytest.approx(sbc.STAGE2P_DOMAIN_AREA / 1.0, rel=2e-6)
    assert (areas > 0).all() and (areas < 1 / 625 * (1 + 1e-6)).any()


@pytest.mark.slow
def test_sbc_stage2p_harness_smoke(tmp_path):
    """Stage-2p mechanics end-to-end at the REAL stage-2 priors on the
    pre-registered polygon: two LGCP NUTS fits, ranks for all 9 primaries
    plus the a_0 supplementary, stage-aware storage, resume refusal on a
    stage flip. NOT calibration (R=2, deliberately)."""
    out_dir = str(tmp_path / "sbc_stage2p_smoke")
    cfg = sbc.SBCConfig(replicates=2, master_seed=0, num_warmup=20,
                        num_samples=44, rank_draws=11, min_ess_ratio=0.0,
                        stage="2p", out_dir=out_dir)
    records, n_new = sbc.run_sbc(cfg)
    assert n_new == 2 and len(records) == 2
    expected = set(sbc.PRIMARY_STAGE2) | {"a_0"}
    for rec in records:
        assert rec["stage"] == "2p" and rec["L"] == 11
        assert set(rec["ranks"]) == expected
        for name, rank in rec["ranks"].items():
            assert 0 <= rank <= rec["L"], f"rank out of range for {name}"
        vals = list(rec["truth_functionals"].values())
        assert np.all(np.isfinite(vals))
    stored = json.load(open(sbc._paths(out_dir)[0], encoding="utf-8"))
    assert stored["identity"]["stage"] == "2p"
    assert stored["identity"]["domain"] == [list(v) for v in
                                            sbc.STAGE2P_DOMAIN_VERTICES]
    _, n2 = sbc.run_sbc(cfg)   # resume: zero new fits
    assert n2 == 0
    cfg_flip = sbc.SBCConfig(replicates=2, master_seed=0, num_warmup=20,
                             num_samples=44, rank_draws=11, min_ess_ratio=0.0,
                             stage=2, out_dir=out_dir)
    with pytest.raises(RuntimeError, match="config mismatch"):
        sbc.run_sbc(cfg_flip)
    out = sbc.report(out_dir, mc_draws=300)
    assert set(out["functionals"]) == set(sbc.PRIMARY_STAGE2)
    assert "0.005" in out["decision"]["rule"]


@pytest.mark.slow
def test_sbc_stage2_harness_smoke(tmp_path):
    """Stage-2 mechanics end-to-end at the REAL stage-2 priors (already
    low-count by design): two LGCP NUTS fits, ranks for all 9 primaries plus
    the a_0 supplementary, stage-aware storage, resume, config-driven report.
    NOT calibration (R=2, deliberately)."""
    out_dir = str(tmp_path / "sbc_stage2_smoke")
    cfg = sbc.SBCConfig(replicates=2, master_seed=0, num_warmup=20,
                        num_samples=44, rank_draws=11, min_ess_ratio=0.0,
                        stage=2, out_dir=out_dir)
    records, n_new = sbc.run_sbc(cfg)
    assert n_new == 2 and len(records) == 2
    expected = set(sbc.PRIMARY_STAGE2) | {"a_0"}
    for rec in records:
        assert rec["stage"] == 2 and rec["L"] == 11
        assert set(rec["ranks"]) == expected
        assert "exc_share" not in rec["ranks"] and "alpha" not in rec["ranks"]
        for name, rank in rec["ranks"].items():
            assert 0 <= rank <= rec["L"], f"rank out of range for {name}"
        assert set(rec["min_quantile_ess"]) == expected
        assert len(rec["truth"]["z_spatial"]) == 20
        vals = list(rec["truth_functionals"].values())
        assert np.all(np.isfinite(vals))

    stored = json.load(open(sbc._paths(out_dir)[0], encoding="utf-8"))
    assert stored["identity"]["stage"] == 2
    assert stored["identity"]["sp_var_mu"] == 0.0

    # resume performs zero new fits; a stage flip must refuse to pool
    _, n2 = sbc.run_sbc(cfg)
    assert n2 == 0
    cfg_flip = sbc.SBCConfig(replicates=2, master_seed=0, num_warmup=20,
                             num_samples=44, rank_draws=11, min_ess_ratio=0.0,
                             stage=1, out_dir=out_dir)
    with pytest.raises(RuntimeError, match="config mismatch"):
        sbc.run_sbc(cfg_flip, priors=smoke_priors())

    out = sbc.report(out_dir, mc_draws=300)
    assert set(out["functionals"]) == set(sbc.PRIMARY_STAGE2)
    assert set(out["supplementary"]) == {"a_0"}
    assert "0.005" in out["decision"]["rule"]
    for entry in out["functionals"].values():
        assert 0.0 <= entry["mc_p_value"] <= 1.0
        assert sum(entry["rank_hist_bins"]) == 2


def test_stage3_config_and_truth_plumbing():
    """Fast stage-3 plumbing: the prior composition (stage 2's background
    prior times stage 1's trigger priors), gain threading, per-component
    truth subkeys (reproducible, matched dims), the traced composition
    identity Itot_txy - Itot_excite == Itot_txy_back at one-ulp float32
    tolerance, grid indices, identity fields, and the stage-aware out_dir."""
    p1, p2, p3 = sbc.stage1_priors(), sbc.stage2_priors(), sbc.stage3_priors()
    assert (p3["a_0"].loc, p3["a_0"].scale) == (p2["a_0"].loc, p2["a_0"].scale)
    for name in ("alpha", "beta", "sigmax_2"):
        assert type(p3[name]) is type(p1[name])
        for arg in p3[name].arg_constraints:
            assert float(np.asarray(getattr(p3[name], arg))) == \
                float(np.asarray(getattr(p1[name], arg)))
    assert (p3["a_0"].loc, p3["a_0"].scale) != (p1["a_0"].loc, p1["a_0"].scale)

    gen = sbc.build_model_stage3(sbc.make_placeholder(), p3)
    assert gen.args["model"] == "cox_hawkes"
    assert gen.args["sp_var_mu"] == sbc.UNIT_GAIN_SP_VAR_MU == 0.0

    seeds = sbc.replicate_seeds(0, 0)
    truth = sbc.draw_truth_stage3(seeds["prior_key"], p3, gen)
    truth_again = sbc.draw_truth_stage3(seeds["prior_key"], p3, gen)
    for name in sbc.LATENT:
        assert isinstance(truth[name], float)
        assert truth[name] == truth_again[name]
    for vec, dim_key in [("z_temporal", "z_dim_temporal"),
                         ("z_seasonal", "z_dim_seasonal"),
                         ("z_spatial", "z_dim_spatial")]:
        assert truth[vec].shape == (gen.args[dim_key],)
        assert np.array_equal(truth[vec], truth_again[vec])
    json.dumps(sbc._jsonable_truth(truth))  # must serialize

    # Composition identity on the traced model: Itot_txy is stored as the
    # ROUNDED float32 sum Itot_excite + Itot_txy_back, so the subtraction is
    # asserted to one ulp (rtol 1e-6), not bitwise -- a float fact, an order
    # of magnitude below any structural composition defect.
    from numpyro import handlers
    import jax
    for r in range(3):
        tr_truth = sbc.draw_truth_stage3(
            sbc.replicate_seeds(0, r)["prior_key"], p3, gen)
        fixed = {k: tr_truth[k] for k in sbc.LATENT_STAGE3}
        tr = handlers.trace(handlers.substitute(
            handlers.seed(gen.model, jax.random.PRNGKey(0)),
            fixed)).get_trace(gen.args)
        txy = float(np.asarray(tr["Itot_txy"]["value"]))
        exc = float(np.asarray(tr["Itot_excite"]["value"]))
        back = float(np.asarray(tr["Itot_txy_back"]["value"]))
        assert back > 0 and exc >= 0
        np.testing.assert_allclose(txy - exc, back, rtol=1e-6, atol=0.0)

    n_xy = int(gen.args["n_xy"])
    for t_i, ix, iy in sbc.STAGE2_GRID_POINTS:
        assert 0 <= t_i < gen.args["n_t"]
        assert 0 <= iy * n_xy + ix < n_xy ** 2

    cfg = sbc.SBCConfig(stage=3)
    assert cfg.out_dir == os.path.join("results", "sbc_stage3")
    ident = cfg.identity(p3)
    assert ident["stage"] == 3 and ident["model"] == "cox_hawkes"
    assert ident["sp_var_mu"] == sbc.UNIT_GAIN_SP_VAR_MU == 0.0
    assert ident["p_threshold"] == 0.004
    assert ident["rank_targets"]["primaries"] == sbc.PRIMARY_STAGE3
    assert ident["rank_targets"]["supplementary"] == ["a_0", "exc_share"]
    assert len(sbc.PRIMARY_STAGE3) == 12
    assert ident["budget_band"] == [20, 2000, 0.05]


@pytest.mark.slow
def test_sbc_stage3_harness_smoke(tmp_path):
    """Stage-3 mechanics end-to-end at the REAL stage-3 priors (the union
    prior is low-count enough by design): two cox-Hawkes NUTS fits, ranks for
    all 12 primaries plus the a_0 and exc_share supplementaries, stage-aware
    storage, resume, stage-flip pooling refusal, config-driven report.
    NOT calibration (R=2, deliberately)."""
    out_dir = str(tmp_path / "sbc_stage3_smoke")
    cfg = sbc.SBCConfig(replicates=2, master_seed=0, num_warmup=20,
                        num_samples=44, rank_draws=11, min_ess_ratio=0.0,
                        stage=3, out_dir=out_dir)
    records, n_new = sbc.run_sbc(cfg)
    assert n_new == 2 and len(records) == 2
    expected = set(sbc.PRIMARY_STAGE3) | {"a_0", "exc_share"}
    for rec in records:
        assert rec["stage"] == 3 and rec["L"] == 11
        assert set(rec["ranks"]) == expected
        for name, rank in rec["ranks"].items():
            assert 0 <= rank <= rec["L"], f"rank out of range for {name}"
        assert set(rec["min_quantile_ess"]) == expected
        assert set(sbc.LATENT).issubset(rec["truth"])
        assert len(rec["truth"]["z_spatial"]) == 20
        vals = list(rec["truth_functionals"].values())
        assert np.all(np.isfinite(vals))
        assert 0.0 <= rec["truth_functionals"]["exc_share"] <= 1.0

    stored = json.load(open(sbc._paths(out_dir)[0], encoding="utf-8"))
    assert stored["identity"]["stage"] == 3
    assert stored["identity"]["model"] == "cox_hawkes"
    assert stored["identity"]["sp_var_mu"] == 0.0

    # resume performs zero new fits; a stage flip must refuse to pool
    _, n2 = sbc.run_sbc(cfg)
    assert n2 == 0
    cfg_flip = sbc.SBCConfig(replicates=2, master_seed=0, num_warmup=20,
                             num_samples=44, rank_draws=11, min_ess_ratio=0.0,
                             stage=2, out_dir=out_dir)
    with pytest.raises(RuntimeError, match="config mismatch"):
        sbc.run_sbc(cfg_flip)

    out = sbc.report(out_dir, mc_draws=300)
    assert set(out["functionals"]) == set(sbc.PRIMARY_STAGE3)
    assert set(out["supplementary"]) == {"a_0", "exc_share"}
    assert "0.004" in out["decision"]["rule"]
    for entry in out["functionals"].values():
        assert 0.0 <= entry["mc_p_value"] <= 1.0
        assert sum(entry["rank_hist_bins"]) == 2
