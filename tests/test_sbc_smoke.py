"""SBC harness smoke test (R=2, short chains) so scripts/sbc.py cannot rot.

Per the runbook's mechanics section: the harness lives in the repo with a tiny
smoke test in the suite; real runs happen on the analysis machine. This test
asserts the MECHANICS -- prior draw -> simulate -> NUTS fit -> ranks ->
resumable storage -> report -- not calibration (R=2 says nothing about
uniformity, deliberately). Priors here are smoke-specific low-count overrides
for speed; they flow through the same matched-by-construction path as the
pre-registered stage-1 priors, so nothing config-shaped is stubbed out.

Marked slow: two real NUTS fits (20 warmup / 44 raw draws -> L=11).
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
    assert os.path.exists(os.path.join(out_dir, "report.json"))


def test_replicate_mcmc_keys_are_distinct_and_reproducible():
    s0a = sbc.replicate_seeds(0, 0)
    s0b = sbc.replicate_seeds(0, 0)
    s1 = sbc.replicate_seeds(0, 1)
    assert np.array_equal(np.asarray(s0a["mcmc_key"]), np.asarray(s0b["mcmc_key"]))
    assert not np.array_equal(np.asarray(s0a["mcmc_key"]), np.asarray(s1["mcmc_key"]))


def test_uniform_thinning_and_ess_helpers():
    thinned, indices, stride = sbc.uniformly_thin(np.arange(508), 127)
    assert stride == 4
    assert len(thinned) == 127
    assert np.array_equal(thinned, indices)
    rng = np.random.default_rng(0)
    assert sbc.min_quantile_ess(rng.normal(size=508)) > 50
