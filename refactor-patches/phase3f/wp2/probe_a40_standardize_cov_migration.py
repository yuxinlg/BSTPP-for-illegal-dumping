"""A-40: how far apart the two ``standardize_cov`` spellings are, in numbers.

WHY THIS EXISTS. Backward compatibility is explicitly not a goal here, so the
API classification stands and no shim is wanted. What is wanted is ONE
DECLARATION at the migration boundary: ``standardize_cov=True`` on ``main``
standardizes each covariate column by its UNWEIGHTED row mean and variance;
``standardize_cov='domain_area'`` on ``refactor`` weights every row by the
exact clipped area ``|C_c intersect A|``. Both are accepted spellings of "yes,
standardize"; they compute DIFFERENT design matrices and NEITHER RAISES. This
is D-36's first clause arriving at the migration boundary: an old figure and a
new figure compared without anyone registering they are not commensurable.

THE MEASUREMENT IS ON THE REAL LAYER, not a fixture. The covariate frame is
the 1338-row Philadelphia block-group layer the downstream analysis actually
passes (``output/cov_cbg.geojson``), the domain is one real park box from
``output/all_boxes_gdf.geojson``, and the events are the real 311 dumping
reports falling inside it. That combination is what makes the gap large: the
unweighted mean/scale are taken over ALL of Philadelphia's block groups,
while the area-weighted ones are taken over only the handful with mass inside
the park -- so the two standardizations do not merely rescale, they centre on
different populations.

TWO NUMBERS ARE REPORTED.
  (1) The design-matrix gap: per-column centre and scale under each rule, and
      the resulting spread of the standardized columns.
  (2) The consequence for a computed value: ``loglik`` traced at IDENTICAL
      substituted latents, differing only in which ``args['spatial_cov']``
      the model carries. Swapping that one array is exactly the change --
      every other prepared object (clipped areas, ``cov_ind``, the
      computational grid) is geometry-side and standardization-independent.

Usage:  python refactor-patches/phase3f/wp2/probe_a40_standardize_cov_migration.py
"""
import os
import sys
import warnings

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

import geopandas as gpd                                          # noqa: E402
import jax                                                       # noqa: E402
import numpy as np                                               # noqa: E402
import numpyro.distributions as dist                             # noqa: E402
import pandas as pd                                              # noqa: E402
from numpyro import handlers                                     # noqa: E402

from bstpp.main import Hawkes_Model                              # noqa: E402
import bstpp                                                     # noqa: E402

print("PROBE_PROVENANCE")
print(f"  repo           : {REPO}")
print(f"  bstpp.__file__ : {bstpp.__file__}")
print()

# The downstream analysis' own column list (batch_park_fits.COLUMN_NAMES),
# restricted to columns present in the committed layer.
WANTED = ["pop_density", "med_inc_avg", "lep_density", "edu_hs_avg",
          "ndvi_mean_4yr", "RLD", "RMD", "RHD", "I", "T", "GW", "vac_area"]
PARK = "tacony_box"

boxes = gpd.read_file(os.path.join(REPO, "output", "all_boxes_gdf.geojson"))
cov = gpd.read_file(os.path.join(REPO, "output", "cov_cbg.geojson"))
box = boxes[boxes["PARKNAME"] == PARK].to_crs(cov.crs)
cov_names = [c for c in WANTED if c in cov.columns]

events = gpd.read_file(os.path.join(REPO, "output",
                                    "illegal_dumping_full.geojson"))
events = events[events.geometry.within(box.union_all())].copy()
# The real block-group layer leaves 3.44% of this park box uncovered, and a
# handful of reports fall in the gap. Those are dropped, and the number
# dropped is printed: the legacy path raises on them ("Spatial covariates are
# not defined for all data points!"), so they are outside BOTH spellings and
# cannot contribute to a difference between them.
n_all = len(events)
events = events[events.geometry.within(cov.union_all())].copy()
n_dropped = n_all - len(events)
events["start_time"] = pd.to_datetime(events["start_time"])
events = events.sort_values("start_time")
t0 = pd.Timestamp(f"{events['start_time'].min().year}-01-01")
tdays = (events["start_time"] - t0).dt.total_seconds() / 86400.0
locs = pd.DataFrame({"X": events.geometry.x.astype(float).values,
                     "Y": events.geometry.y.astype(float).values,
                     "T": tdays.astype(float).values})
T_DAYS = float(np.ceil(locs["T"].max())) + 1.0

print("REPRESENTATIVE FIT")
print(f"  park domain          : {PARK} ({len(box)} row)")
print(f"  covariate layer      : cov_cbg.geojson, {len(cov)} block groups")
print(f"  covariate columns    : {len(cov_names)}")
print(f"  events in domain     : {len(locs)} "
      f"({n_dropped} dropped: in the covariate gap, legacy raises on them)")
print(f"  horizon (days)       : {T_DAYS}")
print()

PRIORS = dict(a_0=dist.Normal(0, 2), alpha=dist.Beta(2, 2),
              beta=dist.Exponential(1.0), sigmax_2=dist.HalfNormal(0.25))


def build(std):
    # data_contracts='report', deliberately. The real layer leaves 3.44% of
    # this park box uncovered and five events fall in the gap, so 'reject'
    # refuses the fit. That is the contract working; it is not what is being
    # measured. 'report' is the declared dry-run instrument and leaves legacy
    # behaviour bit-unchanged, which is the state the migrating figures were
    # produced in. The two violations are printed above for the record.
    #
    # excitation_support='rectangle', deliberately. The park box is a polygon,
    # so refactor requires the mode to be declared (OP-2). `main` had no
    # polygon mode at all, so 'rectangle' is the commensurable choice: it is
    # the only one under which the two spellings differ ONLY in
    # standardization.
    return Hawkes_Model(locs.copy(), box, T_DAYS, cox_background=False,
                        spatial_cov=cov.copy(), cov_names=cov_names,
                        standardize_cov=std, data_contracts="report",
                        excitation_support="rectangle", **PRIORS)


m_dw = build("domain_area")          # refactor's spelling
m_raw = build(None)                  # off, to recover the untouched X_s

X_raw = np.asarray(m_raw.args["spatial_cov"], dtype=np.float64)
X_dw = np.asarray(m_dw.args["spatial_cov"], dtype=np.float64)

# main's `standardize_cov=True`, verbatim:
#     (X_s - X_s.mean(axis=0)) / (X_s.var(axis=0) ** 0.5)
uw_mean = X_raw.mean(axis=0)
uw_scale = X_raw.var(axis=0) ** 0.5
X_uw = (X_raw - uw_mean) / uw_scale

dw_mean = np.asarray(m_dw.standardization["mean"], dtype=np.float64)
dw_scale = np.asarray(m_dw.standardization["scale"], dtype=np.float64)

print("(1) DESIGN-MATRIX GAP -- neither spelling raises")
print(f"  {'column':<16} {'uw_mean':>12} {'dw_mean':>12} "
      f"{'uw_scale':>12} {'dw_scale':>12} {'centre_shift_sd':>16}")
for j, name in enumerate(cov_names):
    # How far apart the two centres are, expressed in unweighted SDs -- the
    # unit an old figure was implicitly read in.
    shift = (dw_mean[j] - uw_mean[j]) / uw_scale[j]
    print(f"  {name:<16} {uw_mean[j]:>12.4g} {dw_mean[j]:>12.4g} "
          f"{uw_scale[j]:>12.4g} {dw_scale[j]:>12.4g} {shift:>16.4f}")
print()
d = X_dw - X_uw
print(f"  max |X_domain_area - X_unweighted|          : {np.abs(d).max():.6g}")
print(f"  mean |X_domain_area - X_unweighted|         : {np.abs(d).mean():.6g}")
print(f"  max |centre shift| in unweighted SDs        : "
      f"{np.abs((dw_mean - uw_mean) / uw_scale).max():.6g}")
print(f"  scale ratio dw/uw, min .. max               : "
      f"{(dw_scale / uw_scale).min():.6g} .. {(dw_scale / uw_scale).max():.6g}")
print(f"  ALLCLOSE(rtol=1e-3)                         : "
      f"{bool(np.allclose(X_dw, X_uw, rtol=1e-3, atol=1e-3))}")
print()

# (2) The consequence for a computed value. Same model object, same latents;
# the ONLY difference is which standardized design matrix args carries.
print("(2) CONSEQUENCE FOR A COMPUTED VALUE -- loglik at identical latents")
seeded = handlers.seed(m_dw.model, jax.random.PRNGKey(0))
tr = handlers.trace(seeded).get_trace(m_dw.args)
latents = {k: v["value"] for k, v in tr.items()
           if v["type"] == "sample" and not v.get("is_observed", False)}
print(f"  substituted sites : {sorted(latents)}")


def loglik_with(matrix):
    saved = m_dw.args["spatial_cov"]
    m_dw.args["spatial_cov"] = np.asarray(matrix, dtype=np.float32)
    try:
        sub = handlers.substitute(
            handlers.seed(m_dw.model, jax.random.PRNGKey(0)), latents)
        return float(handlers.trace(sub).get_trace(m_dw.args)["loglik"]["value"])
    finally:
        m_dw.args["spatial_cov"] = saved


ll_dw = loglik_with(X_dw)
ll_uw = loglik_with(X_uw)
print(f"  loglik | standardize_cov='domain_area' (refactor) : {ll_dw!r}")
print(f"  loglik | standardize_cov=True         (main)      : {ll_uw!r}")
print(f"  absolute difference                               : "
      f"{abs(ll_dw - ll_uw):.6g}")
print(f"  relative difference                               : "
      f"{abs(ll_dw - ll_uw) / abs(ll_dw):.6g}")
print()
print("MIGRATION NOTE. Both spellings mean 'standardize'. On this fit they")
print("differ by the figures above and NEITHER RAISES. A number produced")
print("under one spelling is not comparable with a number produced under the")
print("other; re-derive, do not re-label.")
print("EXIT:0")
