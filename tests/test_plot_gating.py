"""Regression: run_svi(plot_loss=False) must perform NO pyplot calls.

Before the fix, run_svi gated nothing: plt.plot/xlabel/ylabel/show ran
unconditionally, so plot_loss=False still created a figure and called
plt.show(). On machines whose default Tk backend is broken (missing
tcl/tk support files), any script or test calling
run_svi(..., plot_loss=False) without MPLBACKEND=Agg failed
intermittently with _tkinter.TclError depending on which backend earlier
imports had selected.

RED verification: with the unconditional plt.plot restored, the test
fails (a new figure appears in plt.get_fignums()).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ["MPLBACKEND"] = "Agg"  # before pyplot import via bstpp.main

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpyro.distributions as dist

from bstpp.main import Hawkes_Model

A_RECT = np.array([[10.0, 30.0], [5.0, 15.0]])
T_DAYS = 200.0
PRIORS = dict(a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
              beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))


def _interior_data(n=30, seed=7):
    r = np.random.RandomState(seed)
    return pd.DataFrame({
        "X": r.uniform(10.5, 29.5, n),
        "Y": r.uniform(5.5, 14.5, n),
        "T": np.sort(r.uniform(1.0, T_DAYS - 1.0, n)),
    })


def test_run_svi_plot_loss_false_creates_no_figures():
    m = Hawkes_Model(_interior_data(), A_RECT, T_DAYS, cox_background=False,
                     **PRIORS)
    plt.close("all")
    before = plt.get_fignums()
    m.run_svi(10, 0.01, plot_loss=False)
    assert plt.get_fignums() == before
