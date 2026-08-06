"""A-41: what the vendored downstream copy actually depends on, and what that

leaves unmeasurable about WP2.

THE QUESTION. The WP2 opening-conditions proposal listed "the
``cox_hawkes_shared`` modules being reviewed or integrated" as NOT REQUIRED.
That entry is right about review and integration and WRONG as a blanket: WP2
lands ``ModelConfig`` and ``PriorConfig``, and
``Illegal-Dumping/replication/cox_hawkes_offset.py`` is a copy of a model
function that reads config-adjacent state out of ``args`` directly --
including ``args['priors']``, which is precisely ``PriorConfig``'s quantity.
"Not required" and "unmeasurable" are different entries and this probe is what
tells them apart.

WHAT IS MEASURED HERE. The dependency surface of the copy: every ``args[...]``
key it reads, every ``args['priors'][...]`` subkey, every ``self.`` attribute,
and its constructor's keyword surface. Each is then classified against THIS
fork's live ``args``:

  OURS     the key exists in this fork's ``args`` -- a surface we own, and
           therefore one WP2 can move out from under the copy.
  FOREIGN  the key does not exist here; it is supplied by the untracked
           ``bstpp.cox_hawkes_shared`` or by the subclass itself. Outside
           every claim this repository makes.

WHAT IS NOT MEASURED, AND WHY IT CANNOT BE. Two independent reasons, both
recorded rather than worked around:

  (1) ``bstpp/cox_hawkes_shared.py`` -- the base class, ``hawkes_intensity_sum``,
      and the function this file copies -- is in NO reachable tree. It is
      untracked in a downstream working checkout on another machine.
  (2) WP2's change set does not exist yet. Confirming "no WP2-scope change
      alters a signature or default this copy depends on" is a claim about an
      unenumerated set of changes; C4 is what enumerates it. Until then the
      confirmation is not false, it is NOT YET A STATEMENT.

So the answer to "confirm that no WP2-scope change breaks the vendored copy"
is: **not confirmable at round six**, and the honest register entry is
UNMEASURABLE, not NOT-REQUIRED.

PROVENANCE. The source is another repository. With ``--source PATH`` this
probe extracts the surface and rewrites the committed extract beside it; with
no argument it reads that extract, so the comparison is re-runnable in-tree
without the external checkout. The extract records the upstream URL and HEAD.

Usage:
    python refactor-patches/phase3f/wp2/probe_a41_vendored_dependency_surface.py
    python ... --source <path-to>/replication/cox_hawkes_offset.py [--head SHA]
"""
import argparse
import ast
import json
import os
import re
import sys
import warnings

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
EXTRACT = os.path.join(HERE, "a41_vendored_dependency_extract.json")

UPSTREAM = "https://github.com/yuxinlg/Illegal-Dumping.git"

# Quantities WP2 is named for. NOT a scope declaration -- WP2's scope is C4's
# deliverable and does not exist yet. This is "which of the copy's dependencies
# are of the KIND WP2 is about", which is a weaker and checkable statement.
PRIOR_KIND = {"priors"}
MODEL_KIND = {
    "T", "S", "offset_seasonal", "sp_var_mu", "model",
    "z_dim_temporal", "z_dim_seasonal", "z_dim_spatial",
    "hidden_dim_temporal", "hidden_dim1_seasonal", "hidden_dim2_seasonal",
    "hidden_dim1_spatial", "hidden_dim2_spatial",
    "n_t", "n_s", "n_xy", "t_trig", "sp_trig",
}


def extract(path: str, head: str) -> dict:
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src, filename=os.path.basename(path))
    args_keys, prior_keys, attrs, ctor_kwargs = set(), set(), set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(
                node.slice, ast.Constant) and isinstance(node.slice.value, str):
            base = node.value
            if isinstance(base, ast.Name) and base.id == "args":
                args_keys.add(node.slice.value)
            elif (isinstance(base, ast.Attribute) and base.attr == "args"):
                args_keys.add(node.slice.value)
            elif (isinstance(base, ast.Subscript)
                  and isinstance(base.slice, ast.Constant)
                  and base.slice.value == "priors"):
                prior_keys.add(node.slice.value)
        if isinstance(node, ast.Attribute) and isinstance(
                node.value, ast.Name) and node.value.id == "self":
            attrs.add(node.attr)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            a = node.args
            ctor_kwargs = [k.arg for k in a.kwonlyargs] + \
                          [x.arg for x in a.args if x.arg != "self"]
            if a.vararg:
                ctor_kwargs.append("*" + a.vararg.arg)
            if a.kwarg:
                ctor_kwargs.append("**" + a.kwarg.arg)
    # The copy's own provenance line, quoted rather than paraphrased.
    m = re.search(r"\(~/BSTPP @ commit ([0-9a-f]+),\s*([0-9-]+)\)", src)
    return {
        "upstream": UPSTREAM,
        "upstream_head": head,
        "file": "replication/" + os.path.basename(path),
        "lines": src.count("\n") + 1,
        "copied_from": "bstpp.cox_hawkes_shared.spatiotemporal_hawkes_model_shared",
        "copied_at": {"commit": m.group(1), "date": m.group(2)} if m else None,
        "args_keys": sorted(args_keys),
        "prior_subkeys": sorted(prior_keys),
        "self_attrs": sorted(attrs),
        "ctor_kwargs": ctor_kwargs,
        "imports_from_absent_module": sorted(
            n.name for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "bstpp.cox_hawkes_shared"
            for n in node.names),
    }


def live_args_keys():
    """This fork's args, from a representative cox_hawkes construction.

    WITH covariates, deliberately. ``priors['w']`` is added only inside the
    covariate branch (``main.py:531``), so a no-covariate model reports it
    missing and the comparison would manufacture a difference that is really
    just the fixture's. The copy's calls all pass ``spatial_cov=``, so a
    covariate-bearing model is the commensurable one.
    """
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import numpyro.distributions as dist
    from shapely.geometry import box as shp_box
    from bstpp.main import Hawkes_Model
    t_days = 2.5 * 365.0
    rng = np.random.RandomState(0)
    n = 60
    data = pd.DataFrame({"X": rng.uniform(0.05, 0.95, n),
                         "Y": rng.uniform(0.05, 0.95, n),
                         "T": np.sort(rng.uniform(0, t_days, n))})
    dom = np.array([[0., 1.], [0., 1.]])
    cells, vals = [], []
    for i in range(2):
        for j in range(2):
            cells.append(shp_box(i / 2, j / 2, (i + 1) / 2, (j + 1) / 2))
            vals.append(float(2 * i + j))
    cov = gpd.GeoDataFrame({"cov_a": vals}, geometry=cells)
    m = Hawkes_Model(data, dom, t_days, cox_background="cox",
                     spatial_cov=cov, cov_names=["cov_a"],
                     a_0=dist.Normal(0, 5), alpha=dist.Beta(2, 2),
                     beta=dist.HalfNormal(1.0), sigmax_2=dist.HalfNormal(0.25))
    return set(m.args), set(m.args["priors"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None,
                    help="path to cox_hawkes_offset.py in a read-only checkout")
    ap.add_argument("--head", default=None, help="upstream HEAD sha")
    ns = ap.parse_args()

    if ns.source:
        data = extract(ns.source, ns.head or "UNRECORDED")
        with open(EXTRACT, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"EXTRACT_REWRITTEN from {ns.source}")
    else:
        with open(EXTRACT, encoding="utf-8") as fh:
            data = json.load(fh)
        print("EXTRACT_READ (external source not required for this run)")

    import bstpp
    print()
    print("PROBE_PROVENANCE")
    print(f"  repo             : {REPO}")
    print(f"  bstpp.__file__   : {bstpp.__file__}")
    print(f"  upstream         : {data['upstream']}")
    print(f"  upstream_head    : {data['upstream_head']}")
    print(f"  file             : {data['file']}  ({data['lines']} lines)")
    print(f"  copied from      : {data['copied_from']}")
    print(f"  copied at        : {data['copied_at']}")
    print(f"  imports from bstpp.cox_hawkes_shared (ABSENT HERE): "
          f"{data['imports_from_absent_module']}")
    print()

    ours, prior_names = live_args_keys()

    print("DEPENDENCY SURFACE, classified against THIS fork's live args")
    print(f"  {'args key':<32} {'owner':<9} kind")
    n_ours = n_foreign = 0
    wp2_kind = []
    for k in data["args_keys"]:
        owner = "OURS" if k in ours else "FOREIGN"
        if owner == "OURS":
            n_ours += 1
        else:
            n_foreign += 1
        if k in PRIOR_KIND:
            kind = "PriorConfig-kind"
        elif k in MODEL_KIND:
            kind = "ModelConfig-kind"
        else:
            kind = "-"
        if owner == "OURS" and kind != "-":
            wp2_kind.append((k, kind))
        print(f"  {k:<32} {owner:<9} {kind}")
    print()
    print(f"  args keys read      : {len(data['args_keys'])}"
          f"  ({n_ours} OURS, {n_foreign} FOREIGN)")
    print(f"  self attributes     : {data['self_attrs']}")
    print(f"  ctor keyword surface: {data['ctor_kwargs']}")
    print(f"  args['priors'][...] : {data['prior_subkeys']}")
    missing_priors = [p for p in data["prior_subkeys"] if p not in prior_names]
    print(f"    of which absent from this fork's priors dict: "
          f"{missing_priors or 'none'}")
    print()

    print("WP2-KIND DEPENDENCIES WE OWN "
          "(the reason 'not required' was the wrong entry)")
    for k, kind in wp2_kind:
        print(f"  {k:<32} {kind}")
    print(f"  count : {len(wp2_kind)}")
    print("  These are OUR surface, read directly out of args by a copy that")
    print("  does not follow this repository. A WP2 change to any of them is")
    print("  invisible to every gate this repository runs.")
    print()

    print("WHAT REMAINS UNMEASURABLE, AND WHY")
    print("  (1) bstpp/cox_hawkes_shared.py is in NO reachable tree: the base")
    print("      class, hawkes_intensity_sum, and the function this file")
    print("      copies. Its own args reads cannot be enumerated at all.")
    print("  (2) WP2's change set does not exist. 'No WP2-scope change alters")
    print("      a signature this depends on' is a claim about an unenumerated")
    print("      set; C4 is what enumerates it. The confirmation is NOT YET A")
    print("      STATEMENT, which is different from being true.")
    print("  VERDICT: UNMEASURABLE at round six -- not NOT-REQUIRED.")
    print("EXIT:0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
