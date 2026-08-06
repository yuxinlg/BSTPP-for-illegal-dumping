"""A-36: the DENOMINATOR for the silent-collapse census, computed rather than
asserted (per the 183/184 lesson: a count without a method is not a measurement).

METHOD, stated so it can be re-run and disagreed with.

  Step 1  POPULATION OF ENTRY POINTS. AST-walk every module in ``bstpp/``
          except ``__init__.py``. An entry point is a module-level ``def``
          whose name has no leading underscore, or a method of a module-level
          class whose name has no leading underscore (plus ``__init__``).
          This is the surface a user of the package can call.

  Step 2  EXCLUSIONS, each with a reason, counted and printed. Nothing is
          dropped silently.
            E1  D-40 clause renderers and raisers (``*_invariant_clause``,
                ``raise_*_violation``, ``require_config_*``): they render or
                raise, they never compute. Their whole job is the error path.
            E2  entry points with no parameter at all (accessors, exporters).
            E3  entry points whose every parameter is non-scalar.

  Step 3  SCALAR PARAMETERS. Within surviving entry points, a parameter is
          SCALAR-INTENDED when its annotation reduces to int / float / bool
          (including Optional[...] and ``X | None``), or -- where there is no
          annotation -- its default is an int / float / bool / None and the
          name is not in the known non-scalar list. ``self``/``cls``/``*args``/
          ``**kwargs`` are excluded.

  Step 3b DOCSTRING RULE, added after the method was checked against a known
          positive and FAILED it. Steps 1-3 excluded ``Point_Process_Model.T``
          -- a required positional with no annotation and no default -- which
          is the single parameter this census was commissioned to cover
          (``T=True`` collapses the real horizon to 1.0 days). A rule that
          drops its own headline case is not a method. So: a required
          positional with no annotation is SCALAR-INTENDED when the Numpydoc
          block declares it ``name : float`` / ``int``. Recorded as a
          correction rather than folded in silently, because the first
          denominator was computed and printed before it was found
          (``DENOMINATOR_PAIRS=106``, 45 entry points, superseded below).

  The denominator is the number of (entry point, scalar parameter) PAIRS.
  Every pair is either fired by the census probe or listed here as unfired,
  so the census reports coverage as a fraction of a defined population.
"""
import ast
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[3]

CLAUSE_SUFFIXES = ("_invariant_clause",)
RAISER_PREFIXES = ("raise_",)
REQUIRE_PREFIXES = ("require_config_",)

# Names whose intended domain is demonstrably not a scalar even though the
# default is None or a number. Listed explicitly so the exclusion is auditable.
NON_SCALAR_NAMES = {
    "data", "A", "spatial_cov", "cov_names", "spatial_cov_crs", "crs",
    "domain_geom", "domain_gdf", "poly", "geom", "event_x_real",
    "event_y_real", "x", "y", "z", "t", "coords", "prior", "priors",
    "mass_table", "numerical_config", "extra_provenance", "rng", "rng_key",
    "parameters", "pars", "params", "model", "guide", "svi_result", "sites",
    "ax", "df", "checks", "partitions", "domain", "table", "trace", "key",
    "log_sigma", "log_knots_arr", "sigmax_2_samples", "axis_scales",
    "temporal_trig", "spatial_trig", "spatial_trigger", "temporal_trigger",
    "excitation_support", "support_mode", "mode", "cox_background",
    "data_contracts", "standardize_cov", "t_units", "start_date", "origin",
    "t_col", "quantile_method", "plot_mode", "figsize", "result_gdf",
    "events_gdf", "cov_gdf", "union_geometry", "bounds", "init_strategy",
    "init_state", "auto_guide", "decoder_params", "batch", "aux",
    "sample_shape", "value", "name", "remediation", "n_events_or_none",
    "pairs_and_values", "pairs_and_dxdy", "limits", "dif", "ws_unused",
    "cov_grid_size", "temporal_values", "spatial_values", "field_indices",
    "areas", "covariate_indices", "cell_areas", "mu_cells", "mu", "f_t",
    "f_a", "f_xy", "season_overlap", "xy_events", "t_events",
    "rectangular_bounds", "temporal_parameters", "spatial_parameters",
    "x_vals", "y_vals", "tpl", "inter_arrival_times", "title", "path",
    "file_name", "fn", "include_cov", "resume", "plot_loss", "show_gp",
    "show_hist", "rescale", "points_xy", "n_xy_unused", "validate_args",
    "dx_real", "dy_real",
}

SCALAR_ATOMS = {"int", "float", "bool"}


def scalarish(ann: ast.AST | None) -> bool:
    """True when the annotation reduces to a scalar number (or scalar|None)."""
    if ann is None:
        return False
    txt = ast.unparse(ann)
    txt = txt.replace("Optional[", "").replace("]", "")
    parts = [p.strip() for p in txt.replace("|", ",").split(",")]
    parts = [p for p in parts if p and p != "None"]
    return bool(parts) and all(p in SCALAR_ATOMS for p in parts)


def docstring_scalars(fn: ast.FunctionDef) -> set[str]:
    """Names the Numpydoc block declares as a scalar type (Step 3b)."""
    doc = ast.get_docstring(fn) or ""
    out = set()
    for line in doc.splitlines():
        if ":" not in line:
            continue
        lhs, rhs = line.split(":", 1)
        nm = lhs.strip()
        # "lr: float, default=0.001" -- the type is the first comma token.
        # (Those stated defaults are themselves false: num_steps and lr are
        # required positionals. Noted, not fixed here.)
        ty = rhs.strip().lower().split(",")[0].strip()
        if nm.isidentifier() and ty in ("float", "int", "bool"):
            out.add(nm)
    return out


rows, excl, via_doc, unclassified = [], {"E1": [], "E2": [], "E3": []}, [], []

for path in sorted((REPO / "bstpp").glob("*.py")):
    if path.name == "__init__.py":
        continue
    tree = ast.parse(path.read_text(encoding="utf-8"))
    entries = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            entries.append((node.name, node))
        elif isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, ast.FunctionDef) and (
                        not m.name.startswith("_") or m.name == "__init__"):
                    entries.append((f"{node.name}.{m.name}", m))

    for qual, fn in entries:
        short = qual.split(".")[-1]
        if (short.endswith(CLAUSE_SUFFIXES)
                or short.startswith(RAISER_PREFIXES)
                or short.startswith(REQUIRE_PREFIXES)):
            excl["E1"].append(f"{path.name}:{fn.lineno} {qual}")
            continue

        a = fn.args
        allargs = a.posonlyargs + a.args
        defaults = [None] * (len(allargs) - len(a.defaults)) + list(a.defaults)
        named = [(arg, d) for arg, d in zip(allargs, defaults)
                 if arg.arg not in ("self", "cls")]
        named += list(zip(a.kwonlyargs, a.kw_defaults))
        if not named:
            excl["E2"].append(f"{path.name}:{fn.lineno} {qual}")
            continue

        doc_scalars = docstring_scalars(fn)
        scalars = []
        for arg, d in named:
            if arg.arg in NON_SCALAR_NAMES:
                continue
            if scalarish(arg.annotation):
                scalars.append(arg.arg)
                continue
            if arg.annotation is None and d is not None:
                if isinstance(d, ast.Constant) and (
                        isinstance(d.value, (int, float)) or d.value is None):
                    scalars.append(arg.arg)
                    continue
            # Step 3b: required positional, no annotation, declared scalar in
            # the docstring. This is the rule that recovers ``T``.
            if arg.annotation is None and d is None and arg.arg in doc_scalars:
                scalars.append(arg.arg)
                via_doc.append(f"{path.name}:{fn.lineno} {qual}.{arg.arg}")
                continue
            # Step 4: everything the method could not decide. Printed at its
            # size rather than absorbed into either bucket -- a parameter with
            # a sentinel default (``_UNSET``) or no annotation, no scalar
            # default and no docstring type is UNDECIDED, not "not scalar".
            unclassified.append(f"{path.name}:{fn.lineno} {qual}.{arg.arg}")
        if not scalars:
            excl["E3"].append(f"{path.name}:{fn.lineno} {qual}")
            continue
        for s in scalars:
            rows.append((f"{path.name}:{fn.lineno}", qual, s))

print("A-36 census denominator -- method printed above in the module docstring")
print(f"repo = {REPO}")
print()
print("SCALAR-INTENDED (entry point, parameter) PAIRS")
print("-" * 92)
last = None
for loc, qual, param in rows:
    if qual != last:
        print(f"{loc:<26}{qual}")
        last = qual
    print(f"{'':<26}    .{param}")
print("-" * 92)
print(f"DENOMINATOR_PAIRS={len(rows)}")
print(f"DENOMINATOR_ENTRY_POINTS={len({q for _, q, _ in rows})}")
print()
print(f"RECOVERED_BY_STEP_3B={len(via_doc)}  "
      "-- required positionals the annotation/default rule alone dropped")
for item in via_doc:
    print(f"    {item}")
print()
for tag, label in (("E1", "D-40 clause renderers / raisers (never compute)"),
                   ("E2", "no parameters (accessors, exporters)"),
                   ("E3", "no scalar-intended parameter")):
    print(f"EXCLUDED_{tag}={len(excl[tag])}  -- {label}")
    for item in excl[tag]:
        print(f"    {item}")
print()
print(f"UNCLASSIFIED_PARAMS={len(unclassified)}  -- the method could not decide; "
      "sentinel defaults (_UNSET) and undocumented required positionals live here")
for item in unclassified:
    print(f"    {item}")
print("EXIT:0")
