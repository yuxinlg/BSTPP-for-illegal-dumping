# BSTPP results format

Status: normative for WP9 when adopted
Format name: `bstpp-results`
Schema version: 1
Container: trusted Python pickle envelope

This is a self-contained specification. It states rules, not their history.

## 1. Scope

Schema v1 is a bounded Phase 3 persistence contract. It saves posterior results
together with enough typed configuration, data identity, decoder identity, and
approximation provenance to reject attachment to an incompatible model. It is
independent of the legacy `args` dictionary and must remain unchanged when WP10
removes that dictionary.

This is not a portable, language-neutral archive. Pickle files can execute code
while loading and must be opened only when they come from a trusted source.

## 2. API behavior

The existing public methods remain:

```python
model.save_rslts(path)
model.load_rslts(path)
```

WP9 may add pure helpers — for example `build_results_envelope`,
`validate_results_envelope`, and `attach_results` — but does not require a
signature change to the two public methods.

Saving is atomic: write a temporary file in the destination directory, flush and
close it, then replace the destination. Loading and attachment are
transactional: deserialize and validate the complete envelope before changing
`samples`, `svi_results`, `mcmc`, or any other model attribute.

## 3. Envelope

The unpickled top-level object is a mapping with exactly these required
sections; extensions may be added only under the documented `extensions`
mapping.

```python
{
    "format": {
        "name": "bstpp-results",
        "schema_version": 1,
    },
    "created": {
        "utc": "<RFC 3339 timestamp>",
        "writer": "BSTPP",
    },
    "result": {
        "method": "svi" | "mcmc" | "external",
        "samples": {"<site>": <array>, ...},
        "inference_state": {"svi_results": <opaque>, "mcmc": <opaque>},
    },
    "configuration": {
        "model": <ModelConfig record>,
        "prior": <PriorConfig record>,
        "partition_decoder": <PartitionDecoderConfig record>,
        "numerical": <NumericalConfig record>,
        "inference_run": <InferenceRunConfig record>,
        "fingerprints": {"<config>": "<sha256>", ...},
    },
    "data_identity": {
        "events": <identity record>,
        "domain": <identity record>,
        "horizon_days": <float>,
        "offset_seasonal": <float>,
        "covariates": <identity record or None>,
        "standardization": <standardization record>,
        "partitions": <partition identity record>,
    },
    "decoders": {
        "temporal": <DecoderContract record or None>,
        "seasonal": <DecoderContract record or None>,
        "spatial": <DecoderContract record or None>,
    },
    "excitation": {
        "support": <support record or None>,
        "mass_table": <identity/provenance record or None>,
        "cutoffs": <CutoffProvenance record or None>,
    },
    "environment": {
        "bstpp_version": <string or UNKNOWN>,
        "git_commit": <string or UNKNOWN>,
        "python": <string>,
        "jax": <string or UNKNOWN>,
        "jaxlib": <string or UNKNOWN>,
        "numpyro": <string or UNKNOWN>,
        "numpy": <string or UNKNOWN>,
        "scipy": <string or UNKNOWN>,
        "geopandas": <string or UNKNOWN>,
        "shapely": <string or UNKNOWN>,
        "platform": <string>,
    },
    "extensions": {},
}
```

`result.samples` is required. `inference_state` is optional and may omit either
opaque object. Opaque inference state is a convenience; posterior samples and
provenance are the durable content of schema v1.

**Cutoff provenance lives at `excitation.cutoffs`, not at the envelope's top
level.** See section 13 — the existing G2 gate asserts the top-level position
and must be moved with the schema, or it cannot detect its own satisfaction.

## 4. Configuration records

Each frozen config implements a deterministic `to_record()` returning only
versioned mappings, scalars, strings, booleans, `None`, and arrays with explicit
dtype/shape. Records contain resolved values and any user-supplied-versus-
defaulted distinction the model contract requires. A field that was unresolved
at construction and resolved at bind records both the sentinel state and the
resolved value.

The `configuration.fingerprints` values are SHA-256 digests over a canonical
encoding of the corresponding records. Canonicalization must declare key
ordering, array dtype/shape/byte order, and treatment of `UNKNOWN`.

`PriorConfig` may contain arbitrary NumPyro distribution objects. Its record
therefore has two layers:

- a readable distribution description: fully qualified class, batch/event
  shapes, support, and parameter names/values where introspection is stable;
- a trusted-pickle payload digest computed with the schema's declared pickle
  protocol, used to detect same-environment drift when no portable parameter
  encoding exists.

The format does not promise that opaque custom prior payloads can be recreated
under a different NumPyro version. The environment is recorded and drift is
warned. A prior may not disappear from provenance merely because its portable
description is incomplete.

## 5. Data identity

The receiving model must represent the same fitted problem. Schema v1 records:

### Events

- algorithm/version name;
- row count;
- SHA-256 over canonical little-endian float64 `X`, `Y`, and `T` values in model
  row order;
- column identity needed to reconstruct that order.

### Domain

- rectangle versus polygon mode;
- canonical bounds;
- CRS in a stable representation or `None`;
- for polygon domains, a canonical geometry SHA-256 based on the same
  normalization used by `PolygonMassTable` compatibility;
- for rectangle domains, a canonical little-endian float64 bounds digest.

### Covariates

When present:

- covariate names and order;
- declared CRS;
- canonical design-matrix digest after the recorded standardization;
- event-to-covariate membership digest;
- integration/refinement identity needed by the fitted likelihood.

### Partitions

- `n_t`, `n_s`, and spatial dimensions;
- partition-boundary and membership-map digests;
- `season_overlap` digest;
- integration-field, integration-covariate, and integration-area digests when
  present.

Hashes identify exact fitted state; they do not replace readable metadata.

## 6. Standardization record

Always store:

```python
{
    "method": "none" | "domain_area",
    "columns": [...],
    "mean": <array or None>,
    "scale": <array or None>,
    "reference_scope": "fitted_domain" | "external",
}
```

The in-code standardization dict currently carries `method`, `columns`, `mean`,
and `scale`. WP9 adds `reference_scope`, which is `fitted_domain` for the
in-package `domain_area` path and `external` when the caller standardized
against a fixed reference and passed `standardize_cov=None`. The distinction is
the one that makes cross-site coefficient comparison legible; it cannot be
recovered after the fact.

Legacy `standardize_cov=True` results are not silently relabeled as
`domain_area`. An unversioned legacy file lacks adequate provenance and is
rejected under section 10.

## 7. Decoder records

Each required role stores the complete schema-v1 `DecoderContract` record from
`docs/decoder_contract.md`, including:

- identity strength;
- artifact SHA-256 or declared custom ID;
- role and shapes;
- partition dimensions and flattening;
- gain convention and application count;
- explicit `UNKNOWN` training provenance.

Path drift alone is not incompatibility when verified bytes are identical.

## 8. Excitation and approximation provenance

For Hawkes and Cox-Hawkes results, store under `excitation`:

- excitation support mode;
- temporal and spatial windows in declared real/internal units;
- resolved sigma bounds and prior-truncation provenance;
- design scales and tolerance precedence;
- realized omitted-mass bounds;
- every `CutoffProvenance.to_dict()` field;
- polygon mass-table schema, geometry/event identities, knot domain, builder
  settings, residual/budget evidence, and artifact SHA-256 where applicable;
- the **bound** `NumericalConfig` values actually used, not module defaults.

For LGCP results, support/mass/cutoff entries are `None` and the model-family
record makes that absence meaningful.

## 9. Load algorithm

`load_rslts` performs these steps in order:

1. Warn before opening that only trusted pickle files are safe.
2. Deserialize into a local variable; do not mutate the model.
3. Require top-level mapping, format name, and supported schema version.
4. Validate required sections and field types.
5. Recompute internal record fingerprints and artifact identities.
6. If attaching to an existing model, compare model, prior, data, partition,
   decoder, excitation, and standardization compatibility.
7. Validate sample-site names, shapes, and draw-axis consistency.
8. Record nonfatal environment drift warnings.
9. Attach samples and optional inference state in one final transaction.

Any failure before step 9 leaves the receiving object observationally unchanged.

## 10. Hard failures

Raise a named `ResultsCompatibilityError(ValueError)` before attachment for:

- a nonmapping envelope;
- missing format name or schema version;
- unsupported schema version;
- malformed or missing required sections;
- model-family mismatch;
- missing required sample sites or incompatible sample shapes;
- prior/config incompatibility affecting the fitted model;
- event, domain, horizon, covariate, standardization, or partition mismatch;
- decoder role, identity, shape, partition, flattening, or gain mismatch;
- excitation support, mass-table, sigma-bound, or cutoff-provenance mismatch;
- corrupt internal fingerprints;
- an unversioned/legacy result dictionary.

Unsupported schema and incompatible fitted state are different error clauses and
must be distinguishable in tests.

## 11. Warnings, not hard failures

After all identity/compatibility checks pass, warn and record:

- BSTPP package-version drift;
- git-commit drift;
- Python, JAX, NumPyro, NumPy, SciPy, GeoPandas, or Shapely version drift;
- platform drift;
- opaque inference state that cannot be reattached even though samples and
  provenance are valid.

Dependency drift may explain numerical differences, but it does not by itself
prove that results belong to a different fitted problem. The exact identity
checks above govern compatibility.

## 12. Legacy files

The old unversioned dictionary containing only `samples` and optional
`svi_results`/`mcmc` is not schema v1. Automatic fallback is forbidden because it
would attach results without validating the data, model, decoder, support, or
cutoff provenance.

Loading such a file fails with a message identifying it as an unversioned legacy
result and recommending the checkout/environment that created it to inspect or
export the samples. WP9 does not invent missing provenance.

## 13. The existing G2 gate must move with the schema

A strict `xfail` currently names the missing provenance round trip. It asserts:

```python
blob["cutoff_provenance"] == before
```

— a **top-level** key. Schema v1 stores that record at `excitation.cutoffs`.

Consequence, stated so it is planned rather than discovered: **implementing this
contract exactly and completely leaves that test still xfailing.** It would
never turn green, so the gate meant to certify G2 would report nothing at the
moment G2 was satisfied.

The WP9 commit therefore:

1. preserves the pre-rewrite xfail state as its RED capture;
2. rewrites the assertion to read `envelope["excitation"]["cutoffs"]` and to
   check the schema's required sections;
3. removes the `xfail` marker and lands the test green;
4. names the test edit in the commit message as a planned edit with sign-off,
   not an incidental change.

## 14. Required WP9 tests

- SVI and MCMC samples round-trip under schema v1;
- the rewritten G2 cutoff/excitation provenance test passes without an `xfail`
  marker;
- all five config records and fingerprints round-trip;
- standardization (including `reference_scope`), data, partition, and decoder
  identities round-trip;
- each hard-failure category rejects before any model attribute changes;
- package/dependency drift warns after compatibility passes;
- changed decoder path with identical verified bytes remains compatible;
- changed decoder bytes fail;
- `UNKNOWN` provenance survives exactly;
- legacy unversioned files fail with the named remediation;
- a partially written/corrupt file does not alter an existing model;
- schema output contains no `args` key and remains unchanged after WP10.

## 15. Security and portability statement

The API documentation and load-time warning must say plainly:

> BSTPP result files use Python pickle and may execute code while loading. Load
> only files from a trusted source.

Schema v1 prioritizes bounded Phase 3 compatibility and complete model
provenance over cross-language portability. A future portable archive is a new
schema and a separate project; it does not delay WP9.
