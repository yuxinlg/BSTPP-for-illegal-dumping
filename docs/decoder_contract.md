# Decoder contract

Status: normative for WP7 when adopted
Contract schema: `bstpp.decoder-contract`, version 1

This is a self-contained specification. It states rules, not their history.

## 1. Purpose

BSTPP uses pretrained PriorVAE decoders as prior-side model substitutions for
the temporal, seasonal, and spatial latent fields. A decoder is compatible only
when its identity, role, shapes, partition, and gain convention agree with the
model being built.

This contract makes those facts explicit and validates them before inference.
It does not claim that a decoder reproduces its target Gaussian process exactly,
recover missing training history, or retrain any artifact.

## 2. Artifacts under contract

`bstpp/decoders/` ships nine files. Production code reaches three, and those
three are the artifacts under contract:

| Role | Artifact |
|---|---|
| temporal | `bstpp/decoders/decoder_1d_T50_fixed_ls` |
| seasonal | `bstpp/decoders/decoder_1d_T24_circ_small_l8` |
| spatial | `bstpp/decoders/2d_decoder_15_5_large.pkl` |

The remaining six files are unreferenced by production code. They are **out of
contract**: not hashed, not validated, not loaded, and not deleted by this work.
If one of them is later paired with a model, it enters the contract at that
point and is hashed like any other file-backed artifact.

## 3. Normative object

WP7 implements a frozen `DecoderContract` with one public factory and
construction-time validation. The serialized record has this logical shape:

```yaml
schema_name: bstpp.decoder-contract
schema_version: 1
role: temporal | seasonal | spatial
identity:
  strength: verified | declared
  artifact_sha256: <64 lowercase hex or null>
  decoder_id: <stable nonempty string or null>
  locator: <optional package/file locator>
framework:
  name: jax.example_libraries.stax
  parameter_format: <declared format>
shapes:
  latent: [<positive integers>]
  output: [<positive integers>]
partition:
  axis: temporal | seasonal | spatial
  dimensions: [<positive integers>]
  flattening: <declared convention>
gain:
  convention: identity | exp_log_amplitude
  parameter_name: <string or null>
  application_count: 1
provenance:
  training_code: <value or UNKNOWN>
  training_data: <value or UNKNOWN>
  target_prior: <mapping or UNKNOWN>
  notes: <string or null>
```

Paths are locators, not identity. Moving the same verified bytes does not change
identity; changing the bytes does.

## 4. Identity rules

### 4.1 Packaged or file-backed artifacts

`identity.strength` is `verified` and `artifact_sha256` is required. Compute
SHA-256 from the exact artifact bytes the loader will consume, **before
deserialization**. The loader already holds those bytes: `_load_decoder` obtains
`raw` from `pkgutil.get_data` and passes it to `pickle.loads`, so the digest is
taken from `raw` at that one site and needs no separate read path.

The digest is lowercase hexadecimal and is stored in:

- the `DecoderContract`;
- fitted-results provenance;
- any pin record that claims decoder identity.

`decoder_id` may be a human-readable alias but is not sufficient identity for a
file-backed artifact. `locator` is informative and may change without changing
the contract.

### 4.2 Custom in-memory decoders

If stable artifact bytes do not exist, `identity.strength` is `declared` and a
caller-supplied stable `decoder_id` is required. `artifact_sha256` must be
`null`.

BSTPP must not hash a Python callable, `repr`, object address, source-location
guess, or closure state and call that reproducible identity. A declared custom
identity is weaker than a verified artifact hash, and results provenance says
so.

Custom decoders remain subject to every shape, partition, dtype, purity, and
gain check below.

## 5. Roles and shipped configuration

| Role | Latent shape | Output/partition shape | Gain convention |
|---|---:|---:|---|
| temporal | `(11,)` | `(50,)`, 50 temporal cells | `identity` |
| seasonal | `(8,)` | `(24,)`, 24 circular seasonal cells | `identity` |
| spatial | `(20,)` | `(625,)`, row-major flattening of `25 x 25` cells | `exp_log_amplitude`, parameter `sp_var_mu` |

The latent shapes are the shapes of the registered `z_temporal`, `z_seasonal`,
and `z_spatial` sample sites. The table is a compatibility contract for the
shipped artifacts, not a claim that 50, 24, and 25 are mathematical constants. A
future decoder trained on a different partition may declare different dimensions
under a later compatible contract; it must not be paired with the current
partitions silently.

## 6. Partition compatibility

`PartitionDecoderConfig` owns the chosen partition dimensions and the three
decoder-contract references. Binding succeeds only when:

- each required role appears exactly once;
- the decoder output shape equals the corresponding field-vector shape;
- the latent shape equals the shape of the registered `z_*` sample site;
- temporal output length equals `n_t`;
- seasonal output length equals `n_s` and declares the circular/seasonal role;
- spatial output length equals the product of the spatial partition dimensions;
- the spatial flattening order matches the membership/index-map convention;
- decoder dtype is accepted by the online JAX path;
- no unknown field is being used to excuse a required compatibility fact.

The packaged artifacts remain pinned to the current dimensions. WP7 makes the
dimensions typed configuration constrained by decoder output; it does not make
arbitrary dimensions work with those artifacts.

## 7. Gain contract

Gain is part of decoder identity and compatibility because it changes the
effective prior.

### Temporal and seasonal

The convention is `identity`: the decoded field is the decoder output, with no
additional scalar amplitude.

### Spatial

The convention is:

```text
f_xy = exp(sp_var_mu) * D_spatial(z_spatial)
```

Requirements:

- `sp_var_mu` is the fixed log-amplitude value owned by `ModelConfig`. It is a
  gain **paired with the trained decoder parameters**, restoring the
  log-amplitude factored out during VAE training. It is not a sampled prior
  parameter and not a derived formula.
- `DecoderContract` declares that the spatial artifact expects the
  `exp_log_amplitude` convention and the parameter name `sp_var_mu`.
- The multiplier is applied **exactly once**, inside the shared spatial decode
  function used by both inference and simulation. There is one such function;
  every consumer routes through it.
- `UNIT_GAIN_SP_VAR_MU = 0.0` remains the unit-gain identity used by the SBC
  program.
- `sp_var_mu` is a config-owned real. Type validation belongs to the config
  invariant sequence and reuses the existing real-argument helper; a coerced
  `float(...)` at the storage site is not validation.
- A sampled amplitude and a new seasonal amplitude are outside Phase 3.

An unknown training-time amplitude does not permit omitting or duplicating the
runtime convention. The convention is known from the executable pairing even
when the historical rationale is `UNKNOWN`.

## 8. Provenance and `UNKNOWN`

Every provenance field is either supported by evidence or the literal string
`UNKNOWN`. `UNKNOWN` is valid for historical training facts and must survive:

- package metadata loading;
- contract-to-record conversion;
- results save/load;
- user-facing provenance reports.

`UNKNOWN` is **not** valid for facts BSTPP must know to execute safely:

- decoder role;
- identity strength;
- verified artifact digest or declared custom ID;
- latent/output shapes;
- partition dimensions and flattening;
- gain convention and application count.

The seasonal `.meta.txt` sidecar is an honest stub: its architecture fields are
derived from code and its GP-prior fields (kernel, length scale, variance,
standardization, training code) are `UNKNOWN`. WP7 may translate it into a
versioned record, but must not fill those fields by inference from the filename.
The shipped training script does not reproduce this artifact and has no circular
kernel option; that fact is recorded, not worked around.

## 9. Load and validation order

Validation occurs before a decoder can reach inference, simulation, scoring, or
saved-results attachment:

1. Resolve the contract record.
2. For a verified artifact, read bytes and compare SHA-256.
3. Validate schema, role, required fields, and identity mode.
4. Validate latent/output/partition shapes and flattening.
5. Validate gain convention against `ModelConfig` and the shared decode path.
6. Deserialize parameters.
7. Run a small deterministic shape/dtype probe outside NumPyro tracing.
8. Install the decoder/contract pair transactionally.

If any step fails, no model or config object is partially updated.

## 10. Failures

Use one named `DecoderContractError(ValueError)` family with canonical clauses
for each invariant. Fail loudly on at minimum:

- unsupported contract schema;
- missing or malformed verified SHA-256;
- artifact digest mismatch;
- missing declared custom ID;
- duplicate or wrong role;
- latent, output, or partition shape mismatch;
- spatial flattening mismatch;
- unsupported dtype or nonfinite probe output;
- missing, unknown, or multiply applied gain convention;
- result decoder identity incompatible with the receiving model.

One invariant is one clause, rendered byte-for-byte wherever it is detected.
Paths, object addresses, and callable reprs must not appear as the only identity
in an error. User-supplied text reaching an error or warning follows the
project's console-safe message policy.

## 11. Results and pins

Results schema v1 stores, for every role:

- the complete decoder-contract record;
- identity strength;
- verified artifact SHA-256 or declared custom ID;
- role, shapes, partition, flattening, and gain;
- provenance fields, including `UNKNOWN`.

Attaching results to an existing model hard-fails if role, identity, shape,
partition, flattening, or gain is incompatible. A changed file path with the same
verified bytes is compatible. A dependency or package-version change is recorded
and warned on but does not by itself replace the identity check.

Pin configurations exercising Cox/LGCP paths record verified artifact digests. A
decoder test skipped because an artifact is absent is not evidence that decoder
compatibility passed.

## 12. Required WP7 tests

- packaged artifact with correct digest and shapes constructs;
- one-byte artifact modification fails before deserialization;
- moved artifact with the same bytes retains identity;
- custom decoder without `decoder_id` fails;
- custom decoder with a declared ID and correct probe constructs;
- wrong latent/output/partition dimension fails per role;
- wrong spatial flattening fails;
- spatial gain is applied exactly once and unit gain holds at `sp_var_mu=0`;
- temporal/seasonal gain remains identity;
- `UNKNOWN` fields round-trip without being replaced;
- decoder mismatch prevents results attachment without mutating the model;
- LGCP and Cox-Hawkes smoke tests run with the three contracted artifacts
  present.

## 13. Non-goals

- Recovering or inventing training provenance.
- Retraining or replacing decoders.
- Making the shipped artifacts support arbitrary partitions.
- Sampling field amplitudes.
- Treating a filename or path as content identity.
- Claiming custom declared identity is as strong as a verified byte hash.
- Validating, hashing, or deleting the six out-of-contract artifacts.
