# Changelog

All notable changes to this fork are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Scope, and one limit stated up front

This is a fork of [imanring/BSTPP](https://github.com/imanring/BSTPP) adapted
for Philadelphia illegal-dumping analysis. Entries describe changes **relative
to upstream**, which is what a migrating user needs.

**This file is backfilled, and its completeness is not asserted.** The entries
below are the changes identified as breaking from the decision register
(`phase3_record.tex`) and verified against the source. The register does not
carry a "breaking change" label, so there was no list to transcribe and no
mechanism guarantees this one is exhaustive. It is a floor, not a census.
Recorded this way rather than as a confident inventory because a changelog
that overstates its coverage is worse for a migrating user than one that
states its limit -- they can compensate for a limit they can see.

**No release has been cut from this fork.** Everything below is unreleased.

## Distribution name and version are unresolved (OP-32)

`setup.py` still declares `name='BSTPP'`, `version='0.1.3'` -- upstream's PyPI
name and version, unchanged, on a fork carrying the breaking changes listed
below. Publishing as-is would collide with upstream on PyPI and would announce
a patch-level version for a set of changes that are not patch-level.

A `[project]` table in `pyproject.toml` is the natural home for this metadata
and **has deliberately not been added yet**, because `name` and `version` are
mandatory in that table and both are open decisions. See OP-32 in the register.

## [Unreleased]

### Changed -- breaking

- **`standardize_cov` no longer accepts booleans.** Accepted values are now
  `None` (off, default) or `'domain_area'`. Upstream's `standardize_cov=True`
  raises `NumericalConfigError` rather than being silently reinterpreted, so a
  migrating user is told what happened to their argument instead of getting a
  different estimator without notice.

  Validation is unconditional as of CI-9: it runs in
  `Point_Process_Model.__init__` whether or not covariates are supplied.
  Previously the only check lived inside the `spatial_cov is not None` branch,
  so **every** value -- including misspellings -- constructed silently when no
  covariates were passed.

- **`'domain_area'` is a different estimator from upstream's `True`, not a
  weighted variant of it.** Upstream centres and scales each column over every
  supplied covariate row. This fork weights by clipped area `|C_c intersect A|`,
  so rows with no domain mass drop out of the moments entirely -- a different
  population, not different weights.

  Measured on the Philadelphia layer with a park-box domain: centre shift
  0.571 unweighted SD, scale ratios 0.135-1.223, delta-loglik 1574.53 nats
  (9.44%) at identical latents.

  **No figure computed under one is comparable to one computed under the
  other** -- not log-likelihoods, AIC, `w`, `b_0`, or the excitation share.
  Re-derive rather than re-label, including figures already produced.

  Intent is settled (**D-54**, closes OP-28): `'domain_area'` is an in-package
  convenience for one fitted domain. Cross-site work standardizes externally
  once and passes `None`.

- **Configuration arguments are type-checked at construction** (CI-7 real
  arguments, CI-8 integral arguments). Values that previously flowed through
  and failed later, or coerced silently, now raise at the point of supply with
  a canonical message. A float is not accepted where an integral argument is
  required.

- **Two builder sites raise `NumericalConfigError` where they previously
  raised `TypeError`.** Code catching `TypeError` around configuration
  construction will stop catching these. `NumericalConfigError` is the single
  declared type for configuration-invariant violations.

- **The spatial trigger is a real-unit object.** `sigmax_2` is in squared real
  units of the input X/Y columns and **the user must supply its prior -- there
  is no default**; `spatial_window` is a real length. Input coordinates must be
  metric (a projected CRS); the constructor warns on degree-like domains,
  because lon/lat makes an isotropic kernel anisotropic on the ground.

- **Polygon excitation support requires an explicitly prepared mass table.**
  Tables are built only through the public `prepare_polygon_mass_table(...)`,
  and polygon construction hard-requires a compatible supplied table -- there
  is no silent synchronous rebuild. A table is valid only for the exact domain
  union, event coordinates and row order, spatial window, sigma range and grid,
  and build settings recorded in its metadata; equal row counts are not
  evidence of compatibility.

  `log_expected_likelihood(test_data, mass_table=...)` treats held-out data as
  a separate realization and rebuilds all event-indexed state from it. In
  polygon mode an explicit held-out table prepared for those events is
  required.

  The current Hermite backend integrates `Spatial_Symmetric_Gaussian` only.
  Polygon mode rejects any custom or non-Gaussian spatial trigger until that
  trigger supplies a matching polygon-mass backend.

- **`cox_background` is a bool, default `True`.** The retired string `'cox'`
  is rejected (CI-10, A-50). Cox-Hawkes remains `Hawkes_Model(cox_background=True)`,
  not a third class. Snippets copied from a pre-A-50 signature break loudly.

### Added

- `environment.yml` reproducing the development stack, pinned to the versions
  measured in the working environment (A-55).
- Continuous integration for the machine-independent gates (D-63). The golden-pin
  battery is deliberately excluded and remains manual, because pins are
  machine-local artifacts; the reason is recorded in the workflow file.
- A `commit-msg` hook enforcing the `Change-Class` trailer.
- `CONTRIBUTING.md` for the change-class vocabulary, trailer format, and hook
  install.

### Known issues

- **OP-32** -- distribution name and version are still upstream's (`BSTPP`
  0.1.3) and must be resolved before publication.
- **`requirements.txt` pins `rasterio<1.4` while the measured environment holds
  1.4.4** (A-55). The pin's own comment names 1.5 as the constraint. Recorded,
  not silently widened.

[Unreleased]: https://github.com/yuxinlg/BSTPP-for-illegal-dumping/commits/refactor
