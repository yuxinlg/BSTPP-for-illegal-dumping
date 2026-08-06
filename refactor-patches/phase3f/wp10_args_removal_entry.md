# WP10 — legacy-`args` removal: entry, and its inbound items

**What WP10 is.** The work package that removes the legacy `args` dict. The
register has always had this package — A-22's ordering paragraph says *"OP-8
(legacy-`args` removal) is the final work package, not the first"* — but it
has never had an entry document, and the number **WP10** comes from the
round-seven brief rather than from any prior register text. That is recorded
rather than smoothed over: **WP3, WP4, WP6–WP9 have no register text either**,
and WP5 is cited as a destination by six open items without a defining entry
of its own. Work packages are being routed to by number faster than they are
being defined, which is the OP-23 defect (a citation whose target has no text)
in a different place. This file is the first such entry; it does not create
the others.

**Why this file exists at all.** One inbound item is not recoverable from the
source. That is the whole reason it needs an entry rather than a census row.

---

## Inbound item 1 — the internal-unit literals (`T_INTERNAL`, `S_INTERNAL`, `S_DAYS`)

**Routed here at A-42.** Previously it sat only in OP-27's config-external
census rows, which is not where anyone doing this work would look.

### It is not the same object as the other census members

`panel_h_m` and the two cutoff tolerances are quantities with two owners that
**agree today and could disagree later**. This one is different in three ways
and the difference is what earns it a route:

- **A named constant against a bare literal.** `preparation.py` defines
  `T_INTERNAL = 50`, `S_INTERNAL = 24`, `S_DAYS = 365`. `main.py:338/339/365`
  write `args['T'] = 50`, `args['S'] = 24`, `self.S = 365` as literals — in a
  module that **already imports `T_INTERNAL`** (`main.py:41`) and uses it
  elsewhere (`main.py:1787`, `2221`, `2334`).
- **Directional coupling.** Change the constant and `cutoffs.py`'s
  real↔internal conversions follow it; `args['T']` does not. The owners do not
  drift apart symmetrically — one of them moves and the other silently
  doesn't.
- **One edit away.** This is not a latent hazard waiting on a design change.
  It is a broken invariant waiting on a single edit, and the edit —
  "`T_INTERNAL` should be 40, let me change the constant" — is one anybody
  doing unit work would make.

### Say it in unit terms, because that is where the severity is legible

`AGENTS.md` opens its cross-file invariants with **Internal units**, and the
unit contract states that internal/real conversion happens **at exactly three
declared sites**. The rule the codebase has repeatedly paid to learn is: *any
expression crossing the internal/real boundary must go through a declared
conversion.*

`self.S = 365` and `args['S'] = 24` are **the same seasonal quantity on the
two sides of that boundary, written as bare literals, with the conversion
between them named nowhere.** `365` is real days; `24` is the internal
seasonal coordinate; `S_DAYS`/`S_INTERNAL` are the names that say which is
which, and neither name appears at the site. The same holds for the temporal
pair, `T` real-days horizon against `args['T'] = 50`.

So this is not primarily a duplication finding. **It is the internal/real
boundary crossed without the conversion being named** — the single most
productive defect source this codebase has, by its own record.

### Why WP10 and not sooner

`args` removal is the operation that either fixes this or bakes it in. Every
one of these literals is a write *into* `args` (or into the `self.S` that
pairs with one). When `args` is dissolved into typed objects, each write
either becomes a field initialised from the named constant — fixed — or gets
copied as a literal into the new object — baked in, and harder to find
afterwards, because the census heuristic that surfaced it works on
module-level constants and bare literals in the same package.

### What WP10 must not assume

**That these three literals are the same quantity as those three constants
took adjudication to establish and is not recoverable from source.** No
import, no comment, no assertion binds `50` at `main.py:338` to
`preparation.T_INTERNAL`. A reader of `main.py` alone cannot tell. That is
exactly why it is written down here instead of being left to be re-derived.

### Where this item now appears

1. **This file** — WP10's entry, reachable without reading the census.
2. `phase3_record.tex` §11, the **OP-27** row — as three config-external
   census rows, with the adjudication.
3. `phase3_record.tex` **A-41** and **A-42** — the measurement and the
   routing.
4. `AGENTS.md`, the **Internal units** invariant paragraph — where someone
   doing unit work looks first.

### Deliberately not done at A-42

A one-line comment at `main.py:338` naming `T_INTERNAL` would be the cheapest
mitigation and is **not taken here**: round seven's declared scope is C1's
list closure and C4's classification, and a `bstpp/` edit is outside it.
Recorded as a one-line item for WP10's first commit rather than smuggled in.
