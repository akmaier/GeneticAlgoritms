# Implementation plan

Decisions locked: **Python 3.12** baseline, **pymoo 0.6.2** as the evolutionary engine,
**predicted binding probability as the dominant objective** with plausibility and off-target
binding as feasibility constraints. See [README.md](README.md) for the rationale.

Each phase below ends in something runnable. Nothing is deferred to a big-bang integration.

---

## Phase 0 — Environment and skeleton ✅

**Goal:** `pip install -e ".[core,tcr]"` works and an empty `tcrga` imports.

- `pyproject.toml`, `requires-python = ">=3.12,<3.13"`, hatchling backend
- Extras: `core` (numpy, pandas, pymoo, tidytcells, peptides, matplotlib)
  · `tcr` (tcrdist3, olga, biopython, scikit-learn)
  · `dl` (torch==2.2.2, fair-esm, epytope) — pinned, never imported at module scope
  · `dev` (pytest, pytest-cov, ruff, mypy)
- `.venv` on 3.12, `.gitignore`, `ruff`/`mypy` config, minimal CI
- `tcrga/__init__.py` exposing `__version__`

**Done when:** `pytest` collects, `ruff check` and `mypy tcrga` are clean, and importing
`tcrga` on a machine without torch raises nothing.

**Risk to retire here:** install `epytope` and probe which of its wrapped predictors actually
run offline. Several ePytope-TCR models require weights obtained separately from their
original publications, and the unified API does not bundle them. Better to know the real
offline inventory now than to discover it in Phase 3. Record the findings in `docs/predictors.md`.

---

## Phase 1 — Data ✅

**Goal:** McPAS on disk becomes a clean, validated, queryable binder set.

| Module | Contents |
|---|---|
| `tcrga/data/mcpas.py` | CSV loader, schema validation, SHA-256 checksum, informative failure on a stale or truncated download |
| `tcrga/data/normalise.py` | `tidytcells` pass over TRAV/TRAJ/TRBV/TRBJ and MHC; CDR3 sanity checks (alphabet, anchors, length 8-24) |
| `tcrga/data/pairing.py` | alpha/beta pairing; explicit policy for the many beta-only McPAS rows |
| `tcrga/data/splits.py` | **peptide-grouped** train/val/test — no epitope may appear on both sides |
| `tcrga/data/binders.py` | `BinderSet(peptide)` → known binders, plus a background/decoy sampler |

**Done when:** `tcrga data summary` prints per-peptide paired-chain counts, and the split
function is unit-tested to guarantee zero epitope overlap. *Both hold.* The real McPAS CSV
is still a manual download, so the suite runs against `tests/fixtures/make_fixture.py`,
which reproduces the export's defects rather than a clean idealisation: latin-1 bytes,
null sentinels in eight spellings, lowercase junctions, nucleotide strings in amino-acid
columns, stripped anchors, duplicate clones and a 60% beta-only skew.

**Note:** McPAS is Shiny-served with a session-gated CSV export, so the download stays manual
(documented in the README). The loader validates; it does not scrape.

---

## Phase 2 — GA core

**Goal:** the engine runs end-to-end against a stub scorer.

| Module | Contents |
|---|---|
| `tcrga/encoding/chain.py` | `ChainPair` dataclass, anchor and alphabet invariants |
| `tcrga/encoding/substitution.py` | BLOSUM62 substitution weights for biased point mutation |
| `tcrga/ga/operators.py` | `PairSampling`, `PairCrossover` (single/two-point + whole-chain swap), `PairMutation` (BLOSUM-weighted substitution, length indels, anchors immutable), `PairDuplicates` |
| `tcrga/ga/problem.py` | `BindingProblem(Problem)` — `n_var=1`, `vtype=object`, `elementwise=False`, `n_ieq_constr=2`, `out["F"] = -P(bind)` |
| `tcrga/ga/run.py` | `minimize()` wrapper, seeding, callbacks, checkpoint/resume, history |
| `tcrga/ga/islands.py` | multi-population wrapper with periodic migration (pymoo has none) |

The proof of concept validating this shape lives at `tests/poc_pymoo_objects.py` and is kept
as a test: it asserts batched evaluation, anchor preservation and constraint feasibility.

**Done when:** a 40 x 25-generation run against a stub scorer completes, respects both
constraints, preserves anchors, and is bit-reproducible from a seed.

---

## Phase 3 — Binding scorers

**Goal:** real `P(bind)` values, calibrated and benchmarked.

```python
class BindingScorer(Protocol):
    def score(self, pairs: Sequence[ChainPair], peptide: str) -> NDArray[np.float64]:
        """Return P(bind) in [0, 1], one value per (alpha, beta) pair."""
```

Torch-free first, in this order:

1. `tcrga/fitness/tcrdist_knn.py` — paired-chain TCRdist to known binders of the target
   peptide; distances → probability via a logistic fitted on held-out binders vs. sampled
   non-binders
2. `tcrga/fitness/pwm.py` — per-peptide positional weight matrices for alpha and beta,
   log-odds against a background repertoire PWM
3. `tcrga/fitness/gbdt.py` — `scikit-learn` gradient boosting over TCRdist + k-mer +
   physicochemical features, trained on peptide-grouped splits

Then constraints and optional neural scorers:

4. `tcrga/fitness/constraints.py` — `olga` generation probability (plausibility) and
   off-target score against a panel of non-target peptides
5. `tcrga/fitness/ensemble.py` — mean / rank aggregation, and cascade evaluation (cheap
   scorer over the population, expensive one over the top fraction)
6. `tcrga/fitness/neural.py` — ePytope-TCR wrapped predictors and ESM-2, behind `[dl]`,
   imported lazily

**Calibration is part of this phase, not a follow-up.** Every scorer ships a reliability
curve and a Brier score on a peptide-grouped held-out split; the GBDT gets Platt or isotonic
wrapping. Uncalibrated scores make the constraint thresholds and the ensemble mean arbitrary.

**Done when:** `tcrga bench` reports AUROC, AUPRC and Brier per scorer on the held-out split,
and the poly-tryptophan regression test fails any scorer that is trivially exploitable.

---

## Phase 4 — Interface, reporting, validation

- `tcrga/cli.py` — `tcrga evolve --peptide SIINFEKL --scorer tcrdist-knn --generations 200`,
  plus `--multi-objective` for the NSGA-II Pareto mode
- `tcrga/viz/` — convergence traces, score distributions, sequence logos (`logomaker`),
  Pareto fronts
- Run manifests: seed, scorer, calibration stats, package versions, McPAS checksum — so any
  reported result is reproducible
- `examples/` — one worked notebook on a well-represented McPAS epitope
- **Sanity validation:** hold out a peptide entirely, evolve against it, and check whether
  known binders of that peptide are recovered or approached. This is the honest test of
  whether the whole pipeline does anything.

---

## Open questions

- **Beta-only rows.** Counts are now in (`tcrga data summary`): paired-only gives 4,988
  rows over 263 epitopes, include-beta-only gives 14,716 over 354. Only 38 epitopes carry
  ≥10 distinct paired receptors either way, so the extra beta-only data widens epitope
  coverage more than it deepens any single epitope. Still to decide, and it affects every
  scorer.
- **Species.** Mouse is 9% of the raw export but 41% of usable *paired* rows, because
  paired sequencing is commoner in mouse work. Mouse receptors engage H-2, not HLA, so a
  single scorer across both fits two recognition problems at once. `--species Human`
  leaves 2,949 rows and 28 usable epitopes. Filtering is available; the default keeps
  both and records the mix.
- **Off-target panel.** Which non-target peptides define specificity — a fixed common panel,
  or peptides sampled per run by similarity to the target?
- **MHC conditioning.** McPAS carries MHC restriction. Ignored in v1; worth revisiting, since
  recognition is of peptide-MHC, not peptide alone.
