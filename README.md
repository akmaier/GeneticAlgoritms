# GeneticAlgoritms

**Genetic algorithms for optimising T-cell receptor (TCR) amino-acid sequences.**

Given a target peptide (epitope), this project evolves paired TCR **alpha** and **beta**
chain CDR3 sequences to **maximise the predicted binding probability between the (alpha, beta)
chain pair and that peptide**, subject to the designs staying biologically plausible and
free of predicted off-target binding.

Research project, [Friedrich-Alexander-Universität Erlangen-Nürnberg](https://www.fau.eu).

> **Status: early / pre-implementation.** The plan and environment analysis below are
> settled; the package itself is being built. Expect the API to change.

---

## Why

Designing a TCR that recognises a chosen peptide-MHC complex is a search problem over a
combinatorial sequence space: two chains, ~10-18 residues of hypervariable CDR3 each, over
a 20-letter alphabet. Gradient-based methods need a differentiable objective, but the most
useful scoring functions available today (structural metrics, repertoire statistics, wrapped
pre-trained predictors) are black boxes. Genetic algorithms handle exactly that setting.

### The fitness function

**The fitness is the predicted binding probability, and it dominates.** Everything else is a
guard rail, not a co-equal goal:

| Term | Role | Direction |
|---|---|---|
| **P(bind \| CDR3-alpha, CDR3-beta, peptide)** | **the objective** | **maximise** |
| Biological plausibility (generation probability) | feasibility constraint | above threshold |
| Predicted off-target / cross-reactive binding | feasibility constraint | below threshold |

Ranking candidates by predicted binding probability is the whole point, so the default run is
**single-objective**: maximise `P(bind)` over the feasible set. The two guard-rail terms enter
as constraints — a candidate that violates one is penalised into infeasibility rather than
being allowed to trade binding away against it. This keeps selection pressure pointed at
binding while still stopping the search from drifting into sequences no thymus could produce.

Crucially, `P(bind)` is a **paired-chain** score. Alpha and beta contribute jointly and
non-additively to peptide recognition, so the scorer takes the pair and the peptide together;
it is never the sum of two independent per-chain scores. The chromosome is therefore the pair,
and crossover can swap whole chains between individuals to exploit that structure.

A `--multi-objective` mode is still available, which promotes the two constraints to
objectives and returns an NSGA-II Pareto front — useful for inspecting how much binding a
given plausibility level costs, but not the default.

---

## Data source

[**McPAS-TCR**](https://friedmanlab.weizmann.ac.il/McPAS-TCR/) — a manually curated catalogue
of pathology-associated TCR sequences (last updated 2022-09-10). It supplies paired
CDR3-alpha / CDR3-beta sequences together with TRAV/TRAJ/TRBV/TRBJ gene usage, the antigen
protein, the epitope peptide, MHC restriction and CD4/CD8 lineage — which is precisely the
(TCR, peptide) supervision this project needs.

**The database is not scriptable.** McPAS-TCR is served by a Shiny application and its CSV
export is behind a per-session token, so there is no stable download URL. Fetch it manually:

1. Open <https://friedmanlab.weizmann.ac.il/McPAS-TCR/>
2. Click **Search** with an empty query to return the full database
3. Choose **Download complete database**, and save the CSV to `data/raw/McPAS-TCR.csv`

The loader validates the schema and records a checksum, so a stale or truncated file is
caught early rather than silently degrading the fitness landscape.

**If you use this project, cite McPAS-TCR:** Tickotsky N, Sagiv T, Prilusky J, Shifrut E,
Friedman N (2017). *McPAS-TCR: A manually-curated catalogue of pathology-associated T cell
receptor sequences.* Bioinformatics **33**:2924-2929.

---

## Requirements

**Python 3.12.** This is a deliberate pin, not an accident. PyTorch stopped publishing
macOS-x86_64 wheels after **torch 2.2.2**, which supports Python up to 3.12 — and every
deep-learning TCR-specificity predictor (ERGO-II, NetTCR, pMTnet, TITAN, DeepTCR, ESM-2)
is built on PyTorch. On an Intel Mac, Python 3.13 silently cuts you off from all of them.
On Apple Silicon or Linux, newer Python and newer torch are both fine.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[core,tcr]"        # torch-free: the GA and the default scorers
pip install -e ".[core,tcr,dl]"     # adds the neural predictors
```

The GA core never imports torch. Deep-learning scorers live behind the `dl` extra and are
loaded lazily, so the package stays usable on a machine that cannot install PyTorch at all.

### Dependency map

| Layer | Packages |
|---|---|
| **Evolutionary engine** | [`pymoo`](https://pymoo.org) 0.6.2 — constrained single-objective `GA` by default, `NSGA2` for the multi-objective mode |
| **TCR scoring** | [`tcrdist3`](https://github.com/kmayerb/tcrdist3) (paired-chain TCR distance), [`epytope`](https://github.com/KohlbacherLab/epytope) + ePytope-TCR (unified interface to 21 pre-trained specificity predictors) |
| **Repertoire realism** | [`olga`](https://pypi.org/project/olga/) / [`sonia`](https://pypi.org/project/sonia/) (V(D)J generation probability), [`stitchr`](https://pypi.org/project/stitchr/) (CDR3 to full-length chain) |
| **Normalisation** | [`tidytcells`](https://tidytcells.readthedocs.io) — McPAS gene and MHC nomenclature is inconsistent; this is not optional |
| **Descriptors** | [`peptides`](https://pypi.org/project/peptides/), `biopython`, `propy3` |
| **Optional, `[dl]`** | `torch==2.2.2`, `fair-esm` |

---

## Design

```
tcrga/
  data/      McPAS loader, tidytcells normalisation, alpha/beta pairing,
             peptide-grouped train/test splits
  encoding/  amino-acid alphabet, CDR3 anchor constraints (leading C, trailing F/W),
             length bounds, BLOSUM62 substitution weights
  ga/        pymoo operators over variable-length paired chromosomes:
             PairSampling / PairCrossover / PairMutation / PairDuplicates,
             plus an island-model wrapper (pymoo has no built-in one)
  fitness/   Scorer protocol + composable multi-objective scorers
  viz/       Pareto fronts, convergence traces, sequence logos (logomaker)
  cli.py
```

The central abstraction is a **`Scorer` protocol**: it maps a batch of candidate
`(CDR3-alpha, CDR3-beta)` pairs plus the target peptide to a **binding probability in [0, 1]**.
One signature, many implementations, so the GA never knows which predictor is underneath and
scorers can be swapped or ensembled without touching the search.

```python
class BindingScorer(Protocol):
    def score(self, pairs: Sequence[ChainPair], peptide: str) -> NDArray[np.float64]:
        """Return P(bind) in [0, 1], one value per (alpha, beta) pair."""
```

Because every scorer returns a probability on the same scale, an **ensemble** is just a mean
(or rank-aggregation) over scorers — and a run can cascade: score the whole population with a
cheap scorer, then re-score only the top fraction with an expensive neural one.

**Default, torch-free scorers**, so the project is useful on day one:

| Scorer | How it yields a probability |
|---|---|
| **TCRdist kNN** | paired-chain TCRdist to McPAS TCRs annotated against the target peptide; distances mapped to a probability by a logistic fitted on held-out binders vs. sampled non-binders |
| **Positional PWM** | log-odds of the pair under per-peptide PWMs (alpha and beta) against a background repertoire PWM, squashed to [0, 1] |
| **Gradient-boosted classifier** | `scikit-learn` on TCRdist + k-mer + physicochemical features, trained on McPAS with **peptide-grouped** splits so no epitope leaks between train and test |

**Optional neural scorers** behind `[dl]`: ePytope-TCR's wrapped pre-trained predictors
(ERGO-II, NetTCR and others already emit a binding probability directly) and ESM-2 embeddings
as a learned sequence prior.

**Calibration is part of the deliverable, not an afterthought.** A GA will ruthlessly exploit
a miscalibrated score, so each scorer ships with a reliability curve and a Brier score on a
peptide-grouped held-out split, and the classifier is wrapped in Platt scaling / isotonic
regression. A score that is merely monotone in binding is enough to rank, but a calibrated one
is what makes the constraint thresholds and the ensemble mean meaningful.

---

## The evolutionary engine

`pymoo` 0.6.2, used through its **custom variable type** API rather than its numeric one. A
CDR3 pair is not a fixed-length float vector, so the chromosome is a Python object:

```python
@dataclass
class ChainPair:
    alpha: str   # e.g. CAVRDSNYQLIW
    beta:  str   # e.g. CASSLGQAYEQYF
```

held in a `Problem(n_var=1, vtype=object)`. Three properties of pymoo made this the right
choice over a hand-rolled loop, and all three are verified working on this stack:

- **Variable-length genomes.** Object-typed decision variables carry whatever we want, so
  alpha and beta can differ in length and mutate in length, with custom `Sampling`,
  `Crossover`, `Mutation` and `ElementwiseDuplicateElimination` subclasses enforcing the
  anchor constraints (leading `C`, trailing `F`/`W`).
- **Batched evaluation.** With `elementwise=False` the entire population reaches `_evaluate`
  in a single call, which is what makes a neural `BindingScorer` affordable — one forward
  pass per generation instead of one per individual.
- **Native constraint handling.** `n_ieq_constr` with feasibility-first tournament selection
  implements the guard-rail design directly: plausibility and off-target enter as `out["G"]`
  and an infeasible candidate loses to any feasible one regardless of its binding score.

Two consequences to keep in mind. **pymoo minimises**, so the objective is `-P(bind)`. And
pymoo has **no island model**, so the multi-population/migration layer is ours to write as a
wrapper around repeated `minimize()` calls.

---

## Scope and honest limits

This is a **sequence-design and search** tool. It optimises against *predicted* binding, and
published TCR-epitope predictors generalise poorly to unseen epitopes — a known and
well-documented weakness of the field. Treat the output as a prioritised, diverse shortlist
of hypotheses for experimental validation, not as validated binders.

Optimising hard on a single predicted score makes this sharper, not softer: a GA will happily
find adversarial sequences that score well and bind nothing. This is not hypothetical — a
throwaway scorer that rewarded tryptophan count produced `CKYIWWWWWAVAWHMF` at fitness 1.000
within 25 generations during engine evaluation. That case is kept as a regression test. The plausibility and off-target
constraints, the calibration checks, and scorer ensembling are the three defences against
that, and runs should be reported with the scorer and its held-out calibration stated.

---

## Licence

See [LICENSE](LICENSE).
