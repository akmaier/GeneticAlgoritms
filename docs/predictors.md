# TCR specificity predictors: what is actually usable offline

Findings from the Phase 0 risk-retirement probe, 2026-09-25.

## ePytope-TCR is not on PyPI

`pip install epytope` gives **epytope 4.0.1 with no TCR module**. Its submodules are
`CleavagePrediction, Core, Data, EpitopeAssembly, EpitopePrediction, EpitopeSelection,
HLAtyping, IO, TAPPrediction` — no `TCRSpecificityPrediction`. The same is true of
`KohlbacherLab/epytope` on both `main` and `develop`.

ePytope-TCR lives **only** in the fork used for the benchmark paper:

- Framework: <https://github.com/SchubertLab/epytope> → `epytope/TCRSpecificityPrediction/`
- Benchmark study: <https://github.com/SchubertLab/benchmark_TCRprediction>

## The 22 wrapped predictors

`External.py` (17) — `Ergo1`, `Ergo2`, `pMTnet`, `EpiTCR`, `ATM_TCR`, `AttnTAP`, `TEIM`,
`BERTrand`, `TEINet`, `PanPep`, `DLpTCR`, `TULIP`, `iTCep`, `NetTCR22`, `MixTCRpred`, `TCRGP`

`ML.py` (5) — `ImRex`, `TITAN`, `TCellMatch`, `STAPLER`

## The catch: none of them are self-contained

Every `External` predictor inherits `ARepoTCRSpecificityPrediction`, which **requires the user
to clone that predictor's own repository** and pass its path:

```
raise AttributeError("Please provide 'repository' as a input argument to predict ...
                      'git clone {self.repo}'")
```

Execution then shells out through `conda run -n <env> <interpreter> <cmd>` — each predictor
runs in **its own conda environment**, which is why the benchmark repo ships `docker/envs/`.
Model weights are likewise obtained per-predictor from the original publications; nothing is
bundled.

**This machine has no conda installed.**

## Consequences for `tcrga`

1. **The torch-free scorers are not a fallback, they are the foundation.** TCRdist kNN, the
   per-peptide PWM and the GBDT classifier are the only scorers that will run here without a
   multi-day environment build. Phase 3 order stands.
2. **Subprocess-per-individual is architecturally impossible** in a GA inner loop — conda
   activation plus process spawn per candidate, per generation. If a neural predictor is ever
   wired in, it must be through the **cascade** path: one batched call per generation over the
   top-K survivors only, not per individual. The `BindingScorer` batch signature already
   allows this; `ensemble.py` must enforce it.
3. **The `dl` extra stays optional and unimported.** Do not add ePytope-TCR as a dependency.
   If needed later, vendor a thin adapter behind the existing `BindingScorer` protocol and
   document the per-predictor setup separately.
