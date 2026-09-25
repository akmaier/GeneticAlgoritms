"""Running an evolution and reporting it reproducibly."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from tcrga import __version__
from tcrga.encoding.chain import ChainPair
from tcrga.fitness.base import BindingScorer, Constraint
from tcrga.ga.operators import PairCrossover, PairDuplicates, PairMutation, PairSampling
from tcrga.ga.problem import BindingProblem


class _History(Callback):
    """Records per-generation statistics for convergence plots and sanity checks."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[dict[str, float]] = []

    def notify(self, algorithm) -> None:  # type: ignore[no-untyped-def]
        pop = algorithm.pop
        f = -pop.get("F")[:, 0]  # back to P(bind)
        cv = pop.get("CV")[:, 0] if pop.get("CV") is not None else np.zeros(len(f))
        feasible = cv <= 0
        self.rows.append(
            {
                "generation": int(algorithm.n_gen),
                "best": float(f.max()),
                "mean": float(f.mean()),
                "median": float(np.median(f)),
                "feasible_frac": float(feasible.mean()),
                "unique": float(len({(p.alpha, p.beta) for p in pop.get("X")[:, 0]})),
            }
        )


@dataclass
class EvolutionResult:
    """Everything needed to report, plot, or reproduce a run."""

    peptide: str
    best: ChainPair
    best_score: float
    population: list[ChainPair]
    scores: np.ndarray
    history: pd.DataFrame
    manifest: dict[str, object] = field(default_factory=dict)

    def top(self, n: int = 10) -> pd.DataFrame:
        """The n best candidates, highest predicted binding first."""
        order = np.argsort(-self.scores)[:n]
        return pd.DataFrame(
            {
                "rank": np.arange(1, len(order) + 1),
                "alpha": [self.population[i].alpha for i in order],
                "beta": [self.population[i].beta for i in order],
                "p_bind": self.scores[order],
            }
        )


def evolve(
    scorer: BindingScorer,
    peptide: str,
    *,
    constraints: Sequence[Constraint] = (),
    seeds: Sequence[ChainPair] | None = None,
    pop_size: int = 100,
    n_gen: int = 50,
    seed: int = 0,
    subst_prob: float = 0.15,
    indel_prob: float = 0.05,
    blosum_beta: float = 0.5,
    seed_fraction: float = 0.5,
    verbose: bool = False,
) -> EvolutionResult:
    """Evolve chain pairs maximising `scorer` against `peptide`.

    The run is deterministic given `seed`: the same arguments reproduce the same
    population, which is why the seed and package versions are recorded in the manifest.
    """
    problem = BindingProblem(scorer, peptide, constraints)
    # Each operator gets its own independent, reproducibly-derived stream. pymoo does not
    # pass a seed down to operators, so owning the generators here is what makes a run
    # reproducible; `minimize(seed=...)` only covers pymoo's own selection randomness.
    samp_seed, cx_seed, mut_seed = np.random.SeedSequence(seed).spawn(3)
    algorithm = GA(
        pop_size=pop_size,
        sampling=PairSampling(
            list(seeds or []),
            seed_fraction=seed_fraction,
            rng=np.random.default_rng(samp_seed),
        ),
        crossover=PairCrossover(rng=np.random.default_rng(cx_seed)),
        mutation=PairMutation(
            subst_prob=subst_prob,
            indel_prob=indel_prob,
            beta=blosum_beta,
            rng=np.random.default_rng(mut_seed),
        ),
        eliminate_duplicates=PairDuplicates(),
    )
    history = _History()
    started = time.time()
    res = minimize(
        problem,
        algorithm,
        get_termination("n_gen", n_gen),
        seed=seed,
        callback=history,
        save_history=False,
        verbose=verbose,
    )
    elapsed = time.time() - started

    pop = [row[0] for row in res.pop.get("X")]
    scores = -res.pop.get("F")[:, 0]
    best_idx = int(np.argmax(scores))

    manifest: dict[str, object] = {
        "tcrga_version": __version__,
        "peptide": peptide,
        "scorer": scorer.name,
        "scorer_calibrated": getattr(scorer, "calibrated", False),
        "constraints": [c.name for c in constraints],
        "pop_size": pop_size,
        "n_gen": n_gen,
        "seed": seed,
        "subst_prob": subst_prob,
        "indel_prob": indel_prob,
        "blosum_beta": blosum_beta,
        "evaluations": problem.n_evaluated,
        "cache_hits": problem.n_cache_hits,
        "scorer_batches": problem.n_batches,
        "elapsed_seconds": round(elapsed, 3),
    }
    return EvolutionResult(
        peptide=peptide,
        best=pop[best_idx],
        best_score=float(scores[best_idx]),
        population=pop,
        scores=scores,
        history=pd.DataFrame(history.rows),
        manifest=manifest,
    )
