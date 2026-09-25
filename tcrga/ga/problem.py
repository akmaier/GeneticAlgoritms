"""The pymoo problem: binding probability as the objective, guard rails as constraints.

Two conventions of pymoo shape this file.

*pymoo minimises.* The objective is therefore `-P(bind)`, and every reported fitness is
negated back before it reaches a user.

*Constraints are `g(x) <= 0`.* pymoo's feasibility-first tournament selection then does
exactly what the design calls for: an infeasible candidate loses to any feasible one no
matter how well it binds, so plausibility and specificity cannot be traded away for
binding score. They are guard rails, not competing objectives.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from pymoo.core.problem import Problem

from tcrga.encoding.chain import ChainPair
from tcrga.fitness.base import BindingScorer, Constraint, check_probabilities


class BindingProblem(Problem):
    """Maximise predicted binding to `peptide` over structurally valid chain pairs.

    Evaluation is batched: pymoo hands the whole population to `_evaluate` in one call
    (`elementwise=False`), so a scorer sees the generation as a single batch. This is the
    difference between one neural forward pass per generation and one per individual.
    """

    def __init__(
        self,
        scorer: BindingScorer,
        peptide: str,
        constraints: Sequence[Constraint] = (),
        *,
        cache: bool = True,
    ) -> None:
        super().__init__(
            n_var=1,
            n_obj=1,
            n_ieq_constr=len(constraints),
            vtype=object,
        )
        self.scorer = scorer
        self.peptide = peptide
        self.constraints = list(constraints)
        self._cache: dict[ChainPair, float] | None = {} if cache else None
        # Bookkeeping, so a run can report how much work was actually done.
        self.n_evaluated = 0
        self.n_cache_hits = 0
        self.n_batches = 0

    def _score(self, pairs: list[ChainPair]) -> NDArray[np.float64]:
        """Score a batch, reusing cached values for pairs already seen."""
        if self._cache is None:
            self.n_evaluated += len(pairs)
            self.n_batches += 1
            return check_probabilities(
                np.asarray(self.scorer.score(pairs, self.peptide), dtype=np.float64),
                self.scorer.name,
            )

        out = np.empty(len(pairs), dtype=np.float64)
        # Unique unseen pairs only: a population routinely holds repeats, and scoring the
        # same candidate twice in one generation is pure waste when the scorer is a model.
        todo: dict[ChainPair, list[int]] = {}
        for i, p in enumerate(pairs):
            hit = self._cache.get(p)
            if hit is not None:
                out[i] = hit
                self.n_cache_hits += 1
            elif p in todo:
                todo[p].append(i)
                self.n_cache_hits += 1
            else:
                todo[p] = [i]
        if todo:
            unique = list(todo)
            fresh = check_probabilities(
                np.asarray(self.scorer.score(unique, self.peptide), dtype=np.float64),
                self.scorer.name,
            )
            self.n_evaluated += len(unique)
            self.n_batches += 1
            for p, v in zip(unique, fresh, strict=True):
                self._cache[p] = float(v)
                for i in todo[p]:
                    out[i] = v
        return out

    def _evaluate(self, X: NDArray[np.object_], out: dict, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        pairs = [row[0] for row in X]
        pbind = self._score(pairs)
        out["F"] = -pbind.reshape(-1, 1)  # pymoo minimises
        if self.constraints:
            g = np.column_stack(
                [
                    np.asarray(c.evaluate(pairs, self.peptide), dtype=np.float64)
                    for c in self.constraints
                ]
            )
            out["G"] = g
