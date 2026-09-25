"""The scorer contract.

Everything the GA optimises flows through `BindingScorer`. Two properties of this
interface are load-bearing and should not be weakened:

*Batched.* `score` takes the whole population at once. A neural predictor costs one
forward pass per generation this way instead of one per individual, and the difference is
two orders of magnitude on realistic population sizes.

*Calibrated.* The return value is a probability in [0, 1], not an arbitrary score. This is
what makes ensembling a mean rather than a guess, and what gives the constraint thresholds
a fixed meaning. A scorer that is only monotone in binding can rank, but it cannot be
mixed with others or thresholded, so `calibrated` records which kind it is.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from tcrga.encoding.chain import ChainPair


@runtime_checkable
class BindingScorer(Protocol):
    """Maps candidate chain pairs plus a target peptide to P(bind)."""

    name: str
    calibrated: bool

    def score(self, pairs: Sequence[ChainPair], peptide: str) -> NDArray[np.float64]:
        """Return one probability in [0, 1] per pair, in the order given."""
        ...


class ConstraintResult:
    """Constraint values in pymoo's convention: <= 0 is feasible."""

    __slots__ = ("values", "name")

    def __init__(self, name: str, values: NDArray[np.float64]) -> None:
        self.name = name
        self.values = values

    @property
    def feasible(self) -> NDArray[np.bool_]:
        return self.values <= 0


@runtime_checkable
class Constraint(Protocol):
    """A guard rail. Violation makes a candidate infeasible regardless of its binding score."""

    name: str

    def evaluate(self, pairs: Sequence[ChainPair], peptide: str) -> NDArray[np.float64]:
        """Return g(x) per pair, where g(x) <= 0 means feasible."""
        ...


def check_probabilities(values: NDArray[np.float64], scorer_name: str) -> NDArray[np.float64]:
    """Fail loudly on a scorer that breaks its contract.

    A scorer silently returning NaN or a value outside [0, 1] corrupts selection in a way
    that looks like a converging run, so this is checked on every call rather than in tests.
    """
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{scorer_name}: returned non-finite probabilities")
    if np.any(values < 0.0) or np.any(values > 1.0):
        lo, hi = float(values.min()), float(values.max())
        raise ValueError(f"{scorer_name}: probabilities outside [0, 1] (got [{lo:.3f}, {hi:.3f}])")
    return values
