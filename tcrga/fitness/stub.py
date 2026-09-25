"""Placeholder scorers for exercising the engine before real ones exist (Phase 3).

`MotifScorer` is deliberately naive and deliberately exploitable. It exists so the GA can
be tested end to end, and so the adversarial-exploitation failure mode stays visible: a
scorer that rewards a residue count will be driven to a homopolymer within a few dozen
generations. That behaviour is asserted in the test suite rather than treated as a bug,
because it is the thing the real scorers and the feasibility constraints must defend
against.

Nothing here is a model of binding. Do not report results from it.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from tcrga.encoding.chain import ChainPair
from tcrga.fitness.base import check_probabilities


class MotifScorer:
    """Rewards simple residue-composition similarity between the pair and the peptide.

    A stand-in with the right *shape* (batched, returns [0, 1]) and no biological content.
    """

    def __init__(self, name: str = "motif-stub") -> None:
        self.name = name
        self.calibrated = False

    def score(self, pairs: Sequence[ChainPair], peptide: str) -> NDArray[np.float64]:
        target = set(peptide)
        if not target:
            return np.zeros(len(pairs), dtype=np.float64)
        out = np.empty(len(pairs), dtype=np.float64)
        for i, p in enumerate(pairs):
            joined = p.alpha[1:-1] + p.beta[1:-1]  # ignore the invariant anchors
            shared = sum(1 for aa in joined if aa in target)
            out[i] = shared / max(len(joined), 1)
        return check_probabilities(out, self.name)


class LengthConstraint:
    """Total junction length must stay inside a plausible window.

    A placeholder for the real plausibility constraint (`olga` generation probability),
    kept because an unconstrained GA will otherwise drift to the length bounds.
    """

    def __init__(self, max_total: int = 34, min_total: int = 18) -> None:
        if min_total > max_total:
            raise ValueError("min_total must not exceed max_total")
        self.name = "total-length"
        self.max_total = max_total
        self.min_total = min_total

    def evaluate(self, pairs: Sequence[ChainPair], peptide: str) -> NDArray[np.float64]:
        lengths = np.array([p.total_length for p in pairs], dtype=np.float64)
        # Violation in either direction, expressed as a single g(x) <= 0 value.
        return np.maximum(lengths - self.max_total, self.min_total - lengths)
