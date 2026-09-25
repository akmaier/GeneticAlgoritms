"""pymoo operators for variable-length paired CDR3 chromosomes.

pymoo's built-in operators assume a fixed-length numeric decision vector. A CDR3 pair is
neither fixed-length nor numeric, so we use pymoo's object-typed variable support and
supply the four operators ourselves. Every operator here preserves the encoding
invariants (anchors, alphabet, length bounds) by construction, so an invalid candidate is
never created rather than being created and then penalised.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling

from tcrga.encoding.chain import (
    ALPHA_SPEC,
    AMINO_ACIDS,
    BETA_SPEC,
    ChainPair,
    ChainSpec,
)
from tcrga.encoding.substitution import SubstitutionSampler


def random_cdr3(spec: ChainSpec, rng: np.random.Generator) -> str:
    """Draw a uniformly random but structurally valid CDR3."""
    length = int(rng.integers(spec.min_length, spec.max_length + 1))
    core = "".join(rng.choice(list(AMINO_ACIDS), size=length - 2))
    end = str(rng.choice(list(spec.end_residues)))
    return f"{spec.start_residue}{core}{end}"


class PairSampling(Sampling):
    """Initial population, optionally seeded with known binders.

    Seeding matters more than it might appear. The feasible region of real CDR3 space is a
    vanishing fraction of the random space, and a scorer trained on real repertoires gives
    almost no gradient to random sequences. Starting from known binders of the target
    peptide puts the population somewhere the scorer can actually discriminate.
    """

    def __init__(
        self,
        seeds: list[ChainPair] | None = None,
        seed_fraction: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= seed_fraction <= 1.0:
            raise ValueError("seed_fraction must be in [0, 1]")
        self.seeds = list(seeds or [])
        self.seed_fraction = seed_fraction
        self.rng = rng if rng is not None else np.random.default_rng()

    def _do(self, problem, n_samples: int, **kwargs) -> NDArray[np.object_]:  # type: ignore[no-untyped-def]
        rng = self.rng
        X = np.full((n_samples, 1), None, dtype=object)
        n_seeded = min(int(n_samples * self.seed_fraction), len(self.seeds)) if self.seeds else 0
        if n_seeded:
            picks = rng.choice(len(self.seeds), size=n_seeded, replace=n_seeded > len(self.seeds))
            for i, j in enumerate(picks):
                X[i, 0] = self.seeds[int(j)]
        for i in range(n_seeded, n_samples):
            X[i, 0] = ChainPair(random_cdr3(ALPHA_SPEC, rng), random_cdr3(BETA_SPEC, rng))
        return X


class PairCrossover(Crossover):
    """Two parents, two offspring.

    Two modes, chosen per mating. *Chain swap* exchanges whole alpha/beta chains between
    parents; it is cheap and exploits the fact that the pair is the unit of function while
    each chain is a semi-independent structural module. *Junction crossover* recombines
    within a chain at a point drawn from the variable core, never across the anchors.
    """

    def __init__(
        self,
        prob: float = 0.9,
        chain_swap_prob: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__(n_parents=2, n_offsprings=2, prob=prob)
        self.chain_swap_prob = chain_swap_prob
        self.rng = rng if rng is not None else np.random.default_rng()

    @staticmethod
    def _splice(s1: str, s2: str, rng: np.random.Generator) -> tuple[str, str]:
        """Single-point crossover inside the variable core, anchors untouched."""
        lo, hi = 1, min(len(s1), len(s2)) - 1
        if hi <= lo:
            return s1, s2
        cut = int(rng.integers(lo, hi))
        return s1[:cut] + s2[cut:], s2[:cut] + s1[cut:]

    def _do(self, problem, X: NDArray[np.object_], **kwargs) -> NDArray[np.object_]:  # type: ignore[no-untyped-def]
        rng = self.rng
        _, n_matings, _ = X.shape
        Y = np.full_like(X, None, dtype=object)
        for k in range(n_matings):
            p1, p2 = X[0, k, 0], X[1, k, 0]
            if rng.random() < self.chain_swap_prob:
                Y[0, k, 0] = ChainPair(p1.alpha, p2.beta)
                Y[1, k, 0] = ChainPair(p2.alpha, p1.beta)
            else:
                a1, a2 = self._splice(p1.alpha, p2.alpha, rng)
                b1, b2 = self._splice(p1.beta, p2.beta, rng)
                Y[0, k, 0] = ChainPair(a1, b1)
                Y[1, k, 0] = ChainPair(a2, b2)
        return Y


class PairMutation(Mutation):
    """BLOSUM-weighted substitution plus length-changing indels.

    Indels are what make this a variable-length search: CDR3 length is itself a strong
    determinant of which peptides a receptor can engage, so a search that only substitutes
    residues explores a much smaller space than the biology occupies.
    """

    def __init__(
        self,
        subst_prob: float = 0.15,
        indel_prob: float = 0.05,
        beta: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.subst_prob = subst_prob
        self.indel_prob = indel_prob
        self.sampler = SubstitutionSampler(beta=beta)
        self.rng = rng if rng is not None else np.random.default_rng()

    def _mutate_chain(self, seq: str, spec: ChainSpec, rng: np.random.Generator) -> str:
        core = list(seq[1:-1])
        # Per-residue substitution across the variable core only.
        for i, aa in enumerate(core):
            if rng.random() < self.subst_prob:
                core[i] = self.sampler.sample(aa, rng)
        # Length change, respecting the spec's bounds.
        if core and rng.random() < self.indel_prob:
            length = len(core) + 2
            grow = rng.random() < 0.5
            if grow and length < spec.max_length:
                pos = int(rng.integers(0, len(core) + 1))
                core.insert(pos, str(rng.choice(list(AMINO_ACIDS))))
            elif not grow and length > spec.min_length:
                core.pop(int(rng.integers(0, len(core))))
        return f"{seq[0]}{''.join(core)}{seq[-1]}"

    def _do(self, problem, X: NDArray[np.object_], **kwargs) -> NDArray[np.object_]:  # type: ignore[no-untyped-def]
        rng = self.rng
        for i in range(len(X)):
            p = X[i, 0]
            X[i, 0] = ChainPair(
                self._mutate_chain(p.alpha, ALPHA_SPEC, rng),
                self._mutate_chain(p.beta, BETA_SPEC, rng),
            )
        return X


class PairDuplicates(ElementwiseDuplicateElimination):
    """Keeps the population from collapsing onto one sequence.

    Without this a GA with strong selection converges to many copies of a single
    individual, which wastes the entire evaluation budget re-scoring the same candidate.
    """

    def is_equal(self, a, b) -> bool:  # type: ignore[no-untyped-def]
        return a.X[0] == b.X[0]
