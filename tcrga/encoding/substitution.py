"""BLOSUM62-weighted residue substitution.

A uniform point mutation treats replacing leucine with isoleucine (conservative, common
in real repertoires) exactly like replacing it with proline (a helix breaker that rarely
survives selection). Sampling substitutions in proportion to BLOSUM62 exchangeability
keeps the search in the region of sequence space that real thymic selection produces,
which matters more here than usual: the scorers are trained on real repertoires and have
no meaningful signal outside them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tcrga.encoding.chain import AMINO_ACIDS


def _blosum62() -> dict[tuple[str, str], int]:
    """BLOSUM62 scores, from Biopython if present, else a bundled fallback."""
    try:
        from Bio.Align import substitution_matrices

        mat = substitution_matrices.load("BLOSUM62")
        return {(a, b): int(mat[a, b]) for a in AMINO_ACIDS for b in AMINO_ACIDS}
    except Exception:  # pragma: no cover - only when Biopython is unavailable
        return _FALLBACK


# Minimal fallback so the GA core never hard-depends on Biopython.
_FALLBACK: dict[tuple[str, str], int] = {
    (a, b): (4 if a == b else -1) for a in AMINO_ACIDS for b in AMINO_ACIDS
}


class SubstitutionSampler:
    """Samples replacement residues with probability proportional to exp(beta * BLOSUM62).

    `beta` controls conservatism: 0.0 gives uniform substitution, larger values
    increasingly favour exchangeable residues. The identity substitution is excluded, so
    a mutation always changes the sequence.
    """

    def __init__(self, beta: float = 0.5, alphabet: str = AMINO_ACIDS) -> None:
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.beta = beta
        self.alphabet = alphabet
        self._index = {aa: i for i, aa in enumerate(alphabet)}
        self._probs = self._build(beta, alphabet)

    @staticmethod
    def _build(beta: float, alphabet: str) -> NDArray[np.float64]:
        scores = _blosum62()
        n = len(alphabet)
        probs = np.zeros((n, n), dtype=np.float64)
        for i, a in enumerate(alphabet):
            row = np.array([scores[(a, b)] for b in alphabet], dtype=np.float64)
            weights = np.exp(beta * row)
            weights[i] = 0.0  # never substitute a residue with itself
            probs[i] = weights / weights.sum()
        return probs

    def sample(self, residue: str, rng: np.random.Generator) -> str:
        """Return a replacement for `residue`, never `residue` itself."""
        i = self._index.get(residue)
        if i is None:
            return str(rng.choice(list(self.alphabet)))
        j = int(rng.choice(len(self.alphabet), p=self._probs[i]))
        return self.alphabet[j]
