from collections import Counter

import numpy as np

from tcrga.encoding.substitution import SubstitutionSampler


def test_never_substitutes_identity():
    rng = np.random.default_rng(0)
    s = SubstitutionSampler(beta=0.5)
    assert all(s.sample("L", rng) != "L" for _ in range(500))


def test_blosum_weighting_prefers_conservative_swaps():
    """Leucine should go to I/M/V far more often than to proline."""
    rng = np.random.default_rng(0)
    counts = Counter(SubstitutionSampler(beta=0.5).sample("L", rng) for _ in range(4000))
    conservative = counts["I"] + counts["M"] + counts["V"]
    assert conservative > 4 * counts["P"]


def test_beta_zero_is_uniform():
    rng = np.random.default_rng(0)
    counts = Counter(SubstitutionSampler(beta=0.0).sample("L", rng) for _ in range(6000))
    freqs = np.array([counts[a] for a in "ACDEFGHIKMNPQRSTVWY"], dtype=float)
    freqs /= freqs.sum()
    assert freqs.std() < 0.01
