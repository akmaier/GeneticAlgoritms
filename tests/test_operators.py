import numpy as np

from tcrga.encoding.chain import ALPHA_SPEC, BETA_SPEC, ChainPair
from tcrga.ga.operators import PairCrossover, PairMutation, PairSampling, random_cdr3


def _pop(n=40, seed=0):
    return PairSampling()._do(None, n, seed=seed)


def test_sampling_produces_only_valid_pairs():
    assert all(x[0].is_valid for x in _pop(200))


def test_mutation_preserves_invariants_under_pressure():
    """High mutation rates must still never produce an invalid CDR3."""
    X = _pop(200, seed=3)
    mut = PairMutation(subst_prob=0.9, indel_prob=0.9)
    for gen in range(30):
        X = mut._do(None, X, seed=gen)
        assert all(x[0].is_valid for x in X), f"invalid chain at generation {gen}"


def test_mutation_changes_length():
    X = _pop(200, seed=4)
    before = [x[0].total_length for x in X]
    Y = PairMutation(subst_prob=0.0, indel_prob=1.0)._do(None, X.copy(), seed=5)
    after = [y[0].total_length for y in Y]
    assert before != after


def test_crossover_preserves_invariants():
    X = _pop(64, seed=6)
    parents = np.array([X[:32], X[32:]], dtype=object)
    Y = PairCrossover()._do(None, parents, seed=7)
    assert all(Y[i, k, 0].is_valid for i in range(2) for k in range(32))


def test_chain_swap_recombines_whole_chains():
    p1 = ChainPair("CAAAAAAAAAW", "CBBBBBBBBBF".replace("B", "S"))
    p2 = ChainPair("CDDDDDDDDDW", "CEEEEEEEEEF")
    parents = np.array([[[p1]], [[p2]]], dtype=object)
    Y = PairCrossover(chain_swap_prob=1.0)._do(None, parents, seed=0)
    assert Y[0, 0, 0] == ChainPair(p1.alpha, p2.beta)
    assert Y[1, 0, 0] == ChainPair(p2.alpha, p1.beta)


def test_random_cdr3_respects_spec_bounds():
    rng = np.random.default_rng(0)
    for spec in (ALPHA_SPEC, BETA_SPEC):
        for _ in range(200):
            s = random_cdr3(spec, rng)
            assert spec.min_length <= len(s) <= spec.max_length
            assert s[0] == spec.start_residue and s[-1] in spec.end_residues
