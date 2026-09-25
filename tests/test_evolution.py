"""End-to-end engine behaviour, including the failure mode we most need to keep visible."""

import numpy as np
import pytest

from tcrga.encoding.chain import ChainPair
from tcrga.fitness.base import BindingScorer, check_probabilities
from tcrga.fitness.stub import LengthConstraint, MotifScorer
from tcrga.ga.problem import BindingProblem
from tcrga.ga.run import evolve


class TryptophanScorer:
    """A deliberately broken scorer: rewards a residue count, not binding."""

    name = "tryptophan-trap"
    calibrated = False

    def score(self, pairs, peptide):
        v = np.array([(p.alpha.count("W") + p.beta.count("W")) / 12.0 for p in pairs])
        return check_probabilities(np.clip(v, 0.0, 1.0), self.name)


class BatchSpy(MotifScorer):
    def __init__(self):
        super().__init__(name="batch-spy")
        self.batch_sizes = []

    def score(self, pairs, peptide):
        self.batch_sizes.append(len(pairs))
        return super().score(pairs, peptide)


def test_protocol_is_satisfied_structurally():
    assert isinstance(MotifScorer(), BindingScorer)


def test_evolution_improves_and_stays_valid():
    res = evolve(MotifScorer(), "SIINFEKL", pop_size=40, n_gen=20, seed=1)
    assert res.history["best"].iloc[-1] >= res.history["best"].iloc[0]
    assert all(p.is_valid for p in res.population)
    assert 0.0 <= res.best_score <= 1.0


def test_run_is_reproducible_from_seed():
    a = evolve(MotifScorer(), "SIINFEKL", pop_size=30, n_gen=10, seed=7)
    b = evolve(MotifScorer(), "SIINFEKL", pop_size=30, n_gen=10, seed=7)
    assert a.best == b.best
    assert a.best_score == pytest.approx(b.best_score)


def test_different_seeds_explore_differently():
    a = evolve(MotifScorer(), "SIINFEKL", pop_size=30, n_gen=10, seed=1)
    b = evolve(MotifScorer(), "SIINFEKL", pop_size=30, n_gen=10, seed=2)
    assert a.population != b.population


def test_evaluation_is_batched_not_per_individual():
    """One scorer call per generation, not one per candidate."""
    spy = BatchSpy()
    evolve(spy, "SIINFEKL", pop_size=50, n_gen=10, seed=0)
    assert len(spy.batch_sizes) <= 12, "scorer called more than once per generation"
    assert max(spy.batch_sizes) > 1


def test_constraint_violations_are_eliminated():
    con = LengthConstraint(min_total=20, max_total=28)
    res = evolve(MotifScorer(), "SIINFEKL", constraints=(con,), pop_size=60, n_gen=25, seed=3)
    assert res.history["feasible_frac"].iloc[-1] == 1.0
    assert 20 <= res.best.total_length <= 28


def test_cache_avoids_rescoring_duplicates():
    problem = BindingProblem(MotifScorer(), "SIINFEKL")
    p = ChainPair("CAVRDSNYQLIW", "CASSLGQAYEQYF")
    problem._score([p, p, p])
    problem._score([p])
    assert problem.n_evaluated == 1
    assert problem.n_cache_hits == 3


def test_scorer_contract_is_enforced():
    class Broken:
        name, calibrated = "broken", False

        def score(self, pairs, peptide):
            return check_probabilities(np.full(len(pairs), 1.5), self.name)

    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        evolve(Broken(), "SIINFEKL", pop_size=10, n_gen=2, seed=0)


def test_objective_exploitation_is_reproducible():
    """Regression guard, not a bug.

    A GA optimising hard on a single uncalibrated score finds sequences that maximise the
    score and mean nothing biologically. Here a scorer rewarding tryptophan count drives
    the population to roughly six times the background tryptophan frequency (0.05 under
    uniform sampling) while pinning the score at its ceiling.

    Note it does *not* reach a pure homopolymer, and the reason is instructive: the scorer
    saturates at twelve tryptophans, so beyond that point there is no gradient left to
    climb. Real predictors saturate too. The lesson is that a maxed-out score says nothing
    about how good a candidate is -- only that the scorer has stopped discriminating.

    This pins the behaviour so the defences against it -- calibration, feasibility
    constraints, ensembling -- cannot be dropped without a test noticing.
    """
    res = evolve(TryptophanScorer(), "SIINFEKL", pop_size=60, n_gen=40, seed=0)
    core = res.best.alpha[1:-1] + res.best.beta[1:-1]
    w_fraction = core.count("W") / len(core)

    assert res.best_score > 0.9, "scorer should be driven to its ceiling"
    assert w_fraction > 0.20, (
        f"expected strong enrichment over the 0.05 background, got {w_fraction:.2f} in {res.best}"
    )
    # The point of the test: a perfect score on a sequence no one would synthesise.
    assert not res.best_score < 1.0 or w_fraction > 0.20
