import numpy as np
import pytest

from tcrga.data import build_binder_sets, peptide_grouped_split


def test_no_epitope_appears_in_two_splits(clean_frame):
    """The invariant this module exists to guarantee."""
    splits = peptide_grouped_split(clean_frame, test_size=0.3, seed=0)
    train = set(splits["train"]["peptide"])
    test = set(splits["test"]["peptide"])
    assert train and test
    assert train.isdisjoint(test)


def test_three_way_split_is_disjoint(clean_frame):
    splits = peptide_grouped_split(clean_frame, test_size=0.2, val_size=0.2, seed=1)
    assert set(splits) == {"train", "val", "test"}
    sets = [set(part["peptide"]) for part in splits.values()]
    for i, a in enumerate(sets):
        for b in sets[i + 1 :]:
            assert a.isdisjoint(b)


def test_every_row_is_assigned_exactly_once(clean_frame):
    splits = peptide_grouped_split(clean_frame, test_size=0.25, seed=2)
    assert sum(len(p) for p in splits.values()) == len(clean_frame)


def test_split_is_deterministic(clean_frame):
    a = peptide_grouped_split(clean_frame, test_size=0.3, seed=5)
    b = peptide_grouped_split(clean_frame, test_size=0.3, seed=5)
    assert set(a["test"]["peptide"]) == set(b["test"]["peptide"])


def test_rejects_impossible_proportions(clean_frame):
    with pytest.raises(ValueError):
        peptide_grouped_split(clean_frame, test_size=0.7, val_size=0.5)


def test_requires_normalised_frame(raw_frame):
    with pytest.raises(KeyError, match="normalise"):
        peptide_grouped_split(raw_frame)


def test_binder_sets_are_per_epitope_and_deduplicated(clean_frame):
    sets = build_binder_sets(clean_frame)
    assert sets
    for peptide, bs in sets.items():
        assert bs.peptide == peptide
        assert len(bs.binders) == len(set(bs.binders)), "duplicate clones not collapsed"
        assert all(p.is_valid for p in bs.binders)


def test_negatives_never_include_the_epitopes_own_binders(clean_frame):
    sets = build_binder_sets(clean_frame)
    rng = np.random.default_rng(0)
    for bs in sets.values():
        negatives = bs.sample_negatives(20, rng)
        assert set(negatives).isdisjoint(set(bs.binders))


def test_usability_flag_reflects_sample_size(clean_frame):
    sets = build_binder_sets(clean_frame)
    for bs in sets.values():
        assert bs.is_usable == (len(bs) >= 10)


def test_data_summary_cli_runs(mcpas_csv, capsys):
    from tcrga.cli import main

    assert main(["data", "summary", "--path", str(mcpas_csv), "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert "paired a/b" in out
    assert "peptide-grouped" in out
