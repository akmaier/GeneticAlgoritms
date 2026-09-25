import pytest

from tcrga.encoding.chain import ALPHA_SPEC, BETA_SPEC, ChainPair, InvalidChain, validate_cdr3


def test_valid_pair_round_trips():
    p = ChainPair("CAVRDSNYQLIW", "CASSLGQAYEQYF")
    p.validate()
    assert p.is_valid
    assert p.total_length == 25
    assert str(p) == "CAVRDSNYQLIW/CASSLGQAYEQYF"


def test_pair_is_hashable_and_value_equal():
    a = ChainPair("CAVRDSNYQLIW", "CASSLGQAYEQYF")
    b = ChainPair("CAVRDSNYQLIW", "CASSLGQAYEQYF")
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1


@pytest.mark.parametrize(
    "seq, reason",
    [
        ("AVRDSNYQLIW", "start"),
        ("CAVRDSNYQLIA", "end"),
        ("CAVRDSNYQLIX", "non-standard"),
        ("CAF", "length"),
        ("C" + "A" * 30 + "F", "length"),
        ("", "empty"),
    ],
)
def test_invalid_alpha_rejected(seq, reason):
    with pytest.raises(InvalidChain):
        validate_cdr3(seq, ALPHA_SPEC)


def test_beta_may_not_end_in_tryptophan():
    """Alpha junctions may end in F or W; beta effectively always ends in F."""
    validate_cdr3("CAVRDSNYQLIW", ALPHA_SPEC)
    with pytest.raises(InvalidChain):
        validate_cdr3("CASSLGQAYEQYW", BETA_SPEC)
