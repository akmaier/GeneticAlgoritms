import pandas as pd

from tcrga.data import PairingPolicy, normalise
from tcrga.encoding.chain import ALPHA_SPEC, BETA_SPEC, is_valid_cdr3


def test_every_surviving_sequence_is_a_valid_cdr3(clean_frame):
    for alpha, beta in zip(clean_frame["alpha"], clean_frame["beta"], strict=True):
        assert alpha == "" or is_valid_cdr3(alpha, ALPHA_SPEC)
        assert beta == "" or is_valid_cdr3(beta, BETA_SPEC)


def test_paired_only_policy_leaves_no_single_chain_rows(clean_frame):
    assert (clean_frame["alpha"] != "").all()
    assert (clean_frame["beta"] != "").all()


def test_include_beta_only_keeps_more_rows(raw_frame):
    paired, _ = normalise(raw_frame, policy=PairingPolicy.PAIRED_ONLY)
    both, _ = normalise(raw_frame, policy=PairingPolicy.INCLUDE_BETA_ONLY)
    assert len(both) > len(paired)
    assert (both["beta"] != "").all(), "beta must be present under either policy"


def test_report_accounts_for_every_dropped_row(raw_frame):
    _, report = normalise(raw_frame)
    dropped = (
        report.dropped_missing_peptide
        + report.dropped_invalid_peptide
        + report.dropped_invalid_alpha
        + report.dropped_invalid_beta
        + report.dropped_unpaired
    )
    assert report.n_input - dropped == report.n_output


def test_lowercase_sequences_are_recovered():
    frame = pd.DataFrame(
        {
            "CDR3.alpha.aa": ["cavrdsnyqliw"],
            "CDR3.beta.aa": ["cassLGQAYEQYf"],
            "Epitope.peptide": ["gilgfvftl"],
        }
    )
    out, _ = normalise(frame, normalise_genes=False)
    assert out.loc[0, "alpha"] == "CAVRDSNYQLIW"
    assert out.loc[0, "peptide"] == "GILGFVFTL"


def test_nucleotide_string_in_peptide_column_is_dropped():
    frame = pd.DataFrame(
        {
            "CDR3.alpha.aa": ["CAVRDSNYQLIW"],
            "CDR3.beta.aa": ["CASSLGQAYEQYF"],
            "Epitope.peptide": ["ACGTACGTACGTACGT"],
        }
    )
    out, report = normalise(frame, normalise_genes=False)
    assert len(out) == 0
    assert report.dropped_invalid_peptide == 1


def test_anchorless_junction_is_dropped_not_repaired():
    """A 'fixed' CDR3 is a sequence the curators never reported."""
    frame = pd.DataFrame(
        {
            "CDR3.alpha.aa": ["AVRDSNYQLIW"],
            "CDR3.beta.aa": ["CASSLGQAYEQYF"],
            "Epitope.peptide": ["GILGFVFTL"],
        }
    )
    out, report = normalise(frame, normalise_genes=False)
    assert len(out) == 0
    assert report.dropped_invalid_alpha == 1
