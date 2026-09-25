import pandas as pd
import pytest

from tcrga.data import McPasSchemaError, checksum, load_mcpas, summarise
from tcrga.data.mcpas import clean_sequence


def test_loads_and_keeps_known_columns(raw_frame):
    for col in ("CDR3.alpha.aa", "CDR3.beta.aa", "Epitope.peptide"):
        assert col in raw_frame.columns
    assert len(raw_frame) > 0


def test_missing_file_explains_the_manual_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="friedmanlab"):
        load_mcpas(tmp_path / "absent.csv")


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(McPasSchemaError, match="empty"):
        load_mcpas(path)


def test_wrong_schema_rejected(tmp_path):
    """A CSV that parses fine but is not McPAS must fail loudly, not load empty."""
    path = tmp_path / "wrong.csv"
    pd.DataFrame({"foo": [1, 2], "bar": [3, 4]}).to_csv(path, index=False)
    with pytest.raises(McPasSchemaError, match="missing required column"):
        load_mcpas(path)


def test_headers_without_rows_rejected(tmp_path):
    path = tmp_path / "headers.csv"
    path.write_text("CDR3.alpha.aa,CDR3.beta.aa,Epitope.peptide\n")
    with pytest.raises(McPasSchemaError, match="no rows"):
        load_mcpas(path)


def test_truncated_file_rejected(tmp_path):
    """A half-finished download: ragged rows that pandas cannot parse."""
    path = tmp_path / "trunc.csv"
    path.write_text('CDR3.alpha.aa,CDR3.beta.aa,Epitope.peptide\nA,B,C\n"unterminated,,\n,,,,,,,\n')
    with pytest.raises(McPasSchemaError):
        load_mcpas(path)


@pytest.mark.parametrize("token", ["NA", "n/a", "unknown", "-", "", "  ", "None", "?"])
def test_null_tokens_all_become_empty(token):
    assert clean_sequence(token) == ""


def test_summary_does_not_count_null_tokens_as_chains(raw_frame, mcpas_csv):
    """Regression: `notna()` alone counted the string "unknown" as a present chain,
    inflating the paired count that the pairing policy is chosen on."""
    stats = summarise(raw_frame, mcpas_csv)
    literal_alpha = raw_frame["CDR3.alpha.aa"].notna().sum()
    assert stats.n_paired < literal_alpha
    assert stats.n_paired + stats.n_beta_only + stats.n_alpha_only <= stats.n_rows


def test_checksum_is_stable_and_sensitive(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_text("hello")
    b.write_text("hellp")
    assert checksum(a) == checksum(a)
    assert checksum(a) != checksum(b)
