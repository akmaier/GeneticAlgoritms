"""Reading the McPAS-TCR export.

McPAS-TCR is served by a Shiny application whose CSV export sits behind a per-session
token, so there is no stable URL to fetch and the download stays manual. That makes
validation the loader's real job: the file on disk is unversioned, hand-downloaded, and
easy to truncate or re-export differently. A loader that silently accepts a half-written
file produces a fitness landscape built on missing data, and nothing downstream will
complain.

So this module fails loudly on a schema it does not recognise, and records a checksum of
whatever it did read into every run manifest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: Columns the loader depends on. McPAS ships many more; these are the ones we use.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "CDR3.alpha.aa",
    "CDR3.beta.aa",
    "Epitope.peptide",
)

#: Used when present, tolerated when absent -- McPAS re-exports have varied over time.
OPTIONAL_COLUMNS: tuple[str, ...] = (
    "TRAV",
    "TRAJ",
    "TRBV",
    "TRBD",
    "TRBJ",
    "MHC",
    "Species",
    "Pathology",
    "Category",
    "Antigen.protein",
    "T.Cell.Type",
    "PubMed.ID",
)

DEFAULT_PATH = Path("data/raw/McPAS-TCR.csv")

#: Values McPAS curators have used for "not recorded", in the many forms they typed them.
#: These are ordinary non-empty strings, so anything counting `notna()` alone will report
#: a chain as present when the cell literally reads "unknown".
NULL_TOKENS: frozenset[str] = frozenset(
    {"", "na", "n/a", "nan", "none", "null", "unknown", "-", "?"}
)


def clean_sequence(value: object) -> str:
    """Uppercase and strip, mapping every null sentinel to the empty string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in NULL_TOKENS:
        return ""
    return text.upper().replace(" ", "")


DOWNLOAD_HINT = (
    "McPAS-TCR must be downloaded manually -- it is a Shiny app and its CSV export is "
    "session-gated, so there is no stable URL.\n"
    "  1. Open https://friedmanlab.weizmann.ac.il/McPAS-TCR/\n"
    "  2. Click Search with an empty query to return the whole database\n"
    "  3. Download the complete database and save it to data/raw/McPAS-TCR.csv"
)


class McPasSchemaError(ValueError):
    """The file on disk is not a McPAS-TCR export we can use."""


@dataclass(frozen=True)
class McPasStats:
    """A factual summary of what was loaded, for manifests and for the CLI."""

    path: Path
    sha256: str
    n_rows: int
    n_paired: int
    n_beta_only: int
    n_alpha_only: int
    n_peptides: int
    n_paired_peptides: int

    def as_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "rows": self.n_rows,
            "paired": self.n_paired,
            "beta_only": self.n_beta_only,
            "alpha_only": self.n_alpha_only,
            "peptides": self.n_peptides,
            "peptides_with_paired": self.n_paired_peptides,
        }


def checksum(path: Path) -> str:
    """SHA-256 of the file, so a run manifest records exactly which export produced it."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mcpas(path: Path | str = DEFAULT_PATH) -> pd.DataFrame:
    """Read the McPAS CSV, verifying that it is what it claims to be.

    McPAS is distributed as latin-1 in places (author names, pathology free text), which
    is why the encoding is not left to pandas to guess -- a UnicodeDecodeError halfway
    through a 60k-row file is a confusing way to learn that.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"McPAS-TCR export not found at {path}.\n\n{DOWNLOAD_HINT}")
    if path.stat().st_size == 0:
        raise McPasSchemaError(f"{path} is empty.\n\n{DOWNLOAD_HINT}")

    try:
        frame = pd.read_csv(path, encoding="latin-1", low_memory=False)
    except pd.errors.ParserError as exc:
        raise McPasSchemaError(
            f"{path} is not parseable as CSV -- a truncated or partial download is the "
            f"usual cause.\n\n{DOWNLOAD_HINT}"
        ) from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise McPasSchemaError(
            f"{path} is missing required column(s) {missing}.\n"
            f"Found {len(frame.columns)} columns: {list(frame.columns)[:12]}...\n"
            f"This does not look like a McPAS-TCR export.\n\n{DOWNLOAD_HINT}"
        )
    if frame.empty:
        raise McPasSchemaError(f"{path} has headers but no rows.\n\n{DOWNLOAD_HINT}")

    keep = [c for c in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS) if c in frame.columns]
    return frame[keep].copy()


def summarise(frame: pd.DataFrame, path: Path | str = DEFAULT_PATH) -> McPasStats:
    """Count what is actually usable, which is what decides the pairing policy."""
    path = Path(path)
    # Use the same null-token handling as normalise(): counting notna() alone reports a
    # chain as present when the cell reads "NA" or "unknown", which roughly doubles the
    # apparent number of paired rows -- exactly the figure the pairing policy is chosen on.
    alpha = frame["CDR3.alpha.aa"].map(clean_sequence) != ""
    beta = frame["CDR3.beta.aa"].map(clean_sequence) != ""
    paired = alpha & beta
    peptides = frame["Epitope.peptide"].map(clean_sequence)
    peptides = peptides[peptides != ""]
    return McPasStats(
        path=path,
        sha256=checksum(path) if path.exists() else "",
        n_rows=len(frame),
        n_paired=int(paired.sum()),
        n_beta_only=int((beta & ~alpha).sum()),
        n_alpha_only=int((alpha & ~beta).sum()),
        n_peptides=int(peptides.nunique()),
        n_paired_peptides=int(frame.loc[paired, "Epitope.peptide"].dropna().nunique()),
    )
