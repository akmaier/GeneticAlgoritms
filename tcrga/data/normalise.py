"""Cleaning McPAS rows into something the rest of the package can trust.

Two independent jobs happen here.

*Sequence hygiene.* McPAS is manually curated from the literature, so CDR3 columns carry
the usual artefacts of hand transcription: stray whitespace, lowercase, nucleotide strings
in an amino-acid column, `NA`/`unknown` sentinels, and junctions missing their conserved
anchors. Every one of these produces a plausible-looking string that no scorer has ever
seen, so they are dropped rather than repaired.

*Gene nomenclature.* TRAV/TRBV naming has changed repeatedly and McPAS spans decades of
papers, so the same gene appears as `TRBV20-1`, `TRBV20`, `V20-1` and `TCRBV20S1`.
`tidytcells` reconciles these against IMGT. It is optional here only so the package still
imports without it; when absent, gene columns pass through untouched and a warning is
issued rather than silently producing inconsistent gene statistics.

Species matters here and is easy to get wrong. McPAS is roughly 90% human and 9% mouse,
and mouse gene symbols (`TRAV3N-3`, `TRAV16D/DV11`) simply do not exist in the human IMGT
reference. Standardising every row as human emits thousands of failures and leaves the
mouse rows unreconciled, so the `Species` column drives the lookup per row.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from tcrga.data.mcpas import clean_sequence as _clean_sequence
from tcrga.encoding.chain import ALPHA_SPEC, BETA_SPEC, ChainSpec, is_valid_cdr3

_GENE_COLUMNS = ("TRAV", "TRAJ", "TRBV", "TRBD", "TRBJ")

#: McPAS `Species` values mapped to the identifiers tidytcells expects.
_SPECIES = {"human": "homosapiens", "mouse": "musmusculus"}


class PairingPolicy(StrEnum):
    """What to do with rows that record only one chain.

    McPAS holds far more beta-only rows than paired ones, because bulk TCR-beta
    sequencing is cheap and paired single-cell data is not. The choice between them is a
    real trade-off, not an implementation detail, so it is explicit and recorded in the
    manifest rather than hidden in a default.
    """

    #: Only rows with both chains. Smaller, cleaner, and the only honest input to a
    #: paired-chain scorer.
    PAIRED_ONLY = "paired-only"
    #: Keep beta-only rows too, leaving alpha empty. More data, but any paired-chain
    #: score computed from it is partly imputed.
    INCLUDE_BETA_ONLY = "include-beta-only"


@dataclass(frozen=True)
class NormalisationReport:
    """Where the rows went, so data loss is visible instead of assumed."""

    n_input: int
    n_output: int
    dropped_missing_peptide: int
    dropped_invalid_peptide: int
    dropped_invalid_alpha: int
    dropped_invalid_beta: int
    dropped_unpaired: int
    dropped_species: int
    genes_normalised: bool
    genes_unresolved: int = 0
    species_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "input_rows": self.n_input,
            "output_rows": self.n_output,
            "dropped_missing_peptide": self.dropped_missing_peptide,
            "dropped_invalid_peptide": self.dropped_invalid_peptide,
            "dropped_invalid_alpha": self.dropped_invalid_alpha,
            "dropped_invalid_beta": self.dropped_invalid_beta,
            "dropped_unpaired": self.dropped_unpaired,
            "dropped_species": self.dropped_species,
            "genes_normalised": self.genes_normalised,
            "genes_unresolved": self.genes_unresolved,
            "species_counts": self.species_counts,
        }


def _is_peptide(seq: str, min_len: int = 5, max_len: int = 30) -> bool:
    """A plausible epitope: amino-acid alphabet and an MHC-compatible length.

    Nucleotide strings occasionally appear in peptide columns; a sequence over only
    ACGT of realistic length is far more likely to be DNA than a genuine peptide.
    """
    if not min_len <= len(seq) <= max_len:
        return False
    if not seq.isalpha():
        return False
    from tcrga.encoding.chain import AMINO_ACIDS

    if not set(seq) <= set(AMINO_ACIDS):
        return False
    return not (len(seq) >= 9 and set(seq) <= set("ACGT"))


def _normalise_genes(frame: pd.DataFrame) -> tuple[bool, int]:
    """Reconcile gene symbols against IMGT via tidytcells, per species.

    Returns whether standardisation ran and how many symbols it could not resolve.
    tidytcells reports each failure individually; on 40k rows that is tens of thousands
    of lines of console noise that buries everything else, so failures are counted and
    reported once. `on_fail="keep"` leaves an unresolvable symbol as written rather than
    discarding the row -- the sequence data is still good even when the gene label is not.
    """
    present = [c for c in _GENE_COLUMNS if c in frame.columns]
    if not present:
        return False, 0
    try:
        import tidytcells as tt
    except ImportError:
        warnings.warn(
            "tidytcells is not installed; TRAV/TRBV gene symbols are left as written in "
            "McPAS. Gene-level statistics will conflate naming variants of the same gene. "
            "Install the 'core' extra to fix this.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False, 0

    species_col = (
        frame["Species"].map(lambda v: _SPECIES.get(str(v).strip().lower(), "homosapiens"))
        if "Species" in frame.columns
        else pd.Series("homosapiens", index=frame.index)
    )

    unresolved = 0
    for col in present:
        results = []
        for value, species in zip(frame[col], species_col, strict=True):
            raw = _clean_sequence(value)
            if not raw:
                results.append("")
                continue
            with warnings.catch_warnings(), _quiet_logging():
                warnings.simplefilter("ignore")
                fixed = str(tt.tr.standardise(raw, species=species, on_fail="keep"))
            if fixed == raw and not raw.startswith(("TRA", "TRB")):
                unresolved += 1
            results.append(fixed)
        frame[col] = results
    return True, unresolved


@contextmanager
def _quiet_logging() -> Iterator[None]:
    """Silence tidytcells' per-symbol failure logging.

    It logs one line per unrecognised gene. Across five gene columns and 40k rows that is
    enough output to hide any real problem, so it is aggregated into a single count.
    """
    logger = logging.getLogger("tidytcells")
    previous = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(previous)


def normalise(
    frame: pd.DataFrame,
    *,
    policy: PairingPolicy = PairingPolicy.PAIRED_ONLY,
    normalise_genes: bool = True,
    species: str | None = None,
) -> tuple[pd.DataFrame, NormalisationReport]:
    """Clean a raw McPAS frame and apply the pairing policy.

    Returns the cleaned frame alongside a report of exactly what was discarded. The
    report is not decoration: when a peptide ends up with three usable TCRs instead of
    the thirty the database appears to hold, that is the number a scorer is actually
    trained on and it belongs in the run manifest.

    `species` filters to "Human" or "Mouse". Worth considering rather than defaulting:
    mouse contributes only ~9% of McPAS rows but ~41% of the *paired* ones, because
    paired single-cell sequencing is far more common in mouse work. Mouse receptors
    engage H-2 rather than HLA, so a single scorer fitted across both is fitting two
    different recognition problems at once. Left as None, both are kept and the mix is
    recorded in the report.
    """
    n_input = len(frame)
    out = frame.copy()

    dropped_species = 0
    if species is not None and "Species" in out.columns:
        wanted = species.strip().lower()
        keep = out["Species"].astype(str).str.strip().str.lower() == wanted
        dropped_species = int((~keep).sum())
        out = out[keep]

    out["peptide"] = out["Epitope.peptide"].map(_clean_sequence)
    out["alpha"] = out["CDR3.alpha.aa"].map(_clean_sequence)
    out["beta"] = out["CDR3.beta.aa"].map(_clean_sequence)

    missing_peptide = int((out["peptide"] == "").sum())
    out = out[out["peptide"] != ""]

    valid_peptide = out["peptide"].map(_is_peptide)
    invalid_peptide = int((~valid_peptide).sum())
    out = out[valid_peptide]

    # An anchor-less or non-standard junction is dropped, not repaired: a "fixed" CDR3
    # is a sequence the curators never reported.
    def _keep(seq: str, spec: ChainSpec) -> bool:
        return seq == "" or is_valid_cdr3(seq, spec)

    # `.astype(bool)` matters: on an empty frame `.map()` yields an object-dtype Series
    # whose .sum() is "" rather than 0, which then blows up int().
    alpha_ok = out["alpha"].map(lambda s: _keep(s, ALPHA_SPEC)).astype(bool)
    beta_ok = out["beta"].map(lambda s: _keep(s, BETA_SPEC)).astype(bool)
    # Counted so the reasons partition the dropped rows: a row with two bad junctions is
    # removed once, so attributing it to both chains would make the report fail to
    # reconcile with the row count -- and a report that does not add up is not evidence.
    invalid_alpha = int((~alpha_ok).sum())
    invalid_beta = int((alpha_ok & ~beta_ok).sum())
    out = out[alpha_ok & beta_ok]

    has_alpha = out["alpha"] != ""
    has_beta = out["beta"] != ""
    keep_mask = (has_alpha & has_beta) if policy is PairingPolicy.PAIRED_ONLY else has_beta
    dropped_unpaired = int((~keep_mask).sum())
    out = out[keep_mask]

    genes_normalised, genes_unresolved = _normalise_genes(out) if normalise_genes else (False, 0)
    species_counts = (
        out["Species"].fillna("unknown").astype(str).str.strip().value_counts().to_dict()
        if "Species" in out.columns
        else {}
    )

    out = out.reset_index(drop=True)
    report = NormalisationReport(
        n_input=n_input,
        n_output=len(out),
        dropped_missing_peptide=missing_peptide,
        dropped_invalid_peptide=invalid_peptide,
        dropped_invalid_alpha=invalid_alpha,
        dropped_invalid_beta=invalid_beta,
        dropped_unpaired=dropped_unpaired,
        dropped_species=dropped_species,
        genes_normalised=genes_normalised,
        genes_unresolved=genes_unresolved,
        species_counts={str(k): int(v) for k, v in species_counts.items()},
    )
    return out, report
