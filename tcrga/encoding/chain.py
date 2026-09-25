"""The chromosome: a paired alpha/beta CDR3.

A TCR recognises peptide-MHC through both chains jointly, so the unit of selection here
is the *pair*, never a single chain. `ChainPair` is deliberately immutable and hashable
so it can be used as a dictionary key for scorer caching and for duplicate elimination
inside pymoo.

CDR3 sequences carry structural invariants that a naive mutation operator will happily
destroy. The junction begins at the conserved cysteine of the V gene and ends at the
conserved phenylalanine (or tryptophan) of the J gene. Sequences violating that are not
"low fitness" -- they are not CDR3s at all, and no downstream scorer has ever seen one.
We therefore treat the anchors as hard invariants of the encoding rather than as soft
penalties in the objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

#: The 20 proteinogenic amino acids in single-letter code.
AMINO_ACIDS: Final[str] = "ACDEFGHIKLMNPQRSTVWY"
_AA_SET: Final[frozenset[str]] = frozenset(AMINO_ACIDS)


class InvalidChain(ValueError):
    """A sequence violates a structural invariant of a CDR3 junction."""


@dataclass(frozen=True, slots=True)
class ChainSpec:
    """Structural constraints applied to one chain.

    Length bounds default to the range that covers essentially all of the observed human
    repertoire; McPAS entries outside it are almost always annotation errors.
    """

    start_residue: str = "C"
    end_residues: tuple[str, ...] = ("F", "W")
    min_length: int = 8
    max_length: int = 24

    def __post_init__(self) -> None:
        if self.min_length < 3:
            raise ValueError("min_length must leave room for both anchors and one core residue")
        if self.max_length < self.min_length:
            raise ValueError("max_length must not be below min_length")


#: Alpha junctions conventionally end in F or W; beta junctions almost always in F.
ALPHA_SPEC: Final[ChainSpec] = ChainSpec()
BETA_SPEC: Final[ChainSpec] = ChainSpec(end_residues=("F",))


def validate_cdr3(seq: str, spec: ChainSpec, *, label: str = "chain") -> None:
    """Raise `InvalidChain` if `seq` is not a well-formed CDR3 under `spec`."""
    if not seq:
        raise InvalidChain(f"{label}: empty sequence")
    if not _AA_SET.issuperset(seq):
        bad = sorted(set(seq) - _AA_SET)
        raise InvalidChain(f"{label}: non-standard residues {bad} in {seq!r}")
    if not spec.min_length <= len(seq) <= spec.max_length:
        raise InvalidChain(
            f"{label}: length {len(seq)} outside [{spec.min_length}, {spec.max_length}] in {seq!r}"
        )
    if seq[0] != spec.start_residue:
        raise InvalidChain(f"{label}: must start with {spec.start_residue!r}, got {seq!r}")
    if seq[-1] not in spec.end_residues:
        raise InvalidChain(f"{label}: must end with one of {spec.end_residues}, got {seq!r}")


def is_valid_cdr3(seq: str, spec: ChainSpec) -> bool:
    """Non-raising form of `validate_cdr3`."""
    try:
        validate_cdr3(seq, spec)
    except InvalidChain:
        return False
    return True


@dataclass(frozen=True, slots=True)
class ChainPair:
    """A candidate TCR: the alpha and beta CDR3 junctions together."""

    alpha: str
    beta: str

    def validate(self) -> None:
        """Raise `InvalidChain` if either chain breaks its structural invariants."""
        validate_cdr3(self.alpha, ALPHA_SPEC, label="alpha")
        validate_cdr3(self.beta, BETA_SPEC, label="beta")

    @property
    def is_valid(self) -> bool:
        try:
            self.validate()
        except InvalidChain:
            return False
        return True

    @property
    def total_length(self) -> int:
        return len(self.alpha) + len(self.beta)

    def __str__(self) -> str:
        return f"{self.alpha}/{self.beta}"
