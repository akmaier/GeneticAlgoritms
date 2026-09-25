"""Loading McPAS-TCR and turning it into training and seeding material."""

from tcrga.data.binders import BinderSet, build_binder_sets
from tcrga.data.mcpas import (
    McPasSchemaError,
    McPasStats,
    checksum,
    load_mcpas,
    summarise,
)
from tcrga.data.normalise import PairingPolicy, normalise
from tcrga.data.splits import peptide_grouped_split

__all__ = [
    "BinderSet",
    "McPasSchemaError",
    "McPasStats",
    "PairingPolicy",
    "build_binder_sets",
    "checksum",
    "load_mcpas",
    "normalise",
    "peptide_grouped_split",
    "summarise",
]
