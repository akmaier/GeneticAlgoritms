"""Generate a synthetic McPAS-like CSV.

The real McPAS export is session-gated and cannot be committed (licence and provenance
aside, it is not ours to redistribute). This fixture reproduces the *shape* and, more
importantly, the *defects* of the real file -- latin-1 bytes, null sentinels in several
spellings, lowercase sequences, nucleotide strings in amino-acid columns, anchor-less
junctions, duplicate clones, and a heavy beta-only skew -- so the loader's validation is
exercised on the things that actually go wrong.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

AA = "ACDEFGHIKLMNPQRSTVWY"
PEPTIDES = ["GILGFVFTL", "NLVPMVATV", "SIINFEKL", "GLCTLVAML", "KLGGALQAK", "YVLDHLIVV"]
HEADER = [
    "CDR3.alpha.aa",
    "CDR3.beta.aa",
    "TRAV",
    "TRAJ",
    "TRBV",
    "TRBD",
    "TRBJ",
    "Epitope.peptide",
    "MHC",
    "Species",
    "Pathology",
    "Category",
    "Antigen.protein",
    "T.Cell.Type",
    "PubMed.ID",
]


def cdr3(rng: random.Random, end: str) -> str:
    return "C" + "".join(rng.choice(AA) for _ in range(rng.randint(8, 14))) + end


def build(path: Path, n_rows: int = 600, seed: int = 0) -> None:
    rng = random.Random(seed)
    rows = []
    for _ in range(n_rows):
        peptide = rng.choice(PEPTIDES)
        a, b = cdr3(rng, "W"), cdr3(rng, "F")

        # ~60% beta-only, mirroring the real database's bulk-sequencing skew.
        if rng.random() < 0.60:
            a = rng.choice(["", "NA", "n/a", "unknown", "-"])
        # Curation artefacts.
        if rng.random() < 0.04:
            b = b.lower()
        if rng.random() < 0.03:
            a = "".join(rng.choice("ACGT") for _ in range(30))  # nucleotides in an aa column
        if rng.random() < 0.03:
            b = b[1:]  # anchor stripped
        if rng.random() < 0.03:
            peptide = rng.choice(["", "NA", "XXXX1"])  # unusable epitope

        rows.append(
            [
                a,
                b,
                rng.choice(["TRAV12-2", "TRAV12", "TCRAV12S2", ""]),
                rng.choice(["TRAJ30", ""]),
                rng.choice(["TRBV20-1", "TRBV20", "TCRBV20S1", ""]),
                "",
                rng.choice(["TRBJ2-7", ""]),
                peptide,
                rng.choice(["HLA-A*02:01", "HLA-A2", ""]),
                "Human",
                rng.choice(["Influenza", "CMV", "Café-au-lait syndrome"]),  # non-ASCII on purpose
                rng.choice(["Pathogens", "Autoimmune"]),
                rng.choice(["M1", "pp65", ""]),
                rng.choice(["CD8", "CD4", ""]),
                str(rng.randint(10000000, 39999999)),
            ]
        )

    # Duplicate clones: the same receptor reported by several studies.
    for _ in range(25):
        rows.append(list(rng.choice(rows)))
    rng.shuffle(rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="latin-1") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)


if __name__ == "__main__":
    build(Path(__file__).parent / "mcpas_like.csv")
