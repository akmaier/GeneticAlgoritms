"""Known binders of a peptide: the GA's seeds and the scorers' reference set.

Seeding the initial population with real binders is not an optimisation, it is what makes
the search work at all. Random CDR3s fall outside the region any scorer was trained on, so
a scorer assigns them all roughly the same uninformative value and the GA has no gradient
to climb for many generations. Starting from receptors that genuinely engage the target
puts the population where the scorer can discriminate.

`BinderSet` also supplies the negatives. Sampling those correctly matters more than it
looks: drawing "non-binders" by shuffling within the same epitope produces a task that is
trivially solvable from epitope identity alone, which is one of the documented ways
TCR-epitope benchmarks have been inflated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tcrga.encoding.chain import ChainPair


@dataclass
class BinderSet:
    """The TCRs recorded as recognising one peptide, plus a matched negative sampler."""

    peptide: str
    binders: list[ChainPair]
    _background: list[ChainPair] = field(default_factory=list, repr=False)

    def __len__(self) -> int:
        return len(self.binders)

    @property
    def is_usable(self) -> bool:
        """Whether there is enough here to fit anything against.

        Most McPAS epitopes carry a handful of TCRs. Fitting a per-epitope model on three
        sequences produces a number, not a model, so callers should check this rather
        than discovering it from a meaningless AUROC later.
        """
        return len(self.binders) >= 10

    def sample_negatives(
        self,
        n: int,
        rng: np.random.Generator,
        *,
        exclude: set[ChainPair] | None = None,
    ) -> list[ChainPair]:
        """Draw putative non-binders from TCRs recorded against *other* epitopes.

        These are assumed negatives, not verified ones: a TCR annotated against another
        epitope may still bind this one, and cross-reactivity is common. The assumption is
        standard in the field and unavoidable without negative-binding experiments, but it
        caps how much any calibration curve built on it can be trusted.
        """
        exclude = (exclude or set()) | set(self.binders)
        pool = [p for p in self._background if p not in exclude]
        if not pool:
            return []
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [pool[int(i)] for i in idx]


def build_binder_sets(
    frame: pd.DataFrame,
    *,
    min_binders: int = 1,
    peptide_column: str = "peptide",
) -> dict[str, BinderSet]:
    """Group normalised rows into one `BinderSet` per epitope.

    Rows missing an alpha chain are skipped: a paired-chain `ChainPair` cannot be built
    from a beta chain alone, and inventing one would put fabricated sequence into the
    seeds. Under `PairingPolicy.INCLUDE_BETA_ONLY` those rows still reach beta-only
    scorers; they simply cannot seed a paired search.
    """
    for col in (peptide_column, "alpha", "beta"):
        if col not in frame.columns:
            raise KeyError(f"{col!r} not in frame; run normalise() first")

    by_peptide: dict[str, list[ChainPair]] = {}
    everything: list[ChainPair] = []
    for peptide, group in frame.groupby(peptide_column, sort=False):
        pairs: list[ChainPair] = []
        for alpha, beta in zip(group["alpha"], group["beta"], strict=True):
            if not alpha or not beta:
                continue
            pairs.append(ChainPair(str(alpha), str(beta)))
        # Deduplicate: McPAS records the same clone from several studies.
        unique = list(dict.fromkeys(pairs))
        if len(unique) >= min_binders:
            by_peptide[str(peptide)] = unique
            everything.extend(unique)

    sets: dict[str, BinderSet] = {}
    for peptide, pairs in by_peptide.items():
        own = set(pairs)
        sets[peptide] = BinderSet(
            peptide=peptide,
            binders=pairs,
            _background=[p for p in everything if p not in own],
        )
    return sets
