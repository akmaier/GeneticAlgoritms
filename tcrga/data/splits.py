"""Train/validation/test splits that do not lie.

The default in this field is to split TCR-epitope pairs at random, and it inflates every
reported metric. A model that has seen thirty TCRs binding `GILGFVFTL` during training and
is then tested on a thirty-first has only to recognise the epitope, not to generalise to
it -- and the epitope is right there in the input. Reported AUROC collapses once the test
epitopes are genuinely unseen, which is the well-documented weakness of published
TCR-epitope predictors.

`tcrga` optimises against these scores, so an inflated one is not a cosmetic problem: the
GA will drive candidates straight into the region where the score is confidently wrong.
Splits here are therefore grouped by epitope, always. There is no random-split option,
because a flag that produces a flattering number is a flag someone will eventually use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def peptide_grouped_split(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    val_size: float = 0.0,
    seed: int = 0,
    peptide_column: str = "peptide",
) -> dict[str, pd.DataFrame]:
    """Partition rows so that no epitope appears in more than one split.

    Epitopes are assigned whole, and in descending order of how many TCRs they carry, so
    the rare very-abundant epitopes do not all land in one split and skew its size. The
    returned proportions are therefore approximate -- exact proportions are impossible
    when groups are indivisible and wildly unequal in size.
    """
    if not 0.0 <= test_size < 1.0:
        raise ValueError("test_size must be in [0, 1)")
    if not 0.0 <= val_size < 1.0:
        raise ValueError("val_size must be in [0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must leave room for a training set")
    if peptide_column not in frame.columns:
        raise KeyError(f"{peptide_column!r} not in frame; run normalise() first")

    counts = frame[peptide_column].value_counts()
    rng = np.random.default_rng(seed)
    # Shuffle within equal counts so ties are not broken by pandas' ordering.
    order = counts.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).sort_values(
        ascending=False, kind="stable"
    )

    n_total = len(frame)
    targets = {"test": test_size * n_total, "val": val_size * n_total}
    assigned: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    filled = {"test": 0, "val": 0}

    for peptide, n in order.items():
        # Greedy: give each epitope to whichever holdout is furthest from its target.
        deficits = {k: targets[k] - filled[k] for k in ("test", "val") if targets[k] > 0}
        best = max(deficits, key=lambda k: deficits[k]) if deficits else None
        if best is not None and deficits[best] > 0:
            assigned[best].append(str(peptide))
            filled[best] += int(n)
        else:
            assigned["train"].append(str(peptide))

    splits = {
        name: frame[frame[peptide_column].isin(peptides)].reset_index(drop=True)
        for name, peptides in assigned.items()
    }
    if val_size == 0.0:
        splits.pop("val")

    _assert_disjoint(splits, peptide_column)
    return splits


def _assert_disjoint(splits: dict[str, pd.DataFrame], peptide_column: str) -> None:
    """Guard the one invariant this module exists to provide."""
    seen: dict[str, str] = {}
    for name, part in splits.items():
        for peptide in part[peptide_column].unique():
            if peptide in seen:
                raise AssertionError(
                    f"epitope {peptide!r} leaked between splits {seen[peptide]!r} and {name!r}"
                )
            seen[peptide] = name
