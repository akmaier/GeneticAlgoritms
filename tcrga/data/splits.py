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
    min_holdout_epitopes: int = 5,
) -> dict[str, pd.DataFrame]:
    """Partition rows so that no epitope appears in more than one split.

    Balancing rows alone is not enough. McPAS epitope abundance is extremely skewed --
    the largest epitope carries over 500 paired TCRs while the median carries one -- so a
    greedy fill by row count hands the test set two enormous epitopes and stops. That
    split has the requested 20% of rows and is useless: measuring generalisation to
    unseen epitopes across two epitopes measures almost nothing.

    So assignment tracks two deficits at once, rows and distinct epitopes, and gives each
    epitope to whichever split is furthest behind on either. Exact proportions remain
    impossible -- groups are indivisible and wildly unequal -- but both quantities stay
    in a usable range.
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
    order = counts.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).sort_values(
        ascending=False, kind="stable"
    )

    n_rows, n_epitopes = len(frame), len(counts)
    holdouts = {k: v for k, v in (("test", test_size), ("val", val_size)) if v > 0}
    row_target = {k: v * n_rows for k, v in holdouts.items()}
    epi_target = {
        k: max(min(min_holdout_epitopes, n_epitopes // (len(holdouts) + 1)), int(v * n_epitopes))
        for k, v in holdouts.items()
    }

    assigned: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    rows_filled = dict.fromkeys(holdouts, 0)

    # Pass 1: fill row budgets, largest epitope first. A holdout that has reached its row
    # target takes nothing more here, which is what stops one 500-TCR epitope from
    # swallowing the split.
    for peptide, n in order.items():
        hungry = {k: row_target[k] - rows_filled[k] for k in holdouts}
        hungry = {k: v for k, v in hungry.items() if v > 0}
        if hungry:
            best = max(hungry, key=lambda k: hungry[k])
            assigned[best].append(str(peptide))
            rows_filled[best] += int(n)
        else:
            assigned["train"].append(str(peptide))

    # Pass 2: top up epitope counts using the *smallest* remaining training epitopes, so
    # a holdout reaches a usable number of distinct epitopes at minimal cost in rows.
    # Without this a 20% row budget buys only two or three epitopes on skewed data, and a
    # test set that small cannot support a claim about unseen-epitope generalisation.
    ascending = [p for p in order.index[::-1] if str(p) in set(assigned["train"])]
    for name in holdouts:
        deficit = epi_target[name] - len(assigned[name])
        if deficit <= 0:
            continue
        # Never strip training down to nothing chasing an epitope target.
        movable = ascending[: max(0, min(deficit, len(assigned["train"]) - n_epitopes // 2))]
        for peptide in movable:
            assigned["train"].remove(str(peptide))
            assigned[name].append(str(peptide))
            ascending.remove(peptide)

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
