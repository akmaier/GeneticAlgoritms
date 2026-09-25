"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tcrga import __version__
from tcrga.data import (
    PairingPolicy,
    build_binder_sets,
    load_mcpas,
    normalise,
    peptide_grouped_split,
    summarise,
)
from tcrga.data.mcpas import DEFAULT_PATH
from tcrga.fitness.stub import LengthConstraint, MotifScorer
from tcrga.ga.run import evolve

SCORERS = {"motif-stub": MotifScorer}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tcrga", description=__doc__)
    parser.add_argument("--version", action="version", version=f"tcrga {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evolve", help="evolve chain pairs against a target peptide")
    ev.add_argument("--peptide", required=True, help="target epitope, e.g. SIINFEKL")
    ev.add_argument("--scorer", default="motif-stub", choices=sorted(SCORERS))
    ev.add_argument("--pop-size", type=int, default=100)
    ev.add_argument("--generations", type=int, default=50)
    ev.add_argument("--seed", type=int, default=0)
    ev.add_argument("--top", type=int, default=10, help="how many candidates to print")
    ev.add_argument("--out", type=Path, default=None, help="directory for results and manifest")
    ev.add_argument("--no-constraints", action="store_true")
    ev.add_argument("--verbose", action="store_true")

    data = sub.add_parser("data", help="inspect the McPAS-TCR export")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    summary = data_sub.add_parser("summary", help="what is actually usable in the export")
    summary.add_argument("--path", type=Path, default=DEFAULT_PATH)
    summary.add_argument(
        "--policy",
        default=PairingPolicy.PAIRED_ONLY.value,
        choices=[p.value for p in PairingPolicy],
    )
    summary.add_argument("--top", type=int, default=20, help="epitopes to list")
    summary.add_argument("--min-binders", type=int, default=10)
    return parser


def _cmd_evolve(args: argparse.Namespace) -> int:
    scorer = SCORERS[args.scorer]()
    if not scorer.calibrated:
        print(
            f"warning: scorer '{scorer.name}' is uncalibrated - rankings are not binding "
            f"probabilities and must not be reported as such\n",
            file=sys.stderr,
        )
    constraints = () if args.no_constraints else (LengthConstraint(),)
    result = evolve(
        scorer,
        args.peptide,
        constraints=constraints,
        pop_size=args.pop_size,
        n_gen=args.generations,
        seed=args.seed,
        verbose=args.verbose,
    )
    print(f"target peptide : {result.peptide}")
    print(f"best P(bind)   : {result.best_score:.4f}")
    print(f"best pair      : {result.best}")
    print(
        f"evaluations    : {result.manifest['evaluations']} "
        f"({result.manifest['cache_hits']} cache hits, "
        f"{result.manifest['scorer_batches']} batches)"
    )
    print(f"\ntop {args.top}:")
    print(result.top(args.top).to_string(index=False))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        result.top(len(result.population)).to_csv(args.out / "candidates.csv", index=False)
        result.history.to_csv(args.out / "history.csv", index=False)
        (args.out / "manifest.json").write_text(json.dumps(result.manifest, indent=2))
        print(f"\nwritten to {args.out}/")
    return 0


def _cmd_data_summary(args: argparse.Namespace) -> int:
    """Report what the export really contains, per pairing policy.

    This is the command that decides the pairing policy, so it deliberately shows both
    the raw counts and what survives cleaning. The gap between them is usually large and
    is the part people miss.
    """
    raw = load_mcpas(args.path)
    stats = summarise(raw, args.path)

    print(f"file           : {stats.path}")
    print(f"sha256         : {stats.sha256[:16]}...")
    print(f"rows           : {stats.n_rows:,}")
    print(f"  paired a/b   : {stats.n_paired:,} ({stats.n_paired / stats.n_rows:.1%})")
    print(f"  beta only    : {stats.n_beta_only:,} ({stats.n_beta_only / stats.n_rows:.1%})")
    print(f"  alpha only   : {stats.n_alpha_only:,}")
    print(f"epitopes       : {stats.n_peptides:,} ({stats.n_paired_peptides:,} with paired data)")

    print("\nafter cleaning, by policy:")
    for policy in PairingPolicy:
        frame, report = normalise(raw, policy=policy)
        kept = report.n_output
        print(f"  {policy.value:18s} {kept:6,} rows   {frame['peptide'].nunique():4,} epitopes")
    print("  (dropped: invalid junctions, unusable epitopes, missing chains)")

    frame, report = normalise(raw, policy=PairingPolicy(args.policy))
    print(f"\nselected policy: {args.policy}")
    for key, value in report.as_dict().items():
        print(f"  {key:26s} {value}")

    sets = build_binder_sets(frame)
    usable = {p: s for p, s in sets.items() if len(s) >= args.min_binders}
    print(
        f"\nepitopes with >= {args.min_binders} distinct paired TCRs: "
        f"{len(usable):,} of {len(sets):,}"
    )
    ranked = sorted(sets.items(), key=lambda kv: len(kv[1]), reverse=True)[: args.top]
    if ranked:
        print(f"\ntop {len(ranked)} epitopes by paired TCR count:")
        print(f"  {'epitope':<20} {'TCRs':>6}  usable")
        for peptide, bs in ranked:
            print(f"  {peptide:<20} {len(bs):>6}  {'yes' if bs.is_usable else 'no'}")

    if len(frame):
        splits = peptide_grouped_split(frame, test_size=0.2, seed=0)
        print("\npeptide-grouped 80/20 split:")
        for name, part in splits.items():
            print(f"  {name:6s} {len(part):6,} rows   {part['peptide'].nunique():4,} epitopes")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "evolve":
        return _cmd_evolve(args)
    if args.command == "data" and args.data_command == "summary":
        return _cmd_data_summary(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
