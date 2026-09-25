"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tcrga import __version__
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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "evolve":
        return _cmd_evolve(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
