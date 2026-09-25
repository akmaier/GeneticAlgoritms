# Environment notes

Baseline: **Python 3.12**, virtualenv at `.venv`.

## Why the upper version bounds in `pyproject.toml`

This project is developed on an **Intel Mac (x86_64)**, and the Python scientific ecosystem
is actively dropping that platform. Several dependencies now publish **arm64-only** macOS
wheels, so pip falls back to a source build — which then fails against the system toolchain.

Observed on 2026-09-25 (macOS 15.8, Apple clang 11.0.3 from Command Line Tools):

| Package | Latest | macOS x86_64 wheel? | Pinned to |
|---|---|---|---|
| `biopython` | 1.88 | ❌ arm64 only | `<1.87` (1.86 has `macosx_10_13_x86_64`) |
| `numba` (via `olga`) | 0.67.0 | ❌ arm64 only | `<0.63` (0.62.1 has `macosx_10_15_x86_64`) |
| `llvmlite` (via `numba`) | 0.49.0 | ❌ arm64 only | `<0.46` (0.45.1 has `macosx_10_15_x86_64`) |
| `torch` | 2.14.0 | ❌ none since 2.2.2 | `==2.2.2` in the `dl` extra |

The source builds fail with `clang: error: invalid version number in
'MACOSX_DEPLOYMENT_TARGET=14'` — the installed Command Line Tools (clang 11.0.3, 2020) are far
older than the OS. Updating them with `xcode-select --install` would allow source builds and
let the bounds be relaxed, but pinning to the last wheel-shipping release is cheaper and
reproducible.

**These bounds are harmless on arm64 and Linux**, where newer wheels exist for everything.
They are a platform workaround, not a compatibility claim — revisit when the dev machine changes.

## torch

`torch` is confined to the `dl` extra, pinned to `2.2.2` (last macOS x86_64 build, Python
≤3.12, CPU only — no MPS on Intel). Nothing in `tcrga` imports it at module scope, so the
package installs and runs fully without it.
