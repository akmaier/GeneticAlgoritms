#!/usr/bin/env python
"""Download the McPAS-TCR database via its Shiny interface.

McPAS-TCR has no static download URL. It is an R/Shiny application, and its
"Download the complete database" link carries a per-session token that is minted when
the page loads, so `curl` against a copied URL returns a 404 a few minutes later. The
only way to obtain the file programmatically is to drive a browser, which is what this
does: it loads the page, waits for the session to establish, and clicks the same link a
human would.

This is a convenience wrapper around a manual click, not a scraper. It fetches one file
that the site offers for download, once per invocation. Please respect the database's
terms and cite it:

    Tickotsky N, Sagiv T, Prilusky J, Shifrut E, Friedman N (2017). McPAS-TCR: A
    manually-curated catalogue of pathology-associated T cell receptor sequences.
    Bioinformatics 33:2924-2929.

Usage:
    python scripts/fetch_mcpas.py [--out data/raw/McPAS-TCR.csv] [--headed]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

URL = "https://friedmanlab.weizmann.ac.il/McPAS-TCR/"
DOWNLOAD_SELECTOR = "#downloadDB"
DEFAULT_OUT = Path("data/raw/McPAS-TCR.csv")


def fetch(out: Path, *, headed: bool = False, timeout_ms: int = 120_000) -> Path:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:  # pragma: no cover - optional tooling
        raise SystemExit(
            "playwright is not installed.\n  pip install playwright && playwright install chromium"
        ) from None

    out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()
        try:
            # Shiny keeps a websocket open, so "networkidle" never fires. Wait for the
            # download control to appear instead -- that is the real readiness signal.
            page.goto(URL, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_selector(DOWNLOAD_SELECTOR, timeout=timeout_ms)
            # The href is minted per session; give Shiny a moment to finish wiring it up.
            page.wait_for_timeout(5_000)

            with page.expect_download(timeout=timeout_ms) as download_info:
                page.click(DOWNLOAD_SELECTOR)
            download = download_info.value
            download.save_as(out)
        finally:
            browser.close()

    if not out.exists() or out.stat().st_size == 0:
        raise SystemExit(f"download produced no usable file at {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--headed", action="store_true", help="show the browser window")
    args = parser.parse_args(argv)

    print(f"fetching McPAS-TCR from {URL} ...")
    path = fetch(args.out, headed=args.headed)
    size = path.stat().st_size
    print(f"saved {size:,} bytes to {path}")

    # Validate immediately: a download that lands but does not parse is worse than one
    # that fails, because everything downstream will quietly use it.
    try:
        from tcrga.data import load_mcpas, summarise

        frame = load_mcpas(path)
        stats = summarise(frame, path)
        print(f"validated: {stats.n_rows:,} rows, {stats.n_peptides:,} epitopes")
        print(f"sha256   : {stats.sha256}")
    except Exception as exc:
        print(f"WARNING: downloaded file did not validate: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
