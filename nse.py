"""Utility helpers for working with NSE index tickers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List
from urllib.parse import quote

import requests

# Default location for the config file that stores the target index and tickers.
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def refresh_tickers(config_path: Path = CONFIG_PATH, verbose: bool = False) -> List[str]:
    """
    Fetch the latest constituents for the configured NSE index and persist them.

    Parameters
    ----------
    config_path:
        Location of the JSON configuration file that contains the index name.
    verbose:
        Enable progress logging that mirrors a CLI `--verbose` switch.

    Returns
    -------
    list[str]:
        Alphabetically sorted list of ticker symbols retrieved from NSE.
    """
    if verbose:
        print(f"[refresh_tickers] Loading configuration from {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as cfg_handle:
        config = json.load(cfg_handle)

    index_name = config.get("index")
    if not isinstance(index_name, str) or not index_name.strip():
        raise ValueError("Config JSON must define a non-empty 'index' string.")

    index_name = index_name.strip()
    if verbose:
        print(f"[refresh_tickers] Fetching tickers for NSE index '{index_name}'")

    # NSE blocks requests without a browser-style header/cookie handshake.
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
        }
    )

    # Warm up the session to obtain the cookies NSE expects for subsequent calls.
    homepage_response = session.get("https://www.nseindia.com", timeout=10)
    homepage_response.raise_for_status()

    api_url = (
        "https://www.nseindia.com/api/equity-stockIndices?index="
        f"{quote(index_name, safe='')}"
    )
    response = session.get(api_url, timeout=10)
    response.raise_for_status()

    payload = response.json()
    raw_entries = payload.get("data")
    if not isinstance(raw_entries, list):
        raise RuntimeError("Unexpected NSE payload format: 'data' not present.")

    ticker_set = set()
    for entry in raw_entries:
        if not isinstance(entry, dict):
            continue
        symbol = entry.get("symbol")
        if not isinstance(symbol, str):
            continue
        symbol = symbol.strip()
        if not symbol:
            continue
        if symbol.upper() == index_name.upper():
            # NSE includes the index itself as a "symbol"; suppress it.
            continue
        ticker_set.add(symbol)

    tickers = sorted(ticker_set)
    if not tickers:
        raise RuntimeError("No tickers returned for the requested NSE index.")

    config["tickers"] = tickers
    if verbose:
        print(f"[refresh_tickers] Retrieved {len(tickers)} tickers; updating config")

    with config_path.open("w", encoding="utf-8") as cfg_handle:
        json.dump(config, cfg_handle, indent=2)
        cfg_handle.write("\n")

    if verbose:
        print(f"[refresh_tickers] Successfully refreshed tickers for '{index_name}'")

    return tickers


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for this module."""
    parser = argparse.ArgumentParser(
        description="Utilities for synchronising NSE index tickers."
    )
    parser.add_argument(
        "--refresh_tickers",
        action="store_true",
        help="Fetch the latest NSE constituents and update the config file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help=f"Alternate path to config JSON (default: {CONFIG_PATH}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for troubleshooting.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    """CLI entry point that backs the `python util.py --refresh_tickers` workflow."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.refresh_tickers:
        parser.print_help()
        return 0

    refresh_tickers(config_path=args.config, verbose=args.verbose)
    return 0


__all__ = ["refresh_tickers", "main"]


if __name__ == "__main__":
    sys.exit(main())
