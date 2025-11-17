"""Utility helpers for working with NSE index tickers."""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Set
from urllib.parse import quote

import requests
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# Default location for the config file that stores the target index and tickers.
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

# Less restricted archive endpoint that mirrors the current index constituents.
ARCHIVE_URL_TEMPLATE = (
    "https://archives.nseindia.com/content/indices/ind_{slug}list.csv"
)


def _make_slug(name: str) -> str:
    """Create the slug used by NSE CSV files, e.g., 'NIFTY 50' -> 'nifty50'."""
    cleaned = "".join(ch for ch in name.lower() if ch.isalnum())
    if not cleaned:
        raise ValueError("Index name cannot be reduced to an alphanumeric slug.")
    return cleaned


def _build_session() -> requests.Session:
    """
    Build a session that resembles a browser and retries transient failures.

    NSE often rejects bare HTTP clients; a browser-like header set plus
    retry/backoff around 403/5xx responses improves resilience on headless
    servers.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/121.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "image/avif,image/webp,image/apng,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/",
            "Connection": "keep-alive",
        }
    )

    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(403, 429, 500, 502, 503, 504),
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _try_warm_session(session: requests.Session, verbose: bool) -> None:
    """
    Prime the session to collect the cookies NSE uses for API access.

    Multiple URLs are attempted to mitigate occasional 403s or timeouts.
    """
    warmup_urls = [
        "https://www.nseindia.com/",
        "https://www.nseindia.com/market-data",
        "https://www1.nseindia.com/",
    ]
    for url in warmup_urls:
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            if verbose:
                print(f"[refresh_tickers] Session warmed via {url}")
            return
        except requests.RequestException as exc:  # pragma: no cover - network dependent
            if verbose:
                print(f"[refresh_tickers] Warm-up via {url} failed: {exc}")
            time.sleep(0.75)
    raise RuntimeError(
        "Unable to warm up NSE session; requests kept failing with 403/timeouts."
    )


def _extract_tickers_from_api_response(
    response: Response, index_name: str
) -> List[str]:
    payload = response.json()
    raw_entries = payload.get("data")
    if not isinstance(raw_entries, list):
        raise RuntimeError("Unexpected NSE payload format: 'data' not present.")

    ticker_set: Set[str] = set()
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
    if not ticker_set:
        raise RuntimeError("No tickers returned for the requested NSE index.")
    return sorted(ticker_set)


def _fetch_from_api(index_name: str, verbose: bool) -> List[str]:
    """Primary path: hit the official NSE JSON API once the session is primed."""
    session = _build_session()
    _try_warm_session(session, verbose)

    api_url = (
        "https://www.nseindia.com/api/equity-stockIndices?index="
        f"{quote(index_name, safe='')}"
    )
    response = session.get(api_url, timeout=10)
    response.raise_for_status()
    return _extract_tickers_from_api_response(response, index_name)


def _fetch_from_archive(index_name: str, verbose: bool) -> List[str]:
    """
    Fallback path: pull constituents from the CSV archives endpoint.

    This endpoint is less picky about cookies/headers and works reliably
    on servers where the main site returns 403.
    """
    slug = _make_slug(index_name)
    url = ARCHIVE_URL_TEMPLATE.format(slug=slug)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    if "Symbol" not in reader.fieldnames:
        raise RuntimeError(
            f"Archive payload did not include expected 'Symbol' column from {url}"
        )
    ticker_set: Set[str] = set()
    for row in reader:
        symbol = (row.get("Symbol") or "").strip()
        if symbol:
            ticker_set.add(symbol)

    if not ticker_set:
        raise RuntimeError(
            f"No tickers were found in the archive CSV at {url}; "
            "NSE may have changed the file format."
        )
    if verbose:
        print(
            f"[refresh_tickers] Retrieved {len(ticker_set)} tickers from archive CSV"
        )
    return sorted(ticker_set)


def _persist_tickers(config_path: Path, config: dict, tickers: Iterable[str]) -> None:
    """Write the updated ticker list back to the config file."""
    config["tickers"] = sorted(tickers)
    with config_path.open("w", encoding="utf-8") as cfg_handle:
        json.dump(config, cfg_handle, indent=2)
        cfg_handle.write("\n")


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

    tickers: List[str]
    try:
        tickers = _fetch_from_api(index_name=index_name, verbose=verbose)
        if verbose:
            print(
                f"[refresh_tickers] Retrieved {len(tickers)} tickers from NSE API; "
                "updating config"
            )
    except Exception as exc:  # pragma: no cover - network dependent
        if verbose:
            print(
                "[refresh_tickers] API path failed "
                f"({exc!r}); falling back to archive CSV"
            )
        tickers = _fetch_from_archive(index_name=index_name, verbose=verbose)

    _persist_tickers(config_path, config, tickers)

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
