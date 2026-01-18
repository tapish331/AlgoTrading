"""Daily fetch pipeline for NSE tickers & Zerodha historical data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from nse import refresh_tickers
from zerodha_broker import ensure_access_token, fetch_history

# Location of the repository configuration.
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
# Directory where historical datasets are persisted.
DATA_DIR = Path(__file__).resolve().parent / "data"


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, object]:
    """Load configuration values that govern the fetch pipeline."""
    if not config_path.exists():
        raise FileNotFoundError(f"Unable to locate config.json at {config_path}")
    with config_path.open("r", encoding="utf-8") as cfg_handle:
        return json.load(cfg_handle)


def load_existing_history(
    ticker: str,
    timeframe: str,
    base_dir: Path,
) -> pd.DataFrame:
    """Load previously cached history for the given ticker/timeframe."""
    history_path = base_dir / ticker / timeframe / "history.csv"
    if not history_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(history_path)
    if "date" in df.columns and not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def write_history_csv(
    ticker: str,
    timeframe: str,
    dataframe: pd.DataFrame,
    base_dir: Path,
    verbose: bool = False,
) -> None:
    """Persist historical data to disk in a tidy CSV structure."""
    output_dir = base_dir / ticker / timeframe
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "history.csv"
    parquet_path = output_dir / "history.parquet"
    df = dataframe.copy()
    if not df.empty:
        df.sort_values(by="date", inplace=True)
        df["date"] = df["date"].apply(_timestamp_to_iso)
    df.to_csv(output_path, index=False)
    if verbose:
        col = None
        for candidate in ("timestamp", "date"):
            if candidate in df.columns:
                col = candidate
                break
        if col and not df.empty:
            first_val = df[col].iloc[0]
            last_val = df[col].iloc[-1]
            print(
                f"[fetch] {ticker} {timeframe} -> Range {first_val} → {last_val} "
                f"| rows={len(df)} | file={output_path}"
            )
        else:
            print(f"[fetch] {ticker} {timeframe} -> rows={len(df)} | file={output_path}")

    # Best-effort Parquet write for faster replay loads.
    try:
        parquet_df = dataframe.copy()
        if not parquet_df.empty and "date" in parquet_df.columns:
            parquet_df["date"] = pd.to_datetime(parquet_df["date"], utc=True, errors="coerce")
            parquet_df.dropna(subset=["date"], inplace=True)
            parquet_df.sort_values(by="date", inplace=True)
            parquet_df.reset_index(drop=True, inplace=True)
        if not parquet_df.empty:
            parquet_df.to_parquet(parquet_path, index=False)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"[fetch] Parquet write failed for {ticker} {timeframe}: {exc}")


def backfill_parquet(
    base_dir: Path = DATA_DIR,
    verbose: bool = False,
    force: bool = False,
) -> None:
    """Generate history.parquet files for existing history.csv files."""
    try:
        import pyarrow  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "pyarrow is required for parquet backfill. Install with `pip install pyarrow`."
        ) from exc

    base_dir = base_dir.resolve()
    if not base_dir.exists():
        if verbose:
            print(f"[fetch] No data directory found at {base_dir}")
        return

    csv_paths = sorted(base_dir.rglob("history.csv"))
    if not csv_paths:
        if verbose:
            print(f"[fetch] No history.csv files found under {base_dir}")
        return

    failures = 0
    updated = 0
    skipped = 0
    for csv_path in csv_paths:
        parquet_path = csv_path.with_suffix(".parquet")
        try:
            if not force and parquet_path.exists():
                try:
                    if parquet_path.stat().st_mtime_ns >= csv_path.stat().st_mtime_ns:
                        skipped += 1
                        if verbose:
                            print(f"[fetch] Parquet up to date for {csv_path}")
                        continue
                except OSError:
                    pass
            df = pd.read_csv(csv_path)
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
                df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            df.to_parquet(parquet_path, index=False)
            updated += 1
            if verbose:
                print(f"[fetch] Parquet written {parquet_path} (rows={len(df)})")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[fetch] Parquet backfill failed for {csv_path}: {exc}", file=sys.stderr)

    if verbose:
        print(
            "[fetch] Parquet backfill complete | "
            f"updated={updated} skipped={skipped} failed={failures}"
        )
    if failures:
        raise RuntimeError(f"Parquet backfill failed for {failures} file(s).")


def run_pipeline(verbose: bool = False) -> None:
    """Execute the full refresh → token ensure → history fetch sequence."""
    config = load_config()

    if verbose:
        print("[fetch] Starting ticker refresh")
    refresh_tickers(verbose=verbose)

    if verbose:
        print("[fetch] Ensuring Zerodha access token")
    ensure_access_token(verbose=verbose)

    timeframes: Iterable[str] = config.get("timeframes") or []
    fetch_config = config.get("fetch") if isinstance(config.get("fetch"), dict) else {}
    num_candles = fetch_config.get("num_candles")

    base_dir = DATA_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(timeframes, list) or not timeframes:
        raise RuntimeError("config.json must define timeframes with at least one entry.")
    if not isinstance(num_candles, int) or num_candles <= 0:
        raise RuntimeError("config.json must define fetch.num_candles as a positive integer.")

    tickers: List[str] = list(config.get("tickers") or [])
    if not tickers:
        raise RuntimeError("config.json does not list any tickers to fetch.")

    if verbose:
        len_tickers = len(tickers)
        print(
            f"[fetch] Fetching {num_candles} candles for {len_tickers} tickers "
            f"across timeframes: {', '.join(timeframes)}"
        )

    for ticker in tickers:
        if verbose:
            print(f"[fetch] Processing ticker {ticker}")
        for timeframe in timeframes:
            if verbose:
                print(f"[fetch]  -> Fetching timeframe {timeframe}")
            try:
                existing_df = load_existing_history(
                    ticker=ticker,
                    timeframe=timeframe,
                    base_dir=base_dir,
                )

                initial_count = len(existing_df)
                combined_df: pd.DataFrame

                if existing_df.empty:
                    if verbose:
                        print(f"[fetch]  -> No existing data, fetching {num_candles} candles from scratch")
                    candles = fetch_history(
                        ticker=ticker,
                        timeframe=timeframe,
                        num_candles=num_candles,
                        verbose=verbose,
                    )
                    new_df = _candles_to_dataframe(candles)
                    combined_df = new_df
                else:
                    existing_count = len(existing_df)
                    if existing_count < num_candles:
                        if verbose:
                            print(
                                "[fetch]  -> Backfilling older data "
                                f"(have {existing_count}, need {num_candles})"
                            )
                        candles = fetch_history(
                            ticker=ticker,
                            timeframe=timeframe,
                            num_candles=num_candles,
                            verbose=verbose,
                        )
                        new_df = _candles_to_dataframe(candles)
                        combined_df = _merge_existing_and_new(existing_df, new_df)
                    else:
                        last_dt = _ensure_datetime(existing_df["date"].max())
                        timeframe_delta = _timeframe_to_timedelta(timeframe)
                        fetch_from_dt = last_dt - timeframe_delta if timeframe_delta else last_dt
                        now_dt = datetime.now(timezone.utc)

                        if fetch_from_dt >= now_dt:
                            if verbose:
                                print("[fetch]  -> No new data available (timestamps up to date)")
                            combined_df = existing_df
                        else:
                            candles = fetch_history(
                                ticker=ticker,
                                timeframe=timeframe,
                                from_=_timestamp_to_iso(fetch_from_dt),
                                to=_timestamp_to_iso(now_dt),
                                verbose=verbose,
                            )
                            new_df = _candles_to_dataframe(candles)
                            combined_df = _merge_existing_and_new(existing_df, new_df)

                if combined_df is None or combined_df.empty:
                    if verbose:
                        print("[fetch]  -> No data to write")
                    continue

                combined_count = len(combined_df)
                updated = combined_count > initial_count

                if not updated:
                    if verbose:
                        print("[fetch]  -> No new candles detected; keeping existing data")
                    continue

                write_history_csv(
                    ticker=ticker,
                    timeframe=timeframe,
                    dataframe=combined_df,
                    base_dir=base_dir,
                    verbose=verbose,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[fetch] Error fetching {ticker} {timeframe}: {exc}", file=sys.stderr)
                continue

    backfill_parquet(base_dir=base_dir, verbose=verbose, force=True)

    if verbose:
        print("[fetch] Completed historical data download")


def _candles_to_dataframe(candles: List[Dict[str, object]]) -> pd.DataFrame:
    """Convert raw candle dictionaries into a DataFrame with datetime index."""
    df = pd.DataFrame(candles)
    if not df.empty and "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _merge_existing_and_new(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine old and new history rows while avoiding duplicates."""
    if existing_df is None or existing_df.empty:
        return new_df
    if new_df is None or new_df.empty:
        return existing_df

    combined = pd.concat([existing_df, new_df], ignore_index=True)
    if "date" in combined.columns:
        combined.drop_duplicates(subset=["date"], keep="last", inplace=True)
        combined.sort_values(by="date", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def _ensure_datetime(value: object) -> datetime:
    """Normalise heterogeneous timestamp representations to timezone-aware datetimes."""
    if value is None or pd.isna(value):
        return datetime.now(timezone.utc)
    if isinstance(value, pd.Timestamp):
        ts = value
    else:
        ts = pd.Timestamp(value)
    if pd.isna(ts):
        return datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


def _timeframe_to_timedelta(timeframe: str) -> Optional[timedelta]:
    """Map timeframe strings to the equivalent timedeltas."""
    mapping = {
        "minute": timedelta(minutes=1),
        "3minute": timedelta(minutes=3),
        "5minute": timedelta(minutes=5),
        "10minute": timedelta(minutes=10),
        "15minute": timedelta(minutes=15),
        "30minute": timedelta(minutes=30),
        "60minute": timedelta(minutes=60),
        "day": timedelta(days=1),
    }
    return mapping.get(timeframe)


def _timestamp_to_iso(value: Optional[object]) -> str:
    """Represent various timestamp types as ISO-8601 strings."""
    if value is None:
        return ""
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _build_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for fetch.py."""
    parser = argparse.ArgumentParser(
        description="Refresh NSE tickers, ensure broker auth, and fetch historical data."
    )
    parser.add_argument(
        "--backfill_parquet",
        action="store_true",
        help="Generate history.parquet for existing history.csv files and exit.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for the fetch pipeline.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.backfill_parquet:
            backfill_parquet(verbose=args.verbose)
            return 0
        run_pipeline(verbose=args.verbose)
    except Exception as exc:  # noqa: BLE001
        print(f"[fetch] Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
