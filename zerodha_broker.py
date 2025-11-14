"""Zerodha brokerage helpers for maintaining a valid access token and history."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import dotenv_values, set_key
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException
from zoneinfo import ZoneInfo

INSTRUMENT_CACHE: Dict[str, int] = {}
HISTORY_REQUEST_GAP_SECONDS = 0.05  # seconds between Kite history calls

# Default path to the environment file that stores Zerodha credentials.
ENV_PATH = Path(__file__).resolve().parent / ".env"
INDIA_TZ = ZoneInfo("Asia/Kolkata")

def _to_ist_naive(ts: datetime) -> datetime:
    """Convert datetime to naive IST (required by Kite historical API)."""
    if ts.tzinfo is None:
        return ts
    return ts.astimezone(INDIA_TZ).replace(tzinfo=None)


def _to_ist(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=INDIA_TZ)
    return ts.astimezone(INDIA_TZ)


def _format_ts_ist(raw_value: Any) -> str:
    if isinstance(raw_value, datetime):
        return _to_ist(raw_value).isoformat()
    try:
        dt = _parse_history_datetime(raw_value)
    except Exception:
        return str(raw_value)
    return _to_ist(dt).isoformat()


def ensure_access_token(
    env_path: Path = ENV_PATH,
    verbose: bool = False,
) -> str:
    """
    Ensure the Zerodha access token in `.env` exists and is still valid.

    If the token is missing or invalid, prompt the user to complete the login flow
    and persist the newly issued access token back into the `.env` file.
    """
    env_path = env_path.expanduser().resolve()
    if verbose:
        print(f"[ensure_access_token] Using environment file at {env_path}")

    env_values = _load_env(env_path)
    api_key = _require_env_value(env_values, "ZERODHA_API_KEY")
    api_secret = _require_env_value(env_values, "ZERODHA_API_SECRET")
    access_token = env_values.get("ZERODHA_ACCESS_TOKEN")

    if access_token:
        if verbose:
            print("[ensure_access_token] Existing access token found; validating with Kite Connect")
        if _is_access_token_valid(api_key, access_token, verbose):
            if verbose:
                print("[ensure_access_token] Existing access token is valid")
            return access_token
        if verbose:
            print("[ensure_access_token] Stored access token is invalid or expired; refreshing")
    elif verbose:
        print("[ensure_access_token] No access token found; starting login flow")

    fresh_token = _perform_login_flow(api_key, api_secret, verbose)
    _persist_access_token(env_path, fresh_token, verbose)
    return fresh_token


def _build_authorized_client(env_path: Path, verbose: bool) -> KiteConnect:
    """Instantiate a Kite client with a freshly validated access token."""
    env_path = env_path.expanduser().resolve()
    if verbose:
        print(f"[ensure_access_token] Building Kite client using credentials at {env_path}")
    env_values = _load_env(env_path)
    api_key = _require_env_value(env_values, "ZERODHA_API_KEY")
    access_token = ensure_access_token(env_path=env_path, verbose=verbose)
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def _resolve_instrument_token(
    kite: KiteConnect,
    instrument_key: str,
    verbose: bool,
) -> int:
    cache_key = instrument_key.upper()
    cached = INSTRUMENT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if verbose:
        print(f"[fetch_history] Resolving instrument token for {instrument_key}")
    try:
        quote_data = kite.quote(instrument_key)
    except KiteException as exc:
        raise RuntimeError(f"Failed to resolve instrument token for {instrument_key}: {exc}") from exc
    quote_entry = quote_data.get(instrument_key)
    if not isinstance(quote_entry, dict):
        raise RuntimeError(
            f"Unexpected response while resolving instrument token for {instrument_key}."
        )
    instrument_token = quote_entry.get("instrument_token")
    if instrument_token is None:
        raise RuntimeError(f"No instrument_token returned for {instrument_key}.")
    INSTRUMENT_CACHE[cache_key] = int(instrument_token)
    return int(instrument_token)


def fetch_history(
    ticker: str,
    timeframe: str,
    from_: Optional[str] = None,
    to: Optional[str] = None,
    num_candles: Optional[int] = None,
    env_path: Path = ENV_PATH,
    verbose: bool = False,
    exchange: str = "NSE",
    kite: Optional[KiteConnect] = None,
    instrument_token: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch historical OHLCV data for the requested ticker and interval.

    Parameters
    ----------
    ticker:
        Zerodha/Kite trading symbol (e.g. `INFY`).
    timeframe:
        Interval understood by Kite (e.g. `minute`, `15minute`, `day`).
    from_:
        When provided alongside `to`, ISO8601 string marking the start of the range.
    to:
        When provided alongside `from_`, ISO8601 string marking the end of the range.
    num_candles:
        Optional number of most recent candles to fetch instead of specifying a date range.
    env_path:
        Location of the `.env` file that contains Zerodha credentials.
    verbose:
        Emit verbose logs similar to passing `--verbose` on the CLI.
    exchange:
        Exchange segment to query (defaults to `NSE`).
    kite:
        Optional Kite client to reuse across multiple history calls.
    instrument_token:
        When provided, skips the quote lookup and uses the supplied token directly.
    """
    if not ticker or not ticker.strip():
        raise ValueError("ticker must be provided")
    if not timeframe or not timeframe.strip():
        raise ValueError("timeframe must be provided")

    use_range = from_ and to
    if num_candles is not None and num_candles <= 0:
        raise ValueError("num_candles must be a positive integer when supplied.")
    if num_candles is None and not use_range:
        raise ValueError(
            "fetch_history requires either both 'from' and 'to' datetimes, or 'num_candles'."
        )
    if num_candles is not None and (from_ or to):
        raise ValueError("Specify either 'num_candles' or 'from'/'to', not both.")

    env_path = env_path.expanduser().resolve()
    if verbose:
        print(f"[fetch_history] Loading credentials from {env_path}")

    client = kite or _build_authorized_client(env_path, verbose)

    instrument_key = f"{exchange.upper()}:{ticker.upper()}"
    resolved_token = (
        instrument_token
        if instrument_token is not None
        else _resolve_instrument_token(client, instrument_key, verbose)
    )

    interval_delta = _interval_to_timedelta(timeframe)
    if verbose:
        print(f"[fetch_history] Timeframe '{timeframe}' maps to duration {interval_delta}")

    if num_candles is not None:
        max_window = _history_window_cap(timeframe)
        max_candles_per_request = None
        if max_window is not None:
            max_candles_per_request = max(1, int(max_window / interval_delta))

        collected: List[Dict[str, Any]] = []
        end_ts = datetime.now(timezone.utc)
        attempts = 0
        last_oldest: Optional[datetime] = None

        while len(collected) < num_candles:
            attempts += 1
            remaining = num_candles - len(collected)
            buffer_units = max(25, remaining // 10 + 1)
            if max_candles_per_request is not None:
                buffer_units = min(buffer_units, max(25, max_candles_per_request // 10))

            request_units = remaining + buffer_units
            if max_candles_per_request is not None:
                request_units = min(request_units, max_candles_per_request)

            span_delta = interval_delta * request_units
            start_ts = end_ts - span_delta
            if start_ts >= end_ts:
                start_ts = end_ts - interval_delta

            if verbose:
                print(
                    "[fetch_history] num_candles fetch "
                    f"(attempt {attempts}): range {_format_ts_ist(start_ts)} -> {_format_ts_ist(end_ts)} "
                    f"(requesting ~{request_units} candles)"
                )

            chunk = _request_history(
                kite=client,
                instrument_token=resolved_token,
                start_ts=start_ts,
                end_ts=end_ts,
                timeframe=timeframe,
            )

            if not chunk:
                break

            combined = _merge_history_chunks(chunk, collected)
            collected = combined

            if len(collected) >= num_candles:
                break

            oldest_dt = _parse_history_datetime(collected[0]["date"])
            if oldest_dt.tzinfo is None:
                oldest_dt = oldest_dt.replace(tzinfo=INDIA_TZ)
            if last_oldest is not None and oldest_dt >= last_oldest:
                # No progress made; stop to avoid infinite loop.
                break
            last_oldest = oldest_dt
            end_ts = oldest_dt - interval_delta

            if verbose:
                print(
                    "[fetch_history] Retrieved "
                    f"{len(chunk)} candles; collected {len(collected)} so far"
                )

        if not collected:
            raise RuntimeError(
                "Zerodha returned no historical data for the requested symbol/timeframe. "
                "This can happen when the market has been closed for an extended period or the "
                "instrument/timeframe combination is unavailable. Try specifying an explicit "
                "date range or choosing a different timeframe."
            )

        if len(collected) < num_candles and verbose:
            print(
                "[fetch_history] Warning: fewer candles available than requested "
                f"({len(collected)} < {num_candles})"
            )

        history = collected[-num_candles:]
    else:
        start_ts = _parse_datetime(from_, "from")
        end_ts = _parse_datetime(to, "to")
        if verbose:
            print(
                "[fetch_history] Requesting data "
                f"{_format_ts_ist(start_ts)} -> {_format_ts_ist(end_ts)} @ {timeframe}"
            )
        history = _request_history(
            kite=client,
            instrument_token=resolved_token,
            start_ts=start_ts,
            end_ts=end_ts,
            timeframe=timeframe,
        )

    if verbose:
        print(f"[fetch_history] Retrieved {len(history)} records")

    return history


def fetch_market_snapshot(
    tickers: Iterable[str],
    timeframes: Iterable[str],
    lookback: int,
    env_path: Path = ENV_PATH,
    exchange: str = "NSE",
    verbose: bool = False,
    kite: Optional[KiteConnect] = None,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Fetch a minimal OHLCV window for each ticker/timeframe pair for RL feature generation.

    Returns a nested dictionary keyed as [timeframe][ticker] -> list of candle dicts.
    """
    tickers_list = [str(t).strip().upper() for t in tickers if str(t).strip()]
    timeframes_list = [str(tf).strip() for tf in timeframes if str(tf).strip()]
    if not tickers_list:
        raise ValueError("tickers must contain at least one symbol.")
    if not timeframes_list:
        raise ValueError("timeframes must contain at least one interval.")
    if lookback <= 0:
        raise ValueError("lookback must be a positive integer.")

    env_path = env_path.expanduser().resolve()
    client = kite or _build_authorized_client(env_path, verbose)
    exchange_code = exchange.upper()

    tokens: Dict[str, int] = {}
    token_errors: Dict[str, str] = {}
    for ticker in tickers_list:
        instrument_key = f"{exchange_code}:{ticker}"
        try:
            tokens[ticker] = _resolve_instrument_token(client, instrument_key, verbose)
        except Exception as exc:  # noqa: BLE001
            token_errors[ticker] = str(exc)
            if verbose:
                print(
                    f"[fetch_market_snapshot] Skipping {instrument_key}: {exc}",
                    file=sys.stderr,
                )

    if not tokens:
        raise RuntimeError(
            "Failed to resolve instrument tokens for any ticker; "
            "verify the configured symbols."
        )

    snapshot: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
        timeframe: {} for timeframe in timeframes_list
    }
    now = datetime.now(timezone.utc)
    min_candles = max(lookback * 2, lookback + 2, 10)
    pad_span = _snapshot_history_padding(now)
    last_history_call = 0.0

    def _throttle_history_requests() -> None:
        nonlocal last_history_call
        gap = HISTORY_REQUEST_GAP_SECONDS
        if gap <= 0:
            return
        now_monotonic = time.monotonic()
        if last_history_call > 0.0:
            wait = gap - (now_monotonic - last_history_call)
            if wait > 0:
                time.sleep(wait)
                now_monotonic = time.monotonic()
        last_history_call = now_monotonic

    for timeframe in timeframes_list:
        interval = _interval_to_timedelta(timeframe)
        span = (interval * min_candles) + pad_span
        max_window = _history_window_cap(timeframe)
        if max_window is not None and span > max_window:
            span = max_window
        start_ts = now - span
        for ticker in tickers_list:
            instrument_token = tokens.get(ticker)
            if instrument_token is None:
                continue
            _throttle_history_requests()
            try:
                rows = _request_history(
                    kite=client,
                    instrument_token=instrument_token,
                    start_ts=start_ts,
                    end_ts=now,
                    timeframe=timeframe,
                )
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(
                        "[fetch_market_snapshot] Failed history request "
                        f"for {ticker}@{timeframe}: {exc}",
                        file=sys.stderr,
                    )
                continue
            normalized = [_normalise_history_row(row) for row in rows]
            preserve = max(lookback * 2, lookback + 5, 10)
            window = normalized[-preserve:]
            if not window:
                continue
            snapshot[timeframe][ticker] = window
            if verbose:
                first_ts = _format_ts_ist(window[0].get("date"))
                last_ts = _format_ts_ist(window[-1].get("date"))
                print(
                    f"[fetch_market_snapshot] {ticker}@{timeframe} -> {len(window)} candles "
                    f"| first={first_ts} last={last_ts}"
                )
    if all(not bucket for bucket in snapshot.values()):
        skipped = ", ".join(sorted(token_errors)) or "all tickers"
        raise RuntimeError(
            "Failed to fetch market snapshot for every ticker/timeframe "
            f"(skipped: {skipped})."
        )
    return snapshot


def _snapshot_history_padding(now: datetime) -> timedelta:
    """Ensure snapshot requests span at least one prior trading session."""
    pad = timedelta(hours=18)
    if now.weekday() == 0:  # Mondays need to bridge the weekend gap.
        pad += timedelta(days=2)
    return pad


def _parse_datetime(raw_value: str, label: str) -> datetime:
    """Parse ISO8601 datetime strings into timezone-naive datetime objects."""
    try:
        dt = datetime.fromisoformat(raw_value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=INDIA_TZ)
        return dt
    except ValueError as exc:
        raise RuntimeError(
            f"Unable to parse '{label}' value '{raw_value}'. Supply ISO8601 date/datetime."
        ) from exc


def _interval_to_timedelta(interval: str) -> timedelta:
    """Translate Kite interval strings to timedeltas."""
    lookup = {
        "minute": timedelta(minutes=1),
        "3minute": timedelta(minutes=3),
        "5minute": timedelta(minutes=5),
        "10minute": timedelta(minutes=10),
        "15minute": timedelta(minutes=15),
        "30minute": timedelta(minutes=30),
        "60minute": timedelta(minutes=60),
        "day": timedelta(days=1),
    }
    try:
        return lookup[interval]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported timeframe '{interval}'. "
            "Choose from minute, 3minute, 5minute, 10minute, 15minute, 30minute, 60minute, day."
        ) from exc


def _history_window_cap(interval: str) -> Optional[timedelta]:
    """Return the maximum lookback window supported by Zerodha for the interval."""
    lookup: Dict[str, timedelta] = {
        "minute": timedelta(days=60),
        "3minute": timedelta(days=60),
        "5minute": timedelta(days=60),
        "10minute": timedelta(days=60),
        "15minute": timedelta(days=60),
        "30minute": timedelta(days=60),
        "60minute": timedelta(days=365),
        "day": timedelta(days=2000),
    }
    return lookup.get(interval)


def _merge_history_chunks(
    new_chunk: List[Dict[str, Any]],
    existing: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge two history slices keeping chronological order and removing duplicates."""
    combined = [
        _normalise_history_row(row)
        for row in new_chunk + existing
        if isinstance(row, dict) and row.get("date") is not None
    ]
    combined.sort(key=lambda row: _parse_history_datetime(row["date"]))

    deduped: List[Dict[str, Any]] = []
    seen_dates = set()
    for row in combined:
        date_key = row["date"]
        if date_key in seen_dates:
            continue
        seen_dates.add(date_key)
        deduped.append(row)
    return deduped


def _parse_history_datetime(raw_value: Any) -> datetime:
    """Parse date strings returned by Kite historical API into datetime objects."""
    if isinstance(raw_value, datetime):
        return raw_value

    value = str(raw_value).strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    # Ensure the separator is acceptable for fromisoformat.
    value = value.replace("T", " ")
    return datetime.fromisoformat(value)


def _normalise_history_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Return a shallow copy of row with normalized ISO8601 date strings."""
    normalized = dict(row)
    date_val = normalized.get("date")
    if isinstance(date_val, datetime):
        normalized["date"] = date_val.isoformat()
    elif isinstance(date_val, str):
        normalized["date"] = date_val.strip()
    else:
        normalized["date"] = str(date_val)
    return normalized


def _request_history(
    kite: KiteConnect,
    instrument_token: int,
    start_ts: datetime,
    end_ts: datetime,
    timeframe: str,
) -> List[Dict[str, Any]]:
    """Wrapper around Kite `historical_data` with consistent error handling."""
    try:
        return kite.historical_data(
            instrument_token=instrument_token,
            from_date=_to_ist_naive(start_ts),
            to_date=_to_ist_naive(end_ts),
            interval=timeframe,
            continuous=False,
            oi=False,
        )
    except KiteException as exc:
        raise RuntimeError(
            f"Failed to fetch history for instrument {instrument_token}: {exc}"
        ) from exc


def submit_market_order(
    ticker: str,
    side: str,
    quantity: int,
    mode: str = "paper",
    env_path: Path = ENV_PATH,
    exchange: str = "NSE",
    product: str = "MIS",
    order_type: str = "MARKET",
    validity: str = "DAY",
    tag: Optional[str] = None,
    price: Optional[float] = None,
    verbose: bool = False,
    kite: Optional[KiteConnect] = None,
) -> Dict[str, Any]:
    """Place a market order or simulate one depending on `mode`."""
    tx_type = side.upper()
    if tx_type not in {"BUY", "SELL"}:
        raise ValueError("side must be either 'buy' or 'sell'.")
    if quantity <= 0:
        raise ValueError("quantity must be a positive integer.")

    normalized_mode = str(mode or "paper").lower()
    if normalized_mode not in {"paper", "live", "real"}:
        raise ValueError("mode must be 'paper' or 'live'.")

    record = {
        "ticker": ticker.upper(),
        "transaction_type": tx_type,
        "quantity": quantity,
        "mode": normalized_mode,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "product": product,
        "order_type": order_type,
        "tag": tag,
    }

    if normalized_mode == "paper":
        record.update({"order_id": None, "price": price})
        if verbose:
            print(
                f"[submit_market_order] PAPER {tx_type} {quantity} {ticker.upper()} "
                f"product={product} tag={tag or '-'}"
            )
        return record

    env_path = env_path.expanduser().resolve()
    client = kite or _build_authorized_client(env_path, verbose)
    try:
        order_id = client.place_order(
            variety=KiteConnect.VARIETY_REGULAR,
            exchange=exchange.upper(),
            tradingsymbol=ticker.upper(),
            transaction_type=tx_type,
            quantity=quantity,
            product=product,
            order_type=order_type,
            price=price,
            validity=validity,
            tag=tag,
        )
    except KiteException as exc:
        raise RuntimeError(
            f"Failed to place {tx_type} order for {ticker.upper()}: {exc}"
        ) from exc

    record.update({"order_id": order_id, "price": price})
    if verbose:
        print(
            f"[submit_market_order] LIVE {tx_type} {quantity} {ticker.upper()} order_id={order_id}"
        )
    return record


def fetch_open_positions(
    *,
    mode: str = "paper",
    env_path: Path = ENV_PATH,
    product: Optional[str] = None,
    exchange: Optional[str] = None,
    verbose: bool = False,
    kite: Optional[KiteConnect] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve currently open positions from Zerodha (live mode) or return an empty mapping
    for paper trading.
    """
    normalized_mode = str(mode or "paper").lower()
    if normalized_mode not in {"paper", "live", "real"}:
        raise ValueError("mode must be 'paper', 'live', or 'real'.")
    if normalized_mode == "paper":
        return {}

    env_path = env_path.expanduser().resolve()
    client = kite or _build_authorized_client(env_path, verbose)
    try:
        positions_snapshot = client.positions()
    except KiteException as exc:
        raise RuntimeError(f"Failed to fetch open positions: {exc}") from exc

    def _iter_buckets(payload: Any) -> Iterable[Dict[str, Any]]:
        if isinstance(payload, dict):
            buckets = payload.get("net") or payload.get("day") or []
        else:
            buckets = payload or []
        for entry in buckets:
            if isinstance(entry, dict):
                yield entry

    open_positions: Dict[str, Dict[str, Any]] = {}
    product_filter = str(product or "").upper() or None
    exchange_filter = str(exchange or "").upper() or None
    for entry in _iter_buckets(positions_snapshot):
        ticker = str(entry.get("tradingsymbol") or "").upper()
        if not ticker:
            continue
        if exchange_filter and str(entry.get("exchange") or "").upper() != exchange_filter:
            continue
        if product_filter and str(entry.get("product") or "").upper() != product_filter:
            continue
        try:
            quantity = int(entry.get("quantity") or 0)
        except (TypeError, ValueError):
            continue
        if quantity == 0:
            continue
        avg_price_raw = entry.get("average_price") or entry.get("avg_price") or 0.0
        try:
            avg_price = float(avg_price_raw)
        except (TypeError, ValueError):
            avg_price = 0.0
        last_price_raw = entry.get("last_price")
        try:
            last_price = float(last_price_raw)
        except (TypeError, ValueError):
            last_price = None
        direction = "long" if quantity > 0 else "short"
        open_positions[ticker] = {
            "ticker": ticker,
            "direction": direction,
            "quantity": abs(quantity),
            "average_price": avg_price,
            "last_price": last_price,
            "product": entry.get("product"),
            "exchange": entry.get("exchange"),
            "value": entry.get("value"),
        }
    if verbose:
        print(f"[fetch_open_positions] Retrieved {len(open_positions)} broker positions")
    return open_positions


def square_off_positions(
    positions: Dict[str, Dict[str, Any]],
    mode: str = "paper",
    env_path: Path = ENV_PATH,
    exchange: str = "NSE",
    product: str = "MIS",
    order_type: str = "MARKET",
    verbose: bool = False,
    kite: Optional[KiteConnect] = None,
) -> List[Dict[str, Any]]:
    """Close all provided positions by submitting the necessary market orders."""
    normalized_mode = str(mode or "paper").lower()
    client = kite
    if normalized_mode in {"live", "real"} and client is None:
        client = _build_authorized_client(env_path, verbose)

    results: List[Dict[str, Any]] = []
    for ticker, meta in positions.items():
        direction = str(meta.get("direction", "")).lower()
        quantity = int(meta.get("quantity", 0))
        if quantity <= 0:
            continue
        if direction == "long":
            side = "sell"
        elif direction == "short":
            side = "buy"
        else:
            continue
        result = submit_market_order(
            ticker=ticker,
            side=side,
            quantity=quantity,
            mode=mode,
            env_path=env_path,
            exchange=exchange,
            product=product,
            order_type=order_type,
            verbose=verbose,
            kite=client,
            tag="square_off",
        )
        fill_price = result.get("price")
        if (
            fill_price is None
            and normalized_mode in {"live", "real"}
            and result.get("order_id")
            and client is not None
        ):
            fill_price = _resolve_order_fill_price(client, str(result["order_id"]), verbose)
        results.append(
            {
                "ticker": ticker.upper(),
                "order_id": result.get("order_id"),
                "price": fill_price,
            }
        )
    return results


def _load_env(env_path: Path) -> Dict[str, str]:
    """Load all key-value pairs from the .env file (empty dict if file missing)."""
    if not env_path.exists():
        return {}
    return {key: value for key, value in dotenv_values(env_path).items() if value is not None}


def _require_env_value(env_values: Dict[str, str], key: str) -> str:
    """Fetch a required environment variable from the loaded map."""
    value = env_values.get(key)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable '{key}' in .env. "
            "Populate it before attempting to ensure the access token."
        )
    return value


def _resolve_order_fill_price(
    kite: KiteConnect,
    order_id: str,
    verbose: bool = False,
) -> Optional[float]:
    """Fetch the executed price for a completed order via order history."""
    try:
        history = kite.order_history(order_id)
    except KiteException as exc:
        if verbose:
            print(f"[resolve_order_fill_price] Failed to fetch order history for {order_id}: {exc}")
        return None
    for event in reversed(history):
        avg_price = event.get("average_price")
        if avg_price in (None, "", 0):
            continue
        try:
            return float(avg_price)
        except (TypeError, ValueError):
            continue
    return None


def _is_access_token_valid(api_key: str, access_token: str, verbose: bool) -> bool:
    """Attempt to verify the current access token via the profile endpoint."""
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    try:
        kite.profile()
    except KiteException as exc:
        if verbose:
            print(f"[ensure_access_token] Token validation failed: {exc}")
        return False
    return True


def _perform_login_flow(api_key: str, api_secret: str, verbose: bool) -> str:
    """
    Guide the user through the Zerodha login flow to obtain an access token.

    The flow requires manual interaction: the user must visit the login URL, complete
    authentication, and copy the `request_token` from the redirected URL.
    """
    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()
    print(
        "\nZerodha login required:\n"
        f"1. Open the following URL in your browser and authenticate:\n   {login_url}\n"
        "2. After logging in, you will be redirected to your registered redirect URL.\n"
        "3. Copy the 'request_token' query parameter from that URL and paste it below.\n"
    )

    request_token = input("Enter request_token: ").strip()
    if not request_token:
        raise RuntimeError("No request_token provided; cannot generate a new access token.")

    if verbose:
        print("[ensure_access_token] Exchanging request_token for access token")

    try:
        session_data = kite.generate_session(request_token, api_secret=api_secret)
    except KiteException as exc:
        raise RuntimeError(f"Failed to generate session with provided request_token: {exc}") from exc

    access_token = session_data.get("access_token")
    if not access_token:
        raise RuntimeError("Zerodha did not return an access_token in the session response.")

    if verbose:
        print("[ensure_access_token] Successfully obtained a new access token")

    return access_token


def _persist_access_token(env_path: Path, access_token: str, verbose: bool) -> None:
    """Write the refreshed access token to the .env file."""
    if verbose:
        print(f"[ensure_access_token] Persisting new access token to {env_path}")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.touch(exist_ok=True)
    set_key(str(env_path), "ZERODHA_ACCESS_TOKEN", access_token)
    if verbose:
        print("[ensure_access_token] Access token saved successfully")


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for the Zerodha broker utilities."""
    parser = argparse.ArgumentParser(
        description="Utilities for managing Zerodha broker authentication."
    )
    parser.add_argument(
        "--ensure_access_token",
        action="store_true",
        help="Validate the stored Zerodha access token or trigger a login flow if needed.",
    )
    parser.add_argument(
        "--fetch_history",
        action="store_true",
        help="Fetch historical OHLCV data for a given instrument.",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        help="Trading symbol for history requests (e.g. INFY).",
    )
    parser.add_argument(
        "--from",
        dest="from_date",
        type=str,
        help="Start of the historical range (ISO8601 string).",
    )
    parser.add_argument(
        "--to",
        dest="to_date",
        type=str,
        help="End of the historical range (ISO8601 string).",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        help="Interval to request (e.g. minute, 15minute, day).",
    )
    parser.add_argument(
        "--num_candles",
        type=int,
        help="Fetch the most recent N candles instead of specifying a date range.",
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="NSE",
        help="Exchange segment for the symbol (default: NSE).",
    )
    parser.add_argument(
        "--env",
        type=Path,
        default=ENV_PATH,
        help=f"Alternate path to the .env file (default: {ENV_PATH}).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for troubleshooting.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point for broker utilities such as token refresh and history fetch."""
    parser = _build_parser()
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error(
            "Unrecognized arguments: "
            f"{', '.join(unknown)}\n"
            "Hint: supply values with flags, e.g. "
            "--ticker INFY --from 2024-10-01 --to 2024-10-31 --timeframe day."
        )

    if args.ensure_access_token:
        ensure_access_token(env_path=args.env, verbose=args.verbose)
        return 0

    if args.fetch_history:
        missing_flags = []
        if not args.ticker:
            missing_flags.append("--ticker")
        if not args.timeframe:
            missing_flags.append("--timeframe")

        has_range = args.from_date and args.to_date
        has_num_candles = args.num_candles is not None

        if has_range and has_num_candles:
            parser.error("Provide either --from/--to or --num_candles, not both.")

        if not has_range and not has_num_candles:
            parser.error(
                "--fetch_history requires either --from/--to or --num_candles.\n"
                "Formats: --ticker=SYMBOL (e.g. INFY), "
                "--from=YYYY-MM-DD[THH:MM:SS], --to=YYYY-MM-DD[THH:MM:SS], "
                "--timeframe in {minute, 3minute, 5minute, 10minute, 15minute, "
                "30minute, 60minute, day}, --num_candles=N"
            )

        if missing_flags:
            parser.error(
                "--fetch_history requires the following arguments: "
                f"{', '.join(missing_flags)}\n"
                "Formats: --ticker=SYMBOL (e.g. INFY), "
                "--from=YYYY-MM-DD[THH:MM:SS], --to=YYYY-MM-DD[THH:MM:SS], "
                "--timeframe in {minute, 3minute, 5minute, 10minute, 15minute, "
                "30minute, 60minute, day}, --num_candles=N"
            )

        history = fetch_history(
            ticker=args.ticker,
            timeframe=args.timeframe,
            from_=args.from_date,
            to=args.to_date,
            num_candles=args.num_candles,
            env_path=args.env,
            verbose=args.verbose,
            exchange=args.exchange,
        )
        print(json.dumps(history, indent=2, default=str))
        return 0

    parser.print_help()
    return 0


__all__ = [
    "ensure_access_token",
    "fetch_history",
    "fetch_market_snapshot",
    "fetch_open_positions",
    "submit_market_order",
    "square_off_positions",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
