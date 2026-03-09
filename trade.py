"""Live trading loop powered by the LightRainbowDQN agent."""

from __future__ import annotations

from collections import deque
import argparse
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import warnings
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from fetch import run_pipeline as fetch_run_pipeline
from ml_rl import LightRainbowDQN, torch  # type: ignore[attr-defined]
from replay import (
    CONFIG_PATH,
    INDIA_TZ,
    MODEL_ACTIONS,
    Trade,
    _require_label,
    build_feature_vector,
    compute_day_positions,
    compute_input_dim,
    compute_net_percent_pnl,
    compute_percent_pnl,
    compute_quantity,
    estimate_roundtrip_cost_pct,
    infer_actions,
    load_config,
)
from zerodha_broker import (
    ensure_access_token,
    fetch_market_snapshot,
    fetch_open_positions,
    square_off_positions,
    submit_market_order,
)

DEFAULT_ENV_PATH = Path(".env")
DEFAULT_PRODUCT = "MIS"
DEFAULT_ORDER_TYPE = "MARKET"
DEFAULT_WINNER_CHECKPOINT = Path("checkpoints/light_rainbow_winner.pt")
STATE_DIR = Path("state")
ACTIVE_TRADES_PATH = STATE_DIR / "active_trades.json"
COMPLETED_TRADES_DIR = STATE_DIR / "completed_trades"
AUTO_REPAIR_DIR = STATE_DIR / "auto_repair"
AUTO_REPAIR_TRADE_DAY_MARKER = AUTO_REPAIR_DIR / "last_trade_log_clear_day.txt"
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE_PREFIX = "trade"
DEFAULT_LOG_RETENTION_DAYS = 5
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_VERBOSE_LOG_LEVEL = "DEBUG"
CODEX_AUTOMATION_DIR = STATE_DIR / ".codex_auto"
CODEX_DECISION_SCHEMA_PATH = CODEX_AUTOMATION_DIR / "decision_schema.json"
CODEX_DECISION_OUTPUT_PATH = CODEX_AUTOMATION_DIR / "trade_decision_output.json"
CODEX_TRADE_MODIFIED_DAY_MARKER_PATH = CODEX_AUTOMATION_DIR / "trade_modified_day.txt"
SHARED_CODE_UPDATE_MARKER_PATH = CODEX_AUTOMATION_DIR / "code_updated_marker.json"
EVIDENCE_EPOCH_STATE_PATH = CODEX_AUTOMATION_DIR / "evidence_epoch.json"
TRADE_EVIDENCE_DIR = CODEX_AUTOMATION_DIR / "trade_evidence"
DEFAULT_CODEX_TIMEOUT_SECONDS = 180.0
CODEX_DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["insufficient", "modified"]},
        "reason": {"type": "string"},
    },
    "required": ["decision", "reason"],
    "additionalProperties": False,
}

_ORIGINAL_STDOUT: Optional[Any] = None
_ORIGINAL_STDERR: Optional[Any] = None


class _ConsoleFilter(logging.Filter):
    """Drop file-only records so stdout/stderr tees don't double-print."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401 - succinct
        return not getattr(record, "file_only", False)


class _StreamTee:
    """Duplicate writes to a stream and the log file, without echoing twice."""

    def __init__(self, stream: Any, logger: logging.Logger, level: int, *, file_only: bool) -> None:
        self._stream = stream
        self._logger = logger
        self._level = level
        self._file_only = file_only
        self._buffer: str = ""

    def write(self, message: Any) -> int:
        if message is None:
            return 0
        if not isinstance(message, str):
            message = str(message)
        self._stream.write(message)
        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._logger.log(
                    self._level,
                    line,
                    extra={"file_only": self._file_only},
                )
        return len(message)

    def flush(self) -> None:
        if self._buffer:
            line = self._buffer.rstrip("\n")
            if line.strip():
                self._logger.log(
                    self._level,
                    line,
                    extra={"file_only": self._file_only},
                )
        self._buffer = ""
        self._stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._stream.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._stream, "encoding", "utf-8")


def _normalize_log_level(value: Any, default: int) -> int:
    if isinstance(value, int):
        return value
    if not value:
        return default
    lookup = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    return lookup.get(str(value).strip().upper(), default)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _coerce_finite_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _load_logging_config(path: Path) -> tuple[Dict[str, Any], Optional[str]]:
    if not path.exists():
        return {}, f"Config file missing at {path}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        logging_cfg = payload.get("logging", {})
        if isinstance(logging_cfg, dict):
            return logging_cfg, None
        return {}, "Config 'logging' section is not a dict"
    except Exception as exc:  # noqa: BLE001
        return {}, f"Failed to read logging config ({exc})"


def _ensure_codex_decision_schema() -> Path:
    CODEX_AUTOMATION_DIR.mkdir(parents=True, exist_ok=True)
    if not CODEX_DECISION_SCHEMA_PATH.exists():
        with CODEX_DECISION_SCHEMA_PATH.open("w", encoding="utf-8") as handle:
            json.dump(CODEX_DECISION_SCHEMA, handle, indent=2)
    return CODEX_DECISION_SCHEMA_PATH


def _truncate_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8"):
        pass


def _coerce_epoch(value: Any) -> Optional[int]:
    try:
        epoch = int(value)
    except (TypeError, ValueError):
        return None
    if epoch <= 0:
        return None
    return epoch


def _read_evidence_epoch_payload() -> Optional[Dict[str, Any]]:
    path = Path(__file__).resolve().parent / EVIDENCE_EPOCH_STATE_PATH
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _write_evidence_epoch_state(epoch: int, source: str, reason: str, verbose: bool) -> None:
    path = Path(__file__).resolve().parent / EVIDENCE_EPOCH_STATE_PATH
    payload_base = {
        "epoch": epoch,
        "source": source,
        "reason": reason,
        "pid": os.getpid(),
        "updated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "nonce": time.time_ns(),
    }
    payload_hash = hashlib.sha256(
        json.dumps(payload_base, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {**payload_base, "hash": payload_hash}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        if verbose:
            print(f"[trade] Failed to write evidence epoch state {path}: {exc}", file=sys.stderr)


def _get_current_evidence_epoch(verbose: bool) -> int:
    payload = _read_evidence_epoch_payload()
    epoch = _coerce_epoch((payload or {}).get("epoch"))
    if epoch is not None:
        return epoch
    epoch = 1
    _write_evidence_epoch_state(epoch, source="bootstrap", reason="initialize", verbose=verbose)
    return epoch


def _bump_evidence_epoch(source: str, reason: str, verbose: bool) -> int:
    current = _get_current_evidence_epoch(verbose=verbose)
    new_epoch = current + 1
    _write_evidence_epoch_state(new_epoch, source=source, reason=reason, verbose=verbose)
    return new_epoch


def _trade_epoch_cursor_path(epoch: int) -> Path:
    return Path(__file__).resolve().parent / TRADE_EVIDENCE_DIR / f"trade_epoch_{epoch}_cursor.json"


def _trade_epoch_evidence_path(session_day: str, epoch: int) -> Path:
    return Path(__file__).resolve().parent / TRADE_EVIDENCE_DIR / f"trade_day_{session_day}_epoch_{epoch}.jsonl"


def _load_trade_epoch_cursor(epoch: int) -> Dict[str, int]:
    path = _trade_epoch_cursor_path(epoch)
    try:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, TypeError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    offsets_raw = payload.get("day_start_offsets")
    if not isinstance(offsets_raw, dict):
        return {}
    offsets: Dict[str, int] = {}
    for day, raw in offsets_raw.items():
        if not isinstance(day, str):
            continue
        try:
            offset = int(raw)
        except (TypeError, ValueError):
            continue
        if offset < 0:
            continue
        offsets[day] = offset
    return offsets


def _persist_trade_epoch_cursor(epoch: int, day_start_offsets: Dict[str, int], verbose: bool) -> None:
    path = _trade_epoch_cursor_path(epoch)
    payload = {
        "epoch": epoch,
        "day_start_offsets": dict(sorted(day_start_offsets.items())),
        "updated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        if verbose:
            print(f"[trade] Failed to persist epoch cursor {path}: {exc}", file=sys.stderr)


def _ensure_trade_epoch_start_offset(
    *,
    session_day: str,
    epoch: int,
    completed_trades_path: Path,
    verbose: bool,
) -> int:
    day_start_offsets = _load_trade_epoch_cursor(epoch)
    existing = day_start_offsets.get(session_day)
    if existing is not None:
        return max(existing, 0)

    try:
        start_offset = completed_trades_path.stat().st_size if completed_trades_path.exists() else 0
    except OSError:
        start_offset = 0
    day_start_offsets[session_day] = max(int(start_offset), 0)
    _persist_trade_epoch_cursor(epoch, day_start_offsets, verbose)
    if verbose:
        print(
            f"[trade] Initialized epoch evidence cursor | epoch={epoch} day={session_day} "
            f"offset={day_start_offsets[session_day]}"
        )
    return day_start_offsets[session_day]


def _materialize_trade_epoch_evidence(
    *,
    session_day: str,
    epoch: int,
    completed_trades_path: Path,
    verbose: bool,
) -> tuple[Path, int]:
    start_offset = _ensure_trade_epoch_start_offset(
        session_day=session_day,
        epoch=epoch,
        completed_trades_path=completed_trades_path,
        verbose=verbose,
    )
    evidence_path = _trade_epoch_evidence_path(session_day, epoch)
    payload = ""
    if completed_trades_path.exists():
        try:
            with completed_trades_path.open("rb") as handle:
                handle.seek(max(start_offset, 0))
                payload = handle.read().decode("utf-8", errors="ignore")
        except OSError:
            payload = ""
    try:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(payload, encoding="utf-8")
    except OSError as exc:
        if verbose:
            print(f"[trade] Failed to write epoch evidence file {evidence_path}: {exc}", file=sys.stderr)
    line_count = 0
    if payload:
        line_count = sum(1 for line in payload.splitlines() if line.strip())
    return evidence_path, line_count


def _run_codex_trade_optimizer(
    evidence_trades_path: Path,
    *,
    session_day: str,
    evidence_epoch: int,
    verbose: bool,
    timeout_seconds: float,
) -> Dict[str, str]:
    codex_bin = shutil.which("codex")
    if not codex_bin:
        return {"decision": "insufficient", "reason": "codex_cli_missing"}
    try:
        if not evidence_trades_path.exists() or evidence_trades_path.stat().st_size <= 0:
            return {"decision": "insufficient", "reason": "trade_evidence_missing_or_empty"}
    except OSError:
        return {"decision": "insufficient", "reason": "trade_evidence_unreadable"}

    schema_path = _ensure_codex_decision_schema()
    output_path = CODEX_DECISION_OUTPUT_PATH
    try:
        _truncate_file(output_path)
    except OSError as exc:
        return {"decision": "insufficient", "reason": f"codex_output_unwritable:{exc}"}
    prompt = (
        "Inspect the epoch-scoped completed-trades evidence and decide whether one best code modification is justified.\n"
        f"Session day (IST): {session_day}\n"
        f"Evidence epoch: {evidence_epoch}\n"
        f"Epoch-scoped completed-trades JSONL path: {evidence_trades_path.resolve()}\n"
        "The file contains only records since the current evidence epoch started.\n"
        "Return decision='insufficient' if data is not strong enough, and do not edit files.\n"
        "Return decision='modified' only if you implemented exactly one best high-impact modification in this repo.\n"
        "Keep edits minimal, avoid destructive git operations, and do not run long live-trading loops.\n"
        "Always include a short reason."
    )
    cmd = [
        codex_bin,
        "exec",
        "--ephemeral",
        "--dangerously-bypass-approvals-and-sandbox",
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
        prompt,
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            capture_output=True,
            text=True,
            timeout=max(timeout_seconds, 1.0),
            check=False,
        )
    except FileNotFoundError:
        return {"decision": "insufficient", "reason": "codex_cli_missing"}
    except subprocess.TimeoutExpired:
        return {"decision": "insufficient", "reason": "codex_cli_timeout"}
    except Exception as exc:  # noqa: BLE001
        return {"decision": "insufficient", "reason": f"codex_cli_error:{exc}"}

    if completed.returncode != 0:
        if verbose:
            stderr = (completed.stderr or "").strip()
            if stderr:
                print(f"[trade] Codex CLI failed: {stderr}", file=sys.stderr)
        return {"decision": "insufficient", "reason": f"codex_cli_exit_{completed.returncode}"}

    try:
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:  # noqa: BLE001
        return {"decision": "insufficient", "reason": "codex_cli_invalid_output"}

    decision = str(payload.get("decision", "insufficient")).strip().lower()
    reason = str(payload.get("reason", "")).strip()
    if decision not in {"insufficient", "modified"}:
        return {"decision": "insufficient", "reason": "codex_cli_unknown_decision"}
    return {"decision": decision, "reason": reason}


def _cleanup_old_logs(
    log_dir: Path,
    prefix: str,
    retention_days: int,
    *,
    verbose: bool,
) -> None:
    if retention_days <= 0:
        retention_days = 1
    if not log_dir.exists():
        return
    cutoff = time.time() - (retention_days * 86400)
    removed = 0
    for path in log_dir.glob(f"{prefix}.log*"):
        try:
            if not path.is_file():
                continue
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    if verbose and removed:
        print(f"[trade] Cleaned up {removed} old log file(s) in {log_dir}")


def _setup_logging(verbose: bool) -> logging.Logger:
    global _ORIGINAL_STDOUT, _ORIGINAL_STDERR

    if _ORIGINAL_STDOUT is None:
        _ORIGINAL_STDOUT = sys.stdout
    if _ORIGINAL_STDERR is None:
        _ORIGINAL_STDERR = sys.stderr

    logging_cfg, logging_error = _load_logging_config(CONFIG_PATH)
    log_dir = Path(logging_cfg.get("dir", DEFAULT_LOG_DIR))
    file_prefix = str(logging_cfg.get("file_prefix", DEFAULT_LOG_FILE_PREFIX)).strip() or DEFAULT_LOG_FILE_PREFIX
    retention_days = int(logging_cfg.get("retention_days", DEFAULT_LOG_RETENTION_DAYS) or DEFAULT_LOG_RETENTION_DAYS)
    base_level = logging_cfg.get("level", DEFAULT_LOG_LEVEL)
    verbose_level = logging_cfg.get("verbose_level", DEFAULT_VERBOSE_LOG_LEVEL)
    console_level = _normalize_log_level(verbose_level if verbose else base_level, logging.INFO)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{file_prefix}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(stream=_ORIGINAL_STDOUT)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_ConsoleFilter())

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(console_handler)

    logging.captureWarnings(True)
    warnings.simplefilter("default")

    # Leave stdout/stderr untouched; file logging is handled only on fatal errors.

    _cleanup_old_logs(log_dir, file_prefix, retention_days, verbose=verbose)

    if verbose:
        print(
            "[trade] Logging to console (error log only on fatal) | "
            f"error_log={log_path} retention_days={retention_days} level={logging.getLevelName(console_level)}"
        )
        if logging_error:
            print(f"[trade] Logging config warning: {logging_error}")

    return logger


def _write_fatal_error(exc: BaseException) -> None:
    logging_cfg, _ = _load_logging_config(CONFIG_PATH)
    log_dir = Path(logging_cfg.get("dir", DEFAULT_LOG_DIR))
    file_prefix = str(logging_cfg.get("file_prefix", DEFAULT_LOG_FILE_PREFIX)).strip() or DEFAULT_LOG_FILE_PREFIX
    log_path = log_dir / f"{file_prefix}.log"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=INDIA_TZ).strftime("%Y-%m-%d %H:%M:%S")
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
        message = f"{timestamp} | ERROR | Fatal error: {exc}\n{trace}\n"
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(message)
    except OSError as write_exc:
        target_stderr = _ORIGINAL_STDERR or sys.stderr
        print(f"[trade] Failed to write fatal error log: {write_exc}", file=target_stderr)


def _ist_iso_from_any(value: Optional[Any]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        ts = value.to_pydatetime()
    elif isinstance(value, datetime):
        ts = value
    else:
        ts = pd.to_datetime(value).to_pydatetime()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=INDIA_TZ)
    else:
        ts = ts.astimezone(INDIA_TZ)
    return ts.isoformat()


def _update_trade_extremes(trade: Trade, price: float, timestamp: datetime) -> None:
    ts = timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=INDIA_TZ)
    ts_iso = ts.astimezone(INDIA_TZ).isoformat()
    if price > trade.highest_price or trade.highest_price_time is None:
        trade.highest_price = price
        trade.highest_price_time = ts_iso
    if price < trade.lowest_price or trade.lowest_price_time is None:
        trade.lowest_price = price
        trade.lowest_price_time = ts_iso


def _elapsed_hold_seconds(entry_time: str, now_ist: datetime) -> float:
    try:
        entry_dt = datetime.fromisoformat(entry_time)
    except ValueError:
        return 0.0
    if entry_dt.tzinfo is None:
        entry_dt = entry_dt.replace(tzinfo=INDIA_TZ)
    else:
        entry_dt = entry_dt.astimezone(INDIA_TZ)
    return max((now_ist - entry_dt).total_seconds(), 0.0)


def _parse_time(label: str, raw: Optional[str]) -> dtime:
    if not raw:
        raise ValueError(f"Missing required time configuration for '{label}'.")
    return datetime.strptime(raw, "%H:%M").time()


def _parse_optional_non_negative_float(label: str, raw: Any) -> Optional[float]:
    if raw in (None, ""):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative number when provided.") from exc
    if value < 0:
        raise ValueError(f"{label} must be >= 0 when provided.")
    return value


def _build_history_frames(snapshot: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    frames: Dict[str, Dict[str, pd.DataFrame]] = {}
    for timeframe, ticker_map in snapshot.items():
        frames[timeframe] = {}
        for ticker, rows in ticker_map.items():
            df = pd.DataFrame(rows)
            if df.empty or "date" not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
            frames[timeframe][ticker] = df
    return frames


def _extract_latest_prices(
    history: Dict[str, Dict[str, pd.DataFrame]],
    timeframe: str,
) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    timeframe_history = history.get(timeframe) or {}
    for ticker, df in timeframe_history.items():
        if df is None or df.empty:
            continue
        last_row = df.iloc[-1]
        price = last_row.get("close")
        if pd.notna(price):
            prices[ticker] = float(price)
    return prices


def _latest_atr_value(decision_df: pd.DataFrame, period: int) -> Optional[float]:
    if period <= 0 or decision_df.empty:
        return None
    required = {"high", "low", "close"}
    if any(col not in decision_df.columns for col in required):
        return None
    high = pd.to_numeric(decision_df["high"], errors="coerce")
    low = pd.to_numeric(decision_df["low"], errors="coerce")
    close = pd.to_numeric(decision_df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=(1.0 / period), adjust=False, min_periods=period).mean()
    if atr.empty:
        return None
    value = atr.iloc[-1]
    if pd.isna(value):
        return None
    value_f = float(value)
    return value_f if value_f > 0.0 else None


def _print_active_trades_summary(
    active_trades: Dict[str, Trade],
    prices: Dict[str, float],
    exchange: str,
    status_long_label: str,
) -> None:
    if not active_trades:
        print("[trade] Active trades: none")
        return
    print(f"[trade] Active trades ({len(active_trades)}):")
    for ticker, trade in active_trades.items():
        price = prices.get(ticker)
        pnl_display = "-"
        if price is not None:
            pnl = compute_net_percent_pnl(
                trade.direction == status_long_label,
                trade.entry_price,
                price,
                trade.quantity,
                exchange,
            )
            pnl_display = f"{pnl:.4f}%"
        price_display = f"{price:.2f}" if price is not None else "-"
        print(
            f"  - {ticker}: {trade.direction} qty={trade.quantity} "
            f"entry={trade.entry_price:.2f} latest={price_display} pnl={pnl_display}"
        )


def _sleep(poll_seconds: float, verbose: bool, reason: str) -> None:
    if poll_seconds <= 0:
        return
    if verbose:
        print(f"[trade] Sleeping {poll_seconds:.1f}s ({reason})")
    time.sleep(poll_seconds)


def _persist_active_trades_state(
    path: Path,
    active_trades: Dict[str, Trade],
    *,
    verbose: bool,
    current_prices: Optional[Dict[str, float]] = None,
    current_pnls: Optional[Dict[str, float]] = None,
) -> None:
    payload = {
        "updated_at": datetime.now(tz=INDIA_TZ).isoformat(),
        "active_trades": {
            ticker: {
                "ticker": trade.ticker,
                "direction": trade.direction,
                "quantity": trade.quantity,
                "entry_price": trade.entry_price,
                "entry_time": trade.entry_time,
                "highest_price": trade.highest_price,
                "lowest_price": trade.lowest_price,
                "highest_price_at": trade.highest_price_time,
                "lowest_price_at": trade.lowest_price_time,
                "trailing_stop_price": trade.trailing_stop_price,
                "take_profit_price": trade.take_profit_price,
                "current_price": (current_prices or {}).get(ticker),
                "current_percent_pnl": (current_pnls or {}).get(ticker),
                "data_timestamp": trade.data_timestamp,
            }
            for ticker, trade in active_trades.items()
        },
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(path)
        if verbose:
            print(f"[trade] Persisted {len(active_trades)} active trades to {path}")
    except OSError as exc:
        print(f"[trade] Failed to persist active trades: {exc}", file=sys.stderr)


def _completed_trades_log_path(exit_time: datetime) -> Path:
    ts = exit_time
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()  # type: ignore[assignment]
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=INDIA_TZ)
    session_date = ts.astimezone(INDIA_TZ).date().isoformat()
    return COMPLETED_TRADES_DIR / f"{session_date}.jsonl"


def _record_completed_trade(
    *,
    trade: Trade,
    exit_price: Optional[float],
    exit_reason: str,
    exit_time: datetime,
    mode: str,
    exchange: str,
    product: str,
    order_type: str,
    status_long_label: str,
    verbose: bool,
) -> None:
    ts = exit_time
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()  # type: ignore[assignment]
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=INDIA_TZ)
    ts = ts.astimezone(INDIA_TZ)
    exit_price_value = float(exit_price) if exit_price is not None else None
    pnl_percent: Optional[float] = None
    if exit_price_value is not None:
        pnl_percent = compute_net_percent_pnl(
            trade.direction == status_long_label,
            trade.entry_price,
            exit_price_value,
            trade.quantity,
            exchange,
        )
    record = {
        "session_date": ts.date().isoformat(),
        "ticker": trade.ticker,
        "direction": trade.direction,
        "quantity": trade.quantity,
        "entry_price": trade.entry_price,
        "exit_price": exit_price_value,
        "entry_time": trade.entry_time,
        "exit_time": ts.isoformat(),
        "signal_timestamp": trade.data_timestamp or None,
        "reason": exit_reason,
        "mode": mode,
        "exchange": exchange,
        "product": product,
        "order_type": order_type,
        "highest_price": trade.highest_price,
        "lowest_price": trade.lowest_price,
        "pnl_percent": pnl_percent,
    }
    path = _completed_trades_log_path(ts)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
        if verbose:
            print(f"[trade] Logged completed trade for {trade.ticker} ({record['reason']})")
    except OSError as exc:
        print(f"[trade] Failed to record completed trade for {trade.ticker}: {exc}", file=sys.stderr)


def _load_active_trades_state(
    path: Path,
    allowed_tickers: set[str],
    *,
    verbose: bool,
) -> Dict[str, Trade]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"[trade] Unable to load active trades from {path}: {exc}", file=sys.stderr)
        return {}

    raw_trades = payload.get("active_trades")
    if not isinstance(raw_trades, dict):
        return {}

    restored: Dict[str, Trade] = {}
    for ticker, raw in raw_trades.items():
        if allowed_tickers and ticker not in allowed_tickers:
            continue
        if not isinstance(raw, dict):
            continue
        try:
            data_ts = raw.get("data_timestamp")
            restored[ticker] = Trade(
                ticker=str(raw.get("ticker", ticker)),
                direction=str(raw["direction"]),
                quantity=int(raw["quantity"]),
                entry_price=float(raw["entry_price"]),
                entry_time=str(raw["entry_time"]),
                highest_price=float(raw.get("highest_price", raw["entry_price"])),
                lowest_price=float(raw.get("lowest_price", raw["entry_price"])),
                highest_price_time=raw.get("highest_price_at"),
                lowest_price_time=raw.get("lowest_price_at"),
                trailing_stop_price=(
                    float(raw["trailing_stop_price"])
                    if raw.get("trailing_stop_price") is not None
                    else None
                ),
                take_profit_price=(
                    float(raw["take_profit_price"])
                    if raw.get("take_profit_price") is not None
                    else None
                ),
                data_timestamp=str(data_ts) if data_ts not in (None, "") else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"[trade] Skipping corrupt active trade for {ticker}: {exc}", file=sys.stderr)
            continue

    if restored and verbose:
        print(f"[trade] Restored {len(restored)} active trades from {path}")
    return restored


def _sync_trades_with_broker(
    *,
    active_trades: Dict[str, Trade],
    allowed_tickers: set[str],
    mode: str,
    env_path: Path,
    exchange: str,
    product: str,
    status_long_label: str,
    status_short_label: str,
    verbose: bool,
) -> bool:
    """Align persisted active trades with the broker's current open positions."""
    mutated = False
    allowed = set(allowed_tickers)
    if allowed:
        for ticker in list(active_trades.keys()):
            if ticker not in allowed:
                if verbose:
                    print(f"[trade] Dropping persisted trade for untracked ticker {ticker}")
                del active_trades[ticker]
                mutated = True

    normalized_mode = str(mode or "paper").lower()
    try:
        broker_positions = fetch_open_positions(
            mode=normalized_mode,
            env_path=env_path,
            product=product,
            exchange=exchange,
            verbose=verbose,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[trade] Unable to fetch broker positions: {exc}", file=sys.stderr)
        return mutated

    if not broker_positions:
        if normalized_mode in {"live", "real"} and active_trades:
            if verbose:
                print("[trade] Broker reports no open positions; clearing local state")
            active_trades.clear()
            mutated = True
        return mutated

    processed: set[str] = set()
    now_iso = datetime.now(INDIA_TZ).isoformat()
    for ticker, meta in broker_positions.items():
        ticker_upper = ticker.upper()
        if allowed and ticker_upper not in allowed:
            if verbose:
                print(
                    f"[trade] Broker position for {ticker_upper} ignored; "
                    "not part of the configured tickers"
                )
            continue
        direction_raw = str(meta.get("direction", "")).lower()
        if direction_raw not in {"long", "short"}:
            continue
        quantity_raw = meta.get("quantity", 0)
        try:
            quantity = int(quantity_raw)
        except (TypeError, ValueError):
            continue
        if quantity <= 0:
            continue
        avg_price_raw = meta.get("average_price") or meta.get("entry_price") or 0.0
        try:
            entry_price = float(avg_price_raw)
        except (TypeError, ValueError):
            entry_price = 0.0
        if entry_price <= 0:
            continue
        last_price_raw = meta.get("last_price")
        try:
            last_price = float(last_price_raw)
        except (TypeError, ValueError):
            last_price = entry_price
        entry_time = str(meta.get("entry_time") or now_iso)
        direction_label = status_long_label if direction_raw == "long" else status_short_label
        processed.add(ticker_upper)

        trade = active_trades.get(ticker_upper)
        price_candidates = (
            last_price,
            entry_price,
            getattr(trade, "highest_price", entry_price),
            getattr(trade, "lowest_price", entry_price),
        )
        highest_price = max(price_candidates)
        lowest_price = min(price_candidates)
        if trade is None:
            active_trades[ticker_upper] = Trade(
                ticker=ticker_upper,
                direction=direction_label,
                quantity=quantity,
                entry_price=entry_price,
                entry_time=entry_time,
                highest_price=highest_price,
                lowest_price=lowest_price,
                highest_price_time=now_iso,
                lowest_price_time=now_iso,
                trailing_stop_price=None,
                take_profit_price=None,
                data_timestamp=now_iso,
            )
            mutated = True
            if verbose:
                print(
                    f"[trade] Reconstructed {direction_raw} position for {ticker_upper} "
                    "from broker snapshot"
                )
            continue

        changed = False
        if trade.direction != direction_label:
            trade.direction = direction_label
            trade.trailing_stop_price = None
            trade.take_profit_price = None
            changed = True
        if trade.quantity != quantity:
            trade.quantity = quantity
            changed = True
        if abs(trade.entry_price - entry_price) > 1e-6:
            trade.entry_price = entry_price
            trade.trailing_stop_price = None
            trade.take_profit_price = None
            changed = True
        if not trade.entry_time:
            trade.entry_time = entry_time
            changed = True
        if not getattr(trade, "data_timestamp", None):
            trade.data_timestamp = now_iso
        if highest_price > trade.highest_price:
            trade.highest_price = highest_price
            trade.highest_price_time = now_iso
            changed = True
        if lowest_price < trade.lowest_price:
            trade.lowest_price = lowest_price
            trade.lowest_price_time = now_iso
            changed = True
        trade.highest_price = highest_price
        trade.lowest_price = lowest_price
        mutated = mutated or changed
        if verbose and changed:
            print(f"[trade] Updated persisted trade for {ticker_upper} to match broker state")

    if normalized_mode in {"live", "real"}:
        for ticker in list(active_trades.keys()):
            if ticker not in processed:
                if verbose:
                    print(f"[trade] Broker is flat on {ticker}; removing from active trades")
                del active_trades[ticker]
                mutated = True

    return mutated


def _close_all_positions(
    *,
    active_trades: Dict[str, Trade],
    prices: Dict[str, float],
    mode: str,
    env_path: Path,
    state_path: Path,
    exchange: str,
    product: str,
    order_type: str,
    verbose: bool,
    reason: str,
    status_long_label: str,
    trade_status_indices: Dict[str, int],
    flat_status_idx: int,
) -> None:
    if not active_trades:
        _persist_active_trades_state(state_path, active_trades, verbose=verbose)
        return
    positions = {
        ticker: {"direction": trade.direction, "quantity": trade.quantity}
        for ticker, trade in active_trades.items()
    }
    executions = square_off_positions(
        positions,
        mode=mode,
        env_path=env_path,
        exchange=exchange,
        product=product,
        order_type=order_type,
        verbose=verbose,
    )
    for execution in executions:
        ticker = execution.get("ticker")
        price = execution.get("price")
        if ticker and price is not None:
            prices[ticker] = float(price)
    for ticker, trade in list(active_trades.items()):
        trade_status_indices[ticker] = flat_status_idx
        price = prices.get(ticker)
        if verbose and price is not None:
            is_long = trade.direction == status_long_label
            pnl = compute_net_percent_pnl(
                is_long,
                trade.entry_price,
                price,
                trade.quantity,
                exchange,
            )
            print(
                f"[trade] Closed {ticker} via {reason} | qty={trade.quantity} "
                f"entry={trade.entry_price:.2f} exit={price:.2f} pnl={pnl:.4f}%"
            )
        exit_ts = datetime.now(INDIA_TZ)
        _record_completed_trade(
            trade=trade,
            exit_price=price,
            exit_reason=reason,
            exit_time=exit_ts,
            mode=mode,
            exchange=exchange,
            product=product,
            order_type=order_type,
            status_long_label=status_long_label,
            verbose=verbose,
        )
        del active_trades[ticker]
    current_pnls = {
        t: compute_net_percent_pnl(
            trade.direction == status_long_label,
            trade.entry_price,
            prices.get(t),
            trade.quantity,
            exchange,
        )
        for t, trade in active_trades.items()
        if prices.get(t) is not None
    }
    _persist_active_trades_state(
        state_path,
        active_trades,
        verbose=verbose,
        current_prices=prices,
        current_pnls=current_pnls,
    )


def _load_model(config: Dict[str, Any], verbose: bool) -> LightRainbowDQN:
    if torch is None:
        raise RuntimeError("PyTorch is required to run live trading.")

    timeframes: list[str] = config.get("timeframes", [])
    lookback = int(config.get("train", {}).get("lookback", 1))
    hidden_layers = int(config.get("ml_rl", {}).get("hidden_layers_num", 1))
    input_dim = compute_input_dim(len(timeframes), lookback)
    model = LightRainbowDQN(input_dim=input_dim, hidden_layers_num=hidden_layers)
    model.eval()

    checkpoint = DEFAULT_WINNER_CHECKPOINT
    if not checkpoint.exists():
        raise FileNotFoundError(f"Winner checkpoint not found at {checkpoint}")
    state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    if verbose:
        print(
            f"[trade] Model ready | input_dim={input_dim} hidden_layers={hidden_layers} "
            f"checkpoint={checkpoint}"
        )
    return model


def _start_background_fetch(verbose: bool) -> None:
    if verbose:
        print("[trade] Priming historical dataset via fetch.py (background)")

    def _worker() -> None:
        try:
            fetch_run_pipeline(verbose=verbose)
            if verbose:
                print("[trade] Background fetch pipeline completed")
        except Exception as exc:  # noqa: BLE001
            print(
                f"[trade] Warning: fetch.py failed ({exc}); continuing without refresh.",
                file=sys.stderr,
            )

    thread = threading.Thread(target=_worker, name="fetch_pipeline", daemon=True)
    thread.start()


def _run_post_session_fetch(verbose: bool) -> None:
    if verbose:
        print("[trade] Refreshing historical dataset via fetch.py (post-session)")
    try:
        fetch_run_pipeline(verbose=verbose)
        if verbose:
            print("[trade] Post-session fetch pipeline completed")
    except Exception as exc:  # noqa: BLE001
        print(
            f"[trade] Warning: post-session fetch.py failed ({exc}); continuing without refresh.",
            file=sys.stderr,
        )


def _clear_trade_auto_repair_logs(verbose: bool) -> None:
    base_dir = Path(__file__).resolve().parent
    auto_repair_path = base_dir / AUTO_REPAIR_DIR
    marker_path = base_dir / AUTO_REPAIR_TRADE_DAY_MARKER
    today_ist = datetime.now(INDIA_TZ).date().isoformat()

    last_cleared_day: Optional[str] = None
    try:
        if marker_path.exists():
            last_cleared_day = marker_path.read_text(encoding="utf-8").strip() or None
    except OSError:
        last_cleared_day = None

    if last_cleared_day == today_ist:
        return

    cleared = 0
    if auto_repair_path.exists():
        for path in auto_repair_path.iterdir():
            if not path.is_file():
                continue
            if not path.name.startswith("trade"):
                continue
            try:
                with path.open("w", encoding="utf-8"):
                    pass
                cleared += 1
            except OSError as exc:
                if verbose:
                    print(f"[trade] Failed to clear auto-repair log {path}: {exc}", file=sys.stderr)

    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(today_ist + "\n", encoding="utf-8")
    except OSError as exc:
        if verbose:
            print(f"[trade] Failed to update auto-repair day marker {marker_path}: {exc}", file=sys.stderr)

    if verbose:
        previous = last_cleared_day or "none"
        print(
            f"[trade] Cleared {cleared} trade auto-repair log file(s) in {auto_repair_path} "
            f"(day rollover: {previous} -> {today_ist})"
        )


def _get_today_ist_date_str() -> str:
    return datetime.now(INDIA_TZ).date().isoformat()


def _read_trade_codex_modified_day() -> Optional[str]:
    path = Path(__file__).resolve().parent / CODEX_TRADE_MODIFIED_DAY_MARKER_PATH
    try:
        if not path.exists():
            return None
        value = path.read_text(encoding="utf-8").strip()
        return value or None
    except OSError:
        return None


def _mark_trade_codex_modified_today(verbose: bool) -> None:
    path = Path(__file__).resolve().parent / CODEX_TRADE_MODIFIED_DAY_MARKER_PATH
    today_ist = _get_today_ist_date_str()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(today_ist + "\n", encoding="utf-8")
    except OSError as exc:
        if verbose:
            print(f"[trade] Failed to persist Codex modified-day marker {path}: {exc}", file=sys.stderr)


def _read_shared_code_update_marker_fingerprint() -> Optional[str]:
    path = Path(__file__).resolve().parent / SHARED_CODE_UPDATE_MARKER_PATH
    try:
        if not path.exists():
            return None
        payload = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(payload).hexdigest()


def _write_shared_code_update_marker(
    source: str,
    reason: str,
    verbose: bool,
    evidence_epoch: Optional[int] = None,
) -> None:
    path = Path(__file__).resolve().parent / SHARED_CODE_UPDATE_MARKER_PATH
    payload_base = {
        "source": source,
        "reason": reason,
        "evidence_epoch": _coerce_epoch(evidence_epoch),
        "pid": os.getpid(),
        "updated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "nonce": time.time_ns(),
    }
    payload_hash = hashlib.sha256(
        json.dumps(payload_base, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {**payload_base, "hash": payload_hash}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        if verbose:
            print(f"[trade] Failed to write shared code update marker {path}: {exc}", file=sys.stderr)


def run(args: argparse.Namespace) -> None:
    _clear_trade_auto_repair_logs(args.verbose)

    config = load_config()
    model = _load_model(config, args.verbose)

    trading_cfg = config.get("trading", {})
    poll_seconds = (
        float(args.poll_seconds)
        if args.poll_seconds is not None
        else float(trading_cfg.get("poll_interval_seconds", 60))
    )
    if poll_seconds < 0:
        poll_seconds = 0.0

    mode = str(trading_cfg.get("mode", "paper")).lower()
    codex_timeout_seconds = _coerce_finite_float(trading_cfg.get("codex_cli_timeout_seconds"))
    if codex_timeout_seconds is None or codex_timeout_seconds <= 0:
        codex_timeout_seconds = DEFAULT_CODEX_TIMEOUT_SECONDS
    codex_cli_enabled = _coerce_bool(trading_cfg.get("codex_cli_enabled", True), default=True)
    product = DEFAULT_PRODUCT
    order_type = DEFAULT_ORDER_TYPE
    env_path = DEFAULT_ENV_PATH.expanduser()
    exchange = str(config.get("transaction_exchange") or config.get("exchange") or "NSE").upper()

    # Step 3: ensure Zerodha session is ready before entering the loop.
    ensure_access_token(env_path=env_path, verbose=args.verbose)

    tickers: list[str] = config.get("tickers", [])
    timeframes: list[str] = config.get("timeframes", [])
    if not tickers or not timeframes:
        raise ValueError("Config must define non-empty 'tickers' and 'timeframes'.")

    decision_interval: str = config.get("decision_interval")
    if decision_interval not in timeframes:
        raise ValueError("decision_interval must be one of the configured timeframes")

    lookback = int(config.get("train", {}).get("lookback", 1))
    trailing_stop_atr_multiplier = float(config.get("evaluation_trailing_stop_atr_multiplier", 2.0))
    atr_period = int(config.get("evaluation_trailing_stop_atr_period", 14))
    atr_multiplier = trailing_stop_atr_multiplier
    if atr_period <= 0:
        raise ValueError("evaluation_trailing_stop_atr_period must be > 0.")
    if atr_multiplier <= 0:
        raise ValueError("evaluation_trailing_stop_atr_multiplier must be > 0.")
    take_profit_atr_period = int(config.get("evaluation_take_profit_atr_period", 14))
    take_profit_atr_multiplier = float(config.get("evaluation_take_profit_atr_multiplier", 2.0))
    if take_profit_atr_period <= 0:
        raise ValueError("evaluation_take_profit_atr_period must be > 0.")
    if take_profit_atr_multiplier <= 0:
        raise ValueError("evaluation_take_profit_atr_multiplier must be > 0.")
    atr_lookback_needed = lookback
    atr_lookback_needed = max(atr_lookback_needed, atr_period + 2)
    atr_lookback_needed = max(atr_lookback_needed, take_profit_atr_period + 2)
    snapshot_lookback = atr_lookback_needed
    max_concurrent = int(config.get("max_concurrent_trades", 1))
    capital_per_ticker = float(config.get("capital_per_ticker", 0.0))
    leverage = float(config.get("leverage", 1.0))

    safe_start_time = _parse_time("safe_start_time", config.get("safe_start_time"))
    safe_end_time = _parse_time("safe_end_time", config.get("safe_end_time"))
    hard_end_time = _parse_time("hard_end_time", config.get("hard_end_time"))

    actions: list[str] = config.get("actions", MODEL_ACTIONS)
    action_lookup = {label.lower(): label for label in actions}
    missing = [name for name in MODEL_ACTIONS if name not in action_lookup]
    if missing:
        raise ValueError(f"Config actions missing required entries: {missing}")
    model_to_config_label = {
        idx: action_lookup.get(name, name) for idx, name in enumerate(MODEL_ACTIONS)
    }
    buy_idx = MODEL_ACTIONS.index("buy")
    sell_idx = MODEL_ACTIONS.index("sell")
    hold_idx = MODEL_ACTIONS.index("hold")
    ENTRY_CONFIRM_WINDOW = max(int(trading_cfg.get("entry_confirm_window", 3)), 1)
    signal_exit_min_hold_seconds = max(
        float(
            trading_cfg.get(
                "signal_exit_min_hold_seconds",
                config.get("evaluation_signal_exit_min_hold_seconds", 0),
            )
        ),
        0.0,
    )
    signal_exit_cost_floor_pct = _parse_optional_non_negative_float(
        "signal_exit_cost_floor_pct",
        trading_cfg.get(
            "signal_exit_cost_floor_pct",
            config.get("evaluation_signal_exit_cost_floor_pct"),
        ),
    )
    max_open_loss_pct = _parse_optional_non_negative_float(
        "max_open_loss_pct",
        trading_cfg.get(
            "max_open_loss_pct",
            config.get("evaluation_max_open_loss_pct"),
        ),
    )
    reentry_cooldown_seconds = max(
        float(
            trading_cfg.get(
                "reentry_cooldown_seconds",
                config.get("evaluation_reentry_cooldown_seconds", 0),
            )
        ),
        0.0,
    )
    entry_action_hist = {t: deque(maxlen=ENTRY_CONFIRM_WINDOW) for t in tickers}
    entry_conf_hist = {t: deque(maxlen=ENTRY_CONFIRM_WINDOW) for t in tickers}
    last_seen_signal_ts = {t: None for t in tickers}

    trade_status_list: list[str] = config.get("trade_status", [])
    status_long_label = _require_label(trade_status_list, "long")
    status_short_label = _require_label(trade_status_list, "short")
    status_flat_label = _require_label(trade_status_list, "flat")
    trade_status_map = {label: idx for idx, label in enumerate(trade_status_list)}
    flat_status_idx = trade_status_map[status_flat_label]
    long_status_idx = trade_status_map[status_long_label]
    short_status_idx = trade_status_map[status_short_label]
    total_trade_status = len(trade_status_list)

    allowed_tickers = set(tickers)
    active_trades: Dict[str, Trade] = _load_active_trades_state(
        ACTIVE_TRADES_PATH,
        allowed_tickers,
        verbose=args.verbose,
    )
    broker_sync_mutated = _sync_trades_with_broker(
        active_trades=active_trades,
        allowed_tickers=allowed_tickers,
        mode=mode,
        env_path=env_path,
        exchange=exchange,
        product=product,
        status_long_label=status_long_label,
        status_short_label=status_short_label,
        verbose=args.verbose,
    )
    if broker_sync_mutated:
        _persist_active_trades_state(ACTIVE_TRADES_PATH, active_trades, verbose=args.verbose)

    # --- Intraday-only guard: clear carried state (broker already auto-squares off) ---
    today_ist = datetime.now(INDIA_TZ).date()

    removed = []
    for ticker, trade in list(active_trades.items()):
        try:
            dt = datetime.fromisoformat(trade.entry_time)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=INDIA_TZ)
            if dt.astimezone(INDIA_TZ).date() != today_ist:
                removed.append(ticker)
                del active_trades[ticker]
        except Exception:
            # If entry_time is malformed, drop it rather than risking carry-over
            removed.append(ticker)
            del active_trades[ticker]

    if removed:
        if args.verbose:
            print(f"[trade] Startup intraday guard: cleared stale active_trades: {sorted(removed)}")
        _persist_active_trades_state(ACTIVE_TRADES_PATH, active_trades, verbose=args.verbose)
    # --- end intraday-only guard ---

    trade_status_indices: Dict[str, int] = {ticker: flat_status_idx for ticker in tickers}
    for ticker, trade in active_trades.items():
        if trade.direction == status_long_label:
            trade_status_indices[ticker] = long_status_idx
        elif trade.direction == status_short_label:
            trade_status_indices[ticker] = short_status_idx
        else:
            trade_status_indices[ticker] = flat_status_idx
    last_exit_time_ist: Dict[str, datetime] = {}

    today = datetime.now(INDIA_TZ).date()
    hard_cutoff = datetime.combine(today, hard_end_time, tzinfo=INDIA_TZ)
    evidence_epoch = _get_current_evidence_epoch(args.verbose)
    session_day = today.isoformat()
    session_completed_trades_path = _completed_trades_log_path(datetime.now(INDIA_TZ))
    _ensure_trade_epoch_start_offset(
        session_day=session_day,
        epoch=evidence_epoch,
        completed_trades_path=session_completed_trades_path,
        verbose=args.verbose,
    )

    if args.verbose:
        trailing_stop_desc = f"atr(period={atr_period},mult={atr_multiplier:.4f})"
        take_profit_desc = f"atr(period={take_profit_atr_period},mult={take_profit_atr_multiplier:.4f})"
        signal_exit_cost_floor_desc = (
            f"{signal_exit_cost_floor_pct:.4f}%"
            if signal_exit_cost_floor_pct is not None
            else "auto(roundtrip_cost)"
        )
        max_open_loss_desc = (
            f"{max_open_loss_pct:.4f}%"
            if max_open_loss_pct is not None
            else "disabled"
        )
        print(
            f"[trade] Starting live loop | mode={mode} safe_window={safe_start_time}-{safe_end_time} "
            f"hard_end={hard_end_time} poll={poll_seconds:.1f}s "
            f"trailing_stop={trailing_stop_desc} take_profit={take_profit_desc} "
            f"signal_exit_min_hold={signal_exit_min_hold_seconds:.1f}s "
            f"signal_exit_cost_floor={signal_exit_cost_floor_desc} "
            f"max_open_loss={max_open_loss_desc} "
            f"reentry_cooldown={reentry_cooldown_seconds:.1f}s"
        )

    cycle_index = 0
    seen_marker_fingerprint = _read_shared_code_update_marker_fingerprint()

    def _print_cycle_start(cycle_number: int, started_at: datetime) -> None:
        if args.verbose:
            return
        print(
            f"[trade] c={cycle_number} start={started_at.isoformat(timespec='seconds')} "
            f"active={len(active_trades)} mode={mode} ",
            end="",
            flush=True,
        )

    def _print_cycle_end(
        cycle_number: int,
        started_perf: float,
        reason: str,
        *,
        entries: int,
        exits: int,
        codex_status: str,
    ) -> None:
        if args.verbose:
            return
        elapsed = max(time.perf_counter() - started_perf, 0.0)
        print(
            f"end reason={reason} dt={elapsed:.1f}s "
            f"entry={entries} exit={exits} active={len(active_trades)}",
            flush=True,
        )

    try:
        while True:
            current_marker_fingerprint = _read_shared_code_update_marker_fingerprint()
            if current_marker_fingerprint != seen_marker_fingerprint:
                print("[trade] Shared code update marker changed; restarting process.", flush=True)
                try:
                    os.execv(sys.executable, [sys.executable, *sys.argv])
                except OSError as exc:
                    print(f"[trade] Failed to restart on shared code marker update: {exc}", file=sys.stderr)
                    seen_marker_fingerprint = current_marker_fingerprint

            current_evidence_epoch = _get_current_evidence_epoch(args.verbose)
            if current_evidence_epoch != evidence_epoch:
                if args.verbose:
                    print(
                        f"[trade] Evidence epoch changed ({evidence_epoch} -> {current_evidence_epoch}); "
                        "switching Codex evidence stream."
                    )
                evidence_epoch = current_evidence_epoch
                session_day = _get_today_ist_date_str()
                session_completed_trades_path = _completed_trades_log_path(datetime.now(INDIA_TZ))
                _ensure_trade_epoch_start_offset(
                    session_day=session_day,
                    epoch=evidence_epoch,
                    completed_trades_path=session_completed_trades_path,
                    verbose=args.verbose,
                )

            cycle_index += 1
            cycle_started_perf = time.perf_counter()
            now_ist = datetime.now(INDIA_TZ)
            cycle_entries = 0
            cycle_exits = 0
            _print_cycle_start(cycle_index, now_ist)
            if now_ist >= hard_cutoff:
                if args.verbose:
                    print("[trade] Hard end time reached; square-off and stop.")
                cycle_exits += len(active_trades)
                _close_all_positions(
                    active_trades=active_trades,
                    prices={},
                    mode=mode,
                    env_path=env_path,
                    state_path=ACTIVE_TRADES_PATH,
                    exchange=exchange,
                    product=product,
                    order_type=order_type,
                    verbose=args.verbose,
                    reason="hard_end",
                    status_long_label=status_long_label,
                    trade_status_indices=trade_status_indices,
                    flat_status_idx=flat_status_idx,
                )
                _print_cycle_end(
                    cycle_index,
                    cycle_started_perf,
                    "hard_end",
                    entries=cycle_entries,
                    exits=cycle_exits,
                    codex_status="skip",
                )
                break

            try:
                snapshot = fetch_market_snapshot(
                    tickers,
                    timeframes,
                    snapshot_lookback,
                    env_path=env_path,
                    exchange=exchange,
                    verbose=args.verbose,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[trade] Market snapshot failed: {exc}", file=sys.stderr)
                _print_cycle_end(
                    cycle_index,
                    cycle_started_perf,
                    "snapshot_error",
                    entries=cycle_entries,
                    exits=cycle_exits,
                    codex_status="skip",
                )
                _sleep(poll_seconds, args.verbose, "snapshot_error")
                continue
    
            history = _build_history_frames(snapshot)
            decision_history = history.get(decision_interval)
            if not decision_history:
                if args.verbose:
                    print("[trade] No decision timeframe data available; retrying")
                _print_cycle_end(
                    cycle_index,
                    cycle_started_perf,
                    "missing_decision_interval",
                    entries=cycle_entries,
                    exits=cycle_exits,
                    codex_status="skip",
                )
                _sleep(poll_seconds, args.verbose, "missing_decision_interval")
                continue
    
            local_time_now = now_ist.time()
            if local_time_now < safe_start_time:
                if args.verbose:
                    print(
                        f"[trade] Current time {now_ist} before safe start {safe_start_time}; waiting"
                    )
                _print_cycle_end(
                    cycle_index,
                    cycle_started_perf,
                    "pre_safe_window",
                    entries=cycle_entries,
                    exits=cycle_exits,
                    codex_status="skip",
                )
                _sleep(poll_seconds, args.verbose, "pre_safe_window")
                continue
            decision_positions = {
                ticker: compute_day_positions(decision_history.get(ticker, pd.DataFrame()))
                for ticker in tickers
            }

            ticker_features: Dict[str, Any] = {}
            prices: Dict[str, float] = {}
            trailing_atr_values: Dict[str, float] = {}
            take_profit_atr_values: Dict[str, float] = {}
            missing_feature_reasons: Dict[str, list[str]] = {}
            signal_timestamps: Dict[str, Optional[pd.Timestamp]] = {}
            for ticker in tickers:
                decision_df = decision_history.get(ticker)
                signal_ts = None
                if decision_df is not None and not decision_df.empty:
                    ts = decision_df["date"].iloc[-1]
                    if isinstance(ts, pd.Timestamp):
                        signal_ts = ts
                signal_timestamps[ticker] = signal_ts
                ticker_history = {tf: history.get(tf, {}).get(ticker) for tf in timeframes}
                missing_frames = [
                    tf for tf, df in ticker_history.items() if df is None or df.empty
                ]
                if missing_frames:
                    missing_feature_reasons.setdefault(ticker, []).append(
                        "missing data for " + ", ".join(sorted(missing_frames))
                    )
                    continue
                features = build_feature_vector(
                    ticker,
                    signal_ts,
                    history,
                    timeframes,
                    lookback,
                    decision_positions,
                    trade_status_indices.get(ticker, flat_status_idx),
                    total_trade_status,
                )
                if features is None:
                    missing_feature_reasons.setdefault(ticker, []).append(
                        "insufficient lookback/positions for decision data"
                    )
                    continue
                ticker_features[ticker] = features
                if decision_df is not None and not decision_df.empty:
                    price = decision_df["close"].iloc[-1]
                    if pd.notna(price):
                        prices[ticker] = float(price)
                        atr_value = _latest_atr_value(decision_df, atr_period)
                        if atr_value is not None:
                            trailing_atr_values[ticker] = atr_value
                        if take_profit_atr_period == atr_period:
                            atr_value = trailing_atr_values.get(ticker)
                        else:
                            atr_value = _latest_atr_value(decision_df, take_profit_atr_period)
                        if atr_value is not None:
                            take_profit_atr_values[ticker] = atr_value
                    else:
                        missing_feature_reasons.setdefault(ticker, []).append(
                            "missing close price in decision timeframe"
                        )
                else:
                    missing_feature_reasons.setdefault(ticker, []).append(
                        "no decision timeframe history"
                    )

            if not ticker_features:
                if args.verbose:
                    print("[trade] Features missing for available tickers; retrying")
                    for ticker in tickers:
                        reasons = missing_feature_reasons.get(ticker)
                        if not reasons:
                            continue
                        joined = "; ".join(reasons)
                        print(f"[trade]   -> {ticker}: {joined}")
                _print_cycle_end(
                    cycle_index,
                    cycle_started_perf,
                    "missing_features",
                    entries=cycle_entries,
                    exits=cycle_exits,
                    codex_status="skip",
                )
                _sleep(poll_seconds, args.verbose, "missing_features")
                continue
    
            inference = infer_actions(model, ticker_features, model_to_config_label)
            entry_inference = {t: dict(res) for t, res in inference.items()}
            for ticker, res in entry_inference.items():
                if ticker in active_trades:
                    res["action_idx"] = hold_idx
                    res["action"] = model_to_config_label.get(hold_idx, "hold")
                    res["confidence"] = 0.0
                    continue

                sig_ts = signal_timestamps.get(ticker)
                if sig_ts is None or sig_ts == last_seen_signal_ts.get(ticker):
                    pass
                else:
                    last_seen_signal_ts[ticker] = sig_ts
                    entry_action_hist[ticker].append(int(res.get("action_idx", hold_idx)))
                    entry_conf_hist[ticker].append(float(res.get("confidence", 0.0)))

                if len(entry_action_hist[ticker]) < ENTRY_CONFIRM_WINDOW:
                    confirmed = hold_idx
                else:
                    window = list(entry_action_hist[ticker])
                    confirmed = (
                        window[0]
                        if (window[0] != hold_idx and all(a == window[0] for a in window))
                        else hold_idx
                    )

                res["action_idx"] = confirmed
                res["action"] = model_to_config_label.get(confirmed, "hold")
                res["confidence"] = (
                    sum(entry_conf_hist[ticker]) / ENTRY_CONFIRM_WINDOW
                    if confirmed != hold_idx
                    else 0.0
                )

            actionable = [
                (ticker, data)
                for ticker, data in entry_inference.items()
                if int(data.get("action_idx", hold_idx)) != hold_idx
            ]
            max_conf_ticker = (
                max(actionable, key=lambda item: item[1]["confidence"])[0]
                if actionable
                else None
            )
    
            if local_time_now >= safe_end_time:
                if args.verbose:
                    print("[trade] Safe end window reached; closing open trades")
                cycle_exits += len(active_trades)
                _close_all_positions(
                    active_trades=active_trades,
                    prices=prices,
                    mode=mode,
                    env_path=env_path,
                    state_path=ACTIVE_TRADES_PATH,
                    exchange=exchange,
                    product=product,
                    order_type=order_type,
                    verbose=args.verbose,
                    reason="safe_end",
                    status_long_label=status_long_label,
                    trade_status_indices=trade_status_indices,
                    flat_status_idx=flat_status_idx,
                )
                _print_cycle_end(
                    cycle_index,
                    cycle_started_perf,
                    "safe_end",
                    entries=cycle_entries,
                    exits=cycle_exits,
                    codex_status="skip",
                )
                _sleep(poll_seconds, args.verbose, "safe_end_wait")
                continue

            # Exit logic for active positions (steps 4.e and 4.f)
            for ticker, trade in list(active_trades.items()):
                price = prices.get(ticker)
                if price is None:
                    continue
                _update_trade_extremes(trade, price, now_ist)
                action_idx = int(inference.get(ticker, {}).get("action_idx", hold_idx))
                hold_seconds = _elapsed_hold_seconds(trade.entry_time, now_ist)
                if trade.direction == status_long_label:
                    atr_value = trailing_atr_values.get(ticker)
                    stop_price = trade.trailing_stop_price
                    if atr_value is not None:
                        candidate_stop = trade.highest_price - (atr_multiplier * atr_value)
                        stop_price = (
                            max(stop_price, candidate_stop)
                            if stop_price is not None
                            else candidate_stop
                        )
                        trade.trailing_stop_price = stop_price
                    take_profit_price = trade.take_profit_price
                    if take_profit_price is None:
                        atr_value = take_profit_atr_values.get(ticker)
                        if atr_value is not None:
                            take_profit_price = trade.entry_price + (take_profit_atr_multiplier * atr_value)
                            trade.take_profit_price = take_profit_price
                    exit_signal = action_idx == sell_idx
                    gross_pct_now = compute_percent_pnl(True, trade.entry_price, price)
                    effective_signal_exit_cost_floor = signal_exit_cost_floor_pct
                    if effective_signal_exit_cost_floor is None:
                        effective_signal_exit_cost_floor = estimate_roundtrip_cost_pct(
                            trade.entry_price,
                            trade.quantity,
                            True,
                            exchange,
                        )
                    signal_exit_cost_floor_ok = (
                        gross_pct_now < 0.0
                        or gross_pct_now >= effective_signal_exit_cost_floor
                    )
                    signal_exit_allowed = (
                        exit_signal
                        and (
                            signal_exit_min_hold_seconds <= 0.0
                            or hold_seconds >= signal_exit_min_hold_seconds
                        )
                        and signal_exit_cost_floor_ok
                    )
                    net_pct_now = compute_net_percent_pnl(
                        True,
                        trade.entry_price,
                        price,
                        trade.quantity,
                        exchange,
                    )
                    exit_take_profit = take_profit_price is not None and price >= take_profit_price
                    exit_trailing = stop_price is not None and price <= stop_price
                    exit_max_loss = (
                        max_open_loss_pct is not None
                        and net_pct_now <= (-max_open_loss_pct)
                    )
                    should_exit = (
                        signal_exit_allowed
                        or exit_take_profit
                        or exit_max_loss
                        or exit_trailing
                    )
                    if should_exit:
                        submit_market_order(
                            ticker=ticker,
                            side="sell",
                            quantity=trade.quantity,
                            mode=mode,
                            env_path=env_path,
                            exchange=exchange,
                            product=product,
                            order_type=order_type,
                            verbose=args.verbose,
                            tag="exit_long",
                        )
                        pnl = net_pct_now
                        exit_reason = (
                            "signal"
                            if signal_exit_allowed
                            else "take_profit"
                            if exit_take_profit
                            else "max_loss"
                            if exit_max_loss
                            else "trailing_stop"
                        )
                        if args.verbose:
                            print(
                                f"[trade] Exit long {ticker} via {exit_reason} | qty={trade.quantity} "
                                f"entry={trade.entry_price:.2f} exit={price:.2f} pnl={pnl:.4f}%"
                            )
                        _record_completed_trade(
                            trade=trade,
                            exit_price=price,
                            exit_reason=exit_reason,
                            exit_time=datetime.now(INDIA_TZ),
                            mode=mode,
                            exchange=exchange,
                            product=product,
                            order_type=order_type,
                            status_long_label=status_long_label,
                            verbose=args.verbose,
                        )
                        cycle_exits += 1
                        trade_status_indices[ticker] = flat_status_idx
                        last_exit_time_ist[ticker] = now_ist
                        del active_trades[ticker]
                        _persist_active_trades_state(
                            ACTIVE_TRADES_PATH,
                            active_trades,
                            verbose=args.verbose,
                        )
                elif trade.direction == status_short_label:
                    atr_value = trailing_atr_values.get(ticker)
                    stop_price = trade.trailing_stop_price
                    if atr_value is not None:
                        candidate_stop = trade.lowest_price + (atr_multiplier * atr_value)
                        stop_price = (
                            min(stop_price, candidate_stop)
                            if stop_price is not None
                            else candidate_stop
                        )
                        trade.trailing_stop_price = stop_price
                    take_profit_price = trade.take_profit_price
                    if take_profit_price is None:
                        atr_value = take_profit_atr_values.get(ticker)
                        if atr_value is not None:
                            take_profit_price = trade.entry_price - (take_profit_atr_multiplier * atr_value)
                            trade.take_profit_price = take_profit_price
                    exit_signal = action_idx == buy_idx
                    gross_pct_now = compute_percent_pnl(False, trade.entry_price, price)
                    effective_signal_exit_cost_floor = signal_exit_cost_floor_pct
                    if effective_signal_exit_cost_floor is None:
                        effective_signal_exit_cost_floor = estimate_roundtrip_cost_pct(
                            trade.entry_price,
                            trade.quantity,
                            False,
                            exchange,
                        )
                    signal_exit_cost_floor_ok = (
                        gross_pct_now < 0.0
                        or gross_pct_now >= effective_signal_exit_cost_floor
                    )
                    signal_exit_allowed = (
                        exit_signal
                        and (
                            signal_exit_min_hold_seconds <= 0.0
                            or hold_seconds >= signal_exit_min_hold_seconds
                        )
                        and signal_exit_cost_floor_ok
                    )
                    net_pct_now = compute_net_percent_pnl(
                        False,
                        trade.entry_price,
                        price,
                        trade.quantity,
                        exchange,
                    )
                    exit_take_profit = take_profit_price is not None and price <= take_profit_price
                    exit_trailing = stop_price is not None and price >= stop_price
                    exit_max_loss = (
                        max_open_loss_pct is not None
                        and net_pct_now <= (-max_open_loss_pct)
                    )
                    should_exit = (
                        signal_exit_allowed
                        or exit_take_profit
                        or exit_max_loss
                        or exit_trailing
                    )
                    if should_exit:
                        submit_market_order(
                            ticker=ticker,
                            side="buy",
                            quantity=trade.quantity,
                            mode=mode,
                            env_path=env_path,
                            exchange=exchange,
                            product=product,
                            order_type=order_type,
                            verbose=args.verbose,
                            tag="exit_short",
                        )
                        pnl = net_pct_now
                        exit_reason = (
                            "signal"
                            if signal_exit_allowed
                            else "take_profit"
                            if exit_take_profit
                            else "max_loss"
                            if exit_max_loss
                            else "trailing_stop"
                        )
                        if args.verbose:
                            print(
                                f"[trade] Exit short {ticker} via {exit_reason} | qty={trade.quantity} "
                                f"entry={trade.entry_price:.2f} exit={price:.2f} pnl={pnl:.4f}%"
                            )
                        _record_completed_trade(
                            trade=trade,
                            exit_price=price,
                            exit_reason=exit_reason,
                            exit_time=datetime.now(INDIA_TZ),
                            mode=mode,
                            exchange=exchange,
                            product=product,
                            order_type=order_type,
                            status_long_label=status_long_label,
                            verbose=args.verbose,
                        )
                        cycle_exits += 1
                        trade_status_indices[ticker] = flat_status_idx
                        last_exit_time_ist[ticker] = now_ist
                        del active_trades[ticker]
                        _persist_active_trades_state(
                            ACTIVE_TRADES_PATH,
                            active_trades,
                            verbose=args.verbose,
                        )

            if len(active_trades) >= max_concurrent:
                if args.verbose:
                    _print_active_trades_summary(
                        active_trades,
                        prices,
                        exchange,
                        status_long_label,
                    )
                _persist_active_trades_state(
                    ACTIVE_TRADES_PATH,
                    active_trades,
                    verbose=args.verbose,
                    current_prices=prices,
                    current_pnls={
                        t: compute_net_percent_pnl(
                            trade.direction == status_long_label,
                            trade.entry_price,
                            prices.get(t),
                            trade.quantity,
                            exchange,
                        )
                        for t, trade in active_trades.items()
                        if prices.get(t) is not None
                    },
                )
                _print_cycle_end(
                    cycle_index,
                    cycle_started_perf,
                    "max_concurrent",
                    entries=cycle_entries,
                    exits=cycle_exits,
                    codex_status="skip",
                )
                _sleep(poll_seconds, args.verbose, "max_concurrent_reached")
                continue

            # Entry logic (steps 4.g and 4.h)
            for ticker, result in entry_inference.items():
                if ticker in active_trades:
                    continue
                if max_conf_ticker is None or ticker != max_conf_ticker:
                    continue
                if len(active_trades) >= max_concurrent:
                    break
                if reentry_cooldown_seconds > 0.0:
                    last_exit = last_exit_time_ist.get(ticker)
                    if last_exit is not None:
                        since_exit = (now_ist - last_exit).total_seconds()
                        if since_exit < reentry_cooldown_seconds:
                            if args.verbose:
                                print(
                                    f"[trade] Skip {ticker} entry due to cooldown | "
                                    f"elapsed={since_exit:.1f}s required={reentry_cooldown_seconds:.1f}s"
                                )
                            continue
                action_idx = int(result.get("action_idx", hold_idx))
                price = prices.get(ticker)
                if price is None:
                    continue
                quantity = compute_quantity(capital_per_ticker, leverage, price)
                if quantity <= 0:
                    continue
                signal_ts = signal_timestamps.get(ticker)
                initial_stop_price: Optional[float] = None
                initial_take_profit_price: Optional[float] = None
                if action_idx == buy_idx:
                    trailing_atr = trailing_atr_values.get(ticker)
                    take_profit_atr = take_profit_atr_values.get(ticker)
                    if trailing_atr is None or take_profit_atr is None:
                        continue
                    initial_stop_price = price - (atr_multiplier * trailing_atr)
                    initial_take_profit_price = price + (take_profit_atr_multiplier * take_profit_atr)
                    submit_market_order(
                        ticker=ticker,
                        side="buy",
                        quantity=quantity,
                        mode=mode,
                        env_path=env_path,
                        exchange=exchange,
                        product=product,
                        order_type=order_type,
                        verbose=args.verbose,
                        tag="enter_long",
                    )
                    active_trades[ticker] = Trade(
                        ticker=ticker,
                        direction=status_long_label,
                        quantity=quantity,
                        entry_price=price,
                        entry_time=datetime.now(INDIA_TZ).isoformat(),
                        highest_price=price,
                        lowest_price=price,
                        highest_price_time=now_ist.isoformat(),
                        lowest_price_time=now_ist.isoformat(),
                        trailing_stop_price=initial_stop_price,
                        take_profit_price=initial_take_profit_price,
                        data_timestamp=_ist_iso_from_any(signal_ts),
                    )
                    cycle_entries += 1
                    trade_status_indices[ticker] = long_status_idx
                    if args.verbose:
                        print(
                            f"[trade] Enter long {ticker} | qty={quantity} price={price:.2f} "
                            f"confidence={result.get('confidence'):.4f}"
                        )
                    _persist_active_trades_state(
                        ACTIVE_TRADES_PATH,
                        active_trades,
                        verbose=args.verbose,
                    )
                elif action_idx == sell_idx:
                    trailing_atr = trailing_atr_values.get(ticker)
                    take_profit_atr = take_profit_atr_values.get(ticker)
                    if trailing_atr is None or take_profit_atr is None:
                        continue
                    initial_stop_price = price + (atr_multiplier * trailing_atr)
                    initial_take_profit_price = price - (take_profit_atr_multiplier * take_profit_atr)
                    submit_market_order(
                        ticker=ticker,
                        side="sell",
                        quantity=quantity,
                        mode=mode,
                        env_path=env_path,
                        exchange=exchange,
                        product=product,
                        order_type=order_type,
                        verbose=args.verbose,
                        tag="enter_short",
                    )
                    active_trades[ticker] = Trade(
                        ticker=ticker,
                        direction=status_short_label,
                        quantity=quantity,
                        entry_price=price,
                        entry_time=datetime.now(INDIA_TZ).isoformat(),
                        highest_price=price,
                        lowest_price=price,
                        highest_price_time=now_ist.isoformat(),
                        lowest_price_time=now_ist.isoformat(),
                        trailing_stop_price=initial_stop_price,
                        take_profit_price=initial_take_profit_price,
                        data_timestamp=_ist_iso_from_any(signal_ts),
                    )
                    cycle_entries += 1
                    trade_status_indices[ticker] = short_status_idx
                    if args.verbose:
                        print(
                            f"[trade] Enter short {ticker} | qty={quantity} price={price:.2f} "
                            f"confidence={result.get('confidence'):.4f}"
                        )
                    _persist_active_trades_state(
                        ACTIVE_TRADES_PATH,
                        active_trades,
                        verbose=args.verbose,
                    )
    
            if args.verbose:
                _print_active_trades_summary(
                    active_trades,
                    prices,
                    exchange,
                    status_long_label,
                )
            current_prices = {
                t: p for t, p in prices.items() if t in active_trades and p is not None
            }
            current_pnls: Dict[str, float] = {}
            for t, trade in active_trades.items():
                price = prices.get(t)
                if price is None:
                    continue
                current_pnls[t] = compute_net_percent_pnl(
                    trade.direction == status_long_label,
                    trade.entry_price,
                    price,
                    trade.quantity,
                    exchange,
                )
            _persist_active_trades_state(
                ACTIVE_TRADES_PATH,
                active_trades,
                verbose=args.verbose,
                current_prices=current_prices,
                current_pnls=current_pnls,
            )
            _print_cycle_end(
                cycle_index,
                cycle_started_perf,
                "cycle_complete",
                entries=cycle_entries,
                exits=cycle_exits,
                codex_status=("deferred" if codex_cli_enabled else "disabled"),
            )
            _sleep(poll_seconds, args.verbose, "cycle_complete")
    
    finally:
        final_prices: Dict[str, float] = {}
        if active_trades:
            try:
                snapshot = fetch_market_snapshot(
                    tickers,
                    [decision_interval],
                    max(snapshot_lookback, 2),
                    env_path=env_path,
                    exchange=exchange,
                    verbose=args.verbose,
                )
                history = _build_history_frames(snapshot)
                final_prices = _extract_latest_prices(history, decision_interval)
            except Exception as exc:  # noqa: BLE001
                if args.verbose:
                    print(f"[trade] Warning: failed to refresh exit prices ({exc})")
        _close_all_positions(
            active_trades=active_trades,
            prices=final_prices,
            mode=mode,
            env_path=env_path,
            state_path=ACTIVE_TRADES_PATH,
            exchange=exchange,
            product=product,
            order_type=order_type,
            verbose=args.verbose,
            reason="shutdown",
            status_long_label=status_long_label,
            trade_status_indices=trade_status_indices,
            flat_status_idx=flat_status_idx,
        )
    _run_post_session_fetch(args.verbose)

    if codex_cli_enabled:
        today_ist = _get_today_ist_date_str()
        modified_day = _read_trade_codex_modified_day()
        if modified_day == today_ist:
            print("[trade] End-of-day Codex decision=skipped (already_modified_today).")
        else:
            final_completed_path = _completed_trades_log_path(datetime.now(INDIA_TZ))
            evidence_path, evidence_count = _materialize_trade_epoch_evidence(
                session_day=today_ist,
                epoch=evidence_epoch,
                completed_trades_path=final_completed_path,
                verbose=args.verbose,
            )
            codex_result = _run_codex_trade_optimizer(
                evidence_path,
                session_day=today_ist,
                evidence_epoch=evidence_epoch,
                verbose=args.verbose,
                timeout_seconds=codex_timeout_seconds,
            )
            decision = str(codex_result.get("decision", "insufficient")).strip().lower()
            reason = str(codex_result.get("reason", "")).strip() or "no reason provided"
            if decision == "modified":
                _mark_trade_codex_modified_today(args.verbose)
                next_epoch = _bump_evidence_epoch("trade", reason, args.verbose)
                _write_shared_code_update_marker(
                    "trade",
                    reason,
                    args.verbose,
                    evidence_epoch=next_epoch,
                )
                print(
                    f"[trade] End-of-day Codex applied best modification ({reason}); "
                    f"evidence_records={evidence_count} next_epoch={next_epoch}; exiting."
                )
            else:
                print(
                    f"[trade] End-of-day Codex decision=insufficient ({reason}) "
                    f"epoch={evidence_epoch} evidence_records={evidence_count}"
                )
    else:
        print("[trade] End-of-day Codex decision=skipped (codex_cli_enabled=false).")

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute live trades using the RL agent.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=None,
        help="Override the config poll interval in seconds.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _setup_logging(args.verbose)
    except Exception as exc:  # noqa: BLE001
        print(f"[trade] Warning: logging setup failed ({exc})", file=sys.stderr)
    try:
        run(args)
    except KeyboardInterrupt:
        if args.verbose:
            print("[trade] Interrupted by user; exiting.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[trade] Fatal error: {exc}", file=sys.stderr)
        _write_fatal_error(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
