"""Replay memory generation pipeline driven by historical market data."""

from __future__ import annotations

from collections import deque
import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

import time as _time
try:
    import resource  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    resource = None  # type: ignore[assignment]
try:
    import psutil  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

from ml_rl import LightRainbowDQN, Tensor, _TORCH_IMPORT_ERROR  # type: ignore[attr-defined]
from ml_rl import torch  # type: ignore[attr-defined]

DATA_ROOT = Path(__file__).resolve().parent / "data"
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
TRAINING_REPLAY_PATH = DATA_ROOT / "replay_memory_training.jsonl"
EVALUATION_REPLAY_PATH = DATA_ROOT / "replay_memory_evaluation.jsonl"
INDIA_TZ = ZoneInfo("Asia/Kolkata")
MODEL_ACTIONS = ["hold", "buy", "sell"]

_HISTORY_CACHE: Dict[Path, Tuple[int, int, pd.DataFrame]] = {}

ZERODHA_BROKERAGE_PCT = 0.0003  # 0.03%
ZERODHA_BROKERAGE_CAP = 20.0  # Rs per executed order
ZERODHA_STT_PCT = 0.00025  # 0.025% on sell side
ZERODHA_TRANSACTION_PCT = {
    "NSE": 0.0000297,  # 0.00297%
    "BSE": 0.0000375,  # 0.00375%
}
ZERODHA_GST_PCT = 0.18  # 18%
ZERODHA_SEBI_PER_VALUE = 10.0 / 10_000_000  # ₹10 per crore
ZERODHA_STAMP_PCT = 0.00003  # 0.003% on buy side



@dataclass
class Trade:
    ticker: str
    direction: str
    quantity: int
    entry_price: float
    entry_time: str
    highest_price: float
    lowest_price: float
    highest_price_time: Optional[str] = None
    lowest_price_time: Optional[str] = None
    data_timestamp: Optional[str] = None


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing configuration file at {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_input_dim(timeframe_count: int, lookback: int) -> int:
    return (5 * timeframe_count * lookback) + 1 + 1


def ensure_torch_available() -> None:
    if _TORCH_IMPORT_ERROR is not None or torch is None:
        raise RuntimeError(
            "PyTorch is required for replay generation. Install dependencies via "
            "`pip install -r requirements.txt`. "
            f"Original error: {_TORCH_IMPORT_ERROR}"
        )


def _usage_snapshot() -> Optional[Dict[str, float]]:
    if resource is not None:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
            return {
                "user": float(usage.ru_utime),
                "sys": float(usage.ru_stime),
                "rss_mb": getattr(usage, "ru_maxrss", 0) / 1024.0,
            }
        except Exception:
            pass
    if psutil is not None:
        try:
            proc = psutil.Process()
            times = proc.cpu_times()
            mem = proc.memory_info()
            rss_bytes = getattr(mem, "peak_wset", getattr(mem, "rss", 0))
            return {
                "user": float(getattr(times, "user", 0.0)),
                "sys": float(getattr(times, "system", 0.0)),
                "rss_mb": float(rss_bytes) / (1024.0 * 1024.0),
            }
        except Exception:
            pass
    return None


def build_model(
    input_dim: int,
    hidden_layers: int,
    checkpoint_dir: Path,
    verbose: bool = False,
) -> LightRainbowDQN:
    ensure_torch_available()
    model = LightRainbowDQN(input_dim=input_dim, hidden_layers_num=hidden_layers)
    model.eval()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"light_rainbow_{input_dim}_{hidden_layers}.pt"
    if ckpt_path.exists():
        if verbose:
            print(f"[replay] Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
    elif verbose:
        print(f"[replay] No checkpoint found at {ckpt_path}; starting with fresh weights")
    return model


def _load_history_frame(file_path: Path) -> pd.DataFrame:
    stat = file_path.stat()
    cached = _HISTORY_CACHE.get(file_path)
    if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]
    df = pd.read_parquet(file_path)
    if "date" not in df.columns:
        raise ValueError(f"'date' column missing in {file_path}")
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    _HISTORY_CACHE[file_path] = (stat.st_mtime_ns, stat.st_size, df)
    return df


def load_history_for_ticker_timeframe(ticker: str, timeframe: str) -> pd.DataFrame:
    parquet_path = DATA_ROOT / ticker / timeframe / "history.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Missing historical data at {parquet_path}. "
            "Run fetch.py to generate Parquet history files."
        )
    return _load_history_frame(parquet_path)


def load_all_history(
    tickers: Iterable[str],
    timeframes: Iterable[str],
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, List[str]]]:
    ticker_list = list(tickers)
    timeframe_list = list(timeframes)
    cache: Dict[str, Dict[str, pd.DataFrame]] = {timeframe: {} for timeframe in timeframe_list}
    missing: Dict[str, List[str]] = {}

    for ticker in ticker_list:
        missing_frames: List[str] = []
        for timeframe in timeframe_list:
            try:
                cache[timeframe][ticker] = load_history_for_ticker_timeframe(ticker, timeframe)
            except FileNotFoundError:
                missing_frames.append(timeframe)
        if missing_frames:
            missing[ticker] = missing_frames
            for timeframe in timeframe_list:
                cache[timeframe].pop(ticker, None)

    return cache, missing


def normalize_window(values: np.ndarray) -> np.ndarray:
    mins = values.min(axis=0, keepdims=True)
    maxs = values.max(axis=0, keepdims=True)
    denom = np.where((maxs - mins) == 0, 1.0, maxs - mins)
    return (values - mins) / denom


def extract_window_features(
    df: pd.DataFrame,
    timestamp: Optional[pd.Timestamp],
    lookback: int,
) -> Optional[np.ndarray]:
    window: pd.DataFrame
    if timestamp is not None:
        idx = df["date"].searchsorted(timestamp, side="right")
        window = df.iloc[max(0, idx - lookback):idx]
        if len(window) < lookback:
            window = df.iloc[-lookback:]
    else:
        window = df.iloc[-lookback:]
    if len(window) < lookback:
        return None
    columns = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in columns if col not in window.columns]
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols} in historical data for {timestamp}")
    arr = window[columns].to_numpy(dtype=float)
    return arr


def compute_day_positions(df: pd.DataFrame) -> Dict[pd.Timestamp, Tuple[int, int]]:
    positions: Dict[pd.Timestamp, Tuple[int, int]] = {}
    if df.empty:
        return positions
    local_df = df.copy()
    local_df["day"] = local_df["date"].dt.tz_convert(INDIA_TZ).dt.normalize()
    for _, group in local_df.groupby("day"):
        group_sorted = group.sort_values("date")
        total = len(group_sorted)
        for idx, ts in enumerate(group_sorted["date"]):
            positions[ts] = (idx, total)
    return positions


# --- Proximity-based reward helpers ---

def _proximity_P_for_day(close: float, day_low: float, day_high: float) -> float:
    # P in [-1, +1]; +1 at day's highest high; -1 at day's lowest low
    if day_high <= day_low:
        return 0.0
    p = -1.0 + 2.0 * ((close - day_low) / (day_high - day_low))
    return float(max(-1.0, min(1.0, p)))


def _build_daily_P_lookup(decision_df: pd.DataFrame) -> Dict[pd.Timestamp, float]:
    # decision_df has columns: date (UTC tz-aware), high, low, close
    df = decision_df.copy()
    df["day"] = df["date"].dt.tz_convert(INDIA_TZ).dt.normalize()
    daily = df.groupby("day").agg(day_high=("high", "max"), day_low=("low", "min")).reset_index()
    df = df.merge(daily, on="day", how="left")
    P = [
        _proximity_P_for_day(c, lo, hi)
        for c, lo, hi in zip(
            df["close"].to_numpy(float),
            df["day_low"].to_numpy(float),
            df["day_high"].to_numpy(float),
        )
    ]
    return {ts: p for ts, p in zip(df["date"], P)}


def _build_daily_range_factor_lookup(decision_df: pd.DataFrame) -> Dict[pd.Timestamp, float]:
    """
    For each timestamp, return (day_high - day_low) / day_low * 100
    i.e. the intraday % range, broadcast to all candles of that day.
    """
    df = decision_df.copy()
    df["day"] = df["date"].dt.tz_convert(INDIA_TZ).dt.normalize()

    daily = df.groupby("day").agg(day_high=("high", "max"), day_low=("low", "min")).reset_index()
    df = df.merge(daily, on="day", how="left")

    # range_pct = (day_high - day_low) / day_low * 100, safe against day_low == 0
    day_low = df["day_low"].to_numpy(float)
    day_high = df["day_high"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        range_pct = np.where(day_low > 0.0, (day_high - day_low) / day_low * 100.0, 0.0)

    return {ts: float(rp) for ts, rp in zip(df["date"], range_pct)}


def _f(P: float) -> float:
    return -0.5 * (P**3 - P**2 + P + 1.0)


def _reward_buy(P: float) -> float:
    return _f(P)


def _reward_sell(P: float) -> float:
    return _f(-P)


def _reward_hold_flat(P: float) -> float:
    return 1.0 - 2.0 * (P**2)


def _reward_hold_long(P: float) -> float:
    return -_f(-P)


def _reward_hold_short(P: float) -> float:
    return -_f(P)


def _reward_from_P(
    P: float,
    action_label: str,
    trade_status_idx: int,
    flat_idx: int,
    long_idx: int,
    short_idx: int,
) -> float:
    action = str(action_label).lower()
    if action == "buy":
        return _reward_buy(P)
    if action == "sell":
        return _reward_sell(P)
    if trade_status_idx == long_idx:
        return _reward_hold_long(P)
    if trade_status_idx == short_idx:
        return _reward_hold_short(P)
    return _reward_hold_flat(P)


def get_time_component(ts: pd.Timestamp) -> time:
    if ts.tzinfo is None:
        return ts.time()
    return ts.tz_convert(INDIA_TZ).time()


def normalize_trade_status(idx: int, total_status: int) -> float:
    if total_status <= 1:
        return 0.0
    return idx / (total_status - 1)


def normalize_candle_position(pos: int, total: int) -> float:
    if total <= 1:
        return 0.0
    return pos / (total - 1)


def compute_common_timestamps(decision_data: Dict[str, pd.DataFrame]) -> List[pd.Timestamp]:
    timestamp_sets = [set(df["date"]) for df in decision_data.values()]
    common = set.intersection(*timestamp_sets) if timestamp_sets else set()
    return sorted(common)


def compute_min_valid_timestamp(
    history: Dict[str, Dict[str, pd.DataFrame]],
    timeframes: Iterable[str],
    lookback: int,
) -> Optional[pd.Timestamp]:
    min_candidate: Optional[pd.Timestamp] = None
    for timeframe in timeframes:
        for df in history[timeframe].values():
            if len(df) < lookback:
                return None
            candidate = df["date"].iloc[lookback - 1]
            if min_candidate is None or candidate > min_candidate:
                min_candidate = candidate
    return min_candidate


def build_feature_vector(
    ticker: str,
    timestamp: Optional[pd.Timestamp],
    history: Dict[str, Dict[str, pd.DataFrame]],
    timeframes: List[str],
    lookback: int,
    decision_positions: Dict[str, Dict[pd.Timestamp, Tuple[int, int]]],
    trade_status_idx: int,
    total_trade_status: int,
) -> Optional[np.ndarray]:
    raw_segments: List[np.ndarray] = []
    for timeframe in timeframes:
        df = history[timeframe][ticker]
        window = extract_window_features(df, timestamp, lookback)
        if window is None:
            return None
        raw_segments.append(window)
    if not raw_segments:
        return None
    stacked = np.concatenate(raw_segments, axis=0)
    mins = stacked.min(axis=0, keepdims=True)
    maxs = stacked.max(axis=0, keepdims=True)
    denom = np.where((maxs - mins) == 0.0, 1.0, maxs - mins)
    eps = 1e-6
    safe_mins = np.where(mins == 0.0, eps, mins)
    range_pct = ((maxs - mins) / safe_mins) * 100.0
    normalized_segments: List[np.ndarray] = []
    for segment in raw_segments:
        norm = (segment - mins) / denom
        scaled = norm * range_pct
        normalized_segments.append(scaled.flatten())
    features = np.concatenate(normalized_segments, axis=0)

    position_lookup = decision_positions.get(ticker, {})
    ts_key = timestamp
    if ts_key is None or ts_key not in position_lookup:
        if position_lookup:
            ts_key = max(position_lookup)
    pos_total = position_lookup.get(ts_key)
    if pos_total is None:
        pos_total = (0, max(1, len(position_lookup)))
    pos, total = pos_total
    candle_norm = normalize_candle_position(pos, total)
    status_norm = normalize_trade_status(trade_status_idx, total_trade_status)
    return np.concatenate([features, np.array([candle_norm, status_norm], dtype=np.float32)], axis=0)


def infer_actions(
    model: LightRainbowDQN,
    features: Dict[str, np.ndarray],
    label_lookup: Dict[int, str],
) -> Dict[str, Dict[str, Any]]:
    tickers = list(features.keys())
    matrix = np.stack([features[t] for t in tickers], axis=0).astype(np.float32)
    tensor = torch.from_numpy(matrix)
    with torch.no_grad():
        q_distribution = model(tensor)
        q_vals = q_distribution.mean(dim=2)
    results: Dict[str, Dict[str, Any]] = {}
    for idx, ticker in enumerate(tickers):
        ticker_q = q_vals[idx].cpu().numpy()
        action_idx = int(np.argmax(ticker_q))
        default_label = MODEL_ACTIONS[action_idx] if 0 <= action_idx < len(MODEL_ACTIONS) else str(action_idx)
        action_label = label_lookup.get(action_idx, default_label)
        results[ticker] = {
            "action_idx": action_idx,
            "action": action_label,
            "confidence": float(np.max(ticker_q)),
            "q_values": ticker_q.tolist(),
        }
    return results


def compute_quantity(capital: float, leverage: float, price: float) -> int:
    if price <= 0:
        return 0
    qty = math.floor((capital * leverage) / price)
    return max(qty, 0)


def compute_percent_pnl(is_long: bool, entry_price: float, exit_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    if is_long:
        return ((exit_price - entry_price) / entry_price) * 100
    return ((entry_price - exit_price) / entry_price) * 100


def _resolve_transaction_rate(exchange: str) -> float:
    return ZERODHA_TRANSACTION_PCT.get(exchange.upper(), ZERODHA_TRANSACTION_PCT["NSE"])


def _compute_order_cost(
    order_value: float,
    *,
    is_buy: bool,
    is_sell: bool,
    exchange: str,
) -> float:
    if order_value <= 0:
        return 0.0
    brokerage = min(order_value * ZERODHA_BROKERAGE_PCT, ZERODHA_BROKERAGE_CAP)
    sebi = order_value * ZERODHA_SEBI_PER_VALUE
    transaction = order_value * _resolve_transaction_rate(exchange)
    stt = order_value * ZERODHA_STT_PCT if is_sell else 0.0
    stamp = order_value * ZERODHA_STAMP_PCT if is_buy else 0.0
    gst = ZERODHA_GST_PCT * (brokerage + sebi + transaction)
    return brokerage + sebi + transaction + stt + stamp + gst


def compute_intraday_costs(
    entry_price: float,
    exit_price: float,
    quantity: int,
    is_long: bool,
    exchange: str,
) -> float:
    if quantity <= 0:
        return 0.0
    entry_value = abs(entry_price * quantity)
    exit_value = abs(exit_price * quantity)
    entry_cost = _compute_order_cost(
        entry_value,
        is_buy=is_long,
        is_sell=not is_long,
        exchange=exchange,
    )
    exit_cost = _compute_order_cost(
        exit_value,
        is_buy=not is_long,
        is_sell=is_long,
        exchange=exchange,
    )
    return entry_cost + exit_cost


def compute_net_percent_pnl(
    is_long: bool,
    entry_price: float,
    exit_price: float,
    quantity: int,
    exchange: str,
) -> float:
    if entry_price <= 0 or quantity <= 0:
        return 0.0
    gross_value = (
        (exit_price - entry_price) * quantity if is_long else (entry_price - exit_price) * quantity
    )
    base_value = entry_price * quantity
    costs = compute_intraday_costs(entry_price, exit_price, quantity, is_long, exchange)
    net_value = gross_value - costs
    return (net_value / base_value) * 100 if base_value != 0 else 0.0


def append_replay_event(
    replay: List[Dict[str, Any]],
    timestamp: pd.Timestamp,
    ticker_actions: Dict[str, Dict[str, Any]],
    executed_events: List[Dict[str, Any]],
    active_trades: Dict[str, Trade],
    pnl_snapshot: Dict[str, float],
    percent_pnl_snapshot: Dict[str, float],
) -> None:
    replay.append(
        {
            "timestamp": timestamp.isoformat(),
            "ticker_actions": ticker_actions,
            "executed_events": executed_events,
            "active_trades": {
                ticker: {
                    "direction": trade.direction,
                    "quantity": trade.quantity,
                    "entry_price": trade.entry_price,
                    "entry_time": trade.entry_time,
                    "highest_price": trade.highest_price,
                    "lowest_price": trade.lowest_price,
                }
                for ticker, trade in active_trades.items()
            },
            "pnl_pct": pnl_snapshot,
            "percent_pnl": percent_pnl_snapshot,
        }
    )


def _augment_actions(
    inference_results: Dict[str, Dict[str, Any]],
    feature_map: Dict[str, np.ndarray],
    pnl_snapshot: Dict[str, float],
    model_to_config_label: Dict[int, str],
) -> Dict[str, Dict[str, Any]]:
    enriched: Dict[str, Dict[str, Any]] = {}
    for ticker, data in inference_results.items():
        feats = feature_map.get(ticker)
        if feats is None:
            continue
        model_idx = int(data.get("action_idx", 2))
        default_label = MODEL_ACTIONS[model_idx] if 0 <= model_idx < len(MODEL_ACTIONS) else str(model_idx)
        enriched[ticker] = {
            **data,
            "pnl_pct": pnl_snapshot.get(ticker, 0.0),
            "action_label": model_to_config_label.get(model_idx, default_label),
            "features": feats.tolist(),
        }
    return enriched


def _fill_reward_snapshot_from_P(
    ts: pd.Timestamp,
    pnl_snapshot: Dict[str, float],
    ticker_features: Dict[str, np.ndarray],
    inference_results: Dict[str, Dict[str, Any]],
    executed_events: List[Dict[str, Any]],
    status_before: Dict[str, int],
    P_lookup: Dict[str, Dict[pd.Timestamp, float]],
    range_lookup: Dict[str, Dict[pd.Timestamp, float]],
    flat_status_idx: int,
    long_status_idx: int,
    short_status_idx: int,
) -> None:
    executed_map = {}
    for event in executed_events:
        if event.get("type") not in ("entry", "exit"):
            continue
        ticker = event.get("ticker")
        action = event.get("action")
        if not ticker or not action:
            continue
        executed_map[ticker] = action

    for ticker in ticker_features.keys():
        P = float(P_lookup.get(ticker, {}).get(ts, 0.0))
        range_factor = float(range_lookup.get(ticker, {}).get(ts, 0.0))  # range_pct * 100
        action_label = executed_map.get(ticker)
        if not action_label:
            inference = inference_results.get(ticker, {})
            ai = int(inference.get("action_idx", 0))
            if 0 <= ai < len(MODEL_ACTIONS):
                action_label = MODEL_ACTIONS[ai]
            else:
                action_label = "hold"
        st_idx = int(status_before.get(ticker, flat_status_idx))
        base_reward = _reward_from_P(
            P,
            action_label,
            st_idx,
            flat_status_idx,
            long_status_idx,
            short_status_idx,
        )
        pnl_snapshot[ticker] = base_reward * range_factor


def _update_reward_stats_from_snapshot(
    pnl_snapshot: Dict[str, float],
    inference_results: Dict[str, Dict[str, Any]],
    reward_sum_by_action: Dict[str, float],
    reward_sq_by_action: Dict[str, float],
    reward_cnt_by_action: Dict[str, int],
    reward_neg_by_action: Dict[str, int],
) -> None:
    for ticker, result in inference_results.items():
        action_idx = int(result.get("action_idx", 2))
        canonical_key = None
        if 0 <= action_idx < len(MODEL_ACTIONS):
            canonical_key = MODEL_ACTIONS[action_idx]
        if not canonical_key or canonical_key not in reward_sum_by_action:
            continue
        reward_val = float(pnl_snapshot.get(ticker, 0.0))
        reward_sum_by_action[canonical_key] += reward_val
        reward_sq_by_action[canonical_key] += reward_val * reward_val
        reward_cnt_by_action[canonical_key] += 1
        if reward_val < 0:
            reward_neg_by_action[canonical_key] += 1


def save_replay_memory(records: List[Dict[str, Any]], output_path: Path, verbose: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    if verbose:
        print(f"[replay] Saved {len(records)} transitions to {output_path}")


def _require_label(options: List[str], keyword: str) -> str:
    for value in options:
        if value.lower() == keyword.lower():
            return value
    raise ValueError(f"Config list {options} must include '{keyword}'")


def _select_timestamps(
    timestamps: List[pd.Timestamp],
    evaluation_days: int,
    min_valid_ts: Optional[pd.Timestamp],
    mode: str,
    training_days: Optional[int] = None,
) -> List[pd.Timestamp]:
    if not timestamps:
        return []

    def to_local_date(ts: pd.Timestamp) -> datetime.date:
        return ts.tz_convert(INDIA_TZ).date() if ts.tzinfo else ts.date()

    unique_days = sorted({to_local_date(ts) for ts in timestamps})
    eval_day_set: set[datetime.date] = set()
    if evaluation_days > 0 and evaluation_days <= len(unique_days):
        eval_day_set = set(unique_days[-evaluation_days:])

    if mode == "training":
        available_days = [day for day in unique_days if day not in eval_day_set]
        if training_days and training_days > 0 and training_days < len(available_days):
            allowed_days = set(available_days[-training_days:])
        else:
            allowed_days = set(available_days)
        selected = [ts for ts in timestamps if to_local_date(ts) in allowed_days]
    elif mode == "evaluation":
        selected = [ts for ts in timestamps if to_local_date(ts) in eval_day_set]
    else:
        raise ValueError(f"Unknown replay mode '{mode}'")

    if min_valid_ts is not None:
        selected = [ts for ts in selected if ts >= min_valid_ts]

    return selected


def _generate_replay_memory(
    mode: str,
    output_path: Path,
    verbose: bool = False,
    override_trailing_stop: Optional[float] = None,
    override_take_profit: Optional[float] = None,
) -> None:
    config = load_config()
    tickers: List[str] = config.get("tickers", [])
    timeframes: List[str] = config.get("timeframes", [])
    fetch_cfg = config.get("fetch", {})
    train_cfg = config.get("train", {})
    ml_cfg = config.get("ml_rl", {})
    train_mode = str(train_cfg.get("mode", "RL")).upper()
    if train_mode not in ("RL", "SL"):
        raise ValueError("train.mode must be either 'RL' or 'SL'")
    use_supervised = mode == "training" and train_mode == "SL"

    if not tickers:
        raise ValueError("No tickers configured in config.json")
    if not timeframes:
        raise ValueError("No timeframes configured in config.json")

    if mode == "training":
        ENTRY_CONFIRM_WINDOW = int(config.get("training_entry_confirm_window", 1))
    else:
        ENTRY_CONFIRM_WINDOW = int(config.get("evaluation_entry_confirm_window", 3))
    ENTRY_CONFIRM_WINDOW = max(ENTRY_CONFIRM_WINDOW, 1)
    entry_action_hist = {t: deque(maxlen=ENTRY_CONFIRM_WINDOW) for t in tickers}
    entry_conf_hist = {t: deque(maxlen=ENTRY_CONFIRM_WINDOW) for t in tickers}

    action_labels: List[str] = config.get("actions", [])
    if not action_labels:
        raise ValueError("Config must define an 'actions' list.")
    label_lower_map = {label.lower(): label for label in action_labels}
    missing = [name for name in MODEL_ACTIONS if name not in label_lower_map]
    if missing:
        raise ValueError(f"Config actions missing required entries: {missing}")
    buy_label = label_lower_map["buy"]
    sell_label = label_lower_map["sell"]
    model_to_config_label = {
        idx: label_lower_map.get(name, name) for idx, name in enumerate(MODEL_ACTIONS)
    }
    canonical_index = {name: idx for idx, name in enumerate(MODEL_ACTIONS)}
    buy_idx = canonical_index["buy"]
    sell_idx = canonical_index["sell"]
    hold_idx = canonical_index["hold"]

    lookback = int(train_cfg.get("lookback"))
    hidden_layers = int(ml_cfg.get("hidden_layers_num"))
    checkpoint_dir = Path("checkpoints")

    decision_interval = config.get("decision_interval")
    training_days = int(config.get("training_days_num", 0))
    evaluation_days = int(config.get("evaluation_days_num", 0))
    safe_start_time = datetime.strptime(config.get("safe_start_time"), "%H:%M").time()
    safe_end_time = datetime.strptime(config.get("safe_end_time"), "%H:%M").time()
    trailing_stop_key = (
        "training_trailing_stop_loss_pct" if mode == "training" else "evaluation_trailing_stop_loss_pct"
    )
    trailing_stop = float(config.get(trailing_stop_key))
    take_profit_key = (
        "training_take_profit_pct" if mode == "training" else "evaluation_take_profit_pct"
    )
    take_profit = float(config.get(take_profit_key))
    if override_trailing_stop is not None:
        trailing_stop = float(override_trailing_stop)
    if override_take_profit is not None:
        take_profit = float(override_take_profit)
    max_concurrent_trades = int(config.get("max_concurrent_trades"))
    capital_per_ticker = float(config.get("capital_per_ticker"))
    leverage = float(config.get("leverage"))
    cost_exchange = str(config.get("transaction_exchange") or config.get("exchange") or "NSE").upper()
    trade_status_list: List[str] = config.get("trade_status", [])
    if not trade_status_list:
        raise ValueError("Config must define 'trade_status' list.")
    trade_status_map = {name: idx for idx, name in enumerate(trade_status_list)}
    status_flat_label = _require_label(trade_status_list, "flat")
    status_long_label = _require_label(trade_status_list, "long")
    status_short_label = _require_label(trade_status_list, "short")

    flat_status_idx = trade_status_map[status_flat_label]
    long_status_idx = trade_status_map[status_long_label]
    short_status_idx = trade_status_map[status_short_label]

    num_candles = int(fetch_cfg.get("num_candles"))

    input_dim = compute_input_dim(len(timeframes), lookback)
    if verbose:
        print(
            f"[replay:{mode}] Model configuration -> "
            f"input_dim={input_dim}, hidden_layers={hidden_layers}, lookback={lookback}"
        )
    model = None
    if not use_supervised:
        model = build_model(input_dim, hidden_layers, checkpoint_dir, verbose=verbose)
        if verbose:
            ckpt_name = f"light_rainbow_{input_dim}_{hidden_layers}.pt"
            print(
                "[config] ACTIONS={'hold':%d,'buy':%d,'sell':%d} | "
                "decision_interval=%s | lookback=%d | timeframes=%d | "
                "input_dim=%d | hidden_layers=%d | ckpt=%s"
                % (
                    MODEL_ACTIONS.index("hold"),
                    MODEL_ACTIONS.index("buy"),
                    MODEL_ACTIONS.index("sell"),
                    decision_interval,
                    lookback,
                    len(timeframes),
                    input_dim,
                    hidden_layers,
                    ckpt_name,
                )
            )
            print(
                "[config:reward] reward_fn=proximity_cubic | "
                f"trailing_stop={trailing_stop:.4f} | "
                f"take_profit={take_profit:.4f} | "
                f"safe_window={safe_start_time.strftime('%H:%M')}-{safe_end_time.strftime('%H:%M')} | "
                f"shorting_allowed={'short' in trade_status_list}"
            )
    elif verbose:
        print(
            f"[replay:{mode}] Supervised mode enabled (SL); "
            "skipping model load for training replay generation."
        )

    if decision_interval not in timeframes:
        raise ValueError("decision_interval must be one of the configured timeframes")

    if verbose:
        print(f"[replay:{mode}] Loading historical data for all tickers/timeframes")
    history, missing_history = load_all_history(tickers, timeframes)

    available_tickers = [ticker for ticker in tickers if ticker in history[decision_interval]]
    if missing_history:
        for ticker in sorted(missing_history):
            frames = ", ".join(sorted(missing_history[ticker]))
            print(
                f"[replay:{mode}] Warning: historical data missing for '{ticker}' "
                f"on timeframes: {frames or 'unknown'}; skipping ticker."
            )

    if not available_tickers:
        raise FileNotFoundError(
            "No historical data available for any configured ticker. "
            "Fetch data before generating replay memory."
        )
    if len(available_tickers) < len(tickers) and verbose:
        remaining = ", ".join(available_tickers)
        print(
            f"[replay:{mode}] Proceeding with {len(available_tickers)} of {len(tickers)} tickers: {remaining}"
        )
    tickers = available_tickers

    decision_data = {ticker: history[decision_interval][ticker] for ticker in tickers}
    price_ix = {
        ticker: (
            df.sort_values("date", kind="mergesort")
            .drop_duplicates("date", keep="first")
            .set_index("date")["close"]
        )
        for ticker, df in history[decision_interval].items()
    }
    decision_positions = {
        ticker: compute_day_positions(decision_data[ticker]) for ticker in tickers
    }
    P_lookup: Dict[str, Dict[pd.Timestamp, float]] = {
        ticker: _build_daily_P_lookup(decision_data[ticker]) for ticker in tickers
    }
    range_factor_lookup: Dict[str, Dict[pd.Timestamp, float]] = {
        ticker: _build_daily_range_factor_lookup(decision_data[ticker]) for ticker in tickers
    }

    common_timestamps = compute_common_timestamps(decision_data)
    min_valid_ts = compute_min_valid_timestamp(history, timeframes, lookback)
    selected_timestamps = _select_timestamps(
        common_timestamps,
        evaluation_days,
        min_valid_ts,
        mode,
        training_days if mode == "training" else None,
    )

    if verbose:
        extra = ""
        if mode == "training" and training_days > 0:
            extra = f", training_days_limit={training_days}"
        if mode == "evaluation" and evaluation_days > 0:
            extra = f", evaluation_days={evaluation_days}"
        print(
            f"[replay:{mode}] Total timestamps={len(common_timestamps)}, "
            f"selected timestamps={len(selected_timestamps)}{extra}"
        )
        if min_valid_ts is not None:
            print(f"[replay:{mode}] First timestamp with full lookback coverage: {min_valid_ts.isoformat()}")

    if not selected_timestamps:
        if verbose:
            print(f"[replay:{mode}] No timestamps available for replay generation.")
        save_replay_memory([], output_path, verbose=verbose)
        return {
            "mode": mode,
            "records": 0,
            "timestamps": 0,
            "avg_reward": 0.0,
            "avg_pct_pnl": None,
            "total_pct_pnl": None,
            "std_pct_pnl": None,
            "pnl_tstat": None,
            "completed_trades": 0,
            "positive_trades": 0,
            "negative_trades": 0,
            "flat_trades": 0,
            "path": output_path,
        }

    trade_status_indices: Dict[str, int] = {ticker: flat_status_idx for ticker in tickers}
    active_trades: Dict[str, Trade] = {}
    replay_memory: List[Dict[str, Any]] = []

    total_trade_status = len(trade_status_list)
    total_processed = 0
    reward_accum = 0.0
    reward_events = 0
    action_counts = [0 for _ in MODEL_ACTIONS]
    legal_decisions = 0
    executed_total = 0
    min_reward = float("inf")
    max_reward = float("-inf")
    negative_rewards = 0
    qgap_sum = 0.0
    qmean_sum = 0.0
    qstats_n = 0
    reward_sum_by_action = {name: 0.0 for name in MODEL_ACTIONS}
    reward_sq_by_action = {name: 0.0 for name in MODEL_ACTIONS}
    reward_cnt_by_action = {name: 0 for name in MODEL_ACTIONS}
    reward_neg_by_action = {name: 0 for name in MODEL_ACTIONS}
    entry_long = 0
    entry_short = 0
    exit_long_signal = 0
    exit_long_trail = 0
    exit_long_take_profit = 0
    exit_short_signal = 0
    exit_short_trail = 0
    exit_short_take_profit = 0
    state_action_counts = {
        "flat": {"hold": 0, "buy": 0, "sell": 0},
        "long": {"hold": 0, "buy": 0, "sell": 0},
        "short": {"hold": 0, "buy": 0, "sell": 0},
    }
    reward_trace_n = 0
    hold_minutes_total = 0.0
    hold_minutes_count = 0
    realized_pct_sum = 0.0
    realized_pct_sq_sum = 0.0
    realized_pct_count = 0
    # sign distribution of realized trade PnL (evaluation only)
    realized_pct_pos_count = 0
    realized_pct_neg_count = 0
    realized_pct_zero_count = 0

    def _record_holding(trade: Trade, exit_timestamp: pd.Timestamp) -> None:
        nonlocal hold_minutes_total, hold_minutes_count
        try:
            entry_dt = pd.to_datetime(trade.entry_time, utc=True)
            exit_dt = exit_timestamp if isinstance(exit_timestamp, pd.Timestamp) else pd.to_datetime(exit_timestamp, utc=True)
            delta_minutes = max((exit_dt - entry_dt).total_seconds() / 60.0, 0.0)
            hold_minutes_total += delta_minutes
            hold_minutes_count += 1
        except Exception:
            pass

    def _record_realized_pct(pct_value: float) -> None:
        nonlocal realized_pct_sum, realized_pct_sq_sum, realized_pct_count
        nonlocal realized_pct_pos_count, realized_pct_neg_count, realized_pct_zero_count
        if mode != "evaluation":
            return

        # Track sum and sum of squares for evaluation-only PnL t-statistics.
        realized_pct_sum += pct_value
        realized_pct_sq_sum += pct_value * pct_value
        realized_pct_count += 1

        if pct_value > 0:
            realized_pct_pos_count += 1
        elif pct_value < 0:
            realized_pct_neg_count += 1
        else:
            realized_pct_zero_count += 1

    def _log_reward_trace(
        ticker: str,
        trade: Trade,
        price: float,
        ts: pd.Timestamp,
        executed_events: List[Dict[str, Any]],
        reward_value: float,
    ) -> None:
        nonlocal reward_trace_n
        if not verbose or reward_trace_n >= 10:
            return
        try:
            entry_dt = pd.to_datetime(trade.entry_time, utc=True)
            ts_dt = ts if isinstance(ts, pd.Timestamp) else pd.to_datetime(ts, utc=True)
            if isinstance(ts_dt, pd.Timestamp) and ts_dt.tzinfo is None:
                ts_dt = ts_dt.tz_localize("UTC")
            hold_min = max((ts_dt - entry_dt).total_seconds() / 60.0, 0.0)
        except Exception:
            hold_min = float("nan")
        reason = executed_events[-1].get("reason", "n/a") if executed_events else "n/a"
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        print(
            f"[trace:reward] t={ts_str} ticker={ticker} exit_reason={reason} "
            f"price={price:.4f} r={reward_value:+.4f} hold_min={hold_min:.2f}"
        )
        reward_trace_n += 1

    for ts in selected_timestamps:
        local_time = get_time_component(ts)
        if local_time < safe_start_time:
            continue
        if use_supervised and mode == "training":
            for ticker in tickers:
                P_val = float(P_lookup.get(ticker, {}).get(ts, 0.0))
                range_factor = float(range_factor_lookup.get(ticker, {}).get(ts, 0.0))
                if range_factor == 0.0 and P_val == 0.0:
                    continue
                for status_idx, status_label in enumerate(trade_status_list):
                    feats = build_feature_vector(
                        ticker,
                        ts,
                        history,
                        timeframes,
                        lookback,
                        decision_positions,
                        status_idx,
                        total_trade_status,
                    )
                    if feats is None or len(feats) != input_dim:
                        continue
                    rewards_by_action: List[float] = []
                    for action_name in MODEL_ACTIONS:
                        base = _reward_from_P(
                            P_val,
                            action_name,
                            status_idx,
                            flat_status_idx,
                            long_status_idx,
                            short_status_idx,
                        )
                        rewards_by_action.append(base * range_factor)
                    if status_label.lower() == status_long_label.lower():
                        rewards_by_action[buy_idx] = 0.0
                    elif status_label.lower() == status_short_label.lower():
                        rewards_by_action[sell_idx] = 0.0
                    best_idx = int(max(range(len(MODEL_ACTIONS)), key=lambda i: rewards_by_action[i]))
                    best_label = model_to_config_label.get(best_idx, MODEL_ACTIONS[best_idx])
                    replay_memory.append(
                        {
                            "timestamp": ts.isoformat(),
                            "ticker": ticker,
                            "trade_status_idx": status_idx,
                            "trade_status": status_label,
                            "features": feats.tolist(),
                            "reward_vector": rewards_by_action,
                            "best_action_idx": best_idx,
                            "best_action": best_label,
                            "P": P_val,
                            "range_factor": range_factor,
                        }
                    )
                    total_processed += 1
                    reward_accum += rewards_by_action[best_idx]
                    reward_events += 1
                    for ai, canonical in enumerate(MODEL_ACTIONS):
                        val = rewards_by_action[ai]
                        reward_sum_by_action[canonical] += val
                        reward_sq_by_action[canonical] += val * val
                        reward_cnt_by_action[canonical] += 1
                        if val < 0:
                            reward_neg_by_action[canonical] += 1
                    if 0 <= best_idx < len(action_counts):
                        action_counts[best_idx] += 1
                    min_reward = min(min_reward, min(rewards_by_action))
                    max_reward = max(max_reward, max(rewards_by_action))
                    if rewards_by_action[best_idx] < 0:
                        negative_rewards += 1
            if verbose and total_processed and total_processed % 1000 == 0:
                print(f"[replay:{mode}] Supervised samples generated: {total_processed}")
            continue

        status_before = dict(trade_status_indices)
        pnl_snapshot: Dict[str, float] = {ticker: 0.0 for ticker in tickers}
        percent_pnl_snapshot: Dict[str, float] = {ticker: 0.0 for ticker in tickers}
        ticker_features: Dict[str, np.ndarray] = {}
        prices: Dict[str, float] = {}

        for ticker in tickers:
            features = build_feature_vector(
                ticker,
                ts,
                history,
                timeframes,
                lookback,
                decision_positions,
                trade_status_indices.get(ticker, flat_status_idx),
                total_trade_status,
            )
            if features is None or len(features) != input_dim:
                continue
            ticker_features[ticker] = features
            price_value = price_ix[ticker].get(ts)
            if pd.notna(price_value):
                prices[ticker] = float(price_value)

        if not ticker_features:
            continue

        inference_results = infer_actions(model, ticker_features, model_to_config_label)
        # --- BEGIN: minimal exploration to seed executions ---
        if mode == "training":
            eps = 0.15
            for tkr, res in inference_results.items():
                if np.random.rand() < eps:
                    res["action_idx"] = int(np.random.choice([buy_idx, sell_idx]))
                    res["action"] = model_to_config_label.get(res["action_idx"], "hold")
        # --- END: minimal exploration to seed executions ---
        for ticker, result in inference_results.items():
            ai = int(result.get("action_idx", sell_idx))
            st_idx = trade_status_indices.get(ticker, flat_status_idx)

            is_legal = (
                (st_idx == flat_status_idx and ai in (buy_idx, sell_idx))
                or (st_idx == long_status_idx and ai == sell_idx)
                or (st_idx == short_status_idx and ai == buy_idx)
            )

            if not is_legal:
                result["action_idx"] = hold_idx
                result["action"] = model_to_config_label.get(hold_idx, "hold")

        for ticker, result in inference_results.items():
            st_idx = trade_status_indices.get(ticker, flat_status_idx)
            if st_idx != flat_status_idx:
                continue

            entry_action_hist[ticker].append(int(result.get("action_idx", hold_idx)))
            entry_conf_hist[ticker].append(float(result.get("confidence", 0.0)))

            if ENTRY_CONFIRM_WINDOW <= 1:
                continue

            if len(entry_action_hist[ticker]) < ENTRY_CONFIRM_WINDOW:
                confirmed = hold_idx
            else:
                window = list(entry_action_hist[ticker])
                confirmed = (
                    window[0]
                    if (window[0] != hold_idx and all(a == window[0] for a in window))
                    else hold_idx
                )

            result["action_idx"] = confirmed
            result["action"] = model_to_config_label.get(confirmed, "hold")
            result["confidence"] = (
                sum(entry_conf_hist[ticker]) / ENTRY_CONFIRM_WINDOW
                if confirmed != hold_idx
                else 0.0
            )

        for ticker, result in inference_results.items():
            ai = int(result.get("action_idx", 2))
            st_idx = trade_status_indices.get(ticker, flat_status_idx)
            if (
                (st_idx == flat_status_idx and ai in (buy_idx, sell_idx))
                or (st_idx == long_status_idx and ai == sell_idx)
                or (st_idx == short_status_idx and ai == buy_idx)
            ):
                legal_decisions += 1
            if st_idx == long_status_idx:
                state_key = "long"
            elif st_idx == short_status_idx:
                state_key = "short"
            else:
                state_key = "flat"
            if 0 <= ai < len(MODEL_ACTIONS):
                act_label = MODEL_ACTIONS[ai]
                if act_label in state_action_counts[state_key]:
                    state_action_counts[state_key][act_label] += 1
        actionable = [
            (ticker, result)
            for ticker, result in inference_results.items()
            if int(result.get("action_idx", hold_idx)) != hold_idx
        ]
        max_conf_ticker = (
            max(actionable, key=lambda item: item[1]["confidence"])[0] if actionable else None
        )
        for ticker, result in inference_results.items():
            q_vals = np.array(result.get("q_values", []), dtype=float)
            if q_vals.size >= 2:
                top2 = np.partition(q_vals, -2)[-2:]
                qgap_sum += float(top2[-1] - top2[-2])
                qmean_sum += float(q_vals.mean())
                qstats_n += 1
        for result in inference_results.values():
            action_idx = int(result.get("action_idx", 2))
            if 0 <= action_idx < len(action_counts):
                action_counts[action_idx] += 1
        executed_events: List[Dict[str, Any]] = []
        pending_reward_logs: List[Tuple[str, Trade, float, pd.Timestamp, List[Dict[str, Any]]]] = []

        if local_time >= safe_end_time and active_trades:
            for ticker, trade in list(active_trades.items()):
                price = prices.get(ticker)
                if price is None:
                    continue
                exit_action = sell_label if trade.direction == status_long_label else buy_label
                executed_events.append(
                    {
                        "type": "exit",
                        "reason": "safe_end",
                        "ticker": ticker,
                        "action": exit_action,
                        "price": price,
                        "quantity": trade.quantity,
                    }
                )
                percent_pnl_snapshot[ticker] = compute_net_percent_pnl(
                    trade.direction == status_long_label,
                    trade.entry_price,
                    price,
                    trade.quantity,
                    cost_exchange,
                )
                _record_realized_pct(percent_pnl_snapshot[ticker])
                pending_reward_logs.append((ticker, trade, price, ts, list(executed_events)))
                _record_holding(trade, ts)
                trade_status_indices[ticker] = flat_status_idx
                del active_trades[ticker]

            _fill_reward_snapshot_from_P(
                ts,
                pnl_snapshot,
                ticker_features,
                inference_results,
                executed_events,
                status_before,
                P_lookup,
                range_factor_lookup,
                flat_status_idx,
                long_status_idx,
                short_status_idx,
            )
            _update_reward_stats_from_snapshot(
                pnl_snapshot,
                inference_results,
                reward_sum_by_action,
                reward_sq_by_action,
                reward_cnt_by_action,
                reward_neg_by_action,
            )
            for ticker_log, trade_log, price_log, ts_log, events_snapshot in pending_reward_logs:
                _log_reward_trace(
                    ticker_log,
                    trade_log,
                    price_log,
                    ts_log,
                    events_snapshot,
                    pnl_snapshot.get(ticker_log, 0.0),
                )
            pending_reward_logs.clear()
            enhanced_actions = _augment_actions(
                inference_results,
                ticker_features,
                pnl_snapshot,
                model_to_config_label,
            )
            executed_total += sum(
                1 for event in executed_events if event.get("type") in ("entry", "exit")
            )
            append_replay_event(
                replay_memory,
                ts,
                enhanced_actions,
                executed_events,
                active_trades,
                pnl_snapshot,
                percent_pnl_snapshot,
            )
            total_processed += 1
            continue

        for ticker, result in inference_results.items():
            price = prices.get(ticker)
            if price is None:
                continue

            trade = active_trades.get(ticker)
            action_idx = int(result["action_idx"])

            if trade:
                if trade.direction == status_long_label:
                    trade.highest_price = max(trade.highest_price, price)
                    stop_price = trade.highest_price * (1 - trailing_stop)
                    take_profit_price = trade.entry_price * (1 + take_profit)
                    exit_signal = action_idx == sell_idx
                    exit_take_profit = price >= take_profit_price
                    exit_trailing = price <= stop_price
                    if exit_signal or exit_take_profit or exit_trailing:
                        reason = (
                            "signal"
                            if exit_signal
                            else "take_profit"
                            if exit_take_profit
                            else "trailing_stop"
                        )
                        executed_events.append(
                            {
                                "type": "exit",
                                "reason": reason,
                                "ticker": ticker,
                                "action": sell_label,
                                "price": price,
                                "quantity": trade.quantity,
                            }
                        )
                        if reason == "signal":
                            exit_long_signal += 1
                        elif reason == "take_profit":
                            exit_long_take_profit += 1
                        elif reason == "trailing_stop":
                            exit_long_trail += 1
                        pending_reward_logs.append((ticker, trade, price, ts, list(executed_events)))
                        percent_pnl_snapshot[ticker] = compute_net_percent_pnl(
                            True,
                            trade.entry_price,
                            price,
                            trade.quantity,
                            cost_exchange,
                        )
                        _record_realized_pct(percent_pnl_snapshot[ticker])
                        _record_holding(trade, ts)
                        trade_status_indices[ticker] = flat_status_idx
                        del active_trades[ticker]
                        continue
                elif trade.direction == status_short_label:
                    trade.lowest_price = min(trade.lowest_price, price)
                    stop_price = trade.lowest_price * (1 + trailing_stop)
                    take_profit_price = trade.entry_price * (1 - take_profit)
                    exit_signal = action_idx == buy_idx
                    exit_take_profit = price <= take_profit_price
                    exit_trailing = price >= stop_price
                    if exit_signal or exit_take_profit or exit_trailing:
                        reason = (
                            "signal"
                            if exit_signal
                            else "take_profit"
                            if exit_take_profit
                            else "trailing_stop"
                        )
                        executed_events.append(
                            {
                                "type": "exit",
                                "reason": reason,
                                "ticker": ticker,
                                "action": buy_label,
                                "price": price,
                                "quantity": trade.quantity,
                            }
                        )
                        if reason == "signal":
                            exit_short_signal += 1
                        elif reason == "take_profit":
                            exit_short_take_profit += 1
                        elif reason == "trailing_stop":
                            exit_short_trail += 1
                        pending_reward_logs.append((ticker, trade, price, ts, list(executed_events)))
                        percent_pnl_snapshot[ticker] = compute_net_percent_pnl(
                            False,
                            trade.entry_price,
                            price,
                            trade.quantity,
                            cost_exchange,
                        )
                        _record_realized_pct(percent_pnl_snapshot[ticker])
                        _record_holding(trade, ts)
                        trade_status_indices[ticker] = flat_status_idx
                        del active_trades[ticker]
                        continue

        for ticker, result in inference_results.items():
            price = prices.get(ticker)
            if price is None:
                continue
            if ticker in active_trades:
                continue
            if mode != "training" and len(active_trades) >= max_concurrent_trades:
                break
            if mode != "training" and max_conf_ticker is not None and ticker != max_conf_ticker:
                continue

            action_idx = result["action_idx"]
            direction = None
            entry_action = None
            if action_idx == buy_idx:
                direction = status_long_label
                entry_action = buy_label
            elif action_idx == sell_idx:
                direction = status_short_label
                entry_action = sell_label

            if direction is None:
                continue

            quantity = compute_quantity(capital_per_ticker, leverage, price)
            if quantity <= 0:
                continue

            active_trades[ticker] = Trade(
                ticker=ticker,
                direction=direction,
                quantity=quantity,
                entry_price=price,
                entry_time=ts.isoformat(),
                highest_price=price,
                lowest_price=price,
            )
            trade_status_key = status_long_label if direction == status_long_label else status_short_label
            trade_status_indices[ticker] = trade_status_map.get(trade_status_key, flat_status_idx)
            executed_events.append(
                {
                    "type": "entry",
                    "ticker": ticker,
                    "action": entry_action,
                    "price": price,
                    "quantity": quantity,
                }
            )
            if entry_action == buy_label:
                entry_long += 1
            elif entry_action == sell_label:
                entry_short += 1

        _fill_reward_snapshot_from_P(
            ts,
            pnl_snapshot,
            ticker_features,
            inference_results,
            executed_events,
            status_before,
            P_lookup,
            range_factor_lookup,
            flat_status_idx,
            long_status_idx,
            short_status_idx,
        )
        _update_reward_stats_from_snapshot(
            pnl_snapshot,
            inference_results,
            reward_sum_by_action,
            reward_sq_by_action,
            reward_cnt_by_action,
            reward_neg_by_action,
        )
        for ticker_log, trade_log, price_log, ts_log, events_snapshot in pending_reward_logs:
            _log_reward_trace(
                ticker_log,
                trade_log,
                price_log,
                ts_log,
                events_snapshot,
                pnl_snapshot.get(ticker_log, 0.0),
            )
        pending_reward_logs.clear()

        enhanced_actions = _augment_actions(
            inference_results,
            ticker_features,
            pnl_snapshot,
            model_to_config_label,
        )
        executed_total += sum(
            1 for event in executed_events if event.get("type") in ("entry", "exit")
        )

        append_replay_event(
            replay_memory,
            ts,
            enhanced_actions,
            executed_events,
            active_trades,
            pnl_snapshot,
            percent_pnl_snapshot,
        )
        total_processed += 1

        reward_accum += sum(pnl_snapshot.values())
        reward_events += len(pnl_snapshot)
        for val in pnl_snapshot.values():
            if val is None:
                continue
            if val < min_reward:
                min_reward = val
            if val > max_reward:
                max_reward = val
            if val < 0:
                negative_rewards += 1

        if verbose and total_processed % 100 == 0:
            print(f"[replay:{mode}] Processed {total_processed} timestamps")

    avg_pct_pnl = None
    total_pct_pnl = None
    std_pct_pnl = None
    pnl_tstat = None
    if mode == "evaluation" and realized_pct_count:
        total_pct_pnl = realized_pct_sum
        avg_pct_pnl = realized_pct_sum / realized_pct_count
        if realized_pct_count >= 2:
            mean = avg_pct_pnl
            variance = (realized_pct_sq_sum - realized_pct_count * mean * mean) / (realized_pct_count - 1)
            variance = max(variance, 0.0)
            std_pct_pnl = math.sqrt(variance)
            if std_pct_pnl == 0.0:
                if mean > 0:
                    pnl_tstat = float("inf")
                elif mean < 0:
                    pnl_tstat = float("-inf")
                else:
                    pnl_tstat = 0.0
            else:
                pnl_tstat = mean / (std_pct_pnl / math.sqrt(realized_pct_count))
        else:
            pnl_tstat = 0.0
    if verbose:
        if selected_timestamps:
            first_ts, last_ts = selected_timestamps[0], selected_timestamps[-1]
            print(f"[replay:{mode}] Date range: {first_ts} -> {last_ts}")
        if not use_supervised:
            total_actions = sum(action_counts)
            if total_actions:
                ordered_pairs = []
                for label in action_labels:
                    idx = canonical_index.get(label.lower())
                    if idx is None or idx >= len(action_counts):
                        continue
                    ordered_pairs.append((label, action_counts[idx]))
                label_counts = ", ".join(f"{label}:{count}" for label, count in ordered_pairs)
                print(f"[replay:{mode}] Actions [{label_counts}] (total={total_actions})")
            chosen_total = total_actions
            no_op_rate = 1.0 - (executed_total / max(chosen_total, 1))
            print(
                f"[diag:action_exec] chosen={chosen_total} legal={legal_decisions} "
                f"executed={executed_total} no_op_rate={no_op_rate:.2f}"
            )
            def _mix(mix_counts: Dict[str, int]) -> str:
                tot = sum(mix_counts.values()) or 1
                return (
                    f"hold={mix_counts['hold']/tot:.2%}, "
                    f"buy={mix_counts['buy']/tot:.2%}, "
                    f"sell={mix_counts['sell']/tot:.2%}"
                )
            print(
                "[diag:state_action] flat("
                + _mix(state_action_counts["flat"])
                + ") long("
                + _mix(state_action_counts["long"])
                + ") short("
                + _mix(state_action_counts["short"])
                + ")"
            )
            print(
                f"[diag:exec_mix] open_long={entry_long} open_short={entry_short} "
                f"exit_long_signal={exit_long_signal} exit_long_take_profit={exit_long_take_profit} "
                f"exit_long_trailing_stop={exit_long_trail} exit_short_signal={exit_short_signal} "
                f"exit_short_take_profit={exit_short_take_profit} "
                f"exit_short_trailing_stop={exit_short_trail}"
            )
        safe_min = min_reward if min_reward != float("inf") else 0.0
        safe_max = max_reward if max_reward != float("-inf") else 0.0
        avg_reward = reward_accum / max(reward_events, 1)
        neg_pct = (negative_rewards / reward_events * 100) if reward_events else 0.0
        print(
            f"[replay:{mode}] Reward stats: avg={avg_reward:.4f}, "
            f"min={safe_min:.4f}, max={safe_max:.4f}, "
            f"negatives={negative_rewards}/{reward_events} ({neg_pct:.1f}%)"
        )
        def _fmt(canonical: str) -> str:
            cnt = reward_cnt_by_action[canonical]
            if cnt <= 0:
                display = label_lower_map.get(canonical, canonical)
                return f"{display}(mean=0.0000,std=0.0000,neg%=0.0)"
            mean = reward_sum_by_action[canonical] / cnt
            var = max((reward_sq_by_action[canonical] / cnt) - mean * mean, 0.0)
            std = var ** 0.5
            negp = (reward_neg_by_action[canonical] / cnt * 100.0) if cnt else 0.0
            display = label_lower_map.get(canonical, canonical)
            return f"{display}(mean={mean:.4f},std={std:.4f},neg%={negp:.1f})"
        print("[diag:replay] rewards_by_action=" + ", ".join(_fmt(label) for label in MODEL_ACTIONS))
        if not use_supervised and qstats_n > 0:
            print(
                "[replay:%s] policy_top2_qgap_mean=%.6f | q_mean=%.6f | n=%d"
                % (mode, qgap_sum / qstats_n, qmean_sum / qstats_n, qstats_n)
            )
        import hashlib

        if not use_supervised:
            fp_rows: List[str] = []
            for record in replay_memory[:200]:
                ts = record.get("timestamp", "")
                pnl_map = record.get("pnl_pct", {})
                for ticker, data in sorted(record.get("ticker_actions", {}).items()):
                    ai = int(data.get("action_idx", -1))
                    reward = float(pnl_map.get(ticker, 0.0))
                    fp_rows.append(f"{ts}|{ticker}|{ai}|{reward:.6f}")
            fingerprint = hashlib.sha256("\n".join(fp_rows).encode("utf-8")).hexdigest()
            print(f"[replay:{mode}] fingerprint={fingerprint}")
            print(f"[replay:{mode}] Replay transitions generated: {len(replay_memory)}")
            if mode == "evaluation":
                unique_days = {
                    ts.tz_convert(INDIA_TZ).date() if ts.tzinfo else ts.date()
                    for ts in selected_timestamps
                }
                trades = entry_long + entry_short
                trades_per_day = trades / max(len(unique_days), 1)
                reward_net = reward_accum
                avg_pct_label = (
                    f"{avg_pct_pnl:.4f}% (n={realized_pct_count})"
                    if isinstance(avg_pct_pnl, (float, int))
                    else "n/a"
                )
                total_pct_label = (
                    f"{total_pct_pnl:.4f}%" if isinstance(total_pct_pnl, (float, int)) else "n/a"
                )
                std_pct_label = (
                    f"{std_pct_pnl:.4f}%" if isinstance(std_pct_pnl, (float, int)) else "n/a"
                )
                pnl_tstat_label = (
                    f"{pnl_tstat:.4f}" if isinstance(pnl_tstat, (float, int)) else "n/a"
                )
                print(
                    f"[diag:eval] reward_net={reward_net:.4f}, trades={trades}, "
                    f"trades_per_day={trades_per_day:.2f} | avg_pct_pnl={avg_pct_label} | "
                    f"total_pct_pnl={total_pct_label} | std_pct_pnl={std_pct_label} | "
                    f"pnl_tstat={pnl_tstat_label} | trades_pos={realized_pct_pos_count}, "
                    f"trades_neg={realized_pct_neg_count}, trades_zero={realized_pct_zero_count}"
                )
        else:
            fp_rows = []
            for record in replay_memory[:200]:
                ts = record.get("timestamp", "")
                ticker = record.get("ticker", "")
                best_idx = int(record.get("best_action_idx", -1))
                reward_vec = record.get("reward_vector", [])
                reward_val = 0.0
                if isinstance(reward_vec, list) and 0 <= best_idx < len(reward_vec):
                    try:
                        reward_val = float(reward_vec[best_idx])
                    except Exception:
                        reward_val = 0.0
                fp_rows.append(f"{ts}|{ticker}|{best_idx}|{reward_val:.6f}")
            fingerprint = hashlib.sha256("\n".join(fp_rows).encode("utf-8")).hexdigest()
            print(f"[replay:{mode}] fingerprint={fingerprint}")
            print(f"[replay:{mode}] Supervised samples generated: {len(replay_memory)}")
    save_replay_memory(replay_memory, output_path, verbose=verbose)

    avg_reward = reward_accum / max(reward_events, 1)
    return {
        "mode": mode,
        "records": len(replay_memory),
        "timestamps": len(selected_timestamps),
        "avg_reward": avg_reward,
        "avg_pct_pnl": avg_pct_pnl,
        "total_pct_pnl": total_pct_pnl,
        "std_pct_pnl": std_pct_pnl,
        "pnl_tstat": pnl_tstat,
        "completed_trades": realized_pct_count,
        "positive_trades": realized_pct_pos_count,
        "negative_trades": realized_pct_neg_count,
        "flat_trades": realized_pct_zero_count,
        "path": output_path,
    }


def populate_training_replay_memory(
    verbose: bool = False,
    override_trailing_stop: Optional[float] = None,
    override_take_profit: Optional[float] = None,
) -> Dict[str, Any]:
    if verbose:
        print("[replay:training] Starting training replay generation")

    # measure wall-clock and CPU/memory usage for training replay generation
    start_wall = _time.perf_counter()
    usage_start = _usage_snapshot()

    summary = _generate_replay_memory(
        "training",
        TRAINING_REPLAY_PATH,
        verbose=verbose,
        override_trailing_stop=override_trailing_stop,
        override_take_profit=override_take_profit,
    )

    end_wall = _time.perf_counter()
    usage_end = _usage_snapshot()

    if verbose:
        print(
            f"[replay:training] Saved {summary['records']} records "
            f"from {summary['timestamps']} timestamps"
        )
        wall = end_wall - start_wall
        if usage_start is not None and usage_end is not None:
            user_cpu = usage_end["user"] - usage_start["user"]
            sys_cpu = usage_end["sys"] - usage_start["sys"]
            max_rss_mb = usage_end["rss_mb"]
            print(
                "[perf:replay_training] wall=%.3fs | cpu_user=%.3fs | cpu_sys=%.3fs | max_rss=%.1f MB"
                % (wall, user_cpu, sys_cpu, max_rss_mb)
            )
        else:
            print(f"[perf:replay_training] wall={wall:.3f}s (CPU/mem stats unavailable)")
    return summary


def populate_evaluation_replay_memory(
    verbose: bool = False,
    override_trailing_stop: Optional[float] = None,
    override_take_profit: Optional[float] = None,
) -> Dict[str, Any]:
    if verbose:
        print("[replay:evaluation] Starting evaluation replay generation")

    # measure wall-clock and CPU/memory usage for evaluation replay generation
    start_wall = _time.perf_counter()
    usage_start = _usage_snapshot()

    summary = _generate_replay_memory(
        "evaluation",
        EVALUATION_REPLAY_PATH,
        verbose=verbose,
        override_trailing_stop=override_trailing_stop,
        override_take_profit=override_take_profit,
    )

    end_wall = _time.perf_counter()
    usage_end = _usage_snapshot()

    if verbose:
        print(
            f"[replay:evaluation] Saved {summary['records']} records "
            f"from {summary['timestamps']} timestamps | avg reward {summary['avg_reward']:.4f}"
        )
        wall = end_wall - start_wall
        if usage_start is not None and usage_end is not None:
            user_cpu = usage_end["user"] - usage_start["user"]
            sys_cpu = usage_end["sys"] - usage_start["sys"]
            max_rss_mb = usage_end["rss_mb"]
            print(
                "[perf:replay_evaluation] wall=%.3fs | cpu_user=%.3fs | cpu_sys=%.3fs | max_rss=%.1f MB"
                % (wall, user_cpu, sys_cpu, max_rss_mb)
            )
        else:
            print(f"[perf:replay_evaluation] wall={wall:.3f}s (CPU/mem stats unavailable)")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate replay memory from historical market data.",
    )
    parser.add_argument(
        "--populate_training_replay_memory",
        action="store_true",
        help="Generate replay memory for training using historical data.",
    )
    parser.add_argument(
        "--populate_evaluation_replay_memory",
        action="store_true",
        help="Generate replay memory for evaluation using historical data.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not (args.populate_training_replay_memory or args.populate_evaluation_replay_memory):
        parser.print_help()
        return 0
    try:
        if args.populate_training_replay_memory:
            populate_training_replay_memory(verbose=args.verbose)
        if args.populate_evaluation_replay_memory:
            populate_evaluation_replay_memory(verbose=args.verbose)
    except Exception as exc:  # noqa: BLE001
        print(f"[replay] Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
