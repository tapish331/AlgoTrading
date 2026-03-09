"""Continuous training loop for the LightRainbowDQN agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

from ml_rl import (
    CHECKPOINT_DIR,
    _load_config,
    _resolve_hparams,
    train_agent,
)
from replay import (
    CONFIG_PATH,
    load_config,
    populate_evaluation_replay_memory,
    populate_training_replay_memory,
)
from tune_take_profit_and_trailing_stop_loss import (
    DEFAULT_ATR_MULTIPLIER_CANDIDATES as GRID_CANDIDATES,
    DEFAULT_METRIC as GRID_METRIC,
    tune_take_profit_and_trailing_stop_loss,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WINNER_CHECKPOINT = CHECKPOINT_DIR / "light_rainbow_winner.pt"
META_PATH = CHECKPOINT_DIR / "winner_meta.json"
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE_PREFIX = "train"
CODEX_AUTOMATION_DIR = CHECKPOINT_DIR / ".codex_auto"
CODEX_DECISION_SCHEMA_PATH = CODEX_AUTOMATION_DIR / "decision_schema.json"
CODEX_DECISION_OUTPUT_PATH = CODEX_AUTOMATION_DIR / "decision_output.json"
SHARED_CODEX_AUTOMATION_DIR = BASE_DIR / "state" / ".codex_auto"
SHARED_CODE_UPDATE_MARKER_PATH = SHARED_CODEX_AUTOMATION_DIR / "code_updated_marker.json"
EVIDENCE_EPOCH_STATE_PATH = SHARED_CODEX_AUTOMATION_DIR / "evidence_epoch.json"
TRAIN_EVIDENCE_DIR = SHARED_CODEX_AUTOMATION_DIR / "train_evidence"
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


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "na"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "na"
    if not math.isfinite(num):
        return "na"
    return f"{num:.4f}%"


def _fmt_int(value: Any) -> str:
    if value is None:
        return "na"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "na"
    if not math.isfinite(num):
        return "na"
    return str(int(num))


def _fmt_triplet(avg_pct: Any, total: Any, positive: Any) -> str:
    return f"{_fmt_pct(avg_pct)}/{_fmt_int(total)}/+{_fmt_int(positive)}"


def _coerce_finite_float(value: Any) -> Optional[float]:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(num):
        return None
    return num


def _fmt_float(value: Any, decimals: int = 4, suffix: str = "") -> str:
    num = _coerce_finite_float(value)
    if num is None:
        return "na"
    return f"{num:.{decimals}f}{suffix}"


def _fmt_signed_float(value: Any, decimals: int = 4) -> str:
    num = _coerce_finite_float(value)
    if num is None:
        return "na"
    return f"{num:+.{decimals}f}"


def _resolve_train_log_path(logging_cfg: Dict[str, Any]) -> Path:
    log_dir = Path(logging_cfg.get("dir", DEFAULT_LOG_DIR))
    if not log_dir.is_absolute():
        log_dir = BASE_DIR / log_dir
    file_prefix_raw = (
        logging_cfg.get("file_prefix_train")
        or logging_cfg.get("file_prefix")
        or DEFAULT_LOG_FILE_PREFIX
    )
    file_prefix = str(file_prefix_raw).strip() or DEFAULT_LOG_FILE_PREFIX
    return log_dir / f"{file_prefix}.log"


def _append_train_log_line(line: str, logging_cfg: Dict[str, Any], verbose: bool = False) -> None:
    log_path = _resolve_train_log_path(logging_cfg)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} | INFO | {line}\n")
    except OSError as write_exc:
        if verbose:
            print(f"[train] Failed to append train loop log: {write_exc}", file=sys.stderr)


def _ensure_codex_decision_schema() -> Path:
    CODEX_AUTOMATION_DIR.mkdir(parents=True, exist_ok=True)
    if not CODEX_DECISION_SCHEMA_PATH.exists():
        with CODEX_DECISION_SCHEMA_PATH.open("w", encoding="utf-8") as handle:
            json.dump(CODEX_DECISION_SCHEMA, handle, indent=2)
    return CODEX_DECISION_SCHEMA_PATH


def _truncate_file(path: Path, verbose: bool = False) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8"):
            pass
    except OSError as exc:
        if verbose:
            print(f"[train] Failed to truncate {path}: {exc}", file=sys.stderr)


def _coerce_epoch(value: Any) -> Optional[int]:
    try:
        epoch = int(value)
    except (TypeError, ValueError):
        return None
    if epoch <= 0:
        return None
    return epoch


def _write_evidence_epoch_state(epoch: int, source: str, reason: str, verbose: bool = False) -> None:
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
        EVIDENCE_EPOCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE_EPOCH_STATE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        if verbose:
            print(f"[train] Failed to write evidence epoch state {EVIDENCE_EPOCH_STATE_PATH}: {exc}", file=sys.stderr)


def _read_evidence_epoch_payload() -> Optional[Dict[str, Any]]:
    try:
        if not EVIDENCE_EPOCH_STATE_PATH.exists():
            return None
        with EVIDENCE_EPOCH_STATE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _get_current_evidence_epoch(verbose: bool = False) -> int:
    payload = _read_evidence_epoch_payload()
    epoch = _coerce_epoch((payload or {}).get("epoch"))
    if epoch is not None:
        return epoch
    epoch = 1
    _write_evidence_epoch_state(epoch, source="bootstrap", reason="initialize", verbose=verbose)
    return epoch


def _bump_evidence_epoch(source: str, reason: str, verbose: bool = False) -> int:
    current = _get_current_evidence_epoch(verbose=verbose)
    new_epoch = current + 1
    _write_evidence_epoch_state(new_epoch, source=source, reason=reason, verbose=verbose)
    return new_epoch


def _resolve_train_epoch_evidence_path(epoch: int) -> Path:
    return TRAIN_EVIDENCE_DIR / f"train_epoch_{epoch}.log"


def _append_train_epoch_evidence_line(line: str, epoch: int, verbose: bool = False) -> None:
    log_path = _resolve_train_epoch_evidence_path(epoch)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} | INFO | {line}\n")
    except OSError as write_exc:
        if verbose:
            print(f"[train] Failed to append epoch evidence log: {write_exc}", file=sys.stderr)


def _read_shared_code_update_marker_fingerprint() -> Optional[str]:
    path = SHARED_CODE_UPDATE_MARKER_PATH
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
    evidence_epoch: Optional[int] = None,
    verbose: bool = False,
) -> None:
    path = SHARED_CODE_UPDATE_MARKER_PATH
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
            print(f"[train] Failed to write shared code update marker {path}: {exc}", file=sys.stderr)


def _run_codex_log_optimizer(
    evidence_epoch: int,
    verbose: bool = False,
    timeout_seconds: float = DEFAULT_CODEX_TIMEOUT_SECONDS,
) -> Dict[str, str]:
    log_path = _resolve_train_epoch_evidence_path(evidence_epoch)
    codex_bin = shutil.which("codex")
    if not codex_bin:
        return {"decision": "insufficient", "reason": "codex_cli_missing"}
    if not log_path.exists() or log_path.stat().st_size <= 0:
        return {"decision": "insufficient", "reason": "log_empty"}
    schema_path = _ensure_codex_decision_schema()
    output_path = CODEX_DECISION_OUTPUT_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _truncate_file(output_path, verbose=verbose)

    prompt = (
        "Inspect the epoch-scoped training evidence log and decide whether one best code modification is justified now.\n"
        f"Evidence epoch: {evidence_epoch}\n"
        f"Training evidence log path: {log_path.resolve()}\n"
        "Return decision='insufficient' if evidence is not strong enough and do not edit files.\n"
        "Return decision='modified' only if you implemented exactly one best high-impact modification in this repo.\n"
        "Keep edits minimal, avoid destructive git operations, and do not run long training jobs.\n"
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
            cwd=str(BASE_DIR),
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
                print(f"[train] Codex CLI failed: {stderr}", file=sys.stderr)
        return {"decision": "insufficient", "reason": f"codex_cli_exit_{completed.returncode}"}

    try:
        with output_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:  # noqa: BLE001
        return {"decision": "insufficient", "reason": "codex_cli_invalid_output"}

    decision = str(payload.get("decision", "insufficient")).strip().lower()
    reason = str(payload.get("reason", ""))
    if decision not in {"insufficient", "modified"}:
        return {"decision": "insufficient", "reason": "codex_cli_unknown_decision"}
    return {"decision": decision, "reason": reason}


def _load_logging_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        logging_cfg = payload.get("logging", {})
        if isinstance(logging_cfg, dict):
            return logging_cfg
        return {}
    except Exception:  # noqa: BLE001
        return {}


def _write_fatal_error(exc: BaseException) -> None:
    logging_cfg = _load_logging_config(CONFIG_PATH)
    log_path = _resolve_train_log_path(logging_cfg)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
        message = f"{timestamp} | ERROR | Fatal error: {exc}\n{trace}\n"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message)
    except OSError as write_exc:
        print(f"[train] Failed to write fatal error log: {write_exc}", file=sys.stderr)


def _compute_dataset_signature() -> str:
    """Hash dataset files to detect changes between epochs."""
    hasher = hashlib.sha256()
    if not DATA_DIR.exists():
        return ""
    for path in sorted(DATA_DIR.rglob("*.parquet")):
        stat = path.stat()
        hasher.update(str(path.relative_to(BASE_DIR)).encode())
        hasher.update(str(stat.st_mtime_ns).encode())
        hasher.update(str(stat.st_size).encode())
    return hasher.hexdigest()


def _load_meta() -> Optional[Dict[str, Any]]:
    if not META_PATH.exists():
        return None
    with META_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_meta(meta: Dict[str, Any]) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def _run_pair_tuning(verbose: bool) -> None:
    """Re-evaluate evaluation ATR trailing stop and take profit multipliers after promotion."""
    try:
        cfg = load_config()
        tuning_cfg = cfg.get("tuning", {})
        trailing_candidates = [
            float(v) for v in tuning_cfg.get("trailing_stop_atr_multiplier_candidates", GRID_CANDIDATES)
        ]
        take_profit_candidates = [
            float(v) for v in tuning_cfg.get("take_profit_atr_multiplier_candidates", GRID_CANDIDATES)
        ]
        metric = str(
            tuning_cfg.get("pair_metric")
            or tuning_cfg.get("grid_metric")
            or tuning_cfg.get("trailing_stop_metric")
            or tuning_cfg.get("take_profit_metric")
            or GRID_METRIC
        ).lower()
        tune_take_profit_and_trailing_stop_loss(
            trailing_candidates,
            take_profit_candidates,
            metric,
            verbose=verbose,
            quiet=not verbose,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[train] Take profit + trailing stop tuning failed: {exc}")


def _promote_checkpoint(
    current_ckpt: Path,
    winner_ckpt: Path,
    dataset_signature: str,
    promotion_score: Optional[float],
    pnl_tstat: Optional[float],
    eval_reward: Optional[float],
    avg_pct_pnl: Optional[float] = None,
    total_pct_pnl: Optional[float] = None,
    std_pct_pnl: Optional[float] = None,
    completed_trades: Optional[int] = None,
    positive_trades: Optional[int] = None,
    promotion_metric: Optional[str] = None,
    verbose: bool = False,
) -> None:
    winner_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current_ckpt, winner_ckpt)
    _save_meta(
        {
            "dataset_signature": dataset_signature,
            "promotion_score": promotion_score,
            "promotion_metric": promotion_metric,
            "pnl_tstat": pnl_tstat,
            "pnl_per_trade": avg_pct_pnl,
            "avg_pct_pnl": avg_pct_pnl,
            "total_pct_pnl": total_pct_pnl,
            "std_pct_pnl": std_pct_pnl,
            "completed_trades": completed_trades,
            "positive_trades": positive_trades,
            "avg_reward": eval_reward,
            "winner_checkpoint": str(winner_ckpt),
            "source_checkpoint": str(current_ckpt),
            "timestamp": time.time(),
        }
    )
    if verbose:
        score_label = f"{promotion_score:.4f}" if isinstance(promotion_score, (float, int)) else "n/a"
        metric_label = promotion_metric or "n/a"
        print(f"[train] Promoted checkpoint to {winner_ckpt} (metric={metric_label}, score={score_label})")


def _should_promote(
    meta: Optional[Dict[str, Any]],
    dataset_signature: str,
    promotion_score: Optional[float],
) -> bool:
    if meta is None:
        return True
    if dataset_signature and dataset_signature != meta.get("dataset_signature"):
        return True
    if promotion_score is None:
        return False
    previous_score = meta.get("promotion_score")
    if previous_score is None:
        previous_score = meta.get("pnl_tstat")
    if previous_score is None:
        previous_score = meta.get("pnl_per_trade")
    if previous_score is None:
        previous_score = meta.get("avg_pct_pnl")
    if previous_score is None:
        previous_score = meta.get("avg_reward", float("-inf"))
    return float(promotion_score) > float(previous_score)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous LightRainbowDQN training loop.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between iterations (default: 1).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    run_id = datetime.now().strftime("%Y%m%dT%H%M%S") + f"-p{os.getpid()}"
    iteration = 0
    evidence_epoch = _get_current_evidence_epoch(verbose=args.verbose)
    seen_marker_fingerprint = _read_shared_code_update_marker_fingerprint()

    try:
        while True:
            current_marker_fingerprint = _read_shared_code_update_marker_fingerprint()
            if current_marker_fingerprint != seen_marker_fingerprint:
                print("[train] Shared code update marker changed; restarting process.", flush=True)
                try:
                    os.execv(sys.executable, [sys.executable, *sys.argv])
                except OSError as exc:
                    print(f"[train] Failed to restart on shared code marker update: {exc}", file=sys.stderr)
                    seen_marker_fingerprint = current_marker_fingerprint

            current_evidence_epoch = _get_current_evidence_epoch(verbose=args.verbose)
            if current_evidence_epoch != evidence_epoch:
                if args.verbose:
                    print(
                        f"[train] Evidence epoch changed ({evidence_epoch} -> {current_evidence_epoch}); "
                        "switching Codex evidence stream."
                    )
                evidence_epoch = current_evidence_epoch

            iteration += 1
            if not args.verbose:
                print(f"run={run_id} i={iteration} ", end="", flush=True)
            loop_start = time.perf_counter()
            config = _load_config()
            train_cfg = config.get("train", {})
            logging_cfg = config.get("logging", {})
            if not isinstance(logging_cfg, dict):
                logging_cfg = {}
            codex_timeout_seconds = _coerce_finite_float(train_cfg.get("codex_cli_timeout_seconds"))
            if codex_timeout_seconds is None:
                codex_timeout_seconds = DEFAULT_CODEX_TIMEOUT_SECONDS
            codex_cli_enabled = _coerce_bool(train_cfg.get("codex_cli_enabled", True), default=True)
            train_mode = str(train_cfg.get("mode", "RL")).upper()
            if args.verbose:
                print(f"[train] Using train.mode={train_mode}")
            promotion_metric = str(train_cfg.get("promotion_metric", "pnl_tstat")).lower()
            fallback_metric = str(train_cfg.get("promotion_fallback_metric", "avg_pct_pnl")).lower()
            allowed_metrics = {"pnl_tstat", "avg_pct_pnl", "avg_reward"}
            if promotion_metric not in allowed_metrics:
                if args.verbose:
                    print(f"[train] Unknown promotion_metric '{promotion_metric}', defaulting to 'pnl_tstat'.")
                promotion_metric = "pnl_tstat"
            if fallback_metric not in allowed_metrics:
                if args.verbose:
                    print(
                        f"[train] Unknown promotion_fallback_metric '{fallback_metric}', defaulting to 'avg_pct_pnl'."
                    )
                fallback_metric = "avg_pct_pnl"
            if args.verbose:
                print(f"[train] Promotion metrics: primary={promotion_metric}, fallback={fallback_metric}")
            winner_path = WINNER_CHECKPOINT

            phase_start = time.perf_counter()
            training_summary = populate_training_replay_memory(verbose=args.verbose)
            replay_train_runtime = time.perf_counter() - phase_start
            training_avg_pct_pnl = training_summary.get("avg_pct_pnl")
            training_trades = training_summary.get("completed_trades", 0)
            training_pos_trades = training_summary.get("positive_trades", 0)
            training_records = int(training_summary.get("records", 0) or 0)
            train_args = SimpleNamespace(
                verbose=args.verbose,
                replay_path=str(training_summary["path"]),
                input_dim=None,
                hidden_layers=None,
            )
            phase_start = time.perf_counter()
            train_agent(train_args)
            train_runtime = time.perf_counter() - phase_start

            promotion_trailing_stop = float(
                config.get(
                    "training_trailing_stop_atr_multiplier",
                    config.get("evaluation_trailing_stop_atr_multiplier"),
                )
            )
            promotion_take_profit = float(
                config.get(
                    "training_take_profit_atr_multiplier",
                    config.get("evaluation_take_profit_atr_multiplier"),
                )
            )
            if args.verbose:
                print(
                    "[train] Promotion scoring overrides | "
                    f"trailing_stop_mult={promotion_trailing_stop:.4f} "
                    f"take_profit_mult={promotion_take_profit:.4f}"
                )

            phase_start = time.perf_counter()
            eval_summary = populate_evaluation_replay_memory(
                verbose=args.verbose,
                override_trailing_stop=promotion_trailing_stop,
                override_take_profit=promotion_take_profit,
            )
            replay_eval_runtime = time.perf_counter() - phase_start
            current_avg_reward = eval_summary.get("avg_reward", float("nan"))
            current_avg_pct_pnl = eval_summary.get("avg_pct_pnl")
            current_total_pct_pnl = eval_summary.get("total_pct_pnl")
            current_std_pct_pnl = eval_summary.get("std_pct_pnl")
            current_pnl_tstat = eval_summary.get("pnl_tstat")
            current_trades = int(eval_summary.get("completed_trades", 0) or 0)
            current_pos_trades = int(eval_summary.get("positive_trades", 0) or 0)
            eval_records = int(eval_summary.get("records", 0) or 0)
            metric_values = {
                "pnl_tstat": current_pnl_tstat,
                "avg_pct_pnl": current_avg_pct_pnl,
                "avg_reward": current_avg_reward,
            }
            promotion_score = metric_values.get(promotion_metric)
            promotion_metric_used = promotion_metric
            if promotion_score is None:
                promotion_score = metric_values.get(fallback_metric)
                if promotion_score is not None:
                    promotion_metric_used = fallback_metric
            dataset_sig = _compute_dataset_signature()

            input_dim, hidden_layers = _resolve_hparams(SimpleNamespace(input_dim=None, hidden_layers=None), config)
            current_ckpt = CHECKPOINT_DIR / f"light_rainbow_{input_dim}_{hidden_layers}.pt"

            meta = _load_meta()
            winner_avg_reward = meta.get("avg_reward") if meta else None
            winner_avg_pct_pnl = None
            winner_trades = None
            winner_pos_trades = None
            winner_promotion_score = None
            if meta:
                winner_promotion_score = meta.get("promotion_score")
                if winner_promotion_score is None:
                    winner_promotion_score = meta.get("pnl_tstat")
                if winner_promotion_score is None:
                    winner_promotion_score = meta.get("pnl_per_trade")
                if winner_promotion_score is None:
                    winner_promotion_score = meta.get("avg_pct_pnl")
                winner_avg_pct_pnl = meta.get("avg_pct_pnl")
                if winner_avg_pct_pnl is None:
                    winner_avg_pct_pnl = meta.get("pnl_per_trade")
                winner_trades = meta.get("completed_trades")
                winner_pos_trades = meta.get("positive_trades")
            changed = bool(meta and dataset_sig and dataset_sig != meta.get("dataset_signature"))
            if args.verbose:
                print(
                    "[train] Promotion check | "
                    f"promotion_score={(promotion_score if promotion_score is not None else float('nan')):.4f} "
                    f"({promotion_metric_used}) | "
                    f"winner_score={(winner_promotion_score if winner_promotion_score is not None else float('nan')):.4f} | "
                    f"pnl_tstat={(current_pnl_tstat if current_pnl_tstat is not None else float('nan')):.4f} | "
                    f"avg_pct_pnl={(current_avg_pct_pnl if current_avg_pct_pnl is not None else float('nan')):.4f}% | "
                    f"trades={current_trades} | "
                    f"avg_reward={current_avg_reward:.4f} | "
                    f"winner_avg_reward={(winner_avg_reward if winner_avg_reward is not None else float('nan')):.4f} | "
                    f"dataset_changed={changed} | ckpt={current_ckpt.name}"
                )
                avg_pct_label = (
                    f"{current_avg_pct_pnl:.4f}%" if isinstance(current_avg_pct_pnl, (float, int)) else "n/a"
                )
                total_pct_label = (
                    f"{current_total_pct_pnl:.4f}%" if isinstance(current_total_pct_pnl, (float, int)) else "n/a"
                )
                std_pct_label = (
                    f"{current_std_pct_pnl:.4f}%" if isinstance(current_std_pct_pnl, (float, int)) else "n/a"
                )
                pnl_tstat_label = (
                    f"{current_pnl_tstat:.4f}" if isinstance(current_pnl_tstat, (float, int)) else "n/a"
                )
                promotion_score_label = (
                    f"{promotion_score:.4f}" if isinstance(promotion_score, (float, int)) else "n/a"
                )
                print(
                    "[train] Eval diagnostics | "
                    f"avg_pct_pnl={avg_pct_label} | total_pct_pnl={total_pct_label} | "
                    f"std_pct_pnl={std_pct_label} | pnl_tstat={pnl_tstat_label} | "
                    f"promotion_score={promotion_score_label} ({promotion_metric_used})"
                )

            can_promote = current_ckpt.exists()
            should_promote = can_promote and _should_promote(meta, dataset_sig, promotion_score)
            if not can_promote:
                if args.verbose:
                    print(f"[train] Warning: checkpoint {current_ckpt} missing; skipping promotion check.")
            else:
                if should_promote:
                    _promote_checkpoint(
                        current_ckpt=current_ckpt,
                        winner_ckpt=winner_path,
                        dataset_signature=dataset_sig,
                        promotion_score=promotion_score,
                        pnl_tstat=current_pnl_tstat,
                        eval_reward=eval_summary.get("avg_reward"),
                        avg_pct_pnl=current_avg_pct_pnl,
                        total_pct_pnl=current_total_pct_pnl,
                        std_pct_pnl=current_std_pct_pnl,
                        completed_trades=current_trades,
                        positive_trades=current_pos_trades,
                        promotion_metric=promotion_metric_used,
                        verbose=args.verbose,
                    )
                    _run_pair_tuning(verbose=args.verbose)
                elif args.verbose:
                    print("[train] Existing winner checkpoint still better; no promotion.")

            status_line: Optional[str] = None
            loop_runtime = time.perf_counter() - loop_start
            current_score_num = _coerce_finite_float(promotion_score)
            winner_score_num = _coerce_finite_float(winner_promotion_score)
            score_delta = (
                current_score_num - winner_score_num
                if current_score_num is not None and winner_score_num is not None
                else None
            )
            win_rate_pct = (
                (float(current_pos_trades) / float(current_trades)) * 100.0
                if current_trades > 0
                else None
            )
            warn_count = 0
            if training_records <= 0:
                warn_count += 1
            if eval_records <= 0:
                warn_count += 1
            if current_score_num is None:
                warn_count += 1
            if not can_promote:
                warn_count += 1
            if _coerce_finite_float(current_avg_reward) is None:
                warn_count += 1
            if current_avg_pct_pnl is not None and _coerce_finite_float(current_avg_pct_pnl) is None:
                warn_count += 1
            if current_pnl_tstat is not None and _coerce_finite_float(current_pnl_tstat) is None:
                warn_count += 1
            status_line = (
                f"tr={_fmt_triplet(training_avg_pct_pnl, training_trades, training_pos_trades)} "
                f"ev={_fmt_triplet(current_avg_pct_pnl, current_trades, current_pos_trades)} "
                f"win={_fmt_triplet(winner_avg_pct_pnl, winner_trades, winner_pos_trades)} "
                f"score={promotion_metric_used}:{_fmt_float(current_score_num)}|"
                f"win:{_fmt_float(winner_score_num)}|"
                f"d:{_fmt_signed_float(score_delta)} "
                f"p={'Y' if should_promote else 'N'} rt={loop_runtime:.2f}s"
                f" data={training_records}/{eval_records} "
                f"evrisk={_fmt_float(current_pnl_tstat)}/{_fmt_pct(current_avg_pct_pnl)}/{_fmt_float(win_rate_pct, decimals=1, suffix='%')} "
                f"t={replay_train_runtime:.1f}/{train_runtime:.1f}/{replay_eval_runtime:.1f} "
                f"warn={warn_count}"
            )
            _append_train_log_line(
                f"run={run_id} i={iteration} {status_line}",
                logging_cfg,
                verbose=args.verbose,
            )
            _append_train_epoch_evidence_line(
                f"run={run_id} i={iteration} {status_line}",
                evidence_epoch,
                verbose=args.verbose,
            )

            if codex_cli_enabled:
                codex_result = _run_codex_log_optimizer(
                    evidence_epoch=evidence_epoch,
                    verbose=args.verbose,
                    timeout_seconds=codex_timeout_seconds,
                )
                codex_status = str(codex_result.get("decision", "insufficient")).strip().lower() or "insufficient"
            else:
                codex_result = {"decision": "insufficient", "reason": "codex_cli_disabled"}
                codex_status = "disabled"

            if not args.verbose and status_line is not None:
                status_line = f"{status_line} codex={codex_status}"
                print(status_line)

            if codex_result.get("decision") == "modified":
                reason = codex_result.get("reason", "").strip() or "no reason provided"
                next_epoch = _bump_evidence_epoch("train", reason, verbose=args.verbose)
                _write_shared_code_update_marker(
                    "train",
                    reason,
                    evidence_epoch=next_epoch,
                    verbose=args.verbose,
                )
                if args.verbose:
                    print(
                        f"[train] Codex applied best modification ({reason}); "
                        f"advanced evidence epoch to {next_epoch}; restarting process."
                    )
                try:
                    os.execv(sys.executable, [sys.executable, *sys.argv])
                except OSError as exc:
                    print(f"[train] Failed to restart process after Codex modification: {exc}", file=sys.stderr)
            elif args.verbose:
                reason = codex_result.get("reason", "").strip() or "no reason provided"
                print(f"[train] Codex decision=insufficient ({reason})")

            time.sleep(max(args.sleep, 0.1))
    except KeyboardInterrupt:
        if args.verbose:
            print("[train] Interrupted by user; exiting.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[train] Fatal error: {exc}", file=sys.stderr)
        _write_fatal_error(exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
