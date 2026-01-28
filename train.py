"""Continuous training loop for the LightRainbowDQN agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
    DEFAULT_CANDIDATES as GRID_CANDIDATES,
    DEFAULT_METRIC as GRID_METRIC,
    tune_take_profit_and_trailing_stop_loss,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WINNER_CHECKPOINT = CHECKPOINT_DIR / "light_rainbow_winner.pt"
META_PATH = CHECKPOINT_DIR / "winner_meta.json"
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE_PREFIX = "train"


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
    log_dir = Path(logging_cfg.get("dir", DEFAULT_LOG_DIR))
    file_prefix_raw = (
        logging_cfg.get("file_prefix_train")
        or logging_cfg.get("file_prefix")
        or DEFAULT_LOG_FILE_PREFIX
    )
    file_prefix = str(file_prefix_raw).strip() or DEFAULT_LOG_FILE_PREFIX
    log_path = log_dir / f"{file_prefix}.log"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()
        message = f"{timestamp} | ERROR | Fatal error: {exc}\n{trace}\n"
        with log_path.open("w", encoding="utf-8") as handle:
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
    """Re-evaluate evaluation trailing stop and take profit pct after promoting a new winner."""
    try:
        cfg = load_config()
        tuning_cfg = cfg.get("tuning", {})
        trailing_candidates = [float(v) for v in tuning_cfg.get("trailing_stop_candidates", GRID_CANDIDATES)]
        take_profit_candidates = [float(v) for v in tuning_cfg.get("take_profit_candidates", GRID_CANDIDATES)]
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

    try:
        while True:
            config = _load_config()
            train_cfg = config.get("train", {})
            train_mode = str(train_cfg.get("mode", "RL")).upper()
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

            training_summary = populate_training_replay_memory(verbose=args.verbose)
            train_args = SimpleNamespace(
                verbose=args.verbose,
                replay_path=str(training_summary["path"]),
                input_dim=None,
                hidden_layers=None,
            )
            train_agent(train_args)

            eval_summary = populate_evaluation_replay_memory(verbose=args.verbose)
            current_avg_reward = eval_summary.get("avg_reward", float("nan"))
            current_avg_pct_pnl = eval_summary.get("avg_pct_pnl")
            current_total_pct_pnl = eval_summary.get("total_pct_pnl")
            current_std_pct_pnl = eval_summary.get("std_pct_pnl")
            current_pnl_tstat = eval_summary.get("pnl_tstat")
            current_trades = int(eval_summary.get("completed_trades", 0) or 0)
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
            winner_promotion_score = None
            if meta:
                winner_promotion_score = meta.get("promotion_score")
                if winner_promotion_score is None:
                    winner_promotion_score = meta.get("pnl_tstat")
                if winner_promotion_score is None:
                    winner_promotion_score = meta.get("pnl_per_trade")
                if winner_promotion_score is None:
                    winner_promotion_score = meta.get("avg_pct_pnl")
            changed = bool(meta and dataset_sig and dataset_sig != meta.get("dataset_signature"))
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
            if args.verbose:
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

            if not current_ckpt.exists():
                print(f"[train] Warning: checkpoint {current_ckpt} missing; skipping promotion check.")
            else:
                if _should_promote(meta, dataset_sig, promotion_score):
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
                        promotion_metric=promotion_metric_used,
                        verbose=args.verbose,
                    )
                    _run_pair_tuning(verbose=args.verbose)
                elif args.verbose:
                    print("[train] Existing winner checkpoint still better; no promotion.")

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
