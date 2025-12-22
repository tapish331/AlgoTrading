"""Continuous training loop for the LightRainbowDQN agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
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
    load_config,
    populate_evaluation_replay_memory,
    populate_training_replay_memory,
)
from tune_trailing_stop_loss import DEFAULT_CANDIDATES, DEFAULT_METRIC, tune_trailing_stop_loss

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WINNER_CHECKPOINT = CHECKPOINT_DIR / "light_rainbow_winner.pt"
META_PATH = CHECKPOINT_DIR / "winner_meta.json"


def _compute_dataset_signature() -> str:
    """Hash dataset files to detect changes between epochs."""
    hasher = hashlib.sha256()
    if not DATA_DIR.exists():
        return ""
    for path in sorted(DATA_DIR.rglob("*.csv")):
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


def _run_trailing_stop_tuning(verbose: bool) -> None:
    """Re-evaluate evaluation trailing stop pct after promoting a new winner."""
    try:
        cfg = load_config()
        tuning_cfg = cfg.get("tuning", {})
        candidates = [float(v) for v in tuning_cfg.get("trailing_stop_candidates", DEFAULT_CANDIDATES)]
        metric = str(tuning_cfg.get("trailing_stop_metric", DEFAULT_METRIC)).lower()
        tune_trailing_stop_loss(candidates, metric, verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        print(f"[train] Trailing stop tuning failed: {exc}")


def _promote_checkpoint(
    current_ckpt: Path,
    winner_ckpt: Path,
    dataset_signature: str,
    pnl_per_trade: Optional[float],
    eval_reward: Optional[float],
    eval_pct_pnl: Optional[float] = None,
    completed_trades: Optional[int] = None,
    verbose: bool = False,
) -> None:
    winner_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current_ckpt, winner_ckpt)
    _save_meta(
        {
            "dataset_signature": dataset_signature,
            "pnl_per_trade": pnl_per_trade,
            "avg_pct_pnl": eval_pct_pnl,
            "completed_trades": completed_trades,
            "avg_reward": eval_reward,
            "winner_checkpoint": str(winner_ckpt),
            "source_checkpoint": str(current_ckpt),
            "timestamp": time.time(),
        }
    )
    if verbose:
        print(f"[train] Promoted checkpoint to {winner_ckpt}")


def _should_promote(
    meta: Optional[Dict[str, Any]],
    dataset_signature: str,
    eval_pnl_per_trade: Optional[float],
) -> bool:
    if meta is None:
        return True
    if dataset_signature and dataset_signature != meta.get("dataset_signature"):
        return True
    if eval_pnl_per_trade is None:
        return False
    previous_pnl = meta.get("pnl_per_trade")
    if previous_pnl is None:
        previous_pnl = meta.get("avg_pct_pnl")
    if previous_pnl is None:
        previous_pnl = meta.get("avg_reward", float("-inf"))
    return eval_pnl_per_trade > float(previous_pnl)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous LightRainbowDQN training loop.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Seconds to sleep between iterations (default: 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    while True:
        config = _load_config()
        train_mode = str(config.get("train", {}).get("mode", "RL")).upper()
        print(f"[train] Using train.mode={train_mode}")
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
        current_trades = int(eval_summary.get("completed_trades", 0) or 0)
        pnl_per_trade = current_avg_pct_pnl
        dataset_sig = _compute_dataset_signature()

        input_dim, hidden_layers = _resolve_hparams(SimpleNamespace(input_dim=None, hidden_layers=None), config)
        current_ckpt = CHECKPOINT_DIR / f"light_rainbow_{input_dim}_{hidden_layers}.pt"

        meta = _load_meta()
        winner_avg_reward = meta.get("avg_reward") if meta else None
        winner_pnl_per_trade = None
        if meta:
            winner_pnl_per_trade = meta.get("pnl_per_trade")
            if winner_pnl_per_trade is None:
                winner_pnl_per_trade = meta.get("avg_pct_pnl")
        changed = bool(meta and dataset_sig and dataset_sig != meta.get("dataset_signature"))
        print(
            "[train] Promotion check | "
            f"pnl_per_trade={(pnl_per_trade if pnl_per_trade is not None else float('nan')):.4f}% | "
            f"winner_pnl_per_trade={(winner_pnl_per_trade if winner_pnl_per_trade is not None else float('nan')):.4f}% | "
            f"trades={current_trades} | "
            f"avg_reward={current_avg_reward:.4f} | "
            f"winner_avg_reward={(winner_avg_reward if winner_avg_reward is not None else float('nan')):.4f} | "
            f"dataset_changed={changed} | ckpt={current_ckpt.name}"
        )

        if not current_ckpt.exists():
            print(f"[train] Warning: checkpoint {current_ckpt} missing; skipping promotion check.")
        else:
            if _should_promote(meta, dataset_sig, pnl_per_trade):
                _promote_checkpoint(
                    current_ckpt=current_ckpt,
                    winner_ckpt=winner_path,
                    dataset_signature=dataset_sig,
                    pnl_per_trade=pnl_per_trade,
                    eval_reward=eval_summary.get("avg_reward"),
                    eval_pct_pnl=eval_summary.get("avg_pct_pnl"),
                    completed_trades=current_trades,
                    verbose=args.verbose,
                )
                _run_trailing_stop_tuning(verbose=args.verbose)
            elif args.verbose:
                print("[train] Existing winner checkpoint still better; no promotion.")

        time.sleep(max(args.sleep, 0.1))


if __name__ == "__main__":
    main()
