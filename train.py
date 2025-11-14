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
    populate_evaluation_replay_memory,
    populate_training_replay_memory,
)

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


def _promote_checkpoint(
    current_ckpt: Path,
    winner_ckpt: Path,
    dataset_signature: str,
    eval_reward: float,
    verbose: bool = False,
) -> None:
    winner_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current_ckpt, winner_ckpt)
    _save_meta(
        {
            "dataset_signature": dataset_signature,
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
    eval_reward: float,
) -> bool:
    if meta is None:
        return True
    if dataset_signature and dataset_signature != meta.get("dataset_signature"):
        return True
    previous_reward = meta.get("avg_reward", float("-inf"))
    return eval_reward > previous_reward


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
        dataset_sig = _compute_dataset_signature()

        input_dim, hidden_layers = _resolve_hparams(SimpleNamespace(input_dim=None, hidden_layers=None), config)
        current_ckpt = CHECKPOINT_DIR / f"light_rainbow_{input_dim}_{hidden_layers}.pt"

        meta = _load_meta()
        prev_avg = meta.get("avg_reward") if meta else None
        changed = bool(meta and dataset_sig and dataset_sig != meta.get("dataset_signature"))
        print(
            "[train] Promotion check | "
            f"eval_avg={eval_summary.get('avg_reward', float('nan')):.4f} | "
            f"prev_avg={(prev_avg if prev_avg is not None else float('nan')):.4f} | "
            f"dataset_changed={changed} | ckpt={current_ckpt.name}"
        )

        if not current_ckpt.exists():
            print(f"[train] Warning: checkpoint {current_ckpt} missing; skipping promotion check.")
        else:
            if _should_promote(meta, dataset_sig, eval_summary["avg_reward"]):
                _promote_checkpoint(
                    current_ckpt=current_ckpt,
                    winner_ckpt=winner_path,
                    dataset_signature=dataset_sig,
                    eval_reward=eval_summary["avg_reward"],
                    verbose=args.verbose,
                )
            elif args.verbose:
                print("[train] Existing winner checkpoint still better; no promotion.")

        time.sleep(max(args.sleep, 0.1))


if __name__ == "__main__":
    main()
