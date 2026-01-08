"""Tune the evaluation take profit percentage using replay evaluation."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from replay import CONFIG_PATH, load_config, populate_evaluation_replay_memory

# Default search space if config does not define one.
DEFAULT_CANDIDATES: List[float] = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]
DEFAULT_METRIC = "avg_pct_pnl"


def _write_config(config: Dict[str, Any], path: Path, verbose: bool) -> None:
    """Persist the provided configuration to disk."""
    serialized = json.dumps(config, indent=2)
    path.write_text(serialized + "\n", encoding="utf-8")
    if verbose:
        tp_val = config.get("evaluation_take_profit_pct")
        take_profit = f"{float(tp_val):.4f}" if isinstance(tp_val, (float, int)) else "n/a"
        print(f"[tune] Wrote config to {path} with evaluation_take_profit_pct={take_profit}")


def _score_summary(summary: Dict[str, Any], metric: str) -> float:
    """Extract a numeric score from replay summary."""
    value = summary.get(metric)
    if value is None:
        return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _dedupe_preserve_order(values: Iterable[float]) -> List[float]:
    seen = set()
    ordered: List[float] = []
    for val in values:
        if val in seen:
            continue
        seen.add(val)
        ordered.append(val)
    return ordered


def _parse_candidates(raw: str | None, config: Dict[str, Any]) -> List[float]:
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        candidates = [float(p) for p in parts]
    else:
        tuning_cfg = config.get("tuning", {})
        candidates = [float(v) for v in tuning_cfg.get("take_profit_candidates", DEFAULT_CANDIDATES)]
    return _dedupe_preserve_order(candidates)


def _select_metric(args_metric: str | None, config: Dict[str, Any]) -> str:
    metric = (args_metric or config.get("tuning", {}).get("take_profit_metric") or DEFAULT_METRIC).lower()
    if metric not in {"avg_reward", "avg_pct_pnl"}:
        raise ValueError(f"Unsupported metric '{metric}'. Choose from ['avg_reward', 'avg_pct_pnl'].")
    return metric


def tune_take_profit(candidates: List[float], metric: str, verbose: bool) -> Tuple[float, Dict[str, Any]]:
    if not candidates:
        raise ValueError("No take profit candidates provided for tuning.")
    candidates = [val for val in candidates if val > 0]
    if not candidates:
        raise ValueError("All provided take profit candidates are non-positive.")

    base_config = load_config(CONFIG_PATH)
    original_config = copy.deepcopy(base_config)
    training_ts_raw = base_config.get("training_trailing_stop_loss_pct")
    if training_ts_raw is None:
        raise ValueError("Config must define 'training_trailing_stop_loss_pct' for take profit tuning.")
    training_trailing_stop = float(training_ts_raw)
    best_value = None
    best_score = float("-inf")
    best_summary: Dict[str, Any] = {}
    results: List[Tuple[float, float]] = []

    if verbose:
        joined = ", ".join(f"{c:.4f}" for c in candidates)
        print(f"[tune] Evaluating {len(candidates)} take profit values: [{joined}] using metric '{metric}'")
        print(
            f"[tune] Fixed evaluation_trailing_stop_loss_pct={training_trailing_stop:.4f} "
            "(from training_trailing_stop_loss_pct)"
        )

    try:
        for idx, candidate in enumerate(candidates, start=1):
            working_config = copy.deepcopy(base_config)
            working_config["evaluation_take_profit_pct"] = candidate
            _write_config(working_config, CONFIG_PATH, verbose=verbose)

            if verbose:
                print(f"[tune] ({idx}/{len(candidates)}) Running evaluation with take_profit={candidate:.4f}")

            summary = populate_evaluation_replay_memory(
                verbose=verbose,
                override_trailing_stop=training_trailing_stop,
            )
            score = _score_summary(summary, metric)
            results.append((candidate, score))

            if verbose:
                pnl = summary.get("avg_pct_pnl")
                pnl_str = f"{pnl:.4f}%" if isinstance(pnl, (float, int)) and pnl is not None else "n/a"
                print(
                    f"[tune] completed take_profit={candidate:.4f} | "
                    f"{metric}={score:.4f} | avg_pct_pnl={pnl_str}"
                )

            if score > best_score:
                best_score = score
                best_value = candidate
                best_summary = summary
    except Exception:
        _write_config(original_config, CONFIG_PATH, verbose=False)
        raise

    if best_value is None:
        _write_config(original_config, CONFIG_PATH, verbose=False)
        raise RuntimeError("Unable to determine best take profit value; all candidates failed.")

    final_config = copy.deepcopy(original_config)
    final_config["evaluation_take_profit_pct"] = best_value
    _write_config(final_config, CONFIG_PATH, verbose=verbose)

    if verbose:
        print(
            f"[tune] Best take profit={best_value:.4f} with {metric}={best_score:.4f}; "
            f"config updated at {CONFIG_PATH}"
        )
        print(
            f"[tune] Records={best_summary.get('records', 'n/a')} | "
            f"Timestamps={best_summary.get('timestamps', 'n/a')} | "
            f"Avg reward={best_summary.get('avg_reward', 'n/a')}"
        )
        print("[tune] Full candidate scores:")
        for val, score in results:
            marker = "<- best" if val == best_value else ""
            print(f"  {val:.4f}: {score:.4f} {marker}")
    else:
        print(f"Updated evaluation_take_profit_pct to {best_value:.4f} (metric '{metric}'={best_score:.4f})")

    return best_value, best_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune evaluation take profit percentage and update config.json.",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        help="Comma-separated take profit pct values to evaluate. Example: 0.005,0.01,0.02",
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Metric to maximize when choosing the best take profit (avg_reward or avg_pct_pnl).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(CONFIG_PATH)
        metric = _select_metric(args.metric, config)
        candidates = _parse_candidates(args.candidates, config)
        tune_take_profit(candidates, metric, verbose=bool(args.verbose))
    except Exception as exc:  # noqa: BLE001
        print(f"[tune] Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
