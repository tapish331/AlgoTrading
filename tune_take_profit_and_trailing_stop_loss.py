"""Tune evaluation take profit and trailing stop loss percentages using replay evaluation."""

from __future__ import annotations

import argparse
import copy
import json
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from replay import CONFIG_PATH, load_config, populate_evaluation_replay_memory

# Default search space if config does not define one.
DEFAULT_CANDIDATES: List[float] = [0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]
DEFAULT_METRIC = "pnl_tstat"


def _write_config(config: Dict[str, Any], path: Path, verbose: bool) -> None:
    """Persist the provided configuration to disk."""
    serialized = json.dumps(config, indent=2)
    path.write_text(serialized + "\n", encoding="utf-8")
    if verbose:
        ts_val = config.get("evaluation_trailing_stop_loss_pct")
        tp_val = config.get("evaluation_take_profit_pct")
        trailing_stop = f"{float(ts_val):.4f}" if isinstance(ts_val, (float, int)) else "n/a"
        take_profit = f"{float(tp_val):.4f}" if isinstance(tp_val, (float, int)) else "n/a"
        print(
            f"[tune] Wrote config to {path} with "
            f"evaluation_trailing_stop_loss_pct={trailing_stop} "
            f"evaluation_take_profit_pct={take_profit}"
        )


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


def _parse_candidates(raw: str | None, config: Dict[str, Any], key: str) -> List[float]:
    if raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        candidates = [float(p) for p in parts]
    else:
        tuning_cfg = config.get("tuning", {})
        candidates = [float(v) for v in tuning_cfg.get(key, DEFAULT_CANDIDATES)]
    return _dedupe_preserve_order(candidates)


def _select_metric(args_metric: str | None, config: Dict[str, Any]) -> str:
    tuning_cfg = config.get("tuning", {})
    metric = (
        args_metric
        or tuning_cfg.get("pair_metric")
        or tuning_cfg.get("grid_metric")
        or tuning_cfg.get("trailing_stop_metric")
        or tuning_cfg.get("take_profit_metric")
        or DEFAULT_METRIC
    )
    metric = str(metric).lower()
    if metric not in {"avg_reward", "avg_pct_pnl", "pnl_tstat"}:
        raise ValueError(
            "Unsupported metric '%s'. Choose from ['avg_reward', 'avg_pct_pnl', 'pnl_tstat']." % metric
        )
    return metric


def tune_take_profit_and_trailing_stop_loss(
    trailing_candidates: List[float],
    take_profit_candidates: List[float],
    metric: str,
    verbose: bool,
) -> Tuple[Tuple[float, float], Dict[str, Any]]:
    if not trailing_candidates:
        raise ValueError("No trailing stop candidates provided for tuning.")
    if not take_profit_candidates:
        raise ValueError("No take profit candidates provided for tuning.")
    trailing_candidates = [val for val in trailing_candidates if val > 0]
    take_profit_candidates = [val for val in take_profit_candidates if val > 0]
    if not trailing_candidates:
        raise ValueError("All provided trailing stop candidates are non-positive.")
    if not take_profit_candidates:
        raise ValueError("All provided take profit candidates are non-positive.")

    base_config = load_config(CONFIG_PATH)
    original_config = copy.deepcopy(base_config)
    best_pair: Tuple[float, float] | None = None
    best_score = float("-inf")
    best_summary: Dict[str, Any] = {}
    results: List[Tuple[float, float, float]] = []
    total = len(trailing_candidates) * len(take_profit_candidates)

    if verbose:
        trailing_joined = ", ".join(f"{c:.4f}" for c in trailing_candidates)
        take_profit_joined = ", ".join(f"{c:.4f}" for c in take_profit_candidates)
        print(f"[tune] Evaluating {total} combinations using metric '{metric}'")
        print(f"[tune] Trailing stop candidates ({len(trailing_candidates)}): [{trailing_joined}]")
        print(f"[tune] Take profit candidates ({len(take_profit_candidates)}): [{take_profit_joined}]")

    try:
        # Iterate through the full grid of candidate pairs.
        for idx, (trailing_stop, take_profit) in enumerate(
            product(trailing_candidates, take_profit_candidates),
            start=1,
        ):
            if verbose:
                print(
                    f"[tune] ({idx}/{total}) Running evaluation with "
                    f"trailing_stop={trailing_stop:.4f}, take_profit={take_profit:.4f}"
                )

            summary = populate_evaluation_replay_memory(
                verbose=verbose,
                override_trailing_stop=trailing_stop,
                override_take_profit=take_profit,
            )
            score = _score_summary(summary, metric)
            results.append((trailing_stop, take_profit, score))

            if verbose:
                pnl = summary.get("avg_pct_pnl")
                pnl_str = f"{pnl:.4f}%" if isinstance(pnl, (float, int)) and pnl is not None else "n/a"
                print(
                    f"[tune] completed trailing_stop={trailing_stop:.4f}, take_profit={take_profit:.4f} | "
                    f"{metric}={score:.4f} | avg_pct_pnl={pnl_str}"
                )

            if score > best_score:
                best_score = score
                best_pair = (trailing_stop, take_profit)
                best_summary = summary
    except Exception:
        raise

    if best_pair is None:
        raise RuntimeError("Unable to determine best take profit + trailing stop pair; all candidates failed.")

    final_config = copy.deepcopy(original_config)
    final_config["evaluation_trailing_stop_loss_pct"] = best_pair[0]
    final_config["evaluation_take_profit_pct"] = best_pair[1]
    _write_config(final_config, CONFIG_PATH, verbose=verbose)

    if verbose:
        print(
            f"[tune] Best pair trailing_stop={best_pair[0]:.4f}, take_profit={best_pair[1]:.4f} "
            f"with {metric}={best_score:.4f}; config updated at {CONFIG_PATH}"
        )
        print(
            f"[tune] Records={best_summary.get('records', 'n/a')} | "
            f"Timestamps={best_summary.get('timestamps', 'n/a')} | "
            f"Avg reward={best_summary.get('avg_reward', 'n/a')}"
        )
        print("[tune] Full candidate scores:")
        for trailing_stop, take_profit, score in results:
            marker = "<- best" if (trailing_stop, take_profit) == best_pair else ""
            print(
                f"  trailing_stop={trailing_stop:.4f}, take_profit={take_profit:.4f}: {score:.4f} {marker}"
            )
    else:
        print(
            f"Updated evaluation_trailing_stop_loss_pct to {best_pair[0]:.4f} and "
            f"evaluation_take_profit_pct to {best_pair[1]:.4f} (metric '{metric}'={best_score:.4f})"
        )

    return best_pair, best_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune evaluation take profit and trailing stop loss percentages and update config.json.",
    )
    parser.add_argument(
        "--trailing-stop-candidates",
        type=str,
        help="Comma-separated trailing stop pct values to evaluate. Example: 0.005,0.01,0.02",
    )
    parser.add_argument(
        "--take-profit-candidates",
        type=str,
        help="Comma-separated take profit pct values to evaluate. Example: 0.005,0.01,0.02",
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Metric to maximize when choosing the best pair (avg_reward, avg_pct_pnl, pnl_tstat).",
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
        trailing_candidates = _parse_candidates(
            args.trailing_stop_candidates,
            config,
            "trailing_stop_candidates",
        )
        take_profit_candidates = _parse_candidates(
            args.take_profit_candidates,
            config,
            "take_profit_candidates",
        )
        tune_take_profit_and_trailing_stop_loss(
            trailing_candidates,
            take_profit_candidates,
            metric,
            verbose=bool(args.verbose),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[tune] Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
