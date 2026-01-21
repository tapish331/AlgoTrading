"""Minimal Rainbow-inspired DQN utilities and CLI runner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, cast

import time
try:
    import resource  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    resource = None  # type: ignore[assignment]
try:
    import psutil  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    _TORCH_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:  # pragma: no cover - triggered when torch missing
    torch = cast(Any, None)
    nn = cast(Any, None)
    F = cast(Any, None)
    DataLoader = cast(Any, None)
    Dataset = cast(Any, None)
    WeightedRandomSampler = cast(Any, None)
    _TORCH_IMPORT_ERROR = exc

if TYPE_CHECKING:
    from torch import Tensor
else:
    Tensor = Any  # type: ignore[assignment]

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
TRAINING_REPLAY_PATH = BASE_DIR / "data" / "replay_memory_training.jsonl"
MODEL_ACTIONS = ["hold", "buy", "sell"]
CANONICAL_ACTION_INDEX = {name.lower(): idx for idx, name in enumerate(MODEL_ACTIONS)}


def counts_in_config_order(raw_counts: List[int], config_actions: List[str]) -> List[int]:
    ordered: List[int] = []
    for label in config_actions:
        idx = CANONICAL_ACTION_INDEX.get(label.lower())
        if idx is None or idx >= len(raw_counts):
            ordered.append(0)
        else:
            ordered.append(raw_counts[idx])
    return ordered


def pct_from_counts(ordered_counts: List[int]) -> List[float]:
    total = max(sum(ordered_counts), 1)
    return [100.0 * count / total for count in ordered_counts]


def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing configuration file at {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


if _TORCH_IMPORT_ERROR is None:
    class TradeStatusActionMask(nn.Module):
        """
        Builds an action mask from config-driven trade status rules.

        The last feature produced by `build_feature_vector` is the normalized trade status,
        so we rehydrate the discrete status index from that scalar and zero-out actions
        that are illegal for the current state.
        """

        def __init__(self, input_dim: int, action_dim: int) -> None:
            super().__init__()
            if input_dim <= 0:
                raise ValueError("input_dim must be positive to build an action mask")
            config = _load_config()
            trade_status = config.get("trade_status", [])
            if not trade_status:
                raise ValueError("Config must define a non-empty 'trade_status' list.")
            actions = [str(label).lower() for label in config.get("actions", [])]
            missing = [name for name in MODEL_ACTIONS if name not in actions]
            if missing:
                raise ValueError(f"Config actions missing required entries: {missing}")

            self.status_feature_idx = input_dim - 1  # trade_status_norm appended last
            self.action_dim = action_dim
            self.total_trade_status = len(trade_status)
            status_lookup = {label.lower(): idx for idx, label in enumerate(trade_status)}

            mask = torch.ones(self.total_trade_status, action_dim, dtype=torch.float32)
            hold_idx = CANONICAL_ACTION_INDEX["hold"]
            buy_idx = CANONICAL_ACTION_INDEX["buy"]
            sell_idx = CANONICAL_ACTION_INDEX["sell"]

            def _restrict(status_name: str, allowed: List[int]) -> None:
                idx = status_lookup.get(status_name)
                if idx is None:
                    return
                row = mask[idx]
                row.zero_()
                row[allowed] = 1.0

            _restrict("long", [hold_idx, sell_idx])
            _restrict("short", [hold_idx, buy_idx])
            # flat stays fully legal (mask row already initialized to ones)

            self.register_buffer("mask_lookup", mask)

        def forward(self, features: Tensor) -> Tensor:
            if features.ndim < 2 or features.shape[1] <= self.status_feature_idx:
                raise ValueError("Feature tensor is missing the trade status feature.")
            batch = features.shape[0]
            device = features.device
            dtype = features.dtype

            if self.total_trade_status <= 1:
                full = torch.ones(batch, self.action_dim, device=device, dtype=dtype)
                return full.unsqueeze(-1)

            status_column = features[:, self.status_feature_idx]
            normalized = status_column.clamp(0.0, 1.0)
            idx = torch.round(normalized * (self.total_trade_status - 1)).long()
            idx = idx.clamp(0, self.total_trade_status - 1)
            idx = idx.to(self.mask_lookup.device)
            mask = self.mask_lookup.index_select(0, idx).to(dtype)
            return mask.unsqueeze(-1)

    class LightRainbowDQN(nn.Module):
        """
        One-class, lightest Rainbow-style DQN (dueling + distributional + noisy heads).
        Everything is derived solely from:
          - input_dim
          - hidden_layers_num

        Notes:
          - Actions fixed to 3 (hold/buy/sell).
          - Quantiles K fixed to 8 (minimal but stable).
          - Trunk blocks follow your rule: Linear -> GELU -> LayerNorm -> Dropout(0.05)
          - NoisyLinear is implemented inline (no extra class).
          - Double DQN / n-step / prioritized replay are training-time mechanics.

        API:
          forward(x) -> Q distribution [B, 3, 8]
          q_values(x) -> mean over quantiles [B, 3]
          act(x) -> argmax action index [B]
        """

        def __init__(self, input_dim: int, hidden_layers_num: int):
            super().__init__()
            assert input_dim > 0, "input_dim must be positive"
            self.input_dim = int(input_dim)
            self.L = max(0, int(hidden_layers_num))  # allow 0 for ultra-light heads-only
            self.A = 3
            self.K = 8
            self.dropout_p = 0.05
            self.noisy_sigma0 = 0.5  # standard choice

            # ---- Build tapered trunk (ModuleList of your sandwich blocks) ----
            widths = self._compute_tapered_widths(self.input_dim, self.L)
            self.blocks = nn.ModuleList()
            in_dim = self.input_dim
            for h in widths:
                self.blocks.append(nn.Linear(in_dim, h))
                self.blocks.append(nn.GELU())
                self.blocks.append(nn.LayerNorm(h))
                self.blocks.append(nn.Dropout(p=self.dropout_p))
                in_dim = h
            last_dim = in_dim  # equals input_dim if L==0

            # ---- Noisy heads (parameters only; no extra class) ----
            # Value head: last_dim -> K
            self.v_w_mu = nn.Parameter(torch.empty(self.K, last_dim))
            self.v_w_sigma = nn.Parameter(torch.empty(self.K, last_dim))
            self.v_b_mu = nn.Parameter(torch.empty(self.K))
            self.v_b_sigma = nn.Parameter(torch.empty(self.K))

            # Advantage head: last_dim -> A*K
            out_adv = self.A * self.K
            self.a_w_mu = nn.Parameter(torch.empty(out_adv, last_dim))
            self.a_w_sigma = nn.Parameter(torch.empty(out_adv, last_dim))
            self.a_b_mu = nn.Parameter(torch.empty(out_adv))
            self.a_b_sigma = nn.Parameter(torch.empty(out_adv))

            # Initialize parameters
            self._reset_noisy_(self.v_w_mu, self.v_w_sigma, self.v_b_mu, self.v_b_sigma, last_dim)
            self._reset_noisy_(self.a_w_mu, self.a_w_sigma, self.a_b_mu, self.a_b_sigma, last_dim)
            self.action_mask = TradeStatusActionMask(self.input_dim, self.A)

        # ---------------- public API ----------------
        def forward(self, x: Tensor) -> Tensor:
            """
            Return distributional Q-values: [B, A, K]
            """
            z = self._run_trunk(x)  # [B, last_dim]
            V = self._noisy_linear(z, self.v_w_mu, self.v_w_sigma, self.v_b_mu, self.v_b_sigma)  # [B, K]
            Ahat = self._noisy_linear(z, self.a_w_mu, self.a_w_sigma, self.a_b_mu, self.a_b_sigma)  # [B, A*K]
            Ahat = Ahat.view(-1, self.A, self.K)  # [B, A, K]

            # Dueling combine (per-quantile, no extra params)
            Amean = Ahat.mean(dim=1, keepdim=True)  # [B, 1, K]
            Q = V.unsqueeze(1) + (Ahat - Amean)  # [B, A, K]
            mask = self.action_mask(x)
            Q = Q.masked_fill(mask == 0, -1e9)
            return Q

        @torch.no_grad()
        def q_values(self, x: Tensor) -> Tensor:
            """Mean across quantiles -> standard Q-values per action: [B, A]."""
            return self.forward(x).mean(dim=2)

        @torch.no_grad()
        def act(self, x: Tensor) -> Tensor:
            """Argmax over mean Q-values -> action indices in {0:hold, 1:buy, 2:sell}."""
            return self.q_values(x).argmax(dim=1)

        def parameter_count(self) -> int:
            return sum(p.numel() for p in self.parameters())

        # ---------------- trunk & sizing helpers ----------------
        def _run_trunk(self, x: Tensor) -> Tensor:
            if len(self.blocks) == 0:
                return x
            z = x
            # iterate blocks in groups of 4: Linear -> GELU -> LayerNorm -> Dropout
            for i in range(0, len(self.blocks), 4):
                z = self.blocks[i](z)
                z = self.blocks[i + 1](z)
                z = self.blocks[i + 2](z)
                z = self.blocks[i + 3](z)
            return z

        @staticmethod
        def _compute_tapered_widths(input_dim: int, L: int):
            """
            Lightest rules:
            - L == 0: no trunk
            - L == 1: one tiny nonlinear layer: h1 = max(32, ceil(sqrt(input_dim)))
            - L >= 2: linear taper from S to E (average-of-neighbors sequence)
                S = ceil(0.5 * input_dim)
                E = ceil(sqrt(input_dim))
                ensure S > E; enforce strict decrease after rounding
            """
            if L <= 0:
                return []
            if L == 1:
                return [max(32, math.ceil(math.sqrt(input_dim)))]

            S = math.ceil(0.5 * input_dim)
            E = math.ceil(math.sqrt(input_dim))
            if S <= E:
                S = E + 1  # guarantee downward taper

            hs = []
            for i in range(L):
                val = math.ceil(S + (i / (L - 1)) * (E - S))
                if i > 0 and val >= hs[-1]:
                    val = max(hs[-1] - 1, 1)  # keep strictly decreasing despite rounding
                hs.append(val)
            return hs

        # ---------------- NoisyLinear (inline) ----------------
        def _reset_noisy_(self, w_mu, w_sigma, b_mu, b_sigma, in_features: int):
            mu_range = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(w_mu, -mu_range, +mu_range)
            nn.init.uniform_(b_mu, -mu_range, +mu_range)
            init_sigma = self.noisy_sigma0 / math.sqrt(in_features)
            nn.init.constant_(w_sigma, init_sigma)
            nn.init.constant_(b_sigma, init_sigma)

        @staticmethod
        def _noisy_f(x: Tensor) -> Tensor:
            # factorized noise transform
            return x.sign() * x.abs().sqrt()

        def _noisy_linear(
            self,
            x: Tensor,
            w_mu: Tensor,
            w_sigma: Tensor,
            b_mu: Tensor,
            b_sigma: Tensor,
        ) -> Tensor:
            if self.training:
                eps_in = self._noisy_f(torch.randn(w_mu.size(1), device=x.device, dtype=x.dtype))
                eps_out = self._noisy_f(torch.randn(w_mu.size(0), device=x.device, dtype=x.dtype))
                noise_w = torch.outer(eps_out, eps_in)  # [out, in]
                weight = w_mu + w_sigma * noise_w
                bias = b_mu + b_sigma * eps_out
            else:
                weight, bias = w_mu, b_mu
            return F.linear(x, weight, bias)
else:  # pragma: no cover - executed only when torch missing

    class LightRainbowDQN:  # type: ignore[misc]
        def __init__(self, *_, **__):
            raise RuntimeError(
                "torch is required to instantiate LightRainbowDQN. "
                "Install the dependencies via `pip install -r requirements.txt`. "
                f"Original import error: {_TORCH_IMPORT_ERROR}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lightweight RL model utilities.",
    )
    parser.add_argument(
        "--LightRainbowDQN",
        action="store_true",
        help="Instantiate and inspect the LightRainbowDQN model.",
    )
    parser.add_argument(
        "--input_dim",
        type=int,
        help="Override the input dimension for LightRainbowDQN.",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        help="Override hidden layer count for LightRainbowDQN.",
    )
    parser.add_argument(
        "--train_agent",
        action="store_true",
        help="Train/update the agent from replay memory.",
    )
    parser.add_argument(
        "--replay_path",
        type=str,
        default=None,
        help="Override training replay file path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information during execution.",
    )
    return parser


def _resolve_hparams(args: argparse.Namespace, config: Dict[str, Any]) -> Tuple[int, int]:
    timeframes = config.get("timeframes", [])
    lookback = int(config.get("train", {}).get("lookback", 1))
    default_input = (5 * max(len(timeframes), 1) * max(lookback, 1)) + 2
    default_hidden = int(config.get("ml_rl", {}).get("hidden_layers_num", 1))

    input_dim = args.input_dim if args.input_dim is not None else default_input
    hidden_layers = args.hidden_layers if args.hidden_layers is not None else default_hidden

    return int(input_dim), int(hidden_layers)


class ReplayDataset(Dataset):
    def __init__(self, samples: List[Tuple[List[float], int, float, float, List[float]]]):
        self.features = torch.tensor([s[0] for s in samples], dtype=torch.float32)
        self.actions = torch.tensor([s[1] for s in samples], dtype=torch.long)
        self.rewards = torch.tensor([s[2] for s in samples], dtype=torch.float32)
        self.percent_pnls = torch.tensor([s[3] for s in samples], dtype=torch.float32)
        self.reward_vectors = torch.tensor(
            [s[4] for s in samples],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            self.features[idx],
            self.actions[idx],
            self.rewards[idx],
            self.percent_pnls[idx],
            self.reward_vectors[idx],
        )


def _load_training_samples(
    replay_path: Path,
    input_dim: int,
    verbose: bool = False,
) -> List[Tuple[List[float], int, float, float, List[float]]]:
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay memory file not found at {replay_path}")

    samples: List[Tuple[List[float], int, float, float, List[float]]] = []
    dropped_missing = 0
    dropped_dim_mismatch = 0
    dropped_other = 0
    with replay_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "reward_vector" in record and "features" in record:
                feats = record.get("features")
                reward_vec = record.get("reward_vector")
                best_idx = record.get("best_action_idx")
                if feats is None or reward_vec is None or best_idx is None:
                    dropped_missing += 1
                    continue
                if len(feats) != input_dim:
                    dropped_dim_mismatch += 1
                    continue
                if not isinstance(reward_vec, list) or len(reward_vec) != len(MODEL_ACTIONS):
                    dropped_other += 1
                    continue
                best_idx = int(best_idx)
                reward_scalar = float(reward_vec[best_idx]) if 0 <= best_idx < len(reward_vec) else 0.0
                samples.append((feats, best_idx, reward_scalar, reward_scalar, reward_vec))
                continue

            executed_map = {
                e.get("ticker"): e.get("action")
                for e in record.get("executed_events", [])
                if e.get("type") in ("entry", "exit") and e.get("action")
            }
            ticker_actions = record.get("ticker_actions", {})
            pnl_map = record.get("pnl_pct", {})
            percent_pnl_map = record.get("percent_pnl", {})
            for ticker, data in ticker_actions.items():
                feats = data.get("features")
                action_idx = data.get("action_idx")
                label_name = executed_map.get(ticker)
                if label_name:
                    override = CANONICAL_ACTION_INDEX.get(str(label_name).lower())
                    if override is not None:
                        action_idx = override
                if feats is None or action_idx is None:
                    dropped_missing += 1
                    continue
                if len(feats) != input_dim:
                    dropped_dim_mismatch += 1
                    continue

                # Bare-min fix: drop non-executed BUY/SELL to avoid replay contamination
                if label_name is None and int(action_idx) in (
                    CANONICAL_ACTION_INDEX["buy"],
                    CANONICAL_ACTION_INDEX["sell"],
                ):
                    dropped_other += 1
                    continue
                action_idx = int(action_idx)
                reward = float(pnl_map.get(ticker, 0.0))
                percent_pnl = float(percent_pnl_map.get(ticker, 0.0))
                reward_vec = [0.0] * len(MODEL_ACTIONS)
                if 0 <= action_idx < len(reward_vec):
                    reward_vec[action_idx] = reward
                samples.append((feats, action_idx, reward, percent_pnl, reward_vec))
    if verbose:
        print(f"[ml_rl] Loaded {len(samples)} training samples from {replay_path}")
        print(
            "[diag:samples] loaded=%d | dropped_missing=%d | dropped_dim_mismatch=%d | dropped_other=%d"
            % (len(samples), dropped_missing, dropped_dim_mismatch, dropped_other)
        )
    return samples


def run_light_rainbow(args: argparse.Namespace) -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "torch is not available in the current environment. "
            "Install dependencies and re-run. "
            f"Original error: {_TORCH_IMPORT_ERROR}"
        )

    config = _load_config()
    input_dim, hidden_layers = _resolve_hparams(args, config)
    if args.verbose:
        print(
            "[ml_rl] Instantiating LightRainbowDQN with "
            f"input_dim={input_dim}, hidden_layers={hidden_layers}"
        )
    model = LightRainbowDQN(input_dim=input_dim, hidden_layers_num=hidden_layers)
    params = model.parameter_count()
    if args.verbose:
        print(f"[ml_rl] Model has {params:,} parameters")

    # Demo forward pass with a single zero batch
    sample = torch.zeros(1, input_dim)
    if args.verbose:
        print("[ml_rl] Running forward pass on zero input")
    dist = model(sample)  # [1, 3, 8]
    q_vals = dist.mean(dim=2)
    action = q_vals.argmax(dim=1)

    print("LightRainbowDQN summary")
    print("-----------------------")
    print(f"Input dim      : {input_dim}")
    print(f"Hidden layers  : {hidden_layers}")
    print(f"Parameter count: {params:,}")
    print(f"Sample Q-values: {q_vals.tolist()}")
    print(f"Chosen action  : {action.tolist()}")


def train_agent(args: argparse.Namespace) -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise RuntimeError(
            "torch is not available in the current environment. "
            "Install dependencies and re-run. "
            f"Original error: {_TORCH_IMPORT_ERROR}"
        )

    # measure wall-clock and CPU/memory usage for the full training run
    start_wall = time.perf_counter()
    usage_start = _usage_snapshot()

    config = _load_config()
    input_dim, hidden_layers = _resolve_hparams(args, config)
    config_actions: List[str] = config.get("actions", MODEL_ACTIONS)
    train_cfg = config.get("train", {})
    train_mode = str(train_cfg.get("mode", "RL")).upper()
    if train_mode not in ("RL", "SL"):
        raise ValueError("train.mode must be either 'RL' or 'SL'")
    use_supervised = train_mode == "SL"

    ml_cfg = config.get("ml_rl", {})
    learning_rate = float(ml_cfg.get("learning_rate", 1e-4))
    batch_size = int(ml_cfg.get("batch_size", 64))
    epochs = int(ml_cfg.get("epochs", 1))

    replay_path = Path(args.replay_path) if args.replay_path else TRAINING_REPLAY_PATH
    samples = _load_training_samples(replay_path, input_dim, verbose=args.verbose)
    if not samples:
        raise ValueError(f"No usable training samples found in {replay_path}")

    dataset = ReplayDataset(samples)
    if args.verbose:
        try:
            nonzero_pct = float((dataset.rewards != 0).float().mean().item() * 100.0)
        except Exception:
            nonzero_pct = float("nan")
        print(f"[diag:dataset] reward_nonzero%={nonzero_pct:.2f}")
    dataset_counts_tensor = torch.bincount(dataset.actions, minlength=len(MODEL_ACTIONS))
    dataset_counts = dataset_counts_tensor.cpu().tolist()
    dataset_counts_ordered = counts_in_config_order(dataset_counts, config_actions)
    if args.verbose:
        label_counts = ", ".join(
            f"{label}:{count}" for label, count in zip(config_actions, dataset_counts_ordered)
        )
        print(f"[ml_rl] Class balance [{label_counts}] (total={sum(dataset_counts_ordered)})")
        mapping = {name: idx for idx, name in enumerate(MODEL_ACTIONS)}
        label_counts_map = {label: count for label, count in zip(config_actions, dataset_counts_ordered)}
        mapping_ok = {label.lower() for label in config_actions} == {label.lower() for label in MODEL_ACTIONS}
        print(
            "[diag:dataset] split=train | label_counts=%s | mapping=%s | label_check=%s"
            % (label_counts_map, mapping, "PASS" if mapping_ok else "WARN")
        )
    class_counts = dataset_counts_tensor.to(torch.float32)
    class_weights = torch.zeros_like(class_counts)
    nonzero_mask = class_counts > 0
    if nonzero_mask.any():
        class_weights[nonzero_mask] = class_counts[nonzero_mask].sum() / (
            class_counts[nonzero_mask] * nonzero_mask.sum()
        )
    sample_weights = class_weights[dataset.actions]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    if args.verbose:
        try:
            num_batches = len(loader)
        except Exception:
            num_batches = 0
        print(f"[train] samples={len(dataset)} | batches={num_batches} | batch_size={batch_size}")
        if num_batches == 0:
            print("[train] WARNING: zero batches (dataset may have collapsed).")
        try:
            first_batch = next(iter(loader))
        except StopIteration:
            first_batch = None
        sample_counts = [0] * len(MODEL_ACTIONS)
        if first_batch is not None:
            _, a_samp, _, _, _ = first_batch
            sample_counts = torch.bincount(a_samp, minlength=len(MODEL_ACTIONS)).cpu().tolist()
        sample_counts_ordered = counts_in_config_order(sample_counts, config_actions)
        sample_pct = pct_from_counts(sample_counts_ordered)
        dataset_pct = pct_from_counts(dataset_counts_ordered)
        sample_mix = ", ".join(
            f"{label}%={pct:.2f}" for label, pct in zip(config_actions, sample_pct)
        )
        dataset_mix = ", ".join(
            f"{label}%={pct:.2f}" for label, pct in zip(config_actions, dataset_pct)
        )
        print(
            "[diag:train] sampler_action_mix="
            + sample_mix
            + " | dataset_action_mix="
            + dataset_mix
        )

    checkpoint_path = CHECKPOINT_DIR / f"light_rainbow_{input_dim}_{hidden_layers}.pt"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model = LightRainbowDQN(input_dim=input_dim, hidden_layers_num=hidden_layers)
    if checkpoint_path.exists():
        if args.verbose:
            print(f"[ml_rl] Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    val_feats: Optional[Tensor] = None
    val_slice_actions: Optional[Tensor] = None
    val_slice_rewards: Optional[Tensor] = None
    if len(dataset) > 0:
        val_subset_size = min(1024, len(dataset))
        val_slice_actions = dataset.actions[:val_subset_size]
        val_slice_rewards = dataset.rewards[:val_subset_size]
        if args.verbose:
            val_feats = dataset.features[:val_subset_size].to(device)

    def _policy_mix_val() -> List[float]:
        if val_feats is None:
            return []
        model.eval()
        with torch.no_grad():
            qv = model(val_feats).mean(dim=2)
            acts = qv.argmax(dim=1).cpu()
        model.train()
        act_counts = torch.bincount(acts, minlength=len(MODEL_ACTIONS)).cpu().tolist()
        ordered = counts_in_config_order(act_counts, config_actions)
        return pct_from_counts(ordered)

    def _reward_by_action(actions_t: Tensor, rewards_t: Tensor, split_name: str) -> None:
        stats: List[str] = []
        for label in config_actions:
            canonical_idx = CANONICAL_ACTION_INDEX.get(label.lower())
            if canonical_idx is None:
                stats.append(f"{label}(n=0,mean=0.0000,std=0.0000,neg%=0.0,zero%=0.0)")
                continue
            mask = actions_t == canonical_idx
            n = int(mask.sum().item())
            if n == 0:
                stats.append(f"{label}(n=0,mean=0.0000,std=0.0000,neg%=0.0,zero%=0.0)")
                continue
            rewards_subset = rewards_t[mask]
            mean = float(rewards_subset.mean().item())
            std = float(rewards_subset.std(unbiased=False).item())
            neg_pct = float((rewards_subset < 0).float().mean().item() * 100.0)
            zero_pct = float((rewards_subset == 0).float().mean().item() * 100.0)
            stats.append(
                f"{label}(n={n},mean={mean:.4f},std={std:.4f},neg%={neg_pct:.1f},zero%={zero_pct:.1f})"
            )
        print(f"[diag:reward_by_action] split={split_name} | " + ", ".join(stats))

    if args.verbose:
        _reward_by_action(dataset.actions, dataset.rewards, "train")
        if val_slice_actions is not None and val_slice_rewards is not None:
            _reward_by_action(val_slice_actions, val_slice_rewards, "val")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    logged_first_batch = False

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        reward_accum = 0.0
        reward_count = 0
        pct_pnl_accum = 0.0
        pct_pnl_count = 0
        epoch_qgap_chunks: List[Tensor] = []
        epoch_sample_counts = torch.zeros(len(MODEL_ACTIONS), dtype=torch.long)
        for feats, actions, rewards, percent_pnls, reward_vectors in loader:
            feats = feats.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            reward_vectors = reward_vectors.to(device)
            epoch_sample_counts += torch.bincount(actions.detach().cpu(), minlength=len(MODEL_ACTIONS))

            optimizer.zero_grad()
            dist = model(feats)
            q_values = dist.mean(dim=2)
            if args.verbose:
                with torch.no_grad():
                    if q_values.shape[1] >= 2:
                        top2, _ = torch.topk(q_values, k=2, dim=1)
                        epoch_qgap_chunks.append((top2[:, 0] - top2[:, 1]).detach().cpu())
            if use_supervised:
                loss = loss_fn(q_values, reward_vectors)
            else:
                chosen_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = loss_fn(chosen_q, rewards)
            loss.backward()
            optimizer.step()

            if args.verbose and not logged_first_batch:
                with torch.no_grad():
                    q_mean = float(q_values.mean().item())
                    q_std = float(q_values.std().item())
                grad_sq = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_sq += float(param.grad.data.norm(2).item() ** 2)
                grad_norm = grad_sq ** 0.5
                print(
                    "[train] epoch=1 | first_batch | grad_norm=%.3e | q_mean=%.6f | q_std=%.6f"
                    % (grad_norm, q_mean, q_std)
                )
                logged_first_batch = True

            epoch_loss += loss.item() * feats.size(0)
            if use_supervised:
                reward_accum += reward_vectors.sum().item()
                reward_count += reward_vectors.numel()
            else:
                reward_accum += rewards.sum().item()
                reward_count += rewards.size(0)
            pct_pnl_accum += float(percent_pnls.sum().item())
            pct_pnl_count += percent_pnls.size(0)

        epoch_loss /= len(dataset)
        avg_reward = reward_accum / max(reward_count, 1)
        avg_pct_pnl = pct_pnl_accum / max(pct_pnl_count, 1)
        if args.verbose:
            print(
                f"[ml_rl] Epoch {epoch}/{epochs} - loss: {epoch_loss:.6f} "
                f"- avg reward: {avg_reward:.4f} - avg pct pnl: {avg_pct_pnl:.4f}%"
            )
            total_seen = int(epoch_sample_counts.sum().item())
            epoch_ordered_counts_full: List[int] = []
            for label in config_actions:
                idx = CANONICAL_ACTION_INDEX.get(label.lower())
                epoch_ordered_counts_full.append(
                    int(epoch_sample_counts[idx]) if idx is not None else 0
                )
            epoch_pct = [
                100.0 * count / max(total_seen, 1) for count in epoch_ordered_counts_full
            ]
            mix_epoch = ", ".join(
                f"{label}%={pct:.2f}" for label, pct in zip(config_actions, epoch_pct)
            )
            try:
                loader_batches = len(loader)
            except Exception:
                loader_batches = 0
            print(
                f"[diag:sampler_epoch] epoch={epoch} | batches={loader_batches} | sampler_action_mix={mix_epoch}"
            )
        if args.verbose and epoch_qgap_chunks:
            all_gaps = torch.cat(epoch_qgap_chunks)
            print(f"[diag:train] top2_qgap_med={all_gaps.median().item():.6f}")
        if args.verbose and val_feats is not None and ((epoch % 5 == 0) or (epoch == epochs)):
            mix = _policy_mix_val()
            mix_str = ", ".join(f"{label}%={pct:.2f}" for label, pct in zip(config_actions, mix))
            print(f"[diag:train] policy_argmax_mix_val={mix_str}")
            model.eval()
            with torch.no_grad():
                val_q = model(val_feats).mean(dim=2)
                top2v = torch.topk(val_q, k=2, dim=1).values
                qgap_med_val = float((top2v[:, 0] - top2v[:, 1]).median().item())
                q_mean_val = float(val_q.mean().item())
                val_pred = val_q.argmax(dim=1).cpu()
            model.train()
            print(
                "[diag:policy_val_fixed] epoch=%d | argmax_mix=%s | top2_qgap_med=%.6f | q_mean=%.6f"
                % (epoch, mix_str, qgap_med_val, q_mean_val)
            )
            if val_slice_actions is not None:
                val_true = val_slice_actions.cpu()
                cm = torch.zeros(len(MODEL_ACTIONS), len(MODEL_ACTIONS), dtype=torch.long)
                for truth, pred in zip(val_true, val_pred):
                    t_idx = int(truth.item())
                    p_idx = int(pred.item())
                    if 0 <= t_idx < len(MODEL_ACTIONS) and 0 <= p_idx < len(MODEL_ACTIONS):
                        cm[t_idx, p_idx] += 1
                per_class_acc: List[float] = []
                for i in range(len(MODEL_ACTIONS)):
                    total_i = int(cm[i].sum().item())
                    acc_i = (int(cm[i, i].item()) / max(total_i, 1)) if total_i else 0.0
                    per_class_acc.append(acc_i)
                majority_counts = torch.bincount(val_true, minlength=len(MODEL_ACTIONS))
                majority_class = int(majority_counts.argmax().item())
                majority_total = int(cm[majority_class].sum().item())
                majority_correct = int(cm[majority_class, majority_class].item())
                majority_acc = majority_correct / max(majority_total, 1) if majority_total else 0.0
                macro_f1 = float(torch.tensor(per_class_acc).mean().item()) if per_class_acc else 0.0
                print(
                    "[diag:metrics_val] epoch=%d | per_class_acc=%s | macro_f1=%.3f | majority_baseline_acc=%.2f"
                    % (
                        epoch,
                        " ".join(
                            f"{label}:{acc:.2f}"
                            for label, acc in zip(config_actions, per_class_acc)
                        ),
                        macro_f1,
                        majority_acc,
                    )
                )
                print(f"[diag:confusion_val] epoch={epoch} | rows=truth cols=pred | {cm.tolist()}")

    torch.save(model.state_dict(), checkpoint_path)
    if args.verbose:
        print(f"[ml_rl] Saved updated checkpoint to {checkpoint_path}")
        end_wall = time.perf_counter()
        usage_end = _usage_snapshot()

        wall = end_wall - start_wall
        if usage_start is not None and usage_end is not None:
            user_cpu = usage_end["user"] - usage_start["user"]
            sys_cpu = usage_end["sys"] - usage_start["sys"]
            max_rss_mb = usage_end["rss_mb"]
            print(
                "[perf:train_agent] wall=%.3fs | cpu_user=%.3fs | cpu_sys=%.3fs | max_rss=%.1f MB"
                % (wall, user_cpu, sys_cpu, max_rss_mb)
            )
        else:
            print(f"[perf:train_agent] wall={wall:.3f}s (CPU/mem stats unavailable)")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.train_agent:
        train_agent(args)
        return 0

    if not args.LightRainbowDQN:
        parser.print_help()
        return 0

    run_light_rainbow(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
