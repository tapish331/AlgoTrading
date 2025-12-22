# AlgoTrading

> Reinforcement-learning intraday trader for NSE equities that pairs Zerodha's Kite APIs with a lightweight Rainbow-style DQN, full data-fetch and training pipelines, and battle-tested safety rails.

## Key Capabilities

- **End-to-end workflow**: refresh NSE constituents, pull multi-timeframe OHLCV from Zerodha, build replay buffers, train the LightRainbowDQN agent, then paper or live trade in a single repository.
- **Cost-aware RL**: rewards use net percent PnL after brokerage, STT, and GST so the policy optimizes deployable returns.
- **Stateful live loop**: `trade.py` resumes from `state/active_trades.json`, reconciles with broker positions, enforces safe/hard trading windows, and trails stops per ticker.
- **Config-driven**: `config.json` governs timeframes, tickers, lookback, trade status labels, concurrency limits, and ML hyper-parameters—no code edits required for most tweaks.
- **Safety-first defaults**: trading starts in paper mode, square-off happens automatically at `safe_end_time`/`hard_end_time`, and only the highest-confidence actionable signal per cycle is executed.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Repository Layout](#repository-layout)
3. [Data & Model Flow](#data--model-flow)
4. [Configuration Highlights](#configuration-highlights)
5. [Pipelines & CLI Workflows](#pipelines--cli-workflows)
6. [Key Components](#key-components)
7. [Safety & Observability](#safety--observability)
8. [Troubleshooting](#troubleshooting)
9. [Development Tips](#development-tips)

## Quick Start

### Requirements

- Python 3.11+ (matches the versions used for Torch 2.2 and pandas 2.1).
- Zerodha Kite Connect account with API key/secret and access to historical data.
- NSE market data entitlements for the configured tickers/timeframes.

### Environment setup

```bash
git clone https://github.com/your-org/AlgoTrading.git
cd AlgoTrading
python -m venv .venv
.\.venv\Scripts\activate          # use `source .venv/bin/activate` on Unix
pip install --upgrade pip
pip install -r requirements.txt
```

Create `.env` in the repo root (already git-ignored) with your Zerodha credentials:

```ini
ZERODHA_API_KEY=xxxx
ZERODHA_API_SECRET=xxxx
# ZERODHA_ACCESS_TOKEN will be written/rotated automatically by zerodha_broker.py
```

Finally, copy `config.json` if you want environment-specific overrides and confirm the tickers/timeframes suit your deployment.

## Repository Layout

```text
.
├── config.json                  # Primary runtime + training configuration
├── fetch.py                     # Historical OHLCV fetch + ticker refresh pipeline
├── nse.py                       # NSE index constituent downloader (updates config.json)
├── zerodha_broker.py            # Kite authentication, snapshots, order helpers
├── replay.py                    # Replay-memory builder + feature engineering
├── ml_rl.py                     # LightRainbowDQN definition + trainer utilities
├── train.py                     # Continuous train/eval/promote loop
├── trade.py                     # Live/paper trading loop powered by RL agent
├── requirements.txt
├── data/                        # Historical candles + replay buffers
├── checkpoints/                 # Model checkpoints (winner + intermediates)
└── state/                       # Active trade cache + completed trade logs
```

Artifacts you will generate:

- `data/<TICKER>/<timeframe>/history.csv` – rolling OHLCV store.
- `data/replay_memory_{training,evaluation}.jsonl` – replay buffers.
- `checkpoints/light_rainbow_<input_dim>_<layers>.pt` – current weights.
- `checkpoints/light_rainbow_winner.pt` & `state/completed_trades/YYYY-MM-DD.jsonl` – model promoted for trading and daily execution logs.

## Data & Model Flow

```
NSE index -> nse.py (refresh constituents)
           -> fetch.py (ensure token + download OHLCV into data/)
           -> replay.py (multi-timeframe features + reward shaping)
           -> ml_rl.py (train LightRainbowDQN from replay memory)
           -> train.py (continuous training, evaluation, checkpoint promotion)
           -> trade.py (paper/live loop; resumes active trades; submits orders via Zerodha)
```

Feature vectors are assembled per ticker from normalized OHLCV windows across every configured timeframe (5 features × `timeframes` × `lookback`) plus two metadata scalars: intraday candle position and normalized trade-status index. The LightRainbowDQN (dueling + distributional + noisy heads) outputs a [batch, 3, 8] distribution over the canonical actions `hold/buy/sell`, and an action mask prevents illegal transitions (e.g., opening a new long while already long).

## Configuration Highlights

`config.json` is the single source of truth. Key sections:

| Section | Important fields | Notes |
| --- | --- | --- |
| Top-level | `index`, `tickers`, `timeframes`, `decision_interval`, `actions` | Refresh tickers via `nse.py`; `decision_interval` must be in `timeframes`. |
| Session limits | `safe_start_time`, `safe_end_time`, `hard_end_time` | Trading waits until safe start, closes positions at safe end, and force-square-offs at hard end. |
| Risk & sizing | `trailing_stop_loss_pct`, `max_concurrent_trades`, `capital_per_ticker`, `leverage` | Used directly by `trade.py` for position sizing and trailing stops. |
| Fetch | `fetch.num_candles` | Controls the number of most recent candles stored per timeframe/ticker. |
| Training | `training_days_num`, `evaluation_days_num`, `train.lookback`, `ml_rl.hidden_layers_num`, `ml_rl.learning_rate`, `ml_rl.batch_size`, `ml_rl.epochs` | Replay generation and model hyper-parameters. |
| Trading block | `trading.mode`, `trading.poll_interval_seconds` | `mode` defaults to `paper`. Switch to `live` only after verifying the entire stack. |
| Trade status | `trade_status` array | Drives action masking and the state feature appended to every observation (`flat`, `long`, `short`). |

## Pipelines & CLI Workflows

### 1. Refresh NSE constituents

```bash
python nse.py --refresh_tickers --verbose
```

Downloads the latest constituents for `config.json["index"]`, filters out the index symbol itself, and rewrites `tickers`.

### 2. Fetch historical OHLCV

```bash
python zerodha_broker.py --ensure_access_token --verbose  # run once per day
python fetch.py --verbose
```

`fetch.py` performs: ticker refresh → token check → bulk historical download for every ticker/timeframe into `data/`, deduplicating existing CSVs. Configure `fetch.fetch.num_candles` to control depth.

### 3. Build replay memory

```bash
python replay.py --populate_training_replay_memory --verbose
python replay.py --populate_evaluation_replay_memory --verbose
```

`replay.py` stitches together synchronized multi-timeframe windows, computes proximity-based contextual rewards, applies brokerage/transaction costs, and writes JSONL buffers. Diagnostics report action mixes, reward fingerprints, and coverage.

### 4. Train / inspect the LightRainbowDQN

```bash
python ml_rl.py --train_agent --verbose
# or to inspect architecture
python ml_rl.py --LightRainbowDQN --verbose
```

Training uses a weighted sampler to balance hold/buy/sell classes, logs per-action reward stats, and writes checkpoints under `checkpoints/`. By default it reads `data/replay_memory_training.jsonl`; override via `--replay_path`.

### 5. Continuous trainer & checkpoint promotion

```bash
python train.py --verbose
```

`train.py` loops forever: regenerate replay buffers, train the agent, run evaluation replay, compare against the previous `light_rainbow_winner.pt` using dataset signatures + per-trade PnL (total PnL / trades), and promote when the new model improves or the dataset changed. Promotion metadata is stored in `checkpoints/winner_meta.json`.

### 6. Live / paper trading

```bash
python trade.py --verbose --poll-seconds 5
```

`trade.py` workflow:

1. Loads `config.json`, warms up the LightRainbowDQN with the current winner checkpoint, and ensures Zerodha auth.
2. Restores `state/active_trades.json` and reconciles with live broker positions (in live mode).
3. Each loop fetches a fresh snapshot via `fetch_market_snapshot`, rebuilds features, and infers actions for every ticker.
4. Only the highest-confidence actionable ticker per cycle can open a new position, respecting `max_concurrent_trades`.
5. Trailing stops (`trailing_stop_loss_pct`) and opposite signals close positions. Safe/hard windows force square-offs and completed trades are appended to `state/completed_trades/<date>.jsonl`.
6. The loop defaults to `trading.mode = paper`; set `live` only after dry-running end to end.

Interrupting the script performs an orderly square-off and persists state so the next session resumes safely.

## Key Components

- **LightRainbowDQN (`ml_rl.py`)** – Minimal Rainbow-inspired agent: dueling architecture, distributional (8 quantiles) heads, and NoisyLinear layers. The last feature channel encodes normalized trade status, and a mask buffer enforces legal actions per status.
- **Replay engine (`replay.py`)** – Loads `data/` history, builds normalized OHLCV windows per timeframe/lookback, computes intraday candle positions, injects trade-status indices, and exports JSONL transitions with detailed diagnostics and fingerprints.
- **Broker helpers (`zerodha_broker.py`)** – Handles credential storage in `.env`, performs the manual request-token exchange when needed, backfills historical candles, throttles Kite API calls, simulates orders in paper mode, and reconciles live positions with local state.
- **State directory (`state/`)** – Contains `active_trades.json` (auto-written each cycle) and `completed_trades/*.jsonl` audit trails, making it easy to review fills, trailing-stop exits, and realized PnL long after a session ends.

## Safety & Observability

- **Paper-first**: `config["trading"]["mode"]` defaults to `paper`, and `submit_market_order` turns into a dry-run until you explicitly opt into live.
- **Session guards**: `safe_start_time`, `safe_end_time`, and `hard_end_time` keep entries within your comfort window and guarantee square-off at the hard cutoff (even on KeyboardInterrupt).
- **Trade accounting**: `trade.py` syncs with broker positions on startup (so orphaned live positions are captured) and logs every exit reason (`signal` vs `trailing_stop`) along with %PnL after costs.
- **Diagnostics**: All pipelines emit concise `[diag:*]` lines—action mixes, reward stats, sampler behavior, Q-gap metrics, replay fingerprints—which helps validate new datasets or config tweaks before going live.
- **Data quality**: Feature generation requires complete history per ticker/timeframe. Missing frames are explained in verbose logs so you can re-run `fetch.py` or adjust `lookback`.

## Troubleshooting

- **Missing Zerodha token**: `python zerodha_broker.py --ensure_access_token --verbose` guides you through the login flow and refreshes `ZERODHA_ACCESS_TOKEN` in `.env`.
- **No historical data / insufficient lookback**: rerun `python fetch.py --verbose` after ensuring `fetch.num_candles` ≥ `lookback * 2 + buffer`.
- **Torch import errors**: confirm the virtualenv is active and re-run `pip install -r requirements.txt`. Torch 2.2 wheels require Python 3.11+ on many platforms.
- **`decision_interval` mismatch**: make sure the value exists in `config["timeframes"]`; otherwise both `replay.py` and `trade.py` will abort.
- **Stale winner checkpoint**: delete or move `checkpoints/light_rainbow_winner.pt` if you intentionally want to retrain from scratch, then rerun `train.py` (it will recreate the winner).

## Development Tips

- Run `python ml_rl.py --LightRainbowDQN --verbose` after changing `config.json` to confirm the derived `input_dim` and hidden-layer count.
- Keep the repo tidy by activating the virtualenv before every command (`.\.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on Unix).
- When debugging the live loop, pass `--verbose` and temporarily shorten `trading.poll_interval_seconds` to watch state transitions without waiting a full minute.
- Record dry-run sessions by keeping the generated `state/completed_trades/*.jsonl` files—they capture entry/exit timestamps, prices, reasons, and realized percent PnL for later analysis.

Happy experimenting, and always validate with paper trading before switching `trading.mode` to `live`.
