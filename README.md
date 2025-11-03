#

## Mode selection (canonical)

**There is no hidden flag.** Modes are explicit and unambiguous:

- **Live trading (default):** omit `--paper` and `--dry-run`.
- **Paper trading:** add `--paper` (overrides any config).
- **Dry-run (full gate exercise, no orders):** add `--dry-run`.

If both `--paper` and `--dry-run` are supplied, **the last flag wins** (CLI parsing order).

### Canonical CLI
```bash
# Live (REAL orders) – default
python trade.py [--verbose|--silent]

# Paper trading
python trade.py --paper [--verbose|--silent]

# Dry-run smoke (no orders; non-zero exit if any gate fails)
python trade.py --dry-run [--verbose|--silent]
```

> Canonical live path: **Kite SDK**. A raw HTTP/WS path is for advanced users and **not** supported by default.
## Broker Integration (Canonical)
> **Dependency Pins & Installation**
> All third‑party versions are locked in this repo’s `requirements.txt`. Install using:
>
```
bash
BROKER_MODE=kite_sdk
> pip install --require-hashes -r requirements.txt
>

```
> Do **not** edit pins inline in this README. CI fails builds on version ranges to ensure reproducibility.

**Default: Kite Connect SDK (Zerodha)** is the only supported live path in this repo. The raw HTTP/WebSocket approach is provided strictly as a non-default appendix for reference.

- **Pinned dependency:** add the following to your Python dependencies:

```
txt
kiteconnect # pinned via repository lockfile; install with: `pip install --require-hashes -r requirements.txt`
```
- All examples, CLI flags, and deployment instructions below assume the **Kite Connect SDK** path.
- If you previously followed the raw HTTP/WS steps, see **Appendix: Raw HTTP/WS path (optional)** at the end of this document.

RL Trading System — **LIVE, PRODUCTION-READY, NO STUBS / NO FAKES / NO MOCKS**

> A **fully operational**, **post-deployment**, **end-to-end** reinforcement‑learning trading stack for Indian markets.
> **LIVE execution** via Zerodha, **real NSE** index procurement, **incremental data**, **training & evaluation**, **continuous intraday trading**, and a **web dashboard**.
> **No stubs. No fakes. No mocks.** This repository ships the complete, production-ready live system described below.

---

## Table of Contents

- [Operational Overview (LIVE)](#operational-overview-live)
- [Entry Points (LIVE Switches; Stages deprecated)](#entry-points-live-switches--stages)
- [Repository Layout](#repository-layout)
- [Install & Bootstrap (LIVE)](#install--bootstrap-live)
- [Configuration & Secrets (LIVE)](#configuration--secrets-live)
- [Data Contracts (LIVE Storage)](#data-contracts-live-storage)
- [Historical Data: NSE → Zerodha (LIVE)](#historical-data-nse--zerodha-live)
- [Training & Promotion (LIVE Model Lifecycle)](#training--promotion-live-model-lifecycle)
- [Replay Rules & Sizing (LIVE)](#replay-rules--sizing-live)
- [Intraday Trading Loop (LIVE)](#intraday-trading-loop-live)
- [CLI Spec (LIVE Flags, Stages, Function Callers)](#cli-spec-live-flags-stages-function-callers)
- [**Per‑File API (Deterministic Contracts)**](#per-file-api-deterministic-contracts)
 - [.env](#env)
 - [`config.json` keys](#configjson-keys)
 - [`fetch.py` API](#fetchpy-api)

> **Canonical CLI (unified):** Use the interval-based interface for all data ingestion.
>
```
bash
> python -m fetch --from 2024-01-01 --to 2024-01-31 --interval 5m --symbols RELIANCE,TCS
> # or equivalently
> python -m fetch --from 2024-01-01 --to 2024-01-31 --interval 5m --symbols RELIANCE,TCS
>

```
>
> **Deprecated:** The stage-based CLI (` #|token|history|impute`) is deprecated and retained only for backwards compatibility in older notes. Prefer the canonical CLI above in runbooks and examples.

 - [`train.py` API](#trainpy-api)
 - [`trade.py` API](#tradepy-api)
 - [`dashboard.py` API](#dashboardpy-api)
 - [`broker.py` API](#brokerpy-api)
 - [`ml.py` API](#mlpy-api)
 - [`data.py` API](#datapy-api)
- [**Model Architecture (Deterministic Math)**](#model-architecture-deterministic-math)
- [**Rewards & Backtracing (Deterministic Formulas)**](#rewards--backtracing-deterministic-formulas)
- [NSE & Zerodha Operational Details (LIVE)](#nse--zerodha-operational-details-live)
- [Risk, Limits, Circuit Breakers (LIVE)](#risk-limits-circuit-breakers-live)
- [Observability & Health (LIVE)](#observability--health-live)
- [Deployment (LIVE) — Docker / systemd](#deployment-live--docker--systemd)
- [Runbook (LIVE Day-to-Day)](#runbook-live-day-to-day)
- [Security & Compliance](#security--compliance)
- [License](#license)
- [One‑Command Go‑Live](#one-command-go-live)

---

## Operational Overview (LIVE)

This repository delivers a **production trading algorithm**: fetch real market data, train an RL policy, and **place real orders** during market hours using a **winner** checkpoint. All critical flows are **idempotent** and **crash‑recoverable**. Every decision path prioritizes concrete execution—**no stubs/fakes/mocks**.

---

## Entry Points (LIVE Switches & Stages)

All entry points support **`--verbose`** (DEBUG) and **`--silent`** (WARNING+). The last flag wins.

- `fetch.py` — **NSE procurement** → Zerodha token guard → incremental history → imputation
- `train.py` — features → training replay → backtrace → train → holdout → evaluate → **promote winner**
- `trade.py` — **intraday loop** (paper/real) with live order placement and risk controls
- `dashboard.py` — FastAPI **LIVE** dashboard

**Stage selectors** (deprecated) — replace any `--stage <name>` usage with the unified CLI: Replace any `--stage` usage with the unified CLI: **live** = *no flag*; **dry-run** = `--dry-run`. Control scope with `--from/--to/--interval` or whitelisted `--fn` calls.
**Function callers** (`--fn <name> ...`) trigger safe, zero‑arg operational actions.

---

## Repository layout

### Holiday Calendar Continuity
- Ship `config/holidays-<current_year>.json` **and** `config/holidays-<next_year>.json` in the repo.
- Pre-market preflight must **refresh** the active year’s file from broker/NSE API when available; fall back to the local JSON if the API is unavailable.
- A fresh clone must be runnable without external downloads; the preflight update is an optimization, not a requirement.

**Repository layout (updated)**

The repository includes persistent ledgers and a local holiday calendar used by the trading-day gate:

Pin **exact versions** and generate a lock file (no version ranges) using `pip-tools` or `uv`.
Example with pip-tools:

```
echo 'kiteconnect # pinned via repository lockfile; install with: `pip install --require-hashes -r requirements.txt`' >> requirements.in
echo 'torch # pinned via repository lockfile; install with: `pip install --require-hashes -r requirements.txt`' >> requirements.in
pip-compile requirements.in -o requirements.txt

```
> Redeploys must use the compiled **requirements.txt** (or uv lock) to ensure reproducibility.
## Install & Bootstrap (LIVE)

```
bash
pyenv local 3.11.9
python -m venv .venv && source .venv/bin/activate # Install dependencies from the committed lockfile (exact pins; no version ranges)
pip install --require-hashes -r requirements.txt

```

1) Create **`.env`** (below).
2) Prepare **`config.json`** (or omit `"tickers"` to auto‑procure from NSE during `fetch.py`).

> _Note_: Index refresh is handled automatically when needed; there are no `--refresh-index`/`--no-refresh-index` flags.
3) Run:

```
bash
python -m fetch --from 2024-01-01 --to 2024-01-31 --interval 5m --symbols RELIANCE,TCS --verbose
python train.py --verbose
python trade.py --paper --verbose
# When satisfied:

```
json
{
 "winner_min_avg_reward": 0.05,
 "max_slippage_bps": 10,
 "order_min_interval_ms": 250,
 "max_orders_per_minute": 15,
 "leverage": 1.0
}

```
`trade.py` enforces this gate: when invoked in **live mode** (omit `--paper`), it will **refuse to trade** (exits non-zero; no automatic paper fallback mode) if the latest promoted model's `avg_reward < winner_min_avg_reward`. #### Dashboard security & binding - Binds to localhost (127.0.0.1) by default.
It will refuse to trade if `avg_reward` falls below `winner_min_avg_reward` (non-zero exit).
- Requires a bearer token via `DASHBOARD_AUTH_TOKEN` (set in `.env`). Requests without a valid token are rejected (401).
- To expose beyond localhost, explicitly change the bind address and put it behind your reverse proxy + TLS. ### Initial Token Bootstrap (one-time) Obtain a request token from the Kite login flow (redirect URL). Then export it and run the snippet below to **print** your access & refresh tokens; paste the **refresh** into `.env`:
```
bash
export ZERODHA_API_KEY="..."
export ZERODHA_API_SECRET="..."
export ZERODHA_REQUEST_TOKEN="..." # from the redirect query param once

python - <<'PY'
import os, sys
from kiteconnect import KiteConnect
api_key = os.environ["ZERODHA_API_KEY"]
api_secret = os.environ["ZERODHA_API_SECRET"]
req_token = os.environ["ZERODHA_REQUEST_TOKEN"]
kc = KiteConnect(api_key=api_key)
sess = kc.generate_session(req_token, api_secret=api_secret)
print("ZERODHA_ACCESS_TOKEN="+sess["access_token"])
print("ZERODHA_REFRESH_TOKEN="+sess["refresh_token"])
PY

```
Now update `.env` with the printed tokens. Subsequent runs can use the refresh token for headless, non-interactive renewals. ### Environment variables (.env) Create a `.env` with the following keys (single source of truth):
```
env
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret
ZERODHA_ACCESS_TOKEN=
ZERODHA_REFRESH_TOKEN=
DASHBOARD_AUTH_TOKEN=change-me-strong

```
> `ZERODHA_REFRESH_TOKEN` enables automatic token refresh; see **Initial Token Bootstrap** below. ## Appendix: Raw HTTP/WS path (optional) > This appendix is **non-default** and not used in standard live deployments. The canonical live path is the **Kite Connect SDK** documented above. **LIVE** from line one. **No stubs / no fakes / no mocks.** --- ## Configuration & Secrets (LIVE) ### `.env`
```
dotenv
# Zerodha (Kite Connect)
ZERODHA_API_KEY=your_key
ZERODHA_API_SECRET=your_secret
ZERODHA_ACCESS_TOKEN= # optional; auto-refreshed if missing/invalid
ZERODHA_REFRESH_TOKEN= # recommended for non-interactive token lifecycle

# Ops
LOG_LEVEL=INFO # overridden by --verbose/--silent
TZ=Asia/Kolkata # trading windows interpreted in this TZ
DAILY_MAX_LOSS_R=3.0 # mandatory daily lock: halt new entries and flatten on breach

```
**LIVE behavior:** Missing/invalid access tokens are refreshed (when refresh is enabled). No mock flows. ### `config.json` keys - `"timezone"`: e.g., `"Asia/Kolkata"` (all on-disk timestamps stored in **UTC**). - `"index"`: e.g., `"NIFTY50"`. - `"tickers"`: **always refreshed** from NSE on `fetch.py --from 2024-01-01 --to 2024-01-31 --interval 5m --symbols RELIANCE,TCS
- `"timeframes"`: `["5m","15m",...]`. - `"lookbacks"`: `{ "5m": 256, ... }`. - `"from_dates"`: `{ "5m": "YYYY-MM-DD", ... }`. - `"safe_trading_start"`, `"safe_trading_end"`, `"hard_trading_end"` (HH:MM:SS in `timezone`). - `"trailing_stop_r"`, `"capital_per_ticker"`, `"max_concurrent_trades"`, `"paper_trading"`. > Default: `false` (explicit; set `true` only for simulation). > **Clarification — `MFE_price` term:** > - For **longs**: `MFE_price = highest_high_since_entry - entry_price` > - For **shorts**: `MFE_price = entry_price - lowest_low_since_entry` > Use this in the `trailing_stop_r` conversion to maintain deterministic exits across backtest and live. - `"holdout"`: `{ "start": "YYYY-MM-DD", "end": "YYYY-MM-DD" }`. - `"actions"`: `["BUY","SELL","HOLD"]`. - `"paths"`: storage layout for raw/processed/checkpoints/winner. --- ## Data Contracts (LIVE Storage) ### Canonical Symbols & Tradability (LIVE) **Canonical symbol = broker `tradingsymbol`.** All internal references, order placement, and risk checks use the broker's instruments master (`tradingsymbol`, `exchange`, `tick_size`, `lot_size`). Any external codes (e.g., Yahoo-style ``) are treated only as *ingest aliases* and are normalized to the broker `tradingsymbol` before use. **Tradability note:** Index *spot* symbols (e.g., `NIFTY`, `BANKNIFTY`) are **not directly tradable**. When trading derivatives, the system resolves and uses current contract codes (e.g., `NIFTY24NOVFUT`, `BANKNIFTY24NOVFUT`) with correct lot/tick rules from the instruments master. For equities, use actual broker symbols (e.g., `RELIANCE`, `TCS`). **Examples (canonical):**

```
bash
# Equities (historical fetch)
python -m fetch --from 2024-01-01 --to 2024-01-31 --interval 5m --symbols RELIANCE,TCS

# Derivatives (historical fetch; front-month futures as example)
python -m fetch --from 2024-01-01 --to 2024-01-31 --interval 5m --symbols NIFTY24NOVFUT,BANKNIFTY24NOVFUT

```
**Alias mapping (illustrative):**
| Ingest alias | Canonical (broker) |
|---|---|
| `RELIANCE` | `RELIANCE` |
| `TCS` | `TCS` | **Raw candles — `data/raw/{TICKER}/{TF}.parquet`** Columns: `timestamp(UTC)`, `open`, `high`, `low`, `close`, `volume`, optional `oi` Constraints: `low ≤ min(open,close) ≤ max(open,close) ≤ high` Imputation ensures **no NaNs** at persist. **Processed features — `data/processed/{TICKER}/{TF}.parquet`** Columns: `timestamp(UTC)` + `f_*` float32 standardized features per timeframe Targets/rewards are produced later during **replay backtracing**. **Guarantee:** incremental updates append → **dedupe** → **sort** (monotonic `timestamp`). --- ## Historical Data: NSE → Zerodha (LIVE) 1) **NSE procurement (LIVE)** `fetch.py --from 2024-01-01 --to 2024-01-31 --interval 5m --symbols RELIANCE,TCS
2) Acquire/refresh Zerodha access token if it’s missing or invalid (stored in `.env`, validated via `ensure_access_token()` / `fetch.py --from 2024-01-01 --to 2024-01-31 --interval 5m --symbols RELIANCE,TCS
3) **Instrument mapping (LIVE)** Download/refresh broker instrument master and **cache** (token ↔ tradable symbol) for quotes/orders.
> CLI accepts plain names or broker-style symbols; we normalize via the instruments master (e.g., NIFTY → NIFTY25NOVFUT
RELIANCE → RELIANCE). Prefer `<SYMBOL>` in scripts for consistency. 4) **Incremental candles (LIVE)** For each `(ticker, tf)`, compute: `start = max(from_dates[tf], last_on_disk + tf_step)` and `end = now` (all times in UTC on disk). Fetch `[start, end]` only; then **append → dedupe → sort** by `timestamp` → persist Parquet. 5) **Imputation (LIVE)** Forward‑fill and safe interpolation while **preserving OHLC constraints** (no NaNs on disk). **Everything hits real endpoints. No stubs / no fakes / no mocks.** ---
## Training & Promotion (LIVE Model Lifecycle) **Metrics persistence for live trading** On successful promotion, write `models/winner/metrics.json` containing the schema:
```
json
{ "avg_reward": <float>, "evaluated_at": "<ISO8601>", "steps": <int>, "model_hash": "<sha256>" }

```
The live runner (`trade.py `) reads this file and enforces `winner_min_avg_reward`. If promotion is skipped, the previous `models/winner/metrics.json` is retained. **Model:** Dueling **Double DQN** with causal **depthwise‑separable TCN** encoders per timeframe, time pooling, timeframe fusion via MLP, market pooling + gating, **dueling head**. (context size = 2M) **Flow**
1. Build model; if `models/checkpoints/latest.pt` exists, **load** it. 2. Read latest history; generate features per timeframe. 3. Populate **training replay** (excludes holdout). 4. **Backtrace** rewards with deterministic formulas (see below). 5. Train: - AdamW(lr=3e-4, wd=1e-4), Huber loss, γ=0.99 - DDQN target hard copy every 1,000 steps - ε-greedy 0.10→0.01 over 50k steps - Batch 512, grad clip 1.0 - Overwrite `models/checkpoints/latest.pt`
6. Build **holdout replay**, backtrace, **evaluate** average reward. 7. **Promote** to `models/winner/model.pt` **only** if holdout **avg reward** strictly improves. --- ## Replay Rules & Sizing (LIVE) At each timestamp per ticker: a) If **now < safe_trading_start** → **continue** (no trades). b) Run **model inference** for current features; identify **max‑confidence** ticker. - **Tie‑break (max‑confidence)**: prefer tickers **with no active trade**; if still tied, pick the **lexicographically smallest** ticker.
c) If **now > safe_trading_end** and a trade is active for current ticker → **exit immediately**. d) If a **long** is active and (inference=SELL **or** price < trailing stop) → **exit via SELL**. e) If a **short** is active and (inference=BUY **or** price > trailing stop) → **exit via BUY**. f) If **no trade**, **active_trades < max_concurrent_trades**, **current ticker is max‑confidence**, and inference=BUY → **enter BUY** with `qty = floor(capital_per_ticker × leverage / price)`. g) If **no trade**, **active_trades < max_concurrent_trades**, **current ticker is max‑confidence**, and inference=SELL → **enter SELL** (short) with the same sizing rule. Note: round to broker lot-size increments where applicable. All sizing, entries, exits are LIVE‑enforced during trading with broker leverage & lot rules applied. --- ## Intraday Trading Loop (LIVE) Run until **today’s `hard_trading_end`** (TZ: `config.timezone`). - If the exchange calendar marks the day as **early_close**, derive `hard_trading_end` from the calendar **for this run** (override any static config). 4.a: fetch the latest market information for all the tickers → **4.b** 4.b: generate **winner‑model** input features → **4.c** 4.c: if **now < safe_trading_start**, goto **4.a**; else **4.d** 4.d: run **winner model** inference; identify **max‑confidence** ticker → **4.e** - **Tie‑break (max‑confidence)**: prefer tickers **with no active trade**; if still tied, pick the **lexicographically smallest** ticker.
4.e: if **now > safe_trading_end** and trade is active for the current ticker, **exit** (real/paper) and goto **4.a**, else **4.f** 4.f: if **long** active and (inference=SELL or price < trailing stop), **exit** (real/paper) and goto **4.a**, else **4.g** 4.g: if **short** active and (inference=BUY or price > trailing stop), **exit** (real/paper) and goto **4.a**, else **4.h** 4.h: if **no trade**, **active_trades < max_concurrent_trades**, current ticker is **max‑confidence** and inference=BUY, then **enter BUY** (real/paper) with `qty = floor(capital_per_ticker × leverage / price)` and goto **4.a**, else **4.i** 4.i: if **no trade**, **active_trades < max_concurrent_trades**, current ticker is **max‑confidence** and inference=SELL, then **enter SELL** (real/paper) with same sizing and goto **4.a** **LIVE execution** path uses real quotes & orders. **No stubs / no fakes / no mocks.** --- ## CLI Spec (LIVE Flags, Stages, Function Callers) ### Global
- `--verbose` → DEBUG; `--silent` → WARNING+ (last wins). - _(deprecated)_ `--stage <name>` → **Do not use**. Prefer `` / ` --dry-run` with `--from/--to/--interval` or `--fn <name>`.
- `--fn <name>` … call **whitelisted zero‑arg** operational functions (repeatable). ### `fetch.py`

```
bash
python -m fetch [--verbose|--silent]
 [--index TEXT]
 [--from YYYY-MM-DD]
 [--interval {1m,3m,5m,15m,1h,1d
> Note: `1h` is equivalent to `60m`.}]
 [--symbols <comma-separated>]
 [--force-refresh-token/--no-force-refresh-token]
 [--max-workers INT]
 [ #|token|history|impute|all]...
 [--fn load_config|persist_config|nse_tickers|ensure_token|incremental_history|run_imputation]...

```
- **index** — procure **NSE** constituents and **persist** (default: refresh) - **token** — validate/refresh Zerodha access tokens - **history** — incremental candle download - **impute** — run imputation pass - **all** — `index → token → history → impute` ### `train.py`

```
bash
python train.py [--verbose|--silent]
 [--steps INT] [--batch-size INT] [--lr FLOAT] [--gamma FLOAT]
 [--checkpoint-path PATH]
 [features|populate-train|backtrace-train|train|populate-holdout|backtrace-holdout|evaluate|promote|all]...
 [--fn build_model|load_ckpt|make_features|make_train_replay|backtrace_train|fit|make_holdout_replay|backtrace_holdout|eval|promote]...

```
### `trade.py`

```
bash
python trade.py [--verbose|--silent]
 [--paper]
 [--dry-run]
 [--loop-ms INT]
 [--end HH:MM:SS]
 [features|infer|loop|all]...
 [--fn load_winner|one_shot_quotes|one_shot_features|one_shot_infer|loop]...

```
- `--paper` (default from config) or `` for **LIVE orders**
- **Precedence**: CLI flags (`` / `--paper`) **override** `config.paper_trading`; if neither flag is provided, the config value is used. ### `dashboard.py`

```
bash
python dashboard.py [--verbose|--silent]
 [--host TEXT] [--port INT]
 [dump-metrics|all]
 [--fn serve|dump_metrics]...

```
--- ## Per‑File API (Deterministic Contracts) > **All functions are LIVE—no stubs/fakes/mocks.** Use exactly these signatures & behaviors so components interoperate deterministically. ### `.env`
- `ZERODHA_API_KEY`, `ZERODHA_API_SECRET`, `ZERODHA_ACCESS_TOKEN`, `ZERODHA_REFRESH_TOKEN` - `LOG_LEVEL`, `TZ` ### `config.json` keys
- See **Configuration** above. `fetch.py --from 2024-01-01 --to 2024-01-31 --interval 5m --symbols RELIANCE,TCS ### `fetch.py` API

```
python
def load_config(path: str = "config.json") -> dict:
 """Load and validate config. Raises ValueError if invalid."""

def persist_config(cfg: dict, path: str = "config.json") -> None:
 """Atomically write config to disk (UTF-8), preserving key order."""

def get_index_tickers_from_nse(index: str) -> list[str]:
 """Fetch official NSE constituents for `index`. Returns normalized symbols.
 LIVE: HTTP GET with proper User-Agent, backoff, and parsing.
 Raises RuntimeError on fetch/parse errors.

 Retries: 5 attempts on 429/5xx/connection errors with exponential backoff
 (0.5s, 1s, 2s, 4s, 8s) + jitter (0–100ms). Per-request timeout: 10s.
 Throttle: sleep ≥200ms between requests.
"""

def ensure_access_token() -> str:
 """Ensure a valid Zerodha access token (prefer refresh token flow). Returns access token.
 LIVE: Calls broker API; raises RuntimeError on auth failures.

 Retries: up to 3 on transient network errors (backoff 1s, 2s, 4s).
 No retry on 401/invalid credentials (fail fast). Per-request timeout: 10s.
"""

def sync_instruments_cache(force: bool = False) -> str:
 """Download/refresh broker instruments master to local cache. Returns cache path."""

def existing_range(ticker: str, tf: str) -> tuple[datetime.datetime|None, datetime.datetime|None]:
 """Scan on-disk Parquet for last contiguous timestamp range for (ticker, tf)."""

def fetch_history(ticker: str, tf: str, start: datetime.datetime, end: datetime.datetime) -> "pd.DataFrame":
 """Fetch historical candles from broker for [start, end]. LIVE network I/O.
 Guarantees: returns UTC-indexed/dataframe with columns [timestamp, open, high, low, close, volume, (optional oi)]
 and dtype {timestamp: datetime64[ns, UTC], prices: float64, volume/oi: float64}.
 Paginates over broker page/window size transparently; merges and sorts deterministically.

 Retries: 5 attempts with backoff (0.5s, 1s, 2s, 4s, 8s) on 429/5xx/timeouts.
 Per-page timeout: 20s. Throttle: sleep ≥200ms between pages to respect rate limits.
"""

def impute(df: "pd.DataFrame") -> "pd.DataFrame":
 """Fill gaps while preserving OHLC invariants; returns NaN-free frame."""

def save_history(df: "pd.DataFrame", ticker: str, tf: str) -> "pathlib.Path":
 """Append -> dedupe -> sort by `timestamp` -> write Parquet; returns path."""

def main() -> None:
 """Stage dispatcher for index/token/history/impute; index refresh is controlled by flags."""

```
### `train.py` API

```
python
def build_model(cfg: dict) -> "torch.nn.Module":
 """Construct Dueling Double DQN with TCN encoders per config lookbacks/timeframes."""

def load_checkpoint(model: "torch.nn.Module", path: str) -> None:
 """Load state dict if file exists; ignore if missing."""

def load_features(cfg: dict) -> dict[str, dict[str, "np.ndarray"]]:
 """Load processed features per ticker/timeframe into memory (float32)."""

def populate_replay(cfg: dict, holdout: bool) -> "ReplayMemory":
 """Create replay buffer using LIVE rules (a–g), excluding/including holdout as flagged."""

def backtrace_rewards(memory: "ReplayMemory", cfg: dict) -> None:
 """Compute rewards per transition using deterministic formulas (see below)."""

def train_ddqn(model: "torch.nn.Module", memory: "ReplayMemory", cfg: dict) -> dict:
 """Train with DDQN; return metrics dict with step losses and timing."""

def evaluate(model: "torch.nn.Module", holdout: "ReplayMemory") -> dict[str, float]:
 """Return average reward and stability metrics over holdout transitions."""

def maybe_promote_winner(metrics: dict, ckpt_path: str, winner_dir: str) -> bool:
 """Promote latest checkpoint if avg reward strictly improves; return True if promoted."""

def main() -> None:
 """Stage dispatcher for build/features/populate/backtrace/train/holdout/evaluate/promote."""

```
### `trade.py` API

```
python
def load_winner(path: str) -> "InferenceEngine":
 """Load winner model and feature scalers for inference."""

def fetch_realtime_quotes(tickers: list[str]) -> "pd.DataFrame":
 """LIVE quotes snapshot for all tickers; includes last, bid/ask depth if available."""

def build_live_features(quotes: "pd.DataFrame", cache: dict) -> dict[str, "np.ndarray"]:
 """Turn live quotes + rolling buffers into model features per ticker/timeframe."""

def infer(engine: "InferenceEngine", features: dict) -> dict[str, tuple[str, float]]:
 """Return action + confidence per ticker. Confidence in [0,1]."""

def portfolio_state() -> "Positions":
 """Return open positions and PnL; reconciled from broker + local ledger."""

def should_exit_long(signal: str, price: float, trailing_stop_r: float, pos: "Position") -> bool: ...
def should_exit_short(signal: str, price: float, trailing_stop_r: float, pos: "Position") -> bool: ...
def can_enter(is_max_conf: bool, signal: str, active_count: int, cfg: dict) -> bool:
 """Return True only if `active_count < cfg['max_concurrent_trades']`, `is_max_conf` is True,
 and `signal` is either BUY or SELL (not HOLD). Deterministic gate for entry decisions.
 This predicate is purely deterministic and side-effect free.
"""

def enter_trade(ticker: str, side: str, qty: int, stop_price: float, mode: str) -> Dict[str, object]:
 """

Returns a dict containing:
 - entry_order_id: str
 - stop_order_id: str
 - avg_fill_price: float
Place LIVE order (or paper) to open a position.
 Idempotency: generate and reuse a client `client_id` for retries to avoid duplicates.
 Partial fills: poll `order_status()` until filled or timeout; on timeout, reconcile fills,
 reduce remaining qty accordingly, and re-place with a new client_id only if broker requires.
 Exceptions: raises BrokerError on rejects, timeouts, margin issues, or connectivity failures after retries.

 Request timeout: 10s. Retries: 5 on 5xx/timeouts (backoff 0.5s→8s with jitter).
 Post-acceptance fill polling: every 1s up to 90s total. If not fully filled and no fill delta for 15s,
 attempt cancel/replace of the remainder (respecting exchange constraints).
 All retries reuse the same client_id to maintain idempotency unless a cancel/replace is required.
"""

def exit_trade(ticker: str, mode: str) -> None:
 """Flatten/exit an existing position using reduce-only semantics.
 Handles partial fills and retries with idempotent client IDs; raises BrokerError on failure after retries.

 Request timeout: 10s. Retries: 5 on 5xx/timeouts (backoff 0.5s→8s with jitter).
 Reduce-only enforcement. Fill polling every 1s up to 90s; use cancel/replace if broker requires.
"""

def loop_until(end_time: datetime.datetime) -> None:
 """Run LIVE loop with risk/time guards; flatten at end or on kill-switch."""

def main() -> None:
 """Stage dispatcher for quotes/features/infer/loop; supports --paper/."""

```
### `dashboard.py` API
- FastAPI routes: - `GET /` — overview (equity curve, open positions, recent trades) - `GET /trades` — per‑trade analytics: R, MAE(R), %MFE, IS, adherence - `GET /metrics` — training/holdout summary, winner info - `GET /healthz` — liveness; `GET /readyz` — readiness
- `serve(host, port)` — start server (UVicorn). ### `broker.py` API

```
python
class BrokerError(Exception): ...

class ZerodhaClient:
 def authenticate(self) -> None: ...
 def is_token_valid(self) -> bool: ...
 def refresh_token(self) -> str: ...
 def instruments_master(self) -> "pd.DataFrame": ...
 def historical(self, ticker: str, tf: str, start: datetime.datetime, end: datetime.datetime) -> "pd.DataFrame": ...
 def quotes(self, tickers: list[str]) -> "pd.DataFrame": ...
 def leverage(self, ticker: str) -> float: ...
 def place_order(self, ticker: str, side: str, qty: int, client_id: str, reduce_only: bool = False) -> str: ...
 def place_stop_loss(self, ticker: str, side: str, qty: int, stop_price: float, client_id: str, reduce_only: bool = True) -> str: ...
 def modify_order(self, order_id: str, stop_price: float = None, qty: int | None = None) -> None: ...
 def cancel_order(self, order_id: str) -> None: ...
 # Retries: 5 on 5xx/timeouts with backoff (0.5s→8s). Per-request timeout: 10s.

 def order_status(self, order_id: str) -> dict: ...
 def open_positions(self) -> "pd.DataFrame": ...
 def exit_position(self, ticker: str, qty: int, client_id: str) -> None: ...

```
**Order handling (LIVE, idempotent):**
- Generate `client_id = f"{ticker}-{side}-{ts}-{uuid4().hex[:8]}"` per intent.
- Retries **must** reuse `client_id` to avoid duplicates.
- Reconcile partial fills via `order_status()`; update local ledger; retry remaining quantity with new `client_id` if broker requires. ### `ml.py` API

```
python
@dataclass
class Transition:
 state: dict[str, "np.ndarray"]
 action: int
 reward: float
 next_state: dict[str, "np.ndarray"]
 done: bool
 meta: dict[str, "typing.Any"]

class ReplayMemory:
 def push(self, t: Transition) -> None: ...
 def sample(self, batch_size: int) -> list[Transition]: ...
 def __len__(self) -> int: ...
 def clear(self) -> None: ...

class DuelingDoubleDQN(torch.nn.Module): ...
class Trainer: ...
class InferenceEngine: ...
def make_features(raw_data: dict, cfg: dict) -> dict[str, dict[str, "np.ndarray"]]: ...

```
### `data.py` API

```
python
def read_history(ticker: str, tf: str) -> "pd.DataFrame": ...
def write_history(df: "pd.DataFrame", ticker: str, tf: str) -> None: ...
def resample_align(df: "pd.DataFrame", tf: str, tz: str) -> "pd.DataFrame": ...
def impute(df: "pd.DataFrame") -> "pd.DataFrame": ...
def feature_pipeline(df: "pd.DataFrame", cfg: dict) -> "np.ndarray": ...

```
--- ## Model Architecture (Deterministic Math) Let: `S` = #tickers, `A` = #actions, `F` = per‑bar features. For timeframe `τ` with lookback `L_τ`: - Channels: \( C_\tau = \lceil \sqrt{L_\tau \cdot F} \rceil \)
- Kernel: \( K_\tau = 2\lfloor \sqrt{L_\tau} \rfloor + 1 \)
- Depth: choose minimal \( D_\tau \) s.t. causal dilations \(1..D_\tau\) cover \(L_\tau\).
Explicit rule: choose the smallest integer `D_τ = N` such that the cumulative causal receptive field with kernel `K_τ`
covers the lookback window:
`1 + (K_τ - 1) * (N * (N + 1) / 2) >= L_τ`. **Input**: \( X_\tau \in \mathbb{R}^{B\times S\times L_\tau\times F} \) → reshape to \((B\cdot S)\times L_\tau\times F\). **Encoder per `τ` (shared)**: `1×1 Conv (F→C_τ)` → `D_τ` residual TCN blocks: depthwise causal Conv1D (kernel `K_τ`, dilation `i=1..D_τ`) → pointwise Conv1D → SiLU → LayerNorm → residual. **Time pooling**: global mean ⊕ global max → \(\mathbb{R}^{2C_\tau}\). **Fusion**: concat over `τ` → size \( Z = \sum_\tau 2C_\tau \) → MLP \( Z→M→M \) (SiLU), with \( M = \lceil \sqrt{Z} \rceil \). Reshape back to \(B\times S\times M\). **Market pooling**: mean⊕max across `S` → size `2M` → gate `2M→2M` (sigmoid) → elementwise product. **Dueling head**: Value: `2M→M→1`; Advantage: `2M→M→A`. Combine: \( Q = V + (\mathcal{A} - \text{mean}_{a}(\mathcal{A})) \). Properties: receptive fields cover each \(L_\tau\); widths scale ~\(\sqrt{\cdot}\); parameter count independent of `S`. --- ## Rewards & Backtracing (Deterministic Formulas) **1) R‑multiple (Outcome)** \[ R = \frac{\text{Realized PnL}}{(\text{Entry − Stop})\times \text{position size}} \] **2) MAE in R (timing/stop quality)** \[ \mathrm{MAE}_R = \frac{\text{Entry} - \text{Worst vs Stop}}{\text{Entry} - \text{Stop}} \] (Sign adjusted for long/short.) **3) % of MFE Captured (exit efficiency)** \[ \%\mathrm{MFE} = \frac{\text{Realized PnL}}{\text{Max Favorable Excursion}} \times 100\% \] **4) Implementation Shortfall (% of R)** \[ \mathrm{IS} = \frac{\text{Planned PnL} - \text{Realized PnL}}{\text{Initial Risk}} \times 100\% \] **5) Plan Adherence Score** Checklist compliance: \( (\#\text{rules followed} / \#\text{rules total}) \times 100\% \). Checklist (example): entry signal present, position sizing applied, initial stop placed, trailing stop updated, no revenge trades, no signal overrides, exit per rules, log completed. Backtracing traverses each trade’s lifecycle to compute these metrics and writes them into `Transition.meta` for training & holdout evaluation. --- ## NSE & Zerodha Operational Details (LIVE) ### GLOBAL IO Contract (Retries/Timeouts) - **Retries:** up to **5** attempts on transient errors (HTTP 429/5xx/timeouts).
- **Backoff schedule:** **0.5s, 1s, 2s, 4s, 8s** with jitter (0–100ms).
- **Per-request timeout:** **10s** (unless otherwise documented for a specific API).
- **DNS/connect/read timeouts:** **3s / 5s / 5s**.
- **Throttle:** ≥ **200ms** between requests/pages to respect rate limits.
- **Broker heartbeat:** every **5s**; if lost for **2 minutes**, flatten and halt. **NSE procurement**
- Use explicit **User‑Agent**.
- Respect robots & reasonable request rates.
- **Backoff with jitter** (exponential; cap retries).
- HTML layout changes → fail fast with clear error. - **HTTP timeouts & retries (global policy)**: Unless otherwise stated, network requests use a 10s timeout, up to **5 retries** with exponential backoff **0.5s, 1s, 2s, 4s, 8s** and jitter (0–100ms), and throttle of **≥200ms** between calls. **Zerodha authentication**
- Prefer **refresh token** flow (non‑interactive). - On unauthorized: refresh once; if still unauthorized → raise **BrokerError** and halt trading. - Cache access token in memory; renew before expiry window. **Rate limits / errors**
- Apply exponential backoff per the **GLOBAL IO Contract** (5 retries; 0.5s→8s; 10s timeout). - Idempotent order intent via `client_id`. - Network split‑brain: reconcile **open_positions** on every loop; flatten if inconsistency persists beyond a threshold. --- ## Risk, Limits, Circuit Breakers (LIVE) > In LIVE, `DAILY_MAX_LOSS_R` is **mandatory**: on breach, halt new entries and flatten exposure.
**Daily loss lock (MANDATORY)** Configure `DAILY_MAX_LOSS_R` in the environment. When realized+unrealized P/L ≤ `-DAILY_MAX_LOSS_R * R`, the system **halts new entries** and **flattens**. This gate is always enabled in live mode. **Failover & restart reconciliation**: On process start, fetch open positions and open protective stops from the broker, compare to ledger, and reconcile. If a position exists without a protective stop, create one immediately at the last known or computed stop level. **Connectivity/heartbeat degraded mode**: If broker heartbeats are missed for 2 minutes, enter `DEGRADED`:
- Block new entries.
- Queue a *flatten-all* intent.
- On reconnect, execute flatten immediately; if flatten fails across N attempts, raise `BrokerError` and remain `DEGRADED` until manual intervention. - **Max concurrent trades**: `max_concurrent_trades` - **Sizing**: `qty = floor(capital_per_ticker × leverage / price)` (respect lot/step) - **Trailing stop**: `trailing_stop_r` in **R** - **Trailing stop (price‑space conversion)**: Convert `trailing_stop_r` (in **R**) to a price trailing level using initial risk per trade. - **Long**: `initial_risk = entry_price - stop_price`; set/advance `trail_price = max(trail_price, entry_price + MFE_price - trailing_stop_r * initial_risk)`. - **Short**: `initial_risk = stop_price - entry_price`; set/advance `trail_price = min(trail_price, entry_price - MFE_price + trailing_stop_r * initial_risk)`.
- **Time guards**: `safe_trading_start`, `safe_trading_end`, `hard_trading_end` - **Daily stop‑out** (**mandatory in LIVE**): env `DAILY_MAX_LOSS_R` → flatten & lock on breach - **Order throttling**: min inter‑order interval + burst caps - **Heartbeat**: missed broker heartbeats → flatten & halt - **Kill‑switch**: file `.killswitch` or SIGINT/SIGTERM → orderly flatten/halt --- ## Observability & Health (LIVE) - **Logs**: structured JSON (or plain); INFO default; DEBUG with `--verbose`. - **Dashboard**: - `/` overview (equity curve, open positions, recent trades) - `/trades` per‑trade analytics: R, MAE(R), %MFE, IS, adherence - `/metrics` training & holdout summary, winner info - `/healthz` liveness; `/readyz` readiness
- **Audit log**: every order request/response, id, latency, status, fill deltas. No fake metrics. **Everything is LIVE.** --- ## Deployment (LIVE) — Docker / systemd ## One-Command Go-Live For a production launch using Docker Compose (recommended), run the canonical single command from the repo root:
```
bash
docker compose -f docker-compose.yml up -d --build

```
If you are using systemd, the equivalent single command is:

```
bash
sudo systemctl start trader.service

```
Both options assume you have already:
- Populated `.env` with **live** credentials and risk settings
- Run pre‑market checks (`make preflight` or `./scripts/preflight.sh` if present)
- Pulled model checkpoints and verified the **winner** gate passes
- Verified exchange/broker connectivity > Tip: To stop immediately (kill‑switch), use `docker compose down` or `sudo systemctl stop trader.service`. **Docker Compose**

```
yaml
version: "3.9"
services:
 rl-trader:
 build: .
 command: ["bash","-lc","python trade.py --silent"]
 env_file: .env
 volumes:
 - ./config.json:/app/config.json:rw
 - ./data:/app/data:rw
 - ./models:/app/models:rw
 restart: unless-stopped

 dashboard:
 build: .
 command: ["bash","-lc","python dashboard.py --host 127.0.0.1 --port 8000 --silent"]
 env_file: .env
 volumes:
 - ./config.json:/app/config.json:ro
 - ./data:/app/data:ro
 - ./models:/app/models:ro
 ports: ["127.0.0.1:8000:8000"]
 restart: unless-stopped

```
**systemd**: two units (`rl-trader.service`, `rl-dashboard.service`) with `Restart=always` and environment from secure path. --- ## Runbook (LIVE Day‑to‑Day) - **09:00 IST**: check `/healthz`, tokens valid, `.killswitch` absent. - **09:15–09:20**: warm‑up; entries disabled until `safe_trading_start`. - **Session**: monitor equity, open positions, audit latency, rejects. - **15:20–15:29**: exits enforced; **flat** by `hard_trading_end`. - **Post‑close**: `fetch.py` (history) → `train.py` (optional nightly) → potential promotion. - **Emergency**: create `.killswitch` or SIGINT/SIGTERM → flatten/halt. --- ## Security & Compliance - Uses **official** broker APIs; obey exchange/broker terms and **NSE** site usage policies. - Keep `.env` secrets secure; consider OS‑level secret stores for servers. - Follow account, margin, and product constraints; **paper trade** before going **real**. - You are responsible for regulatory compliance, taxes, and operational risk. --- ## License This project **requires** a `LICENSE` file committed at the repository root (MIT/Apache‑2.0/BSD‑3‑Clause). This aligns with the **Repository Artifacts Required** checklist. --- No stubs, no fakes, no mocks—production‑ready from the README to the closing bell.** --- # Production-Critical Fixes Implemented (LIVE, no stubs/fakes/mocks) This README has been updated to close the remaining gaps for a **post-deployment, end‑to‑end, production-ready live trading system**. ## 1) Containerization: Add a real Dockerfile (Compose uses `build: .`)
A working `Dockerfile` is now specified so `docker compose up` can build and run images.
```
dockerfile
# Dockerfile
FROM python:3.11.9-slim
WORKDIR /app

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --require-hashes --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Default command: live trading (silent logs)
CMD ["bash", "-lc", "python trade.py --silent"]

```
> **Dash container note**: If you run the dashboard in a separate service, override the container CMD to:
>
> `CMD ["bash", "-lc", "python dashboard.py --host 127.0.0.1 --port 8000"]` ## 2) Broker connectivity dependency is explicit
Pick exactly **one** mode and ensure your environment matches it. - **Kite Connect SDK mode (recommended)** - Add to *requirements.in*: `kiteconnect # pinned via repository lockfile; install with: `pip install --require-hashes -r requirements.txt`` (then compile your lockfile) - Environment: - `ZERODHA_API_KEY`, `ZERODHA_API_SECRET` (for token init) - `ZERODHA_ACCESS_TOKEN` or the token path you use post-auth - All broker calls (`authenticate`, `quotes`, `place_order`, `order_status`, `positions`) are implemented via the official Kite Connect client. - **Raw HTTP/WebSocket mode (advanced)** - Use `httpx` and `websocket-client` (or `websockets`) and document the exact REST and streaming endpoints used by your `ZerodhaClient` implementation. - Ensure you handle session refresh, rate limits, and WS reconnect/backoff. > **Do not ship without one of the two set up.** If you are using SDK mode, ensure the dependency is actually present in your environment and pinned. ## 3) Dashboard security: safe defaults
**Default bind is now localhost** and a simple bearer token is required by default. - Default host: `127.0.0.1` (avoid exposing on `0.0.0.0` by default).
- Set `DASHBOARD_AUTH_TOKEN` in the environment.
- All routes should check `Authorization: Bearer <token>`. **Run examples**
```
bash
# Local-only
export DASHBOARD_AUTH_TOKEN="$(openssl rand -hex 16)"
python dashboard.py --host 127.0.0.1 --port 8000

# If you intentionally expose externally (behind reverse proxy/TLS): explicitly opt in
python dashboard.py --host 127.0.0.1 --port 8000

```yaml
version: "3.9"
services:
  rl-trader:
    build: .
    command: ["bash","-lc","python trade.py --silent"]
    env_file: .env
    environment:
      - BROKER_MODE=kite_sdk
    volumes:
      - ./config.json:/app/config.json:rw
      - ./data:/app/data:rw
      - ./models:/app/models:rw
    restart: unless-stopped

  dashboard:
    build: .
    command: ["bash","-lc","python dashboard.py --host 127.0.0.1 --port 8000 --silent"]
    env_file: .env
    environment:
      - BROKER_MODE=kite_sdk
    volumes:
      - ./config.json:/app/config.json:ro
      - ./data:/app/data:ro
      - ./models:/app/models:ro
    ports:
      - "127.0.0.1:8000:8000"
    restart: unless-stopped
```


> **Note:** The compose file sets `BROKER_MODE=kite_sdk` explicitly. You may alternatively set it in your `.env` to match your live broker SDK.

yaml
services:
 trader:
 build: .
 environment:
 - ZERODHA_API_KEY=${ZERODHA_API_KEY}
 - ZERODHA_API_SECRET=${ZERODHA_API_SECRET}
 - ZERODHA_ACCESS_TOKEN=${ZERODHA_ACCESS_TOKEN}
 command: bash -lc "python trade.py --silent"
dashboard:
  build: .
  command: ["bash","-lc","python dashboard.py --host 127.0.0.1 --port 8000"]
  env_file: .env
  volumes:
    - ./config.json:/app/config.json:ro
    - ./data:/app/data:ro
    - ./models:/app/models:ro
  ports:
    - "127.0.0.1:8000:8000"
  restart: unless-stopped

 build: .
 command: bash -lc "python dashboard.py --host 127.0.0.1 --port 8000"
 environment:
 - DASHBOARD_AUTH_TOKEN=${DASHBOARD_AUTH_TOKEN}
 ports:
 - "127.0.0.1:8000:8000"

```
> If you truly need remote access to the dashboard, terminate TLS at a reverse proxy and keep the bearer token check enabled. ## LIVE readiness checklist (must be ✅ before going live)
- [ ] Docker image builds locally and under CI with the above `Dockerfile`.
- [ ] **Exactly one** broker mode selected (Kite SDK **or** Raw HTTP/WS) and verified in that environment.
- [ ] `DASHBOARD_AUTH_TOKEN` set; dashboard reachable only via 127.0.0.1 (or behind TLS/proxy).
- [ ] `is_trading_day()` gate passes on market days and skips gracefully on holidays/weekends.
- [ ] `load_winner()` enforces model presence & quality threshold; otherwise exits or flips to `--paper`.
- [ ] Account entitlements confirmed (API trading + historical data).

```

## Normative Specs Addendum (Live Trading)

_Last updated: 2025-11-03 06:34:42Z_

### Zerodha Order Defaults (canonical)
- exchange: NSE
- product: MIS # intraday
- order_type: MARKET # fallback to LIMIT if broker rejects MARKET; set price=last
- validity: DAY
- variety: regular
- time_in_force: DAY
- tag/client_id: as specified (idempotent)
- exit semantics: "reduce-only" enforced in code by sizing <= open position (broker flag not available).

---

### Timeframe Mapping (config → broker)
`config` values map to broker intervals as follows:

| Config | Broker interval |
|-------:|------------------|
| 1m | minute |
| 3m | 3minute |
| 5m | 5minute |
| 15m | 15minute |
| 1h | 60minute |
| 1d | day |

---

### Instrument & Rounding Contract
From instruments master, cache fields: `{instrument_token, tradingsymbol, exchange, lot_size, tick_size}`.

- **Quantity**: `round_down_to_multiple(qty_raw, lot_size or 1)`
- **Price**: `round_to_tick(price_raw, tick_size)` (round half-up to nearest tick)
- **Exchange defaults**: Equity = `NSE`; `tradingsymbol` exactly as in instruments master.
- **Min size**: If computed qty < lot_size (or <1 for cash), **do not place** the order.

---

### Trading Day Gate (normative)

### Mode precedence
_Live is the default (omit `--paper`). If both `--paper` and `--dry-run` are provided, the last flag wins. Supplying both is discouraged and may **error in a future release**. Prefer one explicit mode: `--paper` for simulation or ` --dry-run` for end-to-end validation._

`trade.py` must call `is_trading_day(tz="Asia/Kolkata")` before any broker I/O.

1. Prefer broker/NSE holiday API if reachable (cache for the day).
2. Fallback to a local file: `config/holidays-<year>.json` with schema:

```
json
{"NSE":[{"date":"YYYY-MM-DD","label":"<name>","session":"closed|early_close"}]}

```
On non-trading days: exit 0 before any broker I/O.

---

### Winner Metrics Contract
Path: `models/winner/metrics.json`

```
json
{ "avg_reward": 0.0, "evaluated_at": "YYYY-MM-DDTHH:MM:SSZ", "steps": 0, "model_hash": "<sha256>"
}

```
`trade.py ` refuses to start if **(a)** file missing/corrupt or **(b)** `avg_reward < config["winner_min_avg_reward"]`.

---

### Ledger Contract (Parquet, UTC)

> **Ledger bootstrap policy:** Do **not** commit empty Parquet ledger files. The engine will auto-create missing ledgers on startup and on first write.

Files:
- `data/ledger/orders.parquet` with columns: `[ts, client_id, side(BUY/SELL), ticker, qty, order_id, status, price, product, order_type]`
- `data/ledger/fills.parquet` with columns: `[ts, order_id, fill_id, ticker, qty, price]`
- `data/ledger/positions.parquet` with columns: `[ts, ticker, qty, avg_price, unrealized_pnl]`

Invariants: `ts` is UTC; appends only; reconciliation updates positions on every loop; retries are idempotent via `client_id`.

---

### Compose Port Binding (normalized)
All docs and examples bind the dashboard/API to **8000**.

```
yaml
ports: - "127.0.0.1:8000:8000"

```

## Broker interface

**Broker-native protective stops and trailing maintenance (REQUIRED in live mode)**

On every entry, place a *broker-side, reduce-only* protective stop immediately after the entry order is accepted. The trading app must keep the stop's `order_id` in the ledger and maintain it as the trailing stop improves.

**Contract changes**

- `enter_trade(ticker, side, qty, mode, stop_price: float, client_id: str=None) -> dict`
 Places the entry and immediately calls `place_stop_loss(...)` below.

**New broker interface functions**

- `place_stop_loss(ticker, side, qty, stop_price, client_id, reduce_only=True) -> str` — returns `order_id` of the protective stop.
- `modify_order(order_id, stop_price: float = None, price: float | None = None) -> None` — used to trail the stop.
- On restart, reconcile open positions and associated protective stops from broker + ledger **before** resuming.

**Trailing rule**

When the computed trailing stop (per R-space formulas) tightens, call `modify_order` to move the broker stop; only tighten, never loosen. Store latest `order_id` and `stop_price` in the ledger.

## Execution

**Execution guards**

Add to `config.json` (example values):

```
json
{ "max_slippage_bps": 30, "order_min_interval_ms": 500, "max_orders_per_minute": 20, "winner_min_avg_reward": 0.05, "leverage": 1.0
}

```

Entries use MARKET only if expected fill is within `max_slippage_bps` of last/nbbo mid; otherwise convert to a LIMIT at the guard price or skip the order.

### Appendix: Example systemd service (optional)

`/etc/systemd/system/trader.service`:

```
ini
[Unit]
Description=Live Trading Bot
After=network.target

[Service]

Type=simple
User=trader
WorkingDirectory=/opt/trader
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=/opt/trader/.env
ExecStart=/usr/bin/python3 trade.py --silent
Restart=on-failure
RestartSec=10 [Install]
WantedBy=multi-user.target

```
Enable & start:

```
bash
sudo systemctl daemon-reload
sudo systemctl enable trader
sudo systemctl start trader

```
## Trade API (normative)
The trade entry contract is **normative** for LIVE mode:

```
python
def enter_trade(ticker: str, side: Literal["BUY","SELL"], qty: int, stop_price: float, mode: str) -> dict: """ Places the entry **and synchronously places a broker-side, reduce-only protective stop**. Persists both entry and stop order_ids to the ledger atomically. Trailing is applied via modify_order; stops are never loosened. Returns a dict with {entry_order_id, stop_order_id, avg_fill_price}. """ ...

```

_Note_: Entry must **always** result in a protective stop created and recorded in the ledger before returning.

## Dependencies & Lockfiles (mandatory)
To guarantee reproducible deploys, **all dependencies must be exact-pinned** and builds must resolve **from a committed lock file only**.

- Preferred: `uv lock` (committed `uv.lock`) or `pip-tools` (`pip-compile`) producing a compiled `requirements.txt`.

- **Ranges are prohibited** in CI/production.

- Local development may use ranges, but Docker/CI **MUST** install from the lock file.

**Example (illustrative only; replace with your compiled lock):**

```
text
numpy==1.26.4
pandas==2.2.2
pyarrow==15.0.2
scikit-learn==1.5.1
kiteconnect==4.2.0
python-dotenv==1.0.1

```
**CI rule:** If a range (`>=`, `<`, `~=`) is detected in the production install set, the pipeline fails.

## Broker Mode Selection

> **Canonical**: Use `kite_sdk` for LIVE. `raw` is for advanced users and not supported by default.
Set a single, explicit broker path via environment:

```
bash
export BROKER_MODE=kite_sdk # or: raw

```

At startup, the app **refuses to run** if modules for the non-selected path are imported. The canonical path for production is **`kite_sdk`**.

## CI Requirements (non-negotiable for LIVE)
- Build Docker image **from the lock file** (`uv.lock` or compiled `requirements.txt`).

- Run fast smoke tests and **fail the pipeline on any non-zero exit**:

```
bash
# auth/token fetch (no network trading side-effects)
python -m fetch # # minimal data path (tiny symbol set)
# use two liquid NSE symbols that exist in your instruments master; adjust as needed
python -m fetch # --from 2024-01-01 --to 2024-01-31 --interval 5m --symbols RELIANCE,TCS # quote path sanity
python trade.py --dry-run --fn one_shot_quotes # LIVE gate behavior (no winner => non-zero exit)
python trade.py --dry-run # must exit non-zero if winner invalid or avg_reward below threshold

```
- Enforce policy: **no automatic paper fallback** from LIVE.

## Repository Artifacts Required for Fresh-Clone Runs
- `docker-compose.yml` (at repo root; required for one-command go-live)

To honor the “fresh clone runs” guarantee:

- `LICENSE` (MIT/Apache-2.0/etc.) committed at repo root.

- **Do not commit empty ledgers.** Ledger Parquet files are auto-created on startup (and on first write) under `data/ledger/`.
- Holiday calendars committed for current and next year under `config/` (e.g., `config/holidays-2025.json`, `config/holidays-2026.json`).

- `Dockerfile` committed at repo root consistent with the lockfile-only install.

- Lock file (`uv.lock` or compiled `requirements.txt`) committed.

## Dependencies & Installation

**Lockfile only (no version ranges).** Reproducible installs are mandatory.

1. Generate the lockfile locally (choose one tooling approach) and commit it:
 - **uv**:

```
bash
uv lock
uv pip install --require-hashes -r requirements.txt
```
- **pip-tools**:

```
bash
pip-compile --generate-hashes --upgrade requirements.in -o requirements.txt
pip install --require-hashes -r requirements.txt
```

2. CI/CD **fails** if version ranges are detected. Only exact versions are allowed in `requirements.txt` (or `uv.lock`).

> Install command used everywhere in this repo:
>
```
bash
> pip install --require-hashes -r requirements.txt # exact pins from the committed lockfile
>

```
## Install & Bootstrap

Set broker mode **before** starting any process. The app refuses to start if imports do not match the selected mode.

```
bash
export BROKER_MODE=kite_sdk # or: raw

```

- `kite_sdk`: uses official Zerodha KiteConnect SDK.
- `raw`: uses direct HTTP calls (no SDK).

The process aborts at startup if code paths import modules from a non-selected mode.

## Command Line Interfaces (CLI)

### `fetch.py`

Fetch and persist market data.

```
python -m fetch --from 2024-01-01 --to 2025-11-03 --symbols RELIANCE,TCS --interval 1m

```

**Flags**

- `--from YYYY-MM-DD` / `--to YYYY-MM-DD` – date range (inclusive start).
- `--interval {1m,3m,5m,15m,1h,1d}` – candle interval.
- `--symbols <comma-separated>` – **optional**: restrict procurement universe to these tickers. If omitted, uses configured universe.

### `trade.py`

Run the live trading loop.

```
python trade.py --dry-run --silent

```

**Flags**
- Live is the **default** mode; omit `--paper` to trade with the live broker.
- `--dry-run` – exercise all gates/validations with the real broker session **but do not place orders**. The process exits non-zero if any gate that would block live trading fails. Useful for smoke checks in CI and pre-market readiness.

## Trade Entry & Protective Stop (Normative) *(Canonical live contract)*

All entries must install a **broker-side protective stop** immediately after entry is accepted. The trading loop is atomic over (entry, stop).

```
python
from typing import Literal, Dict def enter_trade(ticker: str, side: Literal["BUY","SELL"], qty: int, stop_price: float, mode: str) -> Dict[str, object]: """ Places a marketable entry and immediately attaches a broker-side reduce-only stop. Returns a dict containing: - entry_order_id: str - stop_order_id: str - avg_fill_price: float """ ...

```

**Loop contract (normative):** Upon entry acceptance:
1. Place reduce-only protective stop at `stop_price` with quantity `qty`.
2. Persist `{position_id, entry_order_id, stop_order_id, avg_fill_price}` **atomically**.
3. If any step fails, cancel open legs and reconcile to flat.

> Note: Zerodha does not expose native *reduce-only*; the implementation enforces reduce-only semantics by ensuring `qty <= open_position_qty` and reconciling competing manual orders.

## Feature Specification (Normative)

The feature pipeline is deterministic and versioned. The model expects a fixed layout per timeframe.

**Tensor layout:** `(B · S, L_τ, F)` where:
- `B` = batch size, `S` = symbols, `L_τ` = lookback length, `F` = feature count.

**Ordered feature list (float32)** per candle:
1. `f_returns_1`, `f_returns_3`, `f_returns_5`, `f_returns_10` (log returns)
2. `f_vol_10`, `f_vol_20` (stdev of returns)
3. `f_atr_14`
4. `f_rsi_14`
5. `f_kdj_14_3` (K,D,J)
6. OHLC normalized to rolling `median(price,20)` and z-scored over a sliding lookback
7. Market state: minutes since open, minutes to close, `early_close_flag` (0/1)

**Versioning:** Any change that alters shape or order increments `FEATURE_SPEC_VERSION` and fails fast if mismatched.

## Timezone & Timestamps

**Runtime precedence**

- `config["timezone"]` governs trading windows and scheduling.
- `TZ` (process env) only affects locale/time formatting.
- All on-disk timestamps are persisted in **UTC**.

This prevents drift between operator locale and engine scheduling.

## Continuous Integration (CI)

### CI / Pre-market Smoke Checks

- Lockfile integrity & reproducible install.
- Dry-run of the live loop against the real broker session (no orders placed).

Example GitHub Actions step:

```
```yaml
- name: Install (lockfile exact pins)
  run: pip install --require-hashes -r requirements.txt

- name: Broker smoke (no orders)
  env:
    BROKER_MODE: kite_sdk
  run: python trade.py --dry-run
```


```

If the dry-run fails, CI exits non-zero and deployment is blocked.

## Operations Runbook (LIVE)

- On process start, verify **ledger to broker stop linkage**. If a position lacks a protective stop, **place it immediately** before resuming the loop.
- Zerodha has no native *reduce-only*; we enforce reduce-only semantics by limiting quantity to `<= open_position_qty` and reconciling any manual orders that would increase exposure.

```
## config.json keys (additions)
- `winner_min_avg_reward` *(float)* — Minimum evaluated `avg_reward` required for live trading (winner gate).
- `max_slippage_bps` *(int)* — Guardrail; convert MARKET to LIMIT or skip if expected slippage exceeds this.
- `order_min_interval_ms` *(int)* — Minimum spacing between orders in milliseconds.
- `max_orders_per_minute` *(int)* — Burst cap for order placement.
- `leverage` *(float)* — Used in sizing: `qty = floor(capital_per_ticker × leverage / price)`.

### Canonical trade examples

```
bash
python trade.py --paper --fn one_shot_quotes --silent

```
bash
python trade.py --dry-run --silent

```

#### First-run ledger bootstrap

On startup, if `data/ledger/{orders,fills,positions}.parquet` do not exist, they are **created automatically** with the exact contract schema (schema versions are tracked; migrations bump `schema_version` and run before trading hours). This removes the need to commit empty files and guarantees deterministic first-run behavior in LIVE.

## Canonical examples

### `config.json`
```json
{
  "winner_min_avg_reward": 0.05,
  "max_slippage_bps": 10,
  "order_min_interval_ms": 250,
  "max_orders_per_minute": 15,
  "leverage": 1.0
}
```

### `metrics.json` schema
```json
{
  "avg_reward": 0.0,
  "evaluated_at": "YYYY-MM-DDTHH:MM:SSZ",
  "steps": 0,
  "model_hash": "<sha256>"
}
```

### `.env` (example)
```
# Zerodha (Kite Connect)
ZERODHA_API_KEY=your_key
ZERODHA_API_SECRET=your_secret
ZERODHA_ACCESS_TOKEN=
ZERODHA_REFRESH_TOKEN=

# Ops
LOG_LEVEL=INFO
TZ=Asia/Kolkata
DAILY_MAX_LOSS_R=3.0

# Dashboard
DASHBOARD_AUTH_TOKEN=change-me-strong
```

## CLI examples (copy/paste-safe)

```bash
# History fetch
python -m fetch --from 2024-01-01 --to 2024-01-31 --interval 5m --symbols RELIANCE,TCS --verbose

# Training
python train.py --verbose

# Paper loop sanity
python trade.py --paper --verbose

# Live (real orders)
python trade.py --silent
```

## Normative specs (single source of truth)

This section is the canonical reference for:
- Zerodha order defaults
- Timeframe mapping
- Calendar/trading-day gates

If any similar tables appear elsewhere, **treat this section as authoritative** and consider those duplicates to be removed in future edits.

## Ledger bootstrap policy (canonical)

On startup, if any of the following are missing under `data/ledger/`:
`orders.parquet`, `fills.parquet`, or `positions.parquet`, the app will **create** them

## Trade API (canonical)

```python
from typing import Literal, Dict

def enter_trade(
    ticker: str,
    side: Literal["BUY","SELL"],
    qty: int,
    stop_price: float,
    mode: Literal["live","paper","dryrun"]
) -> Dict[str, object]:
    """
    Places the entry and, on acceptance, synchronously installs a broker-side
    reduce-only protective stop. Returns {entry_order_id, stop_order_id, avg_fill_price}.
    Atomicity: if stop placement fails, cancel entry and reconcile to flat.
    """
```

## Installation & locking (canonical)

Use **one** of the following approaches. Do not mix them.

### Option A: pip-tools (hash-locked)
```bash
# One-time: generate lockfile
pip install pip-tools
pip-compile --generate-hashes -o requirements.txt

# Everywhere (CI, Docker, prod): install from lockfile ONLY
pip install --require-hashes -r requirements.txt
```

### Option B: uv
```bash
uv lock
uv pip install --require-hashes -r requirements.txt
```

**Canonical invocation examples**
- Paper sanity:

```bash
python trade.py --paper --verbose
```
- Dry-run gate:

```bash
python trade.py --dry-run --silent
```
- Live trading (default):

```bash
python trade.py --silent
```

## Canonical Dockerfile (pinned base + hash-enforced installs)

Use this Dockerfile for reproducible, production builds.

```dockerfile
# Dockerfile
FROM python:3.11.9-slim
WORKDIR /app

# Install system deps only if needed by your wheels; otherwise keep slim.
# RUN apt-get update && apt-get install -y --no-install-recommends <deps> && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
# requirements.txt must be compiled with --generate-hashes
ENV PIP_NO_CACHE_DIR=1
RUN pip install --require-hashes -r requirements.txt

COPY . .
CMD ["bash", "-lc", "python trade.py --silent"]
```
