# RL Trading System

> End-to-end pipeline for **data fetching**, **model training** (Dueling Double DQN), **live/paper trading**, and a **monitoring dashboard**.
> This README is a **codegen-ready spec**: deterministic, typed, testable.

---

## Table of Contents

* [Build-from-Spec Conventions](#build-from-spec-conventions)
* [Overview & Entry Points](#overview--entry-points)
* [Project Layout](#project-layout)
* [Setup](#setup)
* [Configuration & Secrets](#configuration--secrets)

  * [.env](#env)
  * [`config.json` schema](#configjson-schema)
  * [Example `config.json`](#example-configjson)
* [Data Contracts](#data-contracts)
* [Historical Data Fetch (with NSE procurement)](#historical-data-fetch-with-nse-procurement)
* [Training](#training)
* [Replay Memory Population](#replay-memory-population)
* [Live/Paper Trading](#livepaper-trading)
* [Model Architecture](#model-architecture)
* [Rewards & Evaluation Metrics](#rewards--evaluation-metrics)
* [Per-file (per-function) Guide](#per-file-per-function-guide)
* [CLI Spec](#cli-spec)
* [Logging, Errors, and Idempotency](#logging-errors-and-idempotency)
* [Testing & Acceptance Criteria](#testing--acceptance-criteria)
* [Operational Notes](#operational-notes)
* [Safety & Disclaimers](#safety--disclaimers)
* [License](#license)
* [Appendix: Pseudocode](#appendix-pseudocode)

---

## Build-from-Spec Conventions

**Language & versions**

* Python **3.11** (type hints everywhere; `mypy --strict` clean).
* Style: **black**, **ruff**, **isort**.

**I/O & time**

* Persist tabular data as **Parquet** (`pyarrow`).
* On disk, timestamps are **UTC**; trading window strings are in **`config.timezone`** (default `Asia/Kolkata`) and converted internally.

**Design**

* Public functions have docstrings with **Args/Returns/Raises**.
* Functions are pure unless prefixed with `save_`, `write_`, `place_`, `exit_`, or `loop_`.

---

## Overview & Entry Points

**Top-level scripts**

* `fetch.py` — bootstrap & incrementally update historical market data.
* `train.py` — build/train a Dueling Double DQN; evaluate; promote the winner.
* `trade.py` — intraday trading loop (real or paper).
* `dashboard.py` — local dashboard for PnL, positions, metrics.

**Quick run**

```bash
python fetch.py
python train.py
python trade.py
python dashboard.py
```

---

## Project Layout

```
.
├─ fetch.py
├─ train.py
├─ trade.py
├─ dashboard.py
├─ broker.py
├─ ml.py
├─ data.py
├─ config.json
├─ .env
├─ requirements.txt
├─ Makefile
├─ data/
│  ├─ raw/{TICKER}/{TF}.parquet
│  └─ processed/{TICKER}/{TF}.parquet
└─ models/
   ├─ checkpoints/latest.pt
   └─ winner/model.pt
```

---

## Setup

```bash
pyenv local 3.11.9
python -m venv .venv && source .venv/bin/activate

# Minimal, pinned where stability matters
cat > requirements.txt <<'REQ'
numpy>=1.26,<3
pandas>=2.2,<3
pyarrow>=15,<20
pydantic>=2.5,<3
python-dotenv>=1.0,<2
typer>=0.12,<1
fastapi>=0.112,<1
uvicorn>=0.30,<1
jinja2>=3.1,<4
plotly>=5.22,<6
httpx>=0.27,<1
beautifulsoup4>=4.12,<5
torch>=2.2,<3
torchmetrics>=1.4,<2
tqdm>=4.66,<5
REQ

pip install -r requirements.txt

# Dev tooling (optional)
pip install black ruff mypy isort pytest
```

---

## Configuration & Secrets

### `.env`

Broker/API credentials and runtime toggles.

```dotenv
# Zerodha (example variable names)
ZERODHA_API_KEY=your_key
ZERODHA_API_SECRET=your_secret
ZERODHA_ACCESS_TOKEN=   # optional; auto-fetched/renewed if missing/invalid

# Misc
LOG_LEVEL=INFO          # overridden by CLI --verbose/--silent
TZ=Asia/Kolkata
```

> If `ZERODHA_ACCESS_TOKEN` is missing or invalid, the system will obtain a new one automatically during fetch/trade.

### `config.json` schema

Strict JSON Schema (Draft 2020-12):

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "RL Trading Config",
  "type": "object",
  "required": [
    "timezone","index","timeframes","lookbacks","from_dates",
    "safe_trading_start","safe_trading_end","hard_trading_end",
    "trailing_stop_r","capital_per_ticker","max_concurrent_trades",
    "paper_trading","holdout","actions","paths"
  ],
  "properties": {
    "timezone": { "type": "string" },
    "index": { "type": "string" },
    "tickers": { "type": "array", "items": { "type": "string" }, "default": [] },
    "timeframes": { "type": "array", "items": { "type": "string", "pattern": "^[0-9]+[smhd]$" } },
    "lookbacks": { "type": "object", "additionalProperties": { "type": "integer", "minimum": 8 } },
    "from_dates": { "type": "object", "additionalProperties": { "type": "string", "format": "date" } },
    "safe_trading_start": { "type": "string", "pattern": "^[0-2][0-9]:[0-5][0-9]:[0-5][0-9]$" },
    "safe_trading_end":   { "type": "string", "pattern": "^[0-2][0-9]:[0-5][0-9]:[0-5][0-9]$" },
    "hard_trading_end":   { "type": "string", "pattern": "^[0-2][0-9]:[0-5][0-9]:[0-5][0-9]$" },
    "trailing_stop_r": { "type": "number", "minimum": 0.1 },
    "capital_per_ticker": { "type": "number", "minimum": 1 },
    "max_concurrent_trades": { "type": "integer", "minimum": 1 },
    "paper_trading": { "type": "boolean" },
    "holdout": {
      "type": "object",
      "required": ["start","end"],
      "properties": {
        "start": { "type": "string", "format": "date" },
        "end": { "type": "string", "format": "date" }
      }
    },
    "actions": {
      "type": "array",
      "items": { "type": "string", "enum": ["BUY","SELL","HOLD"] },
      "minItems": 3, "maxItems": 3
    },
    "paths": {
      "type": "object",
      "required": ["data_raw","data_processed","checkpoints","winner"],
      "properties": {
        "data_raw": { "type": "string" },
        "data_processed": { "type": "string" },
        "checkpoints": { "type": "string" },
        "winner": { "type": "string" }
      }
    }
  }
}
```

### Example `config.json`

```json
{
  "timezone": "Asia/Kolkata",
  "index": "NIFTY50",
  "tickers": [],
  "timeframes": ["5m", "15m"],
  "lookbacks": { "5m": 256, "15m": 128 },
  "from_dates": { "5m": "2022-01-01", "15m": "2022-01-01" },
  "safe_trading_start": "09:20:00",
  "safe_trading_end": "15:20:00",
  "hard_trading_end": "15:29:30",
  "trailing_stop_r": 1.0,
  "capital_per_ticker": 10000.0,
  "max_concurrent_trades": 3,
  "paper_trading": true,
  "holdout": { "start": "2024-07-01", "end": "2024-09-30" },
  "actions": ["BUY","SELL","HOLD"],
  "paths": {
    "data_raw": "data/raw",
    "data_processed": "data/processed",
    "checkpoints": "models/checkpoints",
    "winner": "models/winner"
  }
}
```

---

## Data Contracts

### Raw candles — `data/raw/{TICKER}/{TF}.parquet`

Columns:

* `timestamp` (datetime64[ns, UTC]) — **monotonic increasing**, unique
* `open`, `high`, `low`, `close` (float64)
* `volume` (float64 or int64)
* Optional: `oi` (float64)

Constraints:

* `low ≤ min(open, close) ≤ max(open, close) ≤ high`
* No NaNs after **imputation**.

### Processed features — `data/processed/{TICKER}/{TF}.parquet`

Columns:

* `timestamp` (UTC)
* feature columns `f_*` (float32), standardized per timeframe
  (no targets here; rewards computed during replay backtracing)

### Replay Memory (in-RAM structure)

Transition: `(state, action, reward, next_state, done, meta)`

* `state`: dict `{tf: np.ndarray[L_tf, F_tf]}`
* `action`: int in `[0, A-1]`
* `reward`: float32
* `next_state`: same as `state`
* `done`: bool
* `meta`: `{ ticker, timestamp, price_entry, price_exit, side, r_multiple, mae_r, mfe, pct_mfe_captured, impl_shortfall_pct_r, plan_adherence_pct }`

---

## Historical Data Fetch (with NSE procurement)

**Source of tickers:** Constituents from the **NSE India website** for `config.index`.

**Procedure**

1. **Procure index constituents (NSE) & persist**

   * If `config.tickers` is empty → fetch the list for `config.index` from NSE, normalize symbols to broker format if needed, write back to `config.json`.
2. **Access token**

   * If `.env` has no Zerodha token or `is_token_valid()` returns `False` → obtain a fresh token and update `.env`.
3. **Incremental historical download**

   * For each `(ticker, timeframe)`:

     * Determine the last available `timestamp` on disk.
     * Fetch from `max(from_dates[tf], last+1bar)` to **now**.
     * Append, dedupe, sort by time.
4. **Imputation**

   * Forward-fill small gaps; interpolate where safe while preserving OHLC constraints. Save cleansed Parquet.

**Idempotency:** Re-running must not duplicate rows.

---

## Training

**Procedure**

1. **Create model** from the model class and **load checkpoint** if any.
2. **Load latest history & generate features** per timeframe.
3. **Populate training replay memory** for **all** periods **except** the holdout window.
4. **Backtrace rewards** in the training memory.
5. **Train** the model using the replay memory; save/overwrite `models/checkpoints/latest.pt`.
6. **Populate holdout replay memory** only for the holdout window.
7. **Backtrace rewards** in the holdout memory.
8. **Evaluate & promote** — if new model’s **average reward** on holdout > existing winner’s, promote new model to `models/winner/`.

**Default hyperparams**

* Optimizer: AdamW(lr=3e-4, weight_decay=1e-4)
* Loss: Smooth L1 (Huber)
* Discount `γ=0.99`
* DDQN target update: every 1,000 steps (hard copy)
* ε-greedy: 0.10→0.01 over 50k steps
* Batch: 512, Grad clip: 1.0
* Steps: 500k (configurable)

---

## Replay Memory Population

At each timestamp for each ticker:

a) If **current time < safe_trading_start** → continue (no trades).
b) Run **model inference** for current features; identify **max-confidence** ticker.
c) If **current time > safe_trading_end** and a trade is active for current ticker → **exit immediately**.
d) If **long** active and (inference=SELL **or** price < trailing stop) → **exit via SELL**.
e) If **short** active and (inference=BUY **or** price > trailing stop) → **exit via BUY**.
f) If **no trade**, **active_trades < max_concurrent_trades**, **current ticker is max-confidence**, and inference=BUY → **enter BUY** with quantity = `floor(capital_per_ticker × broker_leverage / price)`.
g) If **no trade**, same conditions, inference=SELL → **enter SELL** (short) with the same sizing rule.

Transitions `(s,a,r,s',done)` are recorded; rewards computed by backtracing (see [Rewards](#rewards--evaluation-metrics)).

---

## Live/Paper Trading

Loop until **today’s `hard_trading_end`**:

a) Fetch latest market info for `config.tickers`.
b) Generate input features for the **winner** model.
c) If **time < safe_trading_start** → loop back to (a).
d) Draw inference; identify **max-confidence** ticker.
e) If **time > safe_trading_end** and trade is active → **exit** (real/paper as per config).
f) If **long** active and (inference=SELL or price < trailing stop) → **exit**.
g) If **short** active and (inference=BUY or price > trailing stop) → **exit**.
h) If **no trade**, capacity available, ticker is **max-confidence**, inference=BUY → **enter BUY** → loop.
i) Else if same with inference=SELL → **enter SELL** → loop.

---

## Model Architecture

**Model:** *Dueling Double DQN* with **causal depthwise-separable TCN encoders**, time pooling, timeframe fusion via MLP, market set pooling + gating, and a dueling head.

**From `config.json`**

* `tickers`, `lookbacks { τ: L_τ }`, `actions`

**Runtime**

* `S` = number of tickers; `F` = detected per-bar features; `A` = number of actions.

Per timeframe (τ):

* Channels (C_\tau=\lceil\sqrt{L_\tau\cdot F}\rceil)
* Kernel (K_\tau=2\lfloor\sqrt{L_\tau}\rfloor+1)
* Depth (D_\tau) s.t. causal dilations 1..D cover (L_\tau)

**Layer stack**

* **Input**: (X_\tau\in\mathbb{R}^{B\times S\times L_\tau\times F}) → reshape ((B\cdot S)\times L_\tau\times F)
* **Encoder (per τ, shared)**: 1×1 Conv (F\to C_\tau) → (D_\tau) residual TCN blocks (depthwise causal Conv1D, pointwise Conv1D, SiLU, LayerNorm, residual)
* **Time pooling**: mean ⊕ max → (\mathbb{R}^{2C_\tau})
* **Fusion per stock**: concat across τ → (Z=\sum_\tau 2C_\tau) → MLP (Z\to M\to M) (SiLU) → reshape (B\times S\times M)
* **Market pooling**: mean⊕max over stocks → (2M) → Gate (2M\to 2M) (sigmoid) → product
* **Dueling head**: Value (2M\to M\to 1); Advantage (2M\to M\to A)
  Combine (Q=V+(\mathcal{A}-\text{mean}_a\mathcal{A}))

**Properties**

* Receptive field covers each (L_\tau); widths scale with (\sqrt{\cdot}).
* Parameter count independent of (S).

---

## Rewards & Evaluation Metrics

Per trade:

1. **R-multiple (Outcome)**
   [
   R=\frac{\text{Realized PnL}}{\text{Entry to Stop distance}\times\text{position size}}
   ]

2. **MAE in R (timing/stop quality)**
   [
   \text{MAE}_R=\frac{\text{Entry}-\text{Worst vs Stop}}{\text{Entry}-\text{Stop}}
   ]
   (adjust sign for long/short)

3. **% of MFE Captured (exit efficiency)**
   [
   %=\frac{\text{Realized PnL}}{\text{MFE}}\times 100%
   ]

4. **Implementation Shortfall (% of R)**
   [
   \text{IS}=\frac{\text{Planned PnL}-\text{Realized PnL}}{\text{Initial Risk}}\times 100%
   ]

5. **Plan Adherence Score**
   Checklist compliance: ((#\text{rules followed}/#\text{rules total})\times 100%).

---

## Per-file (per-function) Guide

> If your code uses different names, map one-to-one to these contracts.

### `.env`

* `ZERODHA_API_KEY`, `ZERODHA_API_SECRET`, `ZERODHA_ACCESS_TOKEN` (optional; auto-refresh)
* `LOG_LEVEL`, `TZ`

### `config.json`

* **index** *(string)*: e.g., `"NIFTY50"`.
* **tickers** *(array)*: if empty, **procured from NSE** and persisted.
* **timeframes**, **lookbacks**, **from_dates**.
* **safe_trading_start / safe_trading_end / hard_trading_end** *(HH:MM:SS in `timezone`)*.
* **trailing_stop_r**, **capital_per_ticker**, **max_concurrent_trades**, **paper_trading**.
* **holdout** period `{start,end}`.
* **actions**: `["BUY","SELL","HOLD"]`.
* **paths**: storage layout.

### `fetch.py`

```py
def load_config(path: str = "config.json") -> dict: ...
def persist_config(cfg: dict, path: str = "config.json") -> None: ...

def get_index_tickers_from_nse(index: str) -> list[str]:
    """
    Fetch index constituents from the official NSE website and return normalized symbols.
    """

def ensure_access_token() -> str: ...
def existing_range(ticker: str, tf: str) -> tuple[datetime|None, datetime|None]: ...
def fetch_history(ticker: str, tf: str, start: datetime, end: datetime) -> "pd.DataFrame": ...
def impute(df: "pd.DataFrame") -> "pd.DataFrame": ...
def save_history(df: "pd.DataFrame", ticker: str, tf: str) -> "pathlib.Path": ...
def main() -> None: ...
```

### `train.py`

```py
def build_model(cfg: dict) -> "torch.nn.Module": ...
def load_checkpoint(model: "torch.nn.Module", path: str) -> None: ...
def load_features(cfg: dict) -> dict[str, dict[str, "np.ndarray"]]: ...
def populate_replay(cfg: dict, holdout: bool) -> "ReplayMemory": ...
def backtrace_rewards(memory: "ReplayMemory", cfg: dict) -> None: ...
def train_ddqn(model: "torch.nn.Module", memory: "ReplayMemory", cfg: dict) -> dict: ...
def evaluate(model: "torch.nn.Module", holdout: "ReplayMemory") -> dict[str, float]: ...
def maybe_promote_winner(metrics: dict, ckpt_path: str, winner_dir: str) -> bool: ...
def main() -> None: ...
```

### `trade.py`

```py
def load_winner(path: str) -> "InferenceEngine": ...
def fetch_realtime_quotes(tickers: list[str]) -> "pd.DataFrame": ...
def build_live_features(quotes: "pd.DataFrame", cache: dict) -> dict[str, "np.ndarray"]: ...
def infer(engine: "InferenceEngine", features: dict) -> dict[str, tuple[str, float]]: ...
def portfolio_state() -> "Positions": ...
def should_exit_long(signal: str, price: float, trailing_stop_r: float, pos: "Position") -> bool: ...
def should_exit_short(signal: str, price: float, trailing_stop_r: float, pos: "Position") -> bool: ...
def can_enter(is_max_conf: bool, signal: str, active_count: int, cfg: dict) -> bool: ...
def enter_trade(ticker: str, side: str, qty: int, mode: str) -> None: ...
def exit_trade(ticker: str, mode: str) -> None: ...
def loop_until(end_time: datetime) -> None: ...
def main() -> None: ...
```

### `dashboard.py`

* FastAPI app with routes:

  * `GET /` — overview: equity curve, open positions, recent trades
  * `GET /trades` — table with R, MAE_R, %MFE, IS, adherence
  * `GET /metrics` — training/holdout summary, winner info
  * `GET /logs` — (optional) SSE log stream
* `serve(app, host, port)`

### `broker.py`

```py
class BrokerError(Exception): ...

class ZerodhaClient:
    def authenticate(self) -> None: ...
    def is_token_valid(self) -> bool: ...
    def refresh_token(self) -> str: ...
    def historical(self, ticker: str, tf: str, start: datetime, end: datetime) -> "pd.DataFrame": ...
    def quotes(self, tickers: list[str]) -> "pd.DataFrame": ...
    def leverage(self, ticker: str) -> float: ...
    def place_order(self, ticker: str, side: str, qty: int, **kwargs) -> str: ...
    def exit_order(self, position: "Position") -> None: ...
```

### `ml.py`

```py
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

### `data.py`

```py
def read_history(ticker: str, tf: str) -> "pd.DataFrame": ...
def write_history(df: "pd.DataFrame", ticker: str, tf: str) -> None: ...
def resample_align(df: "pd.DataFrame", tf: str, tz: str) -> "pd.DataFrame": ...
def impute(df: "pd.DataFrame") -> "pd.DataFrame": ...
def feature_pipeline(df: "pd.DataFrame", cfg: dict) -> "np.ndarray": ...
```

---

## CLI Spec

### Global CLI Conventions (all entrypoints)

* `--verbose` sets log level to **DEBUG**.
* `--silent` sets log level to **WARNING** (errors still shown).
* Default is **INFO**.
* `--verbose` and `--silent` are **mutually exclusive** (if both are present, the **last one wins**).

Two ways to target specific work:

1. **Stage selectors** via `--stage <name>` (can repeat; runs in order).
2. **Function callers** via `--fn <name>` (can repeat; runs whitelisted zero-arg wrappers in order).

> **Note on `--fn`**: Each script exposes a **safe list** of callable wrapper names (documented below). These wrappers use current config and sensible defaults so they require no additional parameters.

---

### `fetch.py`

```bash
python fetch.py [--verbose|--silent]
               [--index TEXT]
               [--from-date YYYY-MM-DD]
               [--force-refresh-token/--no-force-refresh-token]
               [--max-workers INT]
               [--stage index|token|history|impute|all]...
               [--fn load_config|persist_config|nse_tickers|
                     ensure_token|incremental_history|run_imputation]...
```

**Stages**

* `index`   — procure tickers **from NSE** and persist to `config.json`.
* `token`   — ensure/refresh Zerodha access token.
* `history` — incremental candle download (uses `timeframes`, `from_dates`).
* `impute`  — run imputation/cleanup only.
* `all`     — default pipeline: `index → token → history → impute`.

**`--fn` whitelist**

* `load_config` — prints validated config summary.
* `persist_config` — writes current config snapshot to disk.
* `nse_tickers` — fetch & print constituents for `config.index` (does not persist).
* `ensure_token` — refresh token if invalid and print token tail.
* `incremental_history` — run one pass of incremental fetch.
* `run_imputation` — re-impute all raw files and save.

---

### `train.py`

```bash
python train.py [--verbose|--silent]
               [--steps INT] [--batch-size INT] [--lr FLOAT] [--gamma FLOAT]
               [--checkpoint-path PATH]
               [--stage build|features|populate-train|backtrace-train|train|
                       populate-holdout|backtrace-holdout|evaluate|promote|all]...
               [--fn build_model|load_ckpt|make_features|
                     make_train_replay|backtrace_train|fit|
                     make_holdout_replay|backtrace_holdout|eval|promote]...
```

**Stages (default `all` runs in this order)**

* `build` → `features` → `populate-train` → `backtrace-train` → `train`
  → `populate-holdout` → `backtrace-holdout` → `evaluate` → `promote`

**`--fn` whitelist**

* `build_model`, `load_ckpt`, `make_features`, `make_train_replay`, `backtrace_train`, `fit`,
  `make_holdout_replay`, `backtrace_holdout`, `eval`, `promote`.

---

### `trade.py`

```bash
python trade.py [--verbose|--silent]
               [--paper|--real]
               [--loop-ms INT]
               [--end HH:MM:SS]
               [--stage quotes|features|infer|loop|all]...
               [--fn load_winner|one_shot_quotes|one_shot_features|one_shot_infer|loop]...
```

**Stages**

* `quotes`   — one-shot: fetch latest quotes and print sample rows.
* `features` — one-shot: build features snapshot for winner model.
* `infer`    — one-shot: run inference and print (ticker, action, confidence).
* `loop`     — full intraday loop until `hard_trading_end` (default).
* `all`      — alias of `loop`.

**`--fn` whitelist**

* `load_winner`, `one_shot_quotes`, `one_shot_features`, `one_shot_infer`, `loop`.

---

### `dashboard.py`

```bash
python dashboard.py [--verbose|--silent]
                   [--host TEXT] [--port INT]
                   [--stage serve|dump-metrics|all]
                   [--fn serve|dump_metrics]...
```

**Stages**

* `serve`        — start FastAPI app (default).
* `dump-metrics` — print a JSON block of key metrics to stdout and exit.
* `all`          — alias of `serve`.

**`--fn` whitelist**

* `serve`, `dump_metrics`.

---

## Logging, Errors, and Idempotency

* Logger name = module path (e.g., `ml.trainer`).
* Levels: **INFO** default; **DEBUG** with `--verbose` (tensor shapes/row counts/HTTP timings); **WARNING** with `--silent`.
* `fetch.py` is **idempotent**: de-duplicates by `timestamp`, sorts, writes Parquet.
* `train.py` overwrites `models/checkpoints/latest.pt`; promotion replaces `models/winner/model.pt`.
* All broker/network exceptions → `BrokerError` with context.

---

## Testing & Acceptance Criteria

**Unit tests** (`pytest`)

* `test_config_schema.py` — validate example `config.json`.
* `test_nse_parser.py` — parse saved NSE fixture → correct symbols normalized.
* `test_impute.py` — OHLCV imputation preserves constraints.
* `test_replay_rules.py` — transitions follow rules a–g on synthetic data.
* `test_model_shapes.py` — forward pass shapes across TFs and S tickers.
* `test_promotion.py` — promotion iff holdout avg reward strictly improves.
* `test_position_sizing.py` — quantity = `floor(capital_per_ticker * leverage / price)`.

**CLI flags**

* `--verbose` emits DEBUG logs; `--silent` suppresses INFO.
* Stage selectors run only the requested stages in order.
* `--fn` calls listed zero-arg wrappers and exits successfully.

**End-to-end smoke**

1. Seed `config.json` with 2 tickers & tiny lookbacks.
2. `python fetch.py --stage index --stage history --stage impute`
3. `python train.py --steps 5000 --stage build --stage features --stage populate-train --stage backtrace-train --stage train`
4. `python train.py --stage populate-holdout --stage backtrace-holdout --stage evaluate --stage promote`
5. `python trade.py --paper --stage infer --end 09:25:00`
6. `python dashboard.py --stage dump-metrics` returns JSON.

**Done definition**: All tests pass; scripts run without unhandled exceptions using the example config.

---

## Operational Notes

* **Timezone** — all config times are in `config.timezone` (default `Asia/Kolkata`); convert to UTC internally.
* **Sizing** — quantity = `floor(capital_per_ticker × leverage / price)` (respect broker lot/step size; assume step=1 if unknown).
* **Trailing Stop in R** — unrealized drawdown from peak must not exceed `trailing_stop_r × initial risk`.
* **Persistence** — checkpoints overwrite by design; winner is separate.
* **HTTP etiquette (NSE)** — set an explicit `User-Agent`, random short jitter between requests, and exponential backoff on 429/5xx.

---

## Safety & Disclaimers

This code is for **research and education**. Markets involve risk; past performance does not guarantee future results. If you disable `paper_trading`, you may place **real orders**; ensure compliance with your broker/exchange rules.

---

## License

Choose a license (MIT/Apache-2.0/BSD-3-Clause) and add it here.

---

## Appendix: Pseudocode

**`fetch.py:main`**

```
cfg = load_config()
if not cfg["tickers"]:
    # 1) Procure from NSE
    symbols = get_index_tickers_from_nse(cfg["index"])
    cfg["tickers"] = symbols
    persist_config(cfg)

# 2) Access token
ensure_access_token()

# 3) Incremental history
for each ticker in cfg.tickers:
  for each tf in cfg.timeframes:
    last_end = existing_range(ticker, tf).end
    start = max(from_dates[tf], last_end + tf_step if last_end else from_dates[tf])
    if start <= now:
        df = fetch_history(ticker, tf, start, now)
        df = impute(df)
        save_history(df, ticker, tf)
```

**`train.py:main`**

```
model = build_model(cfg)
load_checkpoint(model, checkpoints/latest.pt if exists)
features = load_features(cfg)

train_mem = populate_replay(cfg, holdout=False)
backtrace_rewards(train_mem, cfg)
train_ddqn(model, train_mem, cfg)
save checkpoints/latest.pt

holdout_mem = populate_replay(cfg, holdout=True)
backtrace_rewards(holdout_mem, cfg)
metrics = evaluate(model, holdout_mem)
maybe_promote_winner(metrics, checkpoints/latest.pt, winner/)
```

**`trade.py:loop_until(end)`**

```
while now < hard_trading_end:
  quotes = fetch_realtime_quotes(cfg.tickers)
  feats = build_live_features(quotes, cache)
  preds = infer(winner_model, feats)
  max_conf_ticker = argmax_confidence(preds)

  if now < safe_trading_start: continue

  # exit rules
  if time > safe_trading_end and active_trade_for(max_conf_ticker):
      exit_trade(max_conf_ticker, mode); continue
  if long_active and (signal==SELL or price < trailing_stop): exit_trade(...); continue
  if short_active and (signal==BUY  or price > trailing_stop): exit_trade(...); continue

  # entry rules
  if no_trade and active_trades < max_concurrent and ticker == max_conf_ticker:
      if signal==BUY:  enter_trade(...); continue
      if signal==SELL: enter_trade(...); continue

  sleep(loop_ms)
```

**Rewards (formulas)**

* (R=\frac{\text{Realized PnL}}{\text{Entry–Stop}\times\text{size}})
* (\mathrm{MAE}_R=\frac{\text{Entry}-\text{Worst vs Stop}}{\text{Entry}-\text{Stop}})
* (%\mathrm{MFE}=\frac{\text{Realized PnL}}{\text{MFE}}\times100%)
* (\mathrm{IS}=\frac{\text{Planned PnL}-\text{Realized PnL}}{\text{Initial Risk}}\times100%)
* Plan adherence: checklist compliance %.

---

**Happy building!**
