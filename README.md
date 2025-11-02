# AlgoTrading
End-to-end RL trading system for equities: fetches NSE index constituents, ingests and imputes incremental OHLCV history, engineers features, trains a Dueling DDQN with TCN encoders, evaluates on a holdout and promotes winner, then runs a safe-window intraday trading loop (paper/real) with trailing stop and a monitoring dashboard. FastAPI UI &amp; logs
