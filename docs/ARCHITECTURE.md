# Architecture

skewlab is a layered rebuild of a monolithic vol dashboard into a pure quant core plus a thin,
injectable I/O boundary. The guiding rule: **the maths never touches the network, and the
charts never touch a data vendor.**

## Layers

```
skewlab/
  config.py    RunConfig dataclass + scenario presets — every knob, no side effects.
  model.py     PURE math. Black-Scholes price/greeks, SVI fit (+ Durrleman butterfly g),
               wing extrapolation, Breeden-Litzenberger density, butterfly/calendar arb,
               distribution stats, delta-target inversion, realized-vol lookback map,
               the RV-implied ATMF straddle, and the net-liquidity projection.
  data.py      I/O. fetch_snapshot(cfg, cvt, opd) does ALL fetching once and returns an
               immutable Snapshot; CurveState holds the mutable slider knobs. Source helpers
               handle forward/ATF identification (robust to thin chains), the previous-day
               overlay, term-structure curves, IV-history panels, and the RV-vs-IV fields.
  analysis.py  metrics(snap, cs) computes every number once; render_text / render_html
               produce the plain-text and Dash-card narratives.
  charts/      one module per chart, each a pure make(snap, cs) -> Figure, plus a registry
               of Chart(key, title, make, needs, reacts) records and active(snap).
  app.py       the Dash app, built generically by iterating the chart registry, with sliders
               per SD node, scenario presets, and the net-liquidity calculator panel.
  pipeline/    the data-source boundary (see below).
  run.py       entry point: build config -> fetch snapshot -> serve dashboard.
  inspect.py   collect_run_data(snap) -> {name: DataFrame} for interactive inspection.
  positions.py pure book helpers + trade-ledger -> position-book adapters.
```

## Key design decisions

- **Immutable `Snapshot` + mutable `CurveState`.** All slow/stateful work happens once in
  `fetch_snapshot`; charts are pure functions of `(snapshot, curve_state)`. No globals.
- **Injected data backend.** `fetch_snapshot(cfg, cvt, opd)` receives its data sources; it
  never imports a vendor. This is the seam that makes the demo/production split possible.
- **Registry-driven UI.** Adding a chart means writing a `make(snap, cs)` and registering it;
  the app wires controls and callbacks generically. Charts declare `needs(snap)` (is this
  relevant?) and `reacts` (does it change with the sliders?).
- **Arbitrage-aware by default.** SVI is the default smile; the fitted curve is checked for
  butterfly and calendar arbitrage and the result is surfaced in the analysis card.
- **Analytic greeks everywhere** (no bump-and-reprice).

## The data-source boundary

`skewlab.data` depends only on the small surface below. Anything implementing it can drive the
dashboard.

`cvt` (chains + realized vol):
- `get_quick_option_chain(symbol, date, prev_date, target_dte, size, verbose) -> DataFrame`
  indexed by strike, with columns `S, R, Q, dte, T, implied_vol, iv_call, iv_put, straddle,
  mid_call, mid_put, midpoint, expiration`.
- `get_composite_realised_volatility(symbol, lookback, start, end, ...) -> DataFrame` with a
  `Mean` column indexed by date.

`opd` (calendar / prices / panels / ledger):
- `trading_dates`, `last_trading_date`, `second_last_trading_date`, `ny_timezone`
- `get_ohlcv_from_symbol(symbol) -> OHLCV DataFrame`
- `close_tickers -> DataFrame` (columns include `^VIX`, `^VVIX`)
- `build_iv_panels(symbol, start, end, target_dte, ...) -> (iv_atm Series, iv_history DataFrame)`
- `fetch_trades_ledger()` / `trades_dict` / `print_trade_list(...)`

Two implementations live behind this boundary:

- **Production** — the private `CapriciousVolTamer` pipeline (a ThetaData terminal for
  historical option greeks, yfinance for intraday snapshots, and a local Excel trade ledger).
  Not shipped in this repo.
- **Demo** — `skewlab/pipeline/demo.py`, a self-contained numpy/pandas backend that generates
  a reproducible synthetic world per symbol (GBM price path, put-skewed smile, rolling
  realized vol, IV-history panels, VIX/VVIX series, a small trade ledger). No network, no
  terminal, no credentials — so the dashboard, tests and CI all run offline.

`skewlab.run.get_pipeline()` selects the production backend when importable and falls back to
the demo otherwise (or when `SKEWLAB_DEMO=1`).
