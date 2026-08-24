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
               and the RV-implied ATMF straddle.
  rv.py        PURE realised-variance recovery, annualisation rebasing, rolling aggregation,
               no-lookahead percentiles, regime labels, and RV/IV tables.
  data.py      I/O. fetch_snapshot(cfg, cvt, opd) does ALL fetching once and returns an
               immutable Snapshot; CurveState holds the mutable slider knobs. Source helpers
               handle forward/ATF identification (robust to thin chains), the previous-day
               overlay, term curves, IV history, RV-vs-IV fields, and RVTermState assembly.
  analysis.py  metrics(snap, cs) computes every number once; render_text / render_html
               produce the plain-text and Dash-card narratives.
  charts/      one module per chart, each a pure make(snap, cs) -> Figure, plus a registry
               of Chart(key, title, make, needs, reacts) records and active(snap).
  app.py       the Dash app, built generically by iterating the chart registry, with sliders
               per SD node and scenario presets.
  pipeline/    the data-source boundary (see below).
  run.py       entry point: build config -> fetch snapshot -> serve dashboard.
  inspect.py   structured inspection plus a bounded Markdown/CSV LLM-context export.
  positions.py pure helpers for an optional manually supplied position book.
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
- **Explicit variance clocks.** Backward RV windows count completed trading sessions and use
  the configured RV annualisation basis. Forward IV uses actual calendar DTE and calendar/365
  integrated variance. The chart labels both rather than pretending they are the same clock.
- **Partial-session safety.** Daily high-frequency variance carries an explicit completion flag;
  rolling curves discard incomplete sessions before any window is formed.
- **As-of-safe forward IV.** Every auxiliary ATM-IV tenor retains its requested tenor, actual DTE,
  expiry, observation date, alignment flag, calendar year fraction, and integrated variance.
  Misaligned rows remain available for diagnostics but their IV/variance values are cleared and
  omitted from the chart. The main snapshot supplies a trusted aligned anchor/fallback.
- **Local display semantics.** The public Dash app asks Pandas Styler for the same row-wise
  `YlGnBu` cell context as private `opd._format_display(axis=1)`, then places the computed colours
  directly on semantic `html.Td` cells. It imports neither OPD nor emitted Styler HTML/CSS.

## The data-source boundary

`skewlab.data` depends only on the small surface below. Anything implementing it can drive the
dashboard.

`cvt` (chains + realized vol):
- `get_quick_option_chain(symbol, date, prev_date, target_dte, size, verbose) -> DataFrame`
  indexed by strike, with columns `S, R, Q, dte, T, implied_vol, iv_call, iv_put, straddle,
  mid_call, mid_put, midpoint, expiration`.
- `get_composite_realised_volatility(symbol, lookback, start, end, ...) -> DataFrame` with a
  `Mean` column indexed by date.
- `get_rv_term_source(...) -> dict` with annualized RV columns plus completion flags and
  timestamps. `rv.py` converts those vols back to daily variance before rebasing/aggregation.

`opd` (calendar / prices / panels):
- `trading_dates`, `last_trading_date`, `second_last_trading_date`, `ny_timezone`
- `get_ohlcv_from_symbol(symbol) -> OHLCV DataFrame`
- `close_tickers -> DataFrame` (columns include `^VIX`, `^VVIX`)
- `build_iv_panels(symbol, start, end, target_dte, ...) -> (iv_atm Series, iv_history DataFrame)`

The public launcher selects only `skewlab/pipeline/demo.py`, a self-contained numpy/pandas
adapter that produces reproducible option chains, price paths, RV, IV-history panels, and
VIX/VVIX series. No network or credentials are used. The injection seam is retained so the
maths and charts can be tested independently of data generation.
