# skewlab

**An arbitrage-aware option skew & volatility dashboard** — fits the implied smile
with SVI, recovers the risk-neutral density, checks it for arbitrage, compares implied vol
to a composite realized-vol estimate, and renders it all in an interactive Dash app.

[![CI](https://github.com/reubenB412/skewlab/actions/workflows/ci.yml/badge.svg)](https://github.com/reubenB412/skewlab/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

> **Runs offline out of the box.** With no market-data credentials, skewlab falls back to a
> fully synthetic, reproducible backend, so you can `git clone` and launch the whole
> dashboard in one command. See [Data backends](#data-backends).

![SPY skew curve — offline demo run](docs/skew_curve_demo.png)

*Skew panel from the synthetic offline demo (`python skewlab.py`, no credentials): a raw-SVI
fit with a steep put wing and ATF ≈ 13%, plus previous-day and term-structure overlays.*

![SPY implied risk-neutral density (Breeden–Litzenberger)](docs/implied_density_demo.png)

*Risk-neutral density from the fitted SVI smile via Breeden–Litzenberger, vs a flat log-normal
sheet — left-skewed, mode above the forward, fatter downside tail. Non-negative everywhere, so
the smile passes the butterfly no-arbitrage check.*

![SPY implied-vol history vs composite realized](docs/vol_history_demo.png)

*Implied-vol history buckets (ATM, 25Δ/10Δ) against the composite realized-vol estimate — the
implied-vs-realized comparison behind the variance-risk-premium read.*

![SPY realized-vol estimator stack](docs/rv_estimator_stack_demo.png)

*The composite realized-vol estimator stack — C-C, Parkinson, Hodges–Tompkins, Yang–Zhang,
EWMA half-life, GARCH — with the blended Mean, the RV input to the fair-value / VRP reads.*

![SPY IV history & regime](docs/iv_history_regime_demo.png)

*The 2×2 regime panel: ATM implied vs realized (carry / VRP), the ATM-vol regime and its
percentile, 25Δ/10Δ risk-reversal over time, and today's vol-vs-skew position vs its own history.*

---

## What it's for

skewlab is the decision-support cockpit for running a **discretionary options book**. It works
on essentially any symbol with a listed option chain — across equities and index ETFs,
commodities, and bonds/rates — and works best on the most liquid names, where the smile and
greeks are clean (thin or after-hours chains degrade gracefully to a flatter fit). SPY is
simply the running example in the demo. Each session it turns the day's option chain into the handful of reads a
discretionary trader actually acts on: whether
skew is rich or cheap (the SVI smile and 25Δ risk-reversal versus their own history), whether
implied vol is rich or cheap against realized (the composite-RV fair straddle and the
variance-risk premium, now and at the day's open), what the market is actually pricing (the
risk-neutral distribution — mode, skew, tails — and whether the smile is even
arbitrage-consistent), and what the book itself is doing (live greeks and a realized-vol /
vega / delta P&L decomposition against the implied density). The aim is to anchor discretionary
position decisions to one reproducible surface each day instead of a spreadsheet-and-eyeball
workflow.

## What it does

`skewlab` takes an option chain for one expiry and turns it into a decision-support surface:

- **Skew curve (SVI).** Fits Gatheral raw-SVI in log-moneyness — linear wings, a single
  smooth minimum, and a closed-form Durrleman butterfly test. A polynomial fit is available
  as a legacy fallback.
- **Implied distribution (Breeden–Litzenberger).** Recovers the risk-neutral density from
  the fitted call curve and reports its mean/median/mode/std/skew/kurtosis vs a flat
  log-normal sheet.
- **No-arbitrage checks.** Flags negative butterfly density and calendar-spread violations
  directly on the fitted smile.
- **RV vs IV (variance-risk premium).** Turns the most-recent-close **composite realized
  vol** into a *fair* ATM-forward straddle and vol, then compares it to the market now and
  at the day's open — a clean read on how rich/cheap implied is vs realized.
- **Regime context.** Percentile-ranks today's ATM vol and 25Δ risk-reversal against a
  rolling history; overlays VIX/VVIX empirical distributions and a VVIX/VIX convexity ratio.
- **Vol history (IV vs realized).** Plots the implied-vol history buckets (ATM, 25Δ/10Δ
  put+call) against the composite realized-vol Mean, plus the realized-vol estimator stack,
  over an adjustable start date.
- **Position analytics.** Optional book (manual or from a trade ledger) with analytic
  greeks, a P&L decomposition (realized-vol / vega / delta), and payoff context.

Everything is wrapped in a Dash dashboard with live sliders per standard-deviation node,
scenario presets, and a data-inspection layer that exposes every intermediate DataFrame.

## Quickstart

```bash
git clone https://github.com/reubenB412/skewlab.git
cd skewlab
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt                         # or: pip install -e ".[dev]"

python skewlab.py                                       # opens http://127.0.0.1:8050
```

With no production data pipeline present, this launches on **synthetic offline data**. To be
explicit:

```bash
SKEWLAB_DEMO=1 python skewlab.py
```

Edit the `INPUTS` block at the top of [`skewlab.py`](skewlab.py) to change the symbol,
target DTE, skew model, or position book.

## Data backends

skewlab's I/O layer never imports a data vendor directly — it receives two injected objects,
`cvt` (option chains + composite realized vol) and `opd` (calendar, OHLCV, IV-history panels,
VIX/VVIX, trade ledger). Two backends implement that small interface:

| Backend | Source | Use |
|---|---|---|
| **Production** | private `CapriciousVolTamer` pipeline — **ThetaData** (settled option chains + greeks), **yfinance** (underlying history + the intraday snapshot), local trade ledger | live use; **not included** in this repo |
| **Demo** | [`skewlab/pipeline/demo.py`](skewlab/pipeline/demo.py) — reproducible synthetic chains, RV, IV panels, VIX/VVIX, calendar | offline demo, tests, CI |

In live use, settled option chains and option greeks come from **ThetaData**, a professional
options-data feed; the underlying's price history and the intraday ("today") snapshot come
from **yfinance**; and the position book is read from a local trade ledger. The demo backend
fabricates all of this synthetically so nothing is required to run it.

`skewlab.run.get_pipeline()` uses the production backend when importable and falls back to the
demo one otherwise. This clean seam is what lets the project be public while the proprietary
data plumbing stays private.

## Architecture

A layered package with a pure quant core and an injected I/O boundary:

```
skewlab/
  config.py      RunConfig dataclass — every knob, no side effects
  model.py       PURE math: Black-Scholes/greeks, SVI, Breeden-Litzenberger, no-arb, stats
  data.py        I/O: fetch_snapshot(cfg, cvt, opd) -> immutable Snapshot (+ CurveState)
  analysis.py    metrics(snap, cs) + text / HTML narrative
  charts/        one pure make(snap, cs) -> Figure per chart, + a registry
  app.py         Dash app built generically from the chart registry
  pipeline/      the data-source boundary: demo.py (synthetic) | production (private)
  run.py         entry point: config -> snapshot -> serve
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design decisions and
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the maths (SVI, Breeden–Litzenberger,
the RV-vs-IV fair-value calc, and the realized-vol estimator stack).

## Tests

```bash
pip install -e ".[dev]"
pytest                     # put-call parity, BL density ~1, SVI no-arb, demo smoke
ruff check skewlab
```

CI runs the suite on Python 3.10–3.12 against the offline demo backend.

## Roadmap

- Honor non-reacting charts on slider Apply (interactive-latency win)
- Vectorize the delta scans
- Position payoff overlaid on the implied density + risk-neutral E[P&L] / probability of profit
- Vanna/volga in the position panel; a dedicated ATM term-structure (vol-vs-DTE) chart

## Disclaimer

For research and educational purposes only. Nothing here is investment advice. Synthetic demo
data is not market data.
