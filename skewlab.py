"""skewlab.py — the script you run.
================================

Launches the Moontower-style skew/vol dashboard. Edit the INPUTS block, then:

    python skewlab.py                 # opens the browser dashboard
    SKEWLAB_DEMO=1 python skewlab.py  # force the synthetic offline backend
or, in a Jupyter / VS Code interactive window:
    %run skewlab.py                   # same, and keeps `snap` around to poke at

DATA BACKEND
------------
By default skewlab looks for a private production pipeline (`CapriciousVolTamer`, which
talks to a ThetaData terminal + a local trade ledger). That pipeline is NOT part of this
repo. When it isn't importable — as in this public repo — skewlab automatically falls back
to `skewlab.pipeline.demo`: a fully synthetic, reproducible, offline backend so the whole
dashboard runs with no network, no terminal and no credentials. Set `SKEWLAB_DEMO=1` to
force it even if a production pipeline is present.

POSITIONS
---------
A leg is `(strike, "P"/"C", contracts)` — positive = long, negative = short. Fill the book
manually below, or pull OPEN legs from the (demo or real) trade ledger. With no book and no
shares the Position & P&L panels are simply hidden.
"""
from __future__ import annotations

from skewlab.config import RunConfig
from skewlab.run import main, get_pipeline
from skewlab.positions import add_position, remove_position

cvt, opd = get_pipeline()          # production pipeline if present, else synthetic demo

# =====================================================================================
# INPUTS
# =====================================================================================
symbol   = "SPY"
date     = None                    # None -> latest session
prev_date = None                   # explicit previous-obs date, or None to pin via lookback

# --- skew curve model: "svi" (arbitrage-aware, default) or "poly" -------------------
SKEW_MODEL  = "svi"
POLY_DEGREE = 5

cfg = RunConfig(
    symbol=symbol,
    date=date,
    prev_date=prev_date,
    target_dte=30,
    lookback_days=30,              # previous-day overlay ~1 month back (pins same expiry)
    pin_same_expiry=True,
    pin_strict=False,
    monthly_only=False,

    use_intraday=False,            # settled EOD chain (the demo backend is EOD-only)

    use_iv_history=True,           # historical IV panels -> regime + RV-vs-IV section
    iv_hist_start=None,
    show_term_curves=True,
    show_vix_panels=True,

    skew_model=SKEW_MODEL,
    poly_degree=POLY_DEGREE,
    open_in_browser=True,
)

# =====================================================================================
# POSITIONS — pull OPEN legs from the trade ledger, or type a book in by hand
# =====================================================================================
USE_LEDGER          = True         # pull OPEN legs for `symbol` from opd.fetch_trades_ledger()
LEDGER_MATCH_EXPIRY = False        # True = only legs on the analysed expiry; False = all open
LEDGER_ADD_SHARES   = False
LEDGER_REPLACE      = True

# Manual book (used when USE_LEDGER is False, or merged under the ledger if not REPLACE):
POSITIONS = None
SHARES    = None
# POSITIONS = [(555, "P", -1), (650, "C", -1)]   # e.g. a short strangle
# SHARES = 0

if __name__ == "__main__":
    cfg.auto_positions              = USE_LEDGER
    cfg.auto_positions_match_expiry = LEDGER_MATCH_EXPIRY
    cfg.auto_positions_shares       = LEDGER_ADD_SHARES
    cfg.auto_positions_replace      = LEDGER_REPLACE
    cfg.positions, cfg.shares = POSITIONS, (SHARES or 0)

    snap = main(cfg)               # builds the snapshot and serves the dashboard

    # After the run, `snap` holds the whole immutable context. Inspect any dataframe with:
    #     from skewlab.inspect import collect_run_data, describe_run_data
    #     DATA = collect_run_data(snap); describe_run_data(DATA)
    #     DATA["rv_compare"]   # RV-implied fair value vs market (see the dashboard panel)
