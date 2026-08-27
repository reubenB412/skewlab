"""skewlab.py — the script you run.
================================

Launches the skew/vol dashboard on deterministic synthetic data. Edit the inputs, then:

    python skewlab.py                 # opens the browser dashboard
or, in a Jupyter / VS Code interactive window:
    %run skewlab.py                   # same, and keeps `snap` around to poke at

POSITIONS
---------
A leg is `(strike, "P"/"C", contracts)` — positive = long, negative = short. Fill the
manual book below. With no book and no shares the Position & P&L panels are hidden.
"""
from __future__ import annotations

from skewlab.config import RunConfig
from skewlab.run import main

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
    target_dte=23,                 # ~3-week expiry (matches the reference SPY snapshot)
    lookback_days=30,              # previous-day overlay ~1 month back (pins same expiry)
    pin_same_expiry=True,
    pin_strict=False,
    monthly_only=False,

    use_intraday=False,            # settled EOD chain (the demo backend is EOD-only)

    use_iv_history=True,           # historical IV panels -> regime + RV-vs-IV section
    iv_hist_target_dte=30,         # stable history horizon, separate from today's chosen expiry
    iv_hist_start=None,
    show_term_curves=True,
    show_rv_term_structure=True,   # HF RV regime + estimator/IV term structure
    show_vix_panels=True,

    skew_model=SKEW_MODEL,
    poly_degree=POLY_DEGREE,
    open_in_browser=True,
)

# =====================================================================================
# OPTIONAL MANUAL POSITIONS
# =====================================================================================
POSITIONS = None
SHARES    = None
# POSITIONS = [(555, "P", -1), (650, "C", -1)]   # e.g. a short strangle
# SHARES = 0

if __name__ == "__main__":
    cfg.positions, cfg.shares = POSITIONS, (SHARES or 0)
    snap = main(cfg)

    # After the run, `snap` holds the whole immutable context. Inspect any dataframe with:
    #     from skewlab.inspect import collect_run_data, describe_run_data
    #     DATA = collect_run_data(snap); describe_run_data(DATA)
    #     DATA["rv_term_table"]   # current 5..180-session RV estimator table
