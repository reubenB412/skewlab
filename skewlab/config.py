"""skewlab.config — all the knobs in one place (no globals, no side effects).

A `RunConfig` is a plain dataclass you instantiate and pass to `data.fetch_snapshot`.
Everything that used to be a module-level input in moontower_skew_v8.py lives here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# --- market-regime scenario presets (relative to today's ATF vol) -------------------
# atf_mult scales the level; skew is the %-of-ATF premium/discount at each SD node
# [-3,-2,-1,0,+1,+2,+3]; sl/sr are wing slopes (vol pts per SD).
SCENARIOS = {
    "market": None,  # live / use the sliders
    "Calm bull (low vol, gentle put skew)":
        dict(atf_mult=0.80, skew=[0.30, 0.18, 0.08, 0.0, -0.05, -0.06, -0.04], sl=3, sr=2),
    "Risk-off / crash (vol up, steep put skew)":
        dict(atf_mult=1.70, skew=[0.75, 0.48, 0.22, 0.0, -0.08, -0.08, -0.03], sl=9, sr=3),
    "Vol crush (post-event, flat & low)":
        dict(atf_mult=0.68, skew=[0.14, 0.09, 0.03, 0.0, -0.02, -0.01, 0.0], sl=2, sr=2),
    "Squeeze / bubble (call skew, positive skew)":
        dict(atf_mult=1.35, skew=[0.06, 0.0, -0.05, 0.0, 0.16, 0.42, 0.72], sl=2, sr=9),
    "Event smile (both wings bid, fat tails)":
        dict(atf_mult=1.15, skew=[0.45, 0.30, 0.12, 0.0, 0.12, 0.30, 0.45], sl=6, sr=6),
}


@dataclass
class RunConfig:
    # --- what you normally touch ---
    symbol: str = "SPY"
    date: str | None = None              # None -> today; pipeline rolls back to last settled session
    prev_date: str | None = None         # explicit previous obs date, or None

    # --- expiry / pin behaviour ---
    target_dte: int = 30
    pin_same_expiry: bool = True
    lookback_days: int = 30              # used when prev_date is None and pin_same_expiry
    pin_strict: bool = True             # drop the overlay if the pinned prev expiry != today's
    monthly_only: bool = True           # snap to 3rd-Friday monthly nearest target_dte

    # --- live data ---
    use_intraday: bool = False           # live yfinance "today" instead of settled EOD
    reset_intraday: bool = False         # force a fresh yfinance pull (reset_yf=True)
    intraday_cache: bool = True          # cache in-hours snapshots; reuse them when closed

    # --- IV history / regime ---
    use_iv_history: bool = True
    iv_hist_start: str | None = '2025-06-01' # e.g. "2025-12-01"; None -> pipeline default
    iv_hist_end: str | None = None       # None -> the pricing date
    iv_hist_reset: bool = False
    iv_hist_workers: int = 12            # parallel threads for the per-date EOD-greeks fetch
    term_iv_history: bool = False        # also build IV history for EACH term tenor (10d/90d).
                                         # OFF by default: it triples the historical build on a
                                         # fresh symbol and only feeds term-history hovers.

    # --- term-structure overlays ---
    show_term_curves: bool = True
    term_tenors: tuple = (10, 90)
    term_monthly: tuple = (90,)          # tenors here snap to the nearest monthly expiry

    # --- VIX / VVIX panels (cumulative-close distribution + VVIX/VIX convexity ratio) ---
    show_vix_panels: bool = True
    vix_dist_since: str | None = None    # None -> only full-history VIX/VVIX dists;
                                         # a date (e.g. "2025-10-01") ALSO adds "since" dists
    vix_dist_end: str | None = None
    vix_bin_size: float = 1.0            # $ bin width for the ^VIX distribution
    vvix_bin_size: float = 2.0           # $ bin width for the ^VVIX distribution
    ratio_start: str | None = "2024-01-01"
    ratio_end: str | None = None
    ratio_upper_thres: float = 1.75      # log(VVIX/VIX) level flagged as high-convexity stress
    ratio_lookback: int = 21
    ratio_ewma_alpha: float = 0.94
    ratio_percentile_thres: float = 0.75
    ratio_high_regime: str = "upper_thres"   # 'upper_thres' | 'ewma' | 'percentile'

    # --- structural / shape ---
    size: int = 150                      # # strikes each side pulled
    skew_model: str = "svi"              # 'svi' (arbitrage-aware, log-moneyness) or 'poly'
    poly_degree: int = 5                 # used only when skew_model == 'poly'
    n_strikes: int = 120                 # repriced theoretical-chain resolution
    z_grid: np.ndarray = field(default_factory=lambda: np.array([-3., -2., -1., 0., 1., 2., 3.]))
    wings_on: bool = True
    slope_left: float = 0.04             # vol-fraction per SD beyond the left wing (>0 = up)
    slope_right: float = 0.04
    wing_extra_sd: float = 1.5           # extra SDs to draw/price past the grid ends

    # --- positions (covered-strangle / book) ---
    positions: list | None = field(default_factory=lambda: [
        (740, "P", 100), (715, "P", -200), (690, "P", 100), (780, "C", 50), (810, "C", -50)])
    shares: float | None = 0

    # --- auto-populate the book from the trade ledger (opd.fetch_trades_ledger) ---
    auto_positions: bool = False             # pull OPEN ledger legs for `symbol` into the book
    auto_positions_match_expiry: bool = False  # only legs on the analysed expiry (else all open)
    auto_positions_shares: bool = False      # also net OPEN delta-hedge shares into `shares`
    auto_positions_replace: bool = False     # True: ledger legs replace `positions`; False: merge

    # --- RV vs IV (realized-implied fair value) ---
    show_rv_compare: bool = True         # RV-implied fair vol/straddle vs market now & open
    rv_trading_year: float = 252.0       # trading days/yr the composite RV is annualised on;
                                         # the RV straddle's tau uses THIS, not calendar/365
                                         # (day-count consistency, cf. SKEWLAB_TODO #10)

    # --- display ---
    open_in_browser: bool = True
    half_iv_slider: float = 6.0          # +/- vol pts each side of the seeded IV slider value
    day_count: float = 365.0

    scenarios: dict = field(default_factory=lambda: SCENARIOS)
