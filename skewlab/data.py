"""skewlab.data — the I/O layer.

`fetch_snapshot(cfg, cvt, opd)` does ALL the slow/stateful work once (chain fetch with
walk-back / intraday, previous-day pin, term-structure curves, IV-history panels) and
returns an immutable `Snapshot`. Charts then take `(snapshot, curve_state)` and never
touch the pipeline again.
"""
from __future__ import annotations

import datetime as _dt
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
import pandas as pd

from . import model


# =====================================================================================
# State objects
# =====================================================================================
@dataclass
class TermBundle:
    """A self-contained skew curve for one expiry (its own forward/ATF/grid/poly)."""
    tenor: int
    monthly: bool
    poly: Any
    grid_strikes: np.ndarray
    forward: float
    atf: float
    one_sd: float
    t: float
    r: float
    q: float
    spot: float
    dte: float
    expiry: pd.Timestamp
    hist: Optional[pd.DataFrame] = None


@dataclass
class Snapshot:
    """Immutable market context, built once per run."""
    cfg: Any
    symbol: str
    date: str
    requested_date: str
    spot: float
    forward: float
    t: float
    r: float
    q: float
    dte: float
    atf: float
    one_sd: float
    z_grid: np.ndarray
    grid_strikes: np.ndarray
    grid_vols: np.ndarray
    skew_pct: dict
    market_iv: pd.Series
    poly: Any                              # base-case (market) fitted curve
    # frozen "shadow" baselines
    mkt_curve_x: np.ndarray
    mkt_curve_y: np.ndarray
    mkt_pdf_x: np.ndarray
    mkt_pdf_y: np.ndarray
    # previous-day overlay
    chain_prev: Optional[pd.DataFrame]
    prev_poly: Optional[Any]
    prev_label: Optional[str]
    prev_obs_date: Optional[pd.Timestamp]
    # term structure
    term_bundles: list = field(default_factory=list)
    # IV history / regime
    iv_atm: Optional[pd.Series] = None
    iv_history: Optional[pd.DataFrame] = None
    iv_rv: Optional[pd.Series] = None
    # VIX / VVIX empirical panels: (df, current_close, current_bin) tuples + ratio table
    vix_dist: Optional[tuple] = None            # full history
    vvix_dist: Optional[tuple] = None           # full history
    vix_dist_since: Optional[tuple] = None       # since cfg.vix_dist_since (None -> chart hidden)
    vvix_dist_since: Optional[tuple] = None
    vix_vvix_ratio: Optional[pd.DataFrame] = None
    since_when: Optional[str] = None
    # book
    positions: list = field(default_factory=list)
    shares: float = 0.0
    # raw inputs kept for inspection/debugging (the cleaned chain the grid was built from)
    chain: Optional[pd.DataFrame] = None
    # previous-day at-the-forward (for the prev-curve ATMF marker)
    prev_forward: Optional[float] = None
    prev_atf: Optional[float] = None
    # RV-vs-IV comparison (realized-implied): fair vol/straddle from yesterday's-close
    # composite RV, vs the market at the day's open and now. None -> section hidden.
    rv_iv: Optional[float] = None               # composite RV (decimal), most recent close
    rv_straddle: Optional[float] = None          # fair ATMF straddle implied by rv_iv ($)
    rv_asof: Optional[str] = None                # date of the RV observation
    rv_lookback: Optional[int] = None            # trading-day lookback used for the RV
    open_atf: Optional[float] = None             # market ATF at the earliest capture today
    open_straddle: Optional[float] = None        # market ATM straddle at the day's open ($)
    open_capture_ts: Optional[str] = None        # timestamp of that open capture
    now_straddle: Optional[float] = None         # market ATM straddle now ($)
    now_capture_ts: Optional[str] = None         # timestamp of the latest capture ('live' if open)

    # --- convenience wrappers (delegate to pure model funcs with this snapshot's grid) --
    @property
    def wing_extra_sd(self):
        return self.cfg.wing_extra_sd

    @property
    def has_positions(self):
        return bool(self.positions) or bool(self.shares)

    def fine_strikes(self, wings_on, n=320):
        return model.fine_strikes(self.grid_strikes, self.forward, self.one_sd, self.z_grid,
                                  self.cfg.wing_extra_sd, wings_on, n)

    def curve_vol(self, strikes, cs):
        return model.curve_vol(strikes, cs.poly, self.grid_strikes, self.forward, self.one_sd,
                               self.z_grid, cs.slope_left, cs.slope_right, cs.wings_on)


@dataclass
class CurveState:
    """The mutable knobs the dashboard tweaks; `poly` is derived from grid_vols."""
    grid_vols: np.ndarray
    slope_left: float
    slope_right: float
    wings_on: bool
    atf: float
    poly: Any

    @classmethod
    def from_grid(cls, snap: Snapshot, grid_vols, slope_left, slope_right, wings_on,
                  degree=None):
        grid_vols = np.asarray(grid_vols, float)
        atf = float(grid_vols[list(snap.z_grid).index(0.0)])
        poly = model.fit_skew(snap.grid_strikes, grid_vols, model_name=snap.cfg.skew_model,
                              degree=degree or snap.cfg.poly_degree,
                              forward=snap.forward, T=snap.t)
        return cls(grid_vols, slope_left, slope_right, wings_on, atf, poly)

    @classmethod
    def market(cls, snap: Snapshot):
        return cls.from_grid(snap, snap.grid_vols, snap.cfg.slope_left, snap.cfg.slope_right,
                             snap.cfg.wings_on)

    @classmethod
    def from_scenario(cls, snap: Snapshot, name):
        s = snap.cfg.scenarios[name]
        grid = np.array([snap.atf * s["atf_mult"] * (1.0 + p) for p in s["skew"]])
        return cls.from_grid(snap, grid, s["sl"] / 100.0, s["sr"] / 100.0, True)


# =====================================================================================
# Pipeline helpers
# =====================================================================================
def _third_friday(year, month):
    first = pd.Timestamp(year, month, 1)
    first_fri = first + pd.Timedelta(days=(4 - first.weekday()) % 7)
    return first_fri + pd.Timedelta(days=14)


def _monthly_request_dte(obs_date, target_dte):
    obs = pd.to_datetime(obs_date)
    approx = obs + pd.Timedelta(days=int(target_dte))
    cands = []
    for off in (-1, 0, 1, 2):
        y, m = divmod(approx.month - 1 + off, 12)
        cands.append(_third_friday(approx.year + y, m + 1))
    cands = [c for c in cands if c > obs]
    return max(int((min(cands, key=lambda d: abs((d - approx).days)) - obs).days), 1)


def _request_target_dte(cfg, obs_date, target_dte):
    return _monthly_request_dte(obs_date, target_dte) if cfg.monthly_only else int(target_dte)


def _nearest_trading_date(opd, ts):
    td = pd.DatetimeIndex(opd.trading_dates).normalize()
    ts = pd.to_datetime(ts).normalize()
    pos = int(td.searchsorted(ts))
    if pos >= len(td):
        pos = len(td) - 1
    if td[pos] > ts and pos > 0:
        pos -= 1
    return td[pos]


def _chain_expiry(chain, obs_date):
    if "expiration" in chain.columns and chain["expiration"].notna().any():
        return pd.to_datetime(chain["expiration"].dropna().iloc[0])
    return pd.to_datetime(obs_date) + pd.Timedelta(days=int(chain["dte"].dropna().iloc[0]))


def _core(chain, day_count):
    df = chain
    if df is None or len(df) == 0:
        raise RuntimeError("option chain is empty (no rows returned).")

    # ATM-forward strike: prefer min|C-P| (put-call parity); fall back to min straddle,
    # then strike nearest spot -- so thin / after-hours intraday chains (NaN bids/asks ->
    # NaN midpoint) don't blow up with a KeyError: nan.
    key = None
    for col in ("midpoint", "straddle"):
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            key = pd.to_numeric(df[col], errors="coerce").idxmin()
            break
    if key is None or (isinstance(key, float) and np.isnan(key)):
        ks = pd.Series(df.index, index=df.index).astype(float)
        spot_guess = (float(pd.to_numeric(df["S"], errors="coerce").dropna().iloc[0])
                      if "S" in df.columns and df["S"].notna().any() else float(ks.median()))
        key = (ks - spot_guess).abs().idxmin()

    atmf = df.loc[key]
    if isinstance(atmf, pd.DataFrame):          # duplicate strike index -> take first
        atmf = atmf.iloc[0]

    def _row_num(col):
        try:
            return float(pd.to_numeric(atmf.get(col), errors="coerce"))
        except Exception:
            return float("nan")

    def _col_median(col, default):
        if col not in df.columns:
            return default
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(s.median()) if len(s) else default

    # spot / rate / div — robust to a NaN on the chosen ATM row (else fwd->NaN blanks the chart)
    spot = _row_num("S")
    if not np.isfinite(spot) or spot <= 0:
        spot = _col_median("S", float("nan"))
    if not np.isfinite(spot) or spot <= 0:
        raise RuntimeError("option chain has no usable underlying price (S is NaN across the "
                           "chain — bad/stale snapshot?). Try reset_intraday=True or use EOD.")
    r = _row_num("R")
    if not np.isfinite(r):
        r = _col_median("R", 0.0)
    q = _row_num("Q")
    if not np.isfinite(q):
        q = _col_median("Q", 0.0)
    dte = _row_num("dte")
    if not np.isfinite(dte) or dte <= 0:
        dte = _col_median("dte", float("nan"))
    t = dte / day_count
    fwd = spot * np.exp((r - q) * t)

    # ATF vol — robust to a single bad ATM quote (e.g. an intraday 3% print when the smile
    # is ~13%). Anchor on the median IV of the strikes NEAREST the forward; only keep the
    # chosen row's IV if it's present, positive, and not a low-outlier vs those neighbours.
    iv_all = pd.to_numeric(df["implied_vol"], errors="coerce")
    iv_all = iv_all[iv_all > 0].dropna()
    near_atf = float("nan")
    if len(iv_all):
        strikes = pd.to_numeric(pd.Series(df.index, index=df.index), errors="coerce")
        order = (strikes - fwd).abs().sort_values().index
        near = pd.to_numeric(df.loc[order, "implied_vol"], errors="coerce")
        near = near[near > 0].dropna()
        if len(near):
            near_atf = float(near.iloc[:5].median())
    atf = _row_num("implied_vol")
    if (not np.isfinite(atf)) or atf <= 0 or (np.isfinite(near_atf) and atf < 0.5 * near_atf):
        if np.isfinite(near_atf):
            atf = near_atf
        elif len(iv_all):
            atf = float(iv_all.median())
        else:
            raise RuntimeError("option chain has no usable implied vols (thin / after-hours "
                               "intraday quotes?). Try reset_intraday=True, run during market "
                               "hours, or use a more liquid symbol.")
    return spot, r, q, t, dte, atf, fwd


def _atm_straddle(chain):
    """The quoted ATM-forward straddle ($): the `straddle` at the strike the pipeline uses
    as the forward -- min|mid_call-mid_put| (put-call parity), falling back to min straddle,
    then the strike nearest spot. Mirrors _core's ATM-row selection. NaN if unavailable."""
    df = chain
    if df is None or len(df) == 0 or "straddle" not in df.columns:
        return float("nan")
    key = None
    for col in ("midpoint", "straddle"):
        if col in df.columns and pd.to_numeric(df[col], errors="coerce").notna().any():
            key = pd.to_numeric(df[col], errors="coerce").abs().idxmin()
            break
    if key is None or (isinstance(key, float) and np.isnan(key)):
        try:
            ks = pd.Series(df.index, index=df.index).astype(float)
            spot_guess = float(pd.to_numeric(df["S"], errors="coerce").dropna().iloc[0])
            key = (ks - spot_guess).abs().idxmin()
        except Exception:
            return float("nan")
    val = pd.to_numeric(pd.Series(df["straddle"]).get(key, np.nan), errors="coerce")
    return float(val) if np.isfinite(val) else float("nan")


def _rv_benchmark(cfg, cvt, iv_rv):
    """Composite realized vol as of the most recent close -> (sigma, asof_date_str) or
    (None, None). Reuses the already-built `iv_rv` series when present (no extra fetch);
    otherwise pulls a fresh composite RV (price-only, so it works even for symbols with no
    EOD options coverage). The last value = yesterday's-close realized vol -- the fair vol
    the market 'should' be pricing today."""
    ser = None
    if iv_rv is not None and len(pd.Series(iv_rv).dropna()):
        ser = pd.Series(iv_rv).dropna()
    else:
        try:
            rvdf = cvt.get_composite_realised_volatility(
                cfg.symbol, lookback=model.trading_days_for_dte(cfg.target_dte),
                end=str(cfg.date), verbose=False)
            if rvdf is not None and "Mean" in getattr(rvdf, "columns", []):
                ser = rvdf["Mean"].astype(float).dropna()
        except Exception as e:
            print(f"[rv-compare] realized-vol benchmark unavailable: {e}")
    if ser is None or not len(ser):
        return None, None
    # "previous day's close": exclude a partial current-day bar if one snuck in (a live
    # session can carry a forming today row) so the benchmark is a settled close.
    try:
        today = pd.to_datetime(cfg.date).normalize()
        prior = ser[pd.DatetimeIndex(ser.index).normalize() < today]
        if len(prior):
            ser = prior
    except Exception:
        pass
    try:
        asof = str(pd.to_datetime(ser.index[-1]).date())
    except Exception:
        asof = None
    return float(ser.iloc[-1]), asof


def _build_bundle(chain, cfg, day_count):
    spot, r, q, t, dte, atf, fwd = _core(chain, day_count)
    one_sd = fwd * atf * np.sqrt(t)
    grid_strikes = fwd + cfg.z_grid * one_sd
    skew_pct, _ = model.market_skew_pct(chain, fwd, atf, grid_strikes, cfg.z_grid)
    grid_vols = np.array([atf * (1.0 + skew_pct[z]) for z in cfg.z_grid])
    return dict(poly=model.fit_skew(grid_strikes, grid_vols, model_name=cfg.skew_model,
                                    degree=cfg.poly_degree, forward=fwd, T=t),
                grid_strikes=grid_strikes, forward=fwd, atf=atf, one_sd=one_sd, t=t, r=r,
                q=q, spot=spot, dte=dte, expiry=_chain_expiry(chain, cfg.date))


def _us_market_open_now(opd):
    """True only during the regular US options session (NY time, weekday, ~09:46-18:00)."""
    import datetime as _dt
    try:
        now = _dt.datetime.now(opd.ny_timezone)
    except Exception:
        now = _dt.datetime.now()
    if now.weekday() >= 5:                       # Sat/Sun
        return False
    return _dt.time(9, 46) <= now.time() < _dt.time(18, 0)


def _intraday_live(cfg, opd):
    """Whether to use the LIVE yfinance chain: only if requested AND the market is open."""
    return bool(cfg.use_intraday) and _us_market_open_now(opd)


def _fetch_yf_chain(cfg, cvt, opd, target_dte=None, reset=None):
    """One live yfinance pull, sliced to `target_dte`.

    `reset` overrides cfg.reset_intraday: the FIRST expiry of a run may force a fresh
    download, but every later expiry passes reset=False so it re-slices the full chain
    that was just written to the per-day pkl (one download per run, not one per tenor)."""
    tdte = _request_target_dte(cfg, cfg.date, cfg.target_dte) if target_dte is None else target_dte
    reset_yf = cfg.reset_intraday if reset is None else reset
    return cvt.get_quick_option_chain(
        cfg.symbol, cfg.date, None, target_dte=tdte,
        size=cfg.size, intraday=True, reset_yf=reset_yf, verbose=False)


# --- in-hours yfinance snapshot cache ----------------------------------------------
# We keep our OWN cache of the cleaned chain captured *during* a live session, one file
# per (symbol, target_dte) holding the latest good snapshot. Outside market hours we
# reuse it instead of pulling thin after-hours quotes, before ever touching EOD.
def _intraday_cache_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_intraday_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _intraday_cache_path(symbol, target_dte):
    safe = str(symbol).replace("^", "").replace("/", "_")
    return os.path.join(_intraday_cache_dir(), f"{safe}_dte{int(target_dte)}_intraday.pkl")


def _read_cache_raw(symbol, target_dte):
    """Read the raw pickle dict for (symbol, target_dte), or None. Normalises the LEGACY
    flat format {chain, captured} into the newer {day, open, latest} shape so both the
    latest- and open-of-day loaders can read it uniformly."""
    p = _intraday_cache_path(symbol, target_dte)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "rb") as f:
            d = pickle.load(f)
    except Exception as e:
        print(f"[cache] could not read intraday snapshot: {e}")
        return None
    if not isinstance(d, dict):
        return None
    if "latest" in d or "open" in d:          # new format
        return d
    if "chain" in d:                           # legacy flat format -> treat as latest only
        cap = d.get("captured", "?")
        day = str(cap)[:10] if cap and cap != "?" else None
        one = {"chain": d.get("chain"), "captured": cap}
        return {"day": day, "open": one, "latest": one}
    return None


def _save_intraday_snapshot(symbol, target_dte, chain):
    """Persist the newest in-hours capture. Keeps the EARLIEST capture of the current day
    as `open` (never overwritten within a day) and the newest as `latest`; a new calendar
    day resets `open` to this capture. Enables the open-vs-now RV comparison."""
    now = _dt.datetime.now()
    today = now.date().isoformat()
    cap = now.isoformat(timespec="minutes")
    this = {"chain": chain, "captured": cap}
    prev = _read_cache_raw(symbol, target_dte)
    if prev is not None and prev.get("day") == today and prev.get("open") is not None:
        open_snap = prev["open"]               # same day -> preserve the earliest capture
    else:
        open_snap = this                        # first capture of a new day (or no prior)
    try:
        with open(_intraday_cache_path(symbol, target_dte), "wb") as f:
            pickle.dump({"day": today, "open": open_snap, "latest": this}, f)
    except Exception as e:
        print(f"[cache] could not save intraday snapshot: {e}")


def _load_intraday_snapshot(symbol, target_dte):
    """Most-recent cached in-hours chain -> (chain, captured) or None."""
    d = _read_cache_raw(symbol, target_dte)
    if d is None:
        return None
    latest = d.get("latest") or {}
    ch = latest.get("chain")
    if ch is None or len(ch) == 0:
        return None
    return ch, latest.get("captured", "?")


def _load_open_snapshot(symbol, target_dte):
    """EARLIEST cached in-hours chain of the cache's stored day -> (chain, captured, day)
    or None. Pairs with _load_intraday_snapshot (the latest) to show the day's drift."""
    d = _read_cache_raw(symbol, target_dte)
    if d is None:
        return None
    op = d.get("open") or {}
    ch = op.get("chain")
    if ch is None or len(ch) == 0:
        return None
    return ch, op.get("captured", "?"), d.get("day")


def _intraday_chain(cfg, cvt, opd, target_dte, reset=None):
    """Live yfinance chain when the market is open (and cache it); otherwise reuse the most
    recent in-hours snapshot for this (symbol, target_dte). Returns a chain or None.

    `reset` is forwarded to `_fetch_yf_chain` so callers after the first expiry can pass
    reset=False and re-slice the already-downloaded full chain (one download per run)."""
    if _intraday_live(cfg, opd):
        ch = _fetch_yf_chain(cfg, cvt, opd, target_dte, reset=reset)
        if getattr(cfg, "intraday_cache", True):
            _save_intraday_snapshot(cfg.symbol, target_dte, ch)
        return ch
    if getattr(cfg, "intraday_cache", True):
        snap = _load_intraday_snapshot(cfg.symbol, target_dte)
        if snap is not None:
            ch, ts = snap
            print(f"[fetch] market closed -> reusing in-hours yfinance snapshot for "
                  f"{cfg.symbol} @{int(target_dte)}DTE (captured {ts}).")
            return ch
    return None


def _fetch_eod_walkback(cfg, cvt, opd):
    """Settled EOD chain with a multi-session walk-back."""
    td = pd.DatetimeIndex(opd.trading_dates).normalize()
    pos = int(td.searchsorted(pd.to_datetime(cfg.date).normalize()))
    pos = min(max(pos, 0), len(td) - 1)
    if td[pos] > pd.to_datetime(cfg.date).normalize() and pos > 0:
        pos -= 1
    last_err = None
    for d in [td[i] for i in range(pos, max(pos - 7, -1), -1)]:
        ds = d.strftime("%Y-%m-%d")
        try:
            return cvt.get_quick_option_chain(
                cfg.symbol, ds, None, target_dte=_request_target_dte(cfg, ds, cfg.target_dte),
                size=cfg.size, verbose=False)
        except Exception as e:
            last_err = e
            print(f"[fetch] {cfg.symbol}: no data around {d.date()}, walking back...")
    raise RuntimeError(f"No option chain for '{cfg.symbol}' near {cfg.date} (last: {last_err}).")


def _fetch_today_chain(cfg, cvt, opd):
    """Source the 'today' chain when use_intraday:
        1. market OPEN  -> live yfinance (and cache the snapshot),
        2. market CLOSED -> reuse the most recent in-hours yfinance snapshot,
        3. else settled ThetaData EOD (walk-back),
        4. else a fresh (possibly thin) yfinance pull as a last resort.
    With use_intraday=False it's settled EOD only."""
    if not cfg.use_intraday:
        return _fetch_eod_walkback(cfg, cvt, opd)

    tdte = _request_target_dte(cfg, cfg.date, cfg.target_dte)
    ch = _intraday_chain(cfg, cvt, opd, tdte)          # live or cached in-hours snapshot
    if ch is not None:
        return ch
    print(f"[fetch] market closed & no in-hours snapshot for '{cfg.symbol}' -> trying settled EOD.")
    try:
        return _fetch_eod_walkback(cfg, cvt, opd)
    except Exception as e:
        print(f"[fetch] EOD unavailable for '{cfg.symbol}' ({e}); pulling a fresh (possibly thin) "
              f"yfinance chain as a last resort.")
        return _fetch_yf_chain(cfg, cvt, opd, tdte)


def _fetch_prev_chain(cfg, cvt, opd, chain, prev_obs, max_back=5):
    """Previous-day chain with a short walk-back so a single dead/empty session doesn't kill
    the overlay. Recomputes target_dte per candidate session to honour pin_same_expiry.
    Returns (chain_or_None, actual_obs_date)."""
    td = pd.DatetimeIndex(opd.trading_dates).normalize()
    want = pd.to_datetime(prev_obs).normalize()
    pos = int(td.searchsorted(want))
    pos = min(max(pos, 0), len(td) - 1)
    if td[pos] > want and pos > 0:
        pos -= 1
    today_exp = _chain_expiry(chain, cfg.date)
    last_err = None
    for d in [td[i] for i in range(pos, max(pos - max_back, -1), -1)]:
        ds = d.strftime("%Y-%m-%d")
        try:
            tdte = (max(int((today_exp - d).days), 1) if cfg.pin_same_expiry
                    else _request_target_dte(cfg, d, cfg.target_dte))
            return cvt.get_quick_option_chain(cfg.symbol, ds, None, target_dte=tdte,
                                              size=cfg.size, verbose=False), d
        except Exception as e:
            last_err = e
            print(f"[prev] {cfg.symbol}: no chain on {d.date()}, walking back...")
    print(f"[prev] previous-day fetch failed near {want.date()} (last: {last_err}) -> no overlay")
    return None, prev_obs


def _index_closes(opd, symbol, start, end):
    """Daily close series for an index/ETF via the pipeline's OHLCV helper."""
    ohlc = opd.get_ohlcv_from_symbol(symbol)
    closes = ohlc["Close"].dropna()
    if start or end:
        closes = closes.loc[start:end]
    return closes


def _fetch_vix_panels(cfg, cvt, opd):
    """(vix_full, vvix_full, vix_since, vvix_since, ratio) — each best-effort / None.
    *_full use the entire history; *_since start at cfg.vix_dist_since (None -> skipped)."""
    vix_full = vvix_full = vix_since = vvix_since = ratio = None

    def _dist(symbol, bin_size, start):
        return model.cumulative_price_distribution(
            _index_closes(opd, symbol, start, cfg.vix_dist_end), bin_size)

    try:
        vix_full = _dist("^VIX", cfg.vix_bin_size, None)
    except Exception as e:
        print(f"[vix] ^VIX distribution unavailable: {e}")
    try:
        vvix_full = _dist("^VVIX", cfg.vvix_bin_size, None)
    except Exception as e:
        print(f"[vix] ^VVIX distribution unavailable: {e}")
    if cfg.vix_dist_since:
        try:
            vix_since = _dist("^VIX", cfg.vix_bin_size, cfg.vix_dist_since)
        except Exception as e:
            print(f"[vix] ^VIX distribution since {cfg.vix_dist_since} unavailable: {e}")
        try:
            vvix_since = _dist("^VVIX", cfg.vvix_bin_size, cfg.vix_dist_since)
        except Exception as e:
            print(f"[vix] ^VVIX distribution since {cfg.vix_dist_since} unavailable: {e}")
    try:
        tickers = ["^VIX", "^VVIX", "^SPX", "^VIX3M", "^SDEX", "VXX"]
        try:
            data = opd.close_tickers[tickers].dropna().loc[cfg.ratio_start:cfg.ratio_end]
        except Exception:
            data = opd.close_tickers[["^VIX", "^VVIX"]].dropna().loc[cfg.ratio_start:cfg.ratio_end]
        ratio = model.vvix_vix_ratio_table(
            data["^VIX"], data["^VVIX"], ewma_fn=getattr(cvt, "calc_ewma_zscore", None),
            lookback=cfg.ratio_lookback, ewma_alpha=cfg.ratio_ewma_alpha,
            percentile_thres=cfg.ratio_percentile_thres, upper_thres=cfg.ratio_upper_thres,
            high_regime=cfg.ratio_high_regime)
    except Exception as e:
        print(f"[vix] VVIX/VIX ratio unavailable: {e}")
    return vix_full, vvix_full, vix_since, vvix_since, ratio


# =====================================================================================
# The one entry point that builds everything
# =====================================================================================
def fetch_snapshot(cfg, cvt, opd) -> Snapshot:
    # normalise date (None -> today)
    if cfg.date is None:
        cfg.date = str(getattr(opd, "today_ny_strftime", None) or pd.Timestamp.today().strftime("%Y%m%d"))
    requested = cfg.date

    chain = _fetch_today_chain(cfg, cvt, opd)
    spot, r, q, t, dte, atf, fwd = _core(chain, cfg.day_count)

    # startup sanity line — makes a bad/stale snapshot obvious without opening the dashboard
    _n_iv = int(pd.to_numeric(chain["implied_vol"], errors="coerce").gt(0).sum()) if "implied_vol" in chain.columns else 0
    _flag = "  <-- suspect: ATF very low / few IVs" if (atf < 0.05 or _n_iv < 5) else ""
    print(f"[core] {cfg.symbol} spot={spot:,.2f} fwd={fwd:,.2f} {dte:.0f}DTE "
          f"ATF={atf * 100:.1f}%  ({_n_iv} valid strike IVs){_flag}")

    # recover the actual observation date from the chain (expiry - dte)
    cfg.date = (_chain_expiry(chain, cfg.date) - pd.Timedelta(days=int(dte))).strftime("%Y-%m-%d")
    if str(pd.to_datetime(cfg.date).date()) != str(pd.to_datetime(requested).date()):
        print(f"[date] requested {pd.to_datetime(requested).date()} -> using fetched {cfg.date}")

    one_sd = fwd * atf * np.sqrt(t)
    grid_strikes = fwd + cfg.z_grid * one_sd
    skew_pct, market_iv = model.market_skew_pct(chain, fwd, atf, grid_strikes, cfg.z_grid)
    grid_vols = np.array([atf * (1.0 + skew_pct[z]) for z in cfg.z_grid], float)
    # sanitize: a thin smile can leave a NaN at an extreme SD node, which would seed a slider
    # to null and crash the Dash callback. Backfill any non-finite node from the ATF level.
    if not np.all(np.isfinite(grid_vols)):
        _finite = grid_vols[np.isfinite(grid_vols)]
        _fill = atf if np.isfinite(atf) else (float(np.median(_finite)) if _finite.size else 0.20)
        grid_vols = np.where(np.isfinite(grid_vols), grid_vols, _fill)
    poly = model.fit_skew(grid_strikes, grid_vols, model_name=cfg.skew_model,
                          degree=cfg.poly_degree, forward=fwd, T=t)

    # --- previous-day overlay (pin to same expiry when requested) ---
    prev_obs = None
    if cfg.prev_date is not None:
        prev_obs = pd.to_datetime(cfg.prev_date)
    elif cfg.pin_same_expiry and cfg.lookback_days:
        prev_obs = _nearest_trading_date(opd, pd.to_datetime(cfg.date) - pd.Timedelta(days=int(cfg.lookback_days)))
    chain_prev = prev_poly = prev_label = None
    prev_forward = prev_atf = None
    if prev_obs is not None:
        chain_prev, prev_obs = _fetch_prev_chain(cfg, cvt, opd, chain, prev_obs)
    if chain_prev is not None:
        today_exp = _chain_expiry(chain, cfg.date).date()
        prev_exp = _chain_expiry(chain_prev, prev_obs).date()
        prev_dte = float(chain_prev["dte"].dropna().iloc[0])
        matched = today_exp == prev_exp
        if cfg.pin_same_expiry and not matched and cfg.pin_strict:
            print(f"[pin] STRICT: expiry mismatch ({today_exp} vs {prev_exp}) -> dropping overlay")
            chain_prev = None
        else:
            _b = _build_bundle(chain_prev, cfg, cfg.day_count)
            # prev curve on TODAY's strike axis, for clean diffs/overlay
            sp, _iv = model.market_skew_pct(chain_prev, _b["forward"], _b["atf"], grid_strikes, cfg.z_grid)
            gv = np.array([_b["atf"] * (1.0 + sp[z]) for z in cfg.z_grid])
            prev_poly = model.fit_skew(grid_strikes, gv, model_name=cfg.skew_model,
                                       degree=cfg.poly_degree, forward=fwd, T=t)
            prev_forward, prev_atf = float(_b["forward"]), float(_b["atf"])
            tag = "same exp" if matched else f"exp {prev_exp}"
            prev_label = f"{prev_obs.date()} · {prev_dte:.0f}DTE · {tag}"

    # --- IV history + realized vol ---
    iv_atm = iv_history = iv_rv = None
    if cfg.use_iv_history:
        print("[iv-history] building panels (this can take a while)...")
        try:
            iv_atm, iv_history = opd.build_iv_panels(
                symbol=cfg.symbol, start=cfg.iv_hist_start, end=(cfg.iv_hist_end or cfg.date),
                target_dte=cfg.target_dte, reset=cfg.iv_hist_reset,
                n_workers=getattr(cfg, "iv_hist_workers", 8), verbose=True)
            n = 0 if iv_atm is None else len(iv_atm)
            print(f"[iv-history] loaded {n} obs")
            if n == 0:
                print(f"[iv-history] 0 obs for '{cfg.symbol}': the historical EOD options panel is "
                      f"empty. This almost always means the data terminal has no EOD options "
                      f"coverage for this symbol (the same reason a previous-day overlay 474s). "
                      f"Today's live skew still works via the intraday chain, but the IV history & "
                      f"regime panel and term-history hovers will be hidden. Try an ETF/root with "
                      f"EOD coverage (e.g. SPY), or set use_iv_history=False to silence this.")
        except Exception as e:
            import traceback
            print("[iv-history] build_iv_panels raised -- continuing without history:")
            traceback.print_exc()
        if iv_atm is not None and len(iv_atm):
            try:
                rvdf = cvt.get_composite_realised_volatility(
                    cfg.symbol, lookback=model.trading_days_for_dte(cfg.target_dte),
                    start=str(pd.DatetimeIndex(iv_atm.index).min().date()), end=str(cfg.date), verbose=False)
                if rvdf is not None and "Mean" in getattr(rvdf, "columns", []):
                    iv_rv = rvdf["Mean"].astype(float)
            except Exception as e:
                print(f"[iv-history] realized-vol overlay unavailable: {e}")

    # --- term-structure curves ---
    term_bundles = []
    if cfg.show_term_curves:
        for tn in cfg.term_tenors:
            try:
                monthly = tn in cfg.term_monthly
                tdte = _monthly_request_dte(cfg.date, tn) if monthly else int(tn)
                # reset=False: the main expiry above already pulled (and cached) the full
                # chain this run, so the term tenors just re-slice it -> no extra download.
                tc = _intraday_chain(cfg, cvt, opd, tdte, reset=False) if cfg.use_intraday else None
                if tc is None:                    # closed w/ no snapshot, or use_intraday=False
                    tc = cvt.get_quick_option_chain(cfg.symbol, cfg.date, None, target_dte=tdte,
                                                    size=cfg.size, verbose=False)
                b = _build_bundle(tc, cfg, cfg.day_count)
                hist = None
                # per-tenor IV history is OFF by default — it reruns the full ~1yr panel
                # build for each tenor (3x cost on a fresh symbol) just for hover history.
                if cfg.use_iv_history and getattr(cfg, "term_iv_history", False):
                    try:
                        _, hist = opd.build_iv_panels(
                            symbol=cfg.symbol, start=cfg.iv_hist_start, end=(cfg.iv_hist_end or cfg.date),
                            target_dte=tdte, reset=cfg.iv_hist_reset,
                            n_workers=getattr(cfg, "iv_hist_workers", 8), verbose=False)
                    except Exception as e:
                        print(f"[term] {tn}d history unavailable: {e}")
                term_bundles.append(TermBundle(tenor=tn, monthly=monthly, hist=hist, **b))
                print(f"[term] {tn}d -> {b['dte']:.0f}DTE exp {b['expiry'].date()} "
                      f"(hist rows: {0 if hist is None else len(hist)})")
            except Exception as e:
                print(f"[term] {tn}d curve failed: {e}")

    # --- VIX / VVIX empirical panels ---
    vix_dist = vvix_dist = vix_dist_since = vvix_dist_since = vix_vvix_ratio = None
    if cfg.show_vix_panels:
        (vix_dist, vvix_dist, vix_dist_since,
         vvix_dist_since, vix_vvix_ratio) = _fetch_vix_panels(cfg, cvt, opd)

    # --- positions: optionally auto-populate the book from the trade ledger ---
    snap_positions = list(cfg.positions or [])
    snap_shares = float(cfg.shares or 0.0)
    if getattr(cfg, "auto_positions", False):
        from . import positions as _pos
        analysed_exp = _chain_expiry(chain, cfg.date).strftime("%Y%m%d")
        want_exp = analysed_exp if getattr(cfg, "auto_positions_match_expiry", False) else None
        # also pull related instruments (e.g. SPY <- SPX/MES) with strike scaling
        scales = _pos.resolve_symbol_scales(cfg.symbol)
        try:
            ledger_book, _meta = _pos.open_legs_from_ledger(
                opd, cfg.symbol, expiry=want_exp, verbose=True, symbol_scales=scales)
            # If matching the analysed expiry found nothing but open legs DO exist at other
            # expiries, don't silently show an empty book — include them (valued at the
            # analysed expiry, so P&L is approximate) so the position still appears.
            if not ledger_book and want_exp is not None:
                alt_book, _meta = _pos.open_legs_from_ledger(
                    opd, cfg.symbol, expiry=None, verbose=False, symbol_scales=scales)
                if alt_book:
                    exps = sorted({str(m[2]) for m in _meta})
                    print(f"[ledger] no open legs on the analysed expiry {want_exp}; showing all "
                          f"{len(alt_book)} open leg(s) at {exps} instead (valued at the analysed "
                          f"expiry — P&L approximate). Set target_dte to match, or "
                          f"auto_positions_match_expiry=False to silence.")
                    ledger_book = alt_book
        except Exception as e:
            ledger_book = []
            print(f"[ledger] could not read trade ledger ({e}); skipping auto-positions.")
        if getattr(cfg, "auto_positions_replace", False):
            snap_positions = []
        # merge: ledger legs net into any existing legs at the same (strike, kind)
        for strike, kind, n in ledger_book:
            existing = next((p for p in snap_positions
                             if float(p[0]) == float(strike)
                             and str(p[1]).upper()[0] == str(kind).upper()[0]), None)
            base = int(existing[2]) if existing else 0
            snap_positions = _pos.add_position(snap_positions, strike, kind, base + int(n))
        if getattr(cfg, "auto_positions_shares", False):
            try:
                snap_shares += _pos.open_shares_from_ledger(opd, cfg.symbol)
            except Exception as e:
                print(f"[ledger] could not read delta-hedge shares ({e}).")

    # --- RV vs IV (realized-implied): fair vol/straddle from the most-recent-close
    #     composite RV, vs the market straddle now and at the day's open ---
    rv_iv = rv_straddle = rv_asof = rv_lookback = None
    open_atf = open_straddle = open_capture_ts = None
    now_straddle = now_capture_ts = None
    if getattr(cfg, "show_rv_compare", True):
        now_straddle = _atm_straddle(chain)
        rv_iv, rv_asof = _rv_benchmark(cfg, cvt, iv_rv)
        if rv_iv is not None and np.isfinite(rv_iv):
            rv_lookback = int(model.trading_days_for_dte(cfg.target_dte))
            rv_straddle = float(model.rv_atmf_straddle(
                fwd, rv_iv, dte, getattr(cfg, "rv_trading_year", 252.0)))
        # open-vs-now: earliest & latest in-hours captures from the day-stamped cache
        if getattr(cfg, "intraday_cache", True):
            _tdte = _request_target_dte(cfg, requested, cfg.target_dte)
            try:
                _op = _load_open_snapshot(cfg.symbol, _tdte)
            except Exception as e:
                _op = None
                print(f"[rv-compare] open snapshot unavailable: {e}")
            if _op is not None:
                ochain, open_capture_ts, _oday = _op
                try:
                    _os, _or, _oq, _ot, _odte, o_atf, _ofwd = _core(ochain, cfg.day_count)
                    open_atf = float(o_atf)
                    open_straddle = _atm_straddle(ochain)
                except Exception as e:
                    print(f"[rv-compare] open snapshot parse failed: {e}")
            _lat = _load_intraday_snapshot(cfg.symbol, _tdte)
            if _lat is not None:
                now_capture_ts = _lat[1]
        if now_capture_ts is None and _intraday_live(cfg, opd):
            now_capture_ts = "live"

    # --- frozen shadows ---
    snap = Snapshot(
        cfg=cfg, symbol=cfg.symbol, date=cfg.date, requested_date=requested,
        spot=spot, forward=fwd, t=t, r=r, q=q, dte=dte, atf=atf, one_sd=one_sd,
        z_grid=cfg.z_grid, grid_strikes=grid_strikes, grid_vols=grid_vols, skew_pct=skew_pct,
        market_iv=market_iv, poly=poly,
        mkt_curve_x=np.array([]), mkt_curve_y=np.array([]),
        mkt_pdf_x=np.array([]), mkt_pdf_y=np.array([]),
        chain_prev=chain_prev, prev_poly=prev_poly, prev_label=prev_label, prev_obs_date=prev_obs,
        term_bundles=term_bundles, iv_atm=iv_atm, iv_history=iv_history, iv_rv=iv_rv,
        vix_dist=vix_dist, vvix_dist=vvix_dist, vix_dist_since=vix_dist_since,
        vvix_dist_since=vvix_dist_since, vix_vvix_ratio=vix_vvix_ratio,
        since_when=cfg.vix_dist_since,
        positions=snap_positions, shares=snap_shares, chain=chain,
        prev_forward=prev_forward, prev_atf=prev_atf,
        rv_iv=rv_iv, rv_straddle=rv_straddle, rv_asof=rv_asof, rv_lookback=rv_lookback,
        open_atf=open_atf, open_straddle=open_straddle, open_capture_ts=open_capture_ts,
        now_straddle=now_straddle, now_capture_ts=now_capture_ts)

    mkt_fine = snap.fine_strikes(cfg.wings_on)
    snap.mkt_curve_x = mkt_fine
    snap.mkt_curve_y = model.curve_vol(mkt_fine, poly, grid_strikes, fwd, one_sd, cfg.z_grid,
                                       cfg.slope_left, cfg.slope_right, cfg.wings_on) * 100.0
    calls = model.bs_call_vec(mkt_fine, snap.mkt_curve_y / 100.0, spot, t, r, q)
    pdf = model.implied_pdf(mkt_fine, calls, r, t)
    snap.mkt_pdf_x, snap.mkt_pdf_y = mkt_fine[2:-2], pdf[2:-2]
    return snap
