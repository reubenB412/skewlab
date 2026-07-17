"""skewlab.pipeline.demo — a self-contained SYNTHETIC data backend.

Implements the small surface skewlab needs from the (private) production pipeline so the
dashboard, tests and CI run **offline** — no network, no ThetaData terminal, no API keys.
Everything is generated reproducibly from numpy PRNGs seeded per symbol, so a given symbol
always yields the same synthetic world.

    from skewlab.pipeline.demo import get_demo_pipeline
    cvt, opd = get_demo_pipeline()

`cvt` provides:  get_quick_option_chain, get_composite_realised_volatility
`opd` provides:  trading_dates, last_trading_date, second_last_trading_date, ny_timezone,
                 get_ohlcv_from_symbol, close_tickers, build_iv_panels,
                 fetch_trades_ledger / trades_dict / print_trade_list

The synthetic option chains carry exactly the columns skewlab reads
(S, R, Q, dte, T, implied_vol, iv_call, iv_put, straddle, mid_call, mid_put, midpoint,
expiration, root) with a realistic put-skewed smile, so the SVI fit, Breeden–Litzenberger
density, no-arbitrage checks and RV-vs-IV panel all light up.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
from math import erf

import numpy as np
import pandas as pd

_SQRT2 = 2.0 ** 0.5
_INDEX_SYMBOLS = {"^VIX", "^VVIX", "^SPX", "^VIX3M", "^SDEX", "VXX"}


def _ncdf(x):
    """Standard-normal CDF (vectorised, scipy-free)."""
    v = np.vectorize(erf)
    return 0.5 * (1.0 + v(np.asarray(x, float) / _SQRT2))


def _bs(S, K, T, r, q, sigma, call=True):
    """Black–Scholes European price (arrays over K/sigma allowed)."""
    S = np.asarray(S, float); K = np.asarray(K, float)
    sig = np.maximum(np.asarray(sigma, float), 1e-6); T = max(float(T), 1e-6)
    sq = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * sq)
    d2 = d1 - sig * sq
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    if call:
        return S * dq * _ncdf(d1) - K * dr * _ncdf(d2)
    return K * dr * _ncdf(-d2) - S * dq * _ncdf(-d1)


def _seed(symbol):
    """Stable per-symbol seed (independent of PYTHONHASHSEED)."""
    return int.from_bytes(hashlib.md5(str(symbol).encode()).digest()[:4], "big")


class _World:
    """A reproducible synthetic price + vol world for one symbol."""

    def __init__(self, symbol, today, years_back=2.8):
        self.symbol = str(symbol)
        self.is_index = self.symbol in _INDEX_SYMBOLS
        rng = np.random.default_rng(_seed(symbol))
        n = int(years_back * 252)
        self.dates = pd.bdate_range(end=pd.Timestamp(today).normalize(), periods=n)

        # a slowly-varying daily vol (annualises to ~12–26%) + GJR-ish clustering
        base_dv = 0.011 + 0.004 * np.sin(np.linspace(0, 5.5, n))
        shocks = rng.normal(0, 1, n)
        dv = np.clip(base_dv + 0.0016 * pd.Series(shocks).ewm(span=15).mean().values, 0.005, 0.05)
        rets = dv * shocks + 0.0003            # small positive drift
        self.log_ret = rets

        if self.is_index:                       # VIX/VVIX-like level series (mean-reverting)
            lvl = {"^VIX": 16.0, "^VVIX": 95.0, "^VIX3M": 18.0, "^SDEX": 60.0,
                   "VXX": 20.0, "^SPX": 5200.0}.get(self.symbol, 20.0)
            x = np.zeros(n); x[0] = lvl
            for i in range(1, n):
                x[i] = max(0.5, x[i - 1] + 0.06 * (lvl - x[i - 1]) + lvl * 0.05 * shocks[i])
            close = x
        else:
            start = 600.0 if self.symbol.upper() == "SPY" else 60.0 + _seed(symbol) % 400
            close = start * np.exp(np.cumsum(rets))

        hi = close * (1 + np.abs(rng.normal(0, 0.004, n)))
        lo = close * (1 - np.abs(rng.normal(0, 0.004, n)))
        op = np.r_[close[0], close[:-1]]
        self.ohlcv = pd.DataFrame({"Open": op, "High": np.maximum(hi, close),
                                   "Low": np.minimum(lo, close), "Close": close,
                                   "Volume": rng.integers(1e6, 8e6, n)}, index=self.dates)
        # rolling realized vol (annualised, 252) from close-to-close
        lr = pd.Series(rets, index=self.dates)
        self.realized = {w: lr.rolling(w).std() * np.sqrt(252) for w in (5, 7, 10, 15, 21)}

    def spot_on(self, date):
        d = pd.Timestamp(date).normalize()
        s = self.ohlcv["Close"]
        s = s[s.index <= d]
        return float(s.iloc[-1]) if len(s) else float(self.ohlcv["Close"].iloc[-1])

    def rv_on(self, date, lookback=21):
        w = min((5, 7, 10, 15, 21), key=lambda x: abs(x - lookback))
        r = self.realized[w]
        d = pd.Timestamp(date).normalize()
        r = r[r.index <= d].dropna()
        return float(r.iloc[-1]) if len(r) else 0.15


class _DemoBase:
    """Shared world cache for the demo cvt/opd facades."""

    def __init__(self, today=None):
        self.today = pd.Timestamp(today).normalize() if today else pd.Timestamp.today().normalize()
        self._worlds = {}

    def world(self, symbol):
        if symbol not in self._worlds:
            self._worlds[symbol] = _World(symbol, self.today)
        return self._worlds[symbol]


class DemoCVT(_DemoBase):
    """Synthetic option-chain + realized-vol source (the ``cvt`` role)."""

    R, Q = 0.041, 0.010

    def get_quick_option_chain(self, symbol, date=None, prev_date=None, target_dte=30,
                               size=150, verbose=False):
        w = self.world(symbol)
        obs = pd.Timestamp(date).normalize() if date else self.today
        dte = max(int(target_dte), 1)
        T = dte / 365.0
        spot = w.spot_on(obs)
        fwd = spot * np.exp((self.R - self.Q) * T)
        atf = max(w.rv_on(obs, lookback=max(int(round(dte * 5 / 7)), 2)) * 1.08, 0.06)  # small VRP

        step = max(round(spot * 0.004, 2), 0.5)
        ks = np.round(np.arange(spot * 0.62, spot * 1.28, step) / step) * step
        ks = np.unique(ks[ks > 0])
        k = np.log(ks / fwd)                                   # log-moneyness
        # put-skewed smile with a smooth minimum near the forward (SVI-ish shape)
        smile = atf * (1.0 + 3.2 * (k - 0.02) ** 2 - 0.55 * k)
        iv = np.clip(smile, 0.03, 3.0)

        call = _bs(spot, ks, T, self.R, self.Q, iv, call=True)
        put = _bs(spot, ks, T, self.R, self.Q, iv, call=False)
        exp_str = (obs + pd.Timedelta(days=dte)).strftime("%Y%m%d")
        df = pd.DataFrame({
            "expiration": exp_str, "S": spot, "R": self.R, "adjusted_r": self.R, "Q": self.Q,
            "dte": dte, "T": T, "implied_vol": iv, "iv_call": iv, "iv_put": iv,
            "mid_call": call, "mid_put": put, "straddle": call + put,
            "midpoint": np.abs(call - put), "root": str(symbol),
        }, index=pd.Index(ks, name="strike"))
        if verbose:
            print(f"[demo] {symbol} chain {obs.date()} {dte}DTE spot={spot:,.2f} "
                  f"ATF={atf*100:.1f}% ({len(df)} strikes)")
        return df

    def get_composite_realised_volatility(self, symbol, lookback=21, price_data=None,
                                          start=None, end=None, verbose=False, **kw):
        w = self.world(symbol)
        lr = pd.Series(w.log_ret, index=w.dates)
        cc = lr.rolling(int(lookback)).std() * np.sqrt(252)
        ewma = lr.ewm(span=max(int(lookback) // 2, 3)).std() * np.sqrt(252)
        yz = cc * 0.95                                         # stand-in Yang–Zhang
        mean = (0.5 * cc + 0.3 * yz + 0.2 * ewma)
        out = pd.DataFrame({"C-C": cc, "YZ": yz, "EWMA": ewma, "Mean": mean})
        if start is not None:
            out = out.loc[pd.Timestamp(start):]
        if end is not None:
            out = out.loc[:pd.Timestamp(end)]
        return out


class DemoOPD(_DemoBase):
    """Synthetic calendar / OHLCV / IV-history / VIX-VVIX / ledger source (the ``opd`` role)."""

    def __init__(self, today=None):
        super().__init__(today)
        try:
            from zoneinfo import ZoneInfo
            self.ny_timezone = ZoneInfo("America/New_York")
        except Exception:
            self.ny_timezone = _dt.timezone(_dt.timedelta(hours=-5))
        # a small synthetic trade ledger: one OPEN SPY short strangle (renders Position/PnL)
        self.trades_dict = {
            "SPY_short_strangle_demo": {
                "symbol": "SPY", "dte": 30, "s0": 600.0,
                "options": [
                    {"expiry": None, "op_type": "p", "strike": 555.0, "contracts": 1,
                     "tr_type": "s", "closed": 0},
                    {"expiry": None, "op_type": "c", "strike": 650.0, "contracts": 1,
                     "tr_type": "s", "closed": 0},
                ],
                "delta_hedging_log": [],
            }
        }

    # --- calendar ---
    @property
    def trading_dates(self):
        return self.world("SPY").dates

    @property
    def last_trading_date(self):
        return str(self.trading_dates[-1].date())

    @property
    def second_last_trading_date(self):
        return str(self.trading_dates[-2].date())

    # --- prices ---
    def get_ohlcv_from_symbol(self, symbol):
        return self.world(symbol).ohlcv.copy()

    @property
    def close_tickers(self):
        cols = {}
        for s in ("^VIX", "^VVIX", "^SPX", "^VIX3M", "^SDEX", "VXX"):
            cols[s] = self.world(s).ohlcv["Close"]
        return pd.DataFrame(cols)

    # --- IV-history / regime panel ---
    def build_iv_panels(self, symbol, start=None, end=None, target_dte=30, reset=False,
                        n_workers=8, verbose=False):
        w = self.world(symbol)
        idx = w.dates
        if start is not None:
            idx = idx[idx >= pd.Timestamp(start)]
        if end is not None:
            idx = idx[idx <= pd.Timestamp(end)]
        rng = np.random.default_rng(_seed(symbol) + 7)
        rv = pd.Series({d: w.rv_on(d, max(int(round(target_dte * 5 / 7)), 2)) for d in idx})
        atm = (rv * 1.08 + rng.normal(0, 0.004, len(idx))).clip(0.05, 2.0)
        hist = pd.DataFrame({
            "10d_put":  atm * 1.11, "25d_put": atm * 1.06, "atm": atm,
            "25d_call": atm * 0.97, "10d_call": atm * 0.95,
        }, index=idx)
        return atm.rename("atm_iv"), hist

    # --- trade ledger (synthetic) ---
    def fetch_trades_ledger(self, *a, **k):
        return self.trades_dict

    def print_trade_list(self, *a, **k):
        return None


def get_demo_pipeline(today=None):
    """Return a synthetic ``(cvt, opd)`` pair sharing the same 'today'."""
    return DemoCVT(today), DemoOPD(today)
