"""skewlab.model — pure quantitative engine.

Every function here is pure: it takes explicit arguments and returns values, with no
module-level state and no I/O. This is the testable core that the data layer and the
charts both build on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# --- calendar DTE -> realized-vol lookback (trading days) ---------------------------
def trading_days_for_dte(dte, per_week=5, cal_per_week=7, floor=2):
    """Convert calendar days-to-expiry into the equivalent number of TRADING days, for
    the realized-vol lookback passed to get_composite_realised_volatility.

    Uses the 5-trading-days-per-7-calendar-days ratio, so the horizon of the realized
    window matches the option's time to expiry:
        30 -> 21,  21 -> 15,  14 -> 10,  10 -> 7,  7 -> 5
    """
    return max(int(floor), int(round(float(dte) * per_week / cal_per_week)))


# --- realized-vol -> fair ATM-forward straddle --------------------------------------
_SQRT_2_OVER_PI = 0.7978845608028654          # sqrt(2/pi): the ATMF straddle coefficient


def rv_atmf_straddle(forward, sigma, dte, trading_year=252.0,
                     per_week=5, cal_per_week=7):
    """Fair ATM-forward straddle implied by an annualised vol `sigma` (e.g. yesterday's
    close realized vol), priced over the option's remaining life.

        straddle  ~=  sqrt(2/pi) * F * sigma * sqrt(tau)   ( ~= 0.7979 * one_sd )

    This is the leading (undiscounted, first-order) term of the exact ATMF straddle
    2*F*e^{-rT}*(2*Phi(sigma*sqrt(tau)/2) - 1); accurate to a few cents at equity horizons.

    IMPORTANT (day-count consistency, cf. SKEWLAB_TODO #10): a realized vol from daily
    returns is annualised on TRADING days (~252/yr), so `tau` here is measured in trading
    days too -- trading_days_to_expiry / trading_year -- NOT calendar dte/365. Pricing a
    252-basis vol with a 365 calendar tau would bias the straddle high by ~sqrt(365/252)-1
    (~+20% on tau, ~+9% on the straddle). Returns NaN on bad inputs.
    """
    try:
        F, s = float(forward), float(sigma)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(F) and np.isfinite(s)) or F <= 0 or s <= 0:
        return float("nan")
    td = max(int(round(float(dte) * per_week / cal_per_week)), 1)   # trading days to expiry
    tau = td / float(trading_year)
    return _SQRT_2_OVER_PI * F * s * np.sqrt(tau)


# --- net-liquidity growth projection (continuous compounding + annuity-due deposits) -
def net_liquidity_projection(principal, rate, years, contribution=0.0,
                             contrib_freq_years=1.0, sample_years=0.5, start_date=None):
    """Growth of net liquidity under CONTINUOUS compounding, sampled every `sample_years`
    (default 6 months), with optional level contributions made ANNUITY-DUE.

        value(t) = principal * e^(r t)  +  sum_{t_k <= t} contribution * e^(r (t - t_k))

    `rate` is the continuously-compounded annual rate (effective/yr = e^r - 1). Annuity-due
    means deposits land at the START of each period, so t_k = 0, dt, 2dt, ... with the very
    first deposit at t=0 and the last at (N-1)*dt where dt = contrib_freq_years and
    N = floor(years/dt). Each deposit then compounds continuously to the valuation time.
    No withdrawals, taxes or fees.

    Returns a DataFrame, one row per sample point:
        period        0,1,2,...                         (half-year index by default)
        years         elapsed years (0, 0.5, 1.0, ...)
        date          calendar date (start_date or today + round(t*12) months)
        added         cumulative $ deposited so far (excludes the principal)
        net_liq       projected net liquidity ($)
        gain          net_liq - principal - added        (pure investment growth $)
    """
    P, r = float(principal), float(rate)
    yrs, step = float(years), float(sample_years)
    dt, C = float(contrib_freq_years), float(contribution)
    n = max(int(round(yrs / step)), 0)
    t = np.arange(n + 1) * step                              # valuation times (years)

    base = P * np.exp(r * t)
    contrib_val = np.zeros_like(t, dtype=float)
    added = np.zeros_like(t, dtype=float)
    if C != 0.0 and dt > 0:
        n_pay = int(np.floor(yrs / dt + 1e-9))               # annuity-due: pays at 0..(n_pay-1)*dt
        tk = np.arange(max(n_pay, 0)) * dt
        if tk.size:
            paid = tk[None, :] <= (t[:, None] + 1e-9)        # (samples x deposits) already made?
            grown = np.where(paid, np.exp(r * (t[:, None] - tk[None, :])), 0.0)
            contrib_val = C * grown.sum(axis=1)
            added = C * paid.sum(axis=1)

    value = base + contrib_val
    start = pd.Timestamp.today().normalize() if start_date is None else pd.Timestamp(start_date)
    dates = [start + pd.DateOffset(months=int(round(ti * 12))) for ti in t]
    return pd.DataFrame({
        "period":  np.arange(n + 1),
        "years":   t,
        "date":    dates,
        "added":   np.round(added, 2),
        "net_liq": np.round(value, 2),
        "gain":    np.round(value - P - added, 2),
    })


# --- Black-Scholes (European, continuous dividend q) --------------------------------
def bs_price_delta(S, K, T, r, q, sigma, is_call):
    """Price and delta of one option."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return intrinsic, (1.0 if (is_call and S > K) else 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    if is_call:
        return S * dq * norm.cdf(d1) - K * dr * norm.cdf(d2), dq * norm.cdf(d1)
    return K * dr * norm.cdf(-d2) - S * dq * norm.cdf(-d1), -dq * norm.cdf(-d1)


def bs_greeks(S, K, T, r, q, sigma):
    """Vega (per 1.00 vol) and (call) theta (per year) -- shared by call & put."""
    if T <= 0 or sigma <= 0:
        return 0.0, 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
    theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1))
    return vega, theta


def bs_all(S, K, T, r, q, sigma, is_call):
    """Closed-form price + Greeks for one option.
    Returns (price, delta, gamma, vega_per_1.00_vol, theta_per_calendar_day)."""
    if T <= 0 or sigma <= 0:
        intr = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        delta = (1.0 if (is_call and S > K) else (-1.0 if (not is_call and S < K) else 0.0))
        return intr, delta, 0.0, 0.0, 0.0
    sq = np.sqrt(T)
    dq, dr = np.exp(-q * T), np.exp(-r * T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    pdf = norm.pdf(d1)
    if is_call:
        price = S * dq * norm.cdf(d1) - K * dr * norm.cdf(d2)
        delta = dq * norm.cdf(d1)
        theta_y = (-S * dq * pdf * sigma / (2 * sq) - r * K * dr * norm.cdf(d2)
                   + q * S * dq * norm.cdf(d1))
    else:
        price = K * dr * norm.cdf(-d2) - S * dq * norm.cdf(-d1)
        delta = -dq * norm.cdf(-d1)
        theta_y = (-S * dq * pdf * sigma / (2 * sq) + r * K * dr * norm.cdf(-d2)
                   - q * S * dq * norm.cdf(-d1))
    gamma = dq * pdf / (S * sigma * sq)          # d2price/dS2  (per share)
    vega = S * dq * pdf * sq                      # per 1.00 of vol
    return price, delta, gamma, vega, theta_y / 365.0


def bs_call_vec(strikes, sigmas, S, T, r, q):
    """Vectorised European call price across strikes & per-strike vols."""
    K = np.asarray(strikes, float)
    sig = np.asarray(sigmas, float)
    d1 = (np.log(S / K) + (r - q + 0.5 * sig ** 2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


# --- skew curve: polynomial fit + wing extrapolation --------------------------------
def fit_skew_curve(strikes, vols, degree=5):
    """numpy LINEST equivalent: a `degree`-order polynomial of vol vs strike."""
    return np.poly1d(np.polyfit(strikes, vols, degree))


# --- SVI smile (Gatheral raw parameterisation, fit in log-moneyness) ----------------
# Total implied variance  w(k) = sigma_BS(k)^2 * T  is modelled as
#     w(k) = a + b * ( rho*(k - m) + sqrt((k - m)^2 + s^2) )      (k = ln(K / F))
# 5 params (a, b, rho, m, s). This is the industry-standard arbitrage-controllable
# smile: linear wings, a single smooth minimum, and a closed-form butterfly test.
def svi_total_variance(k, p):
    a, b, rho, m, s = p
    km = np.asarray(k, float) - m
    return a + b * (rho * km + np.sqrt(km * km + s * s))


def _svi_deriv(k, p):
    """w'(k), w''(k) for the Durrleman butterfly check."""
    a, b, rho, m, s = p
    km = np.asarray(k, float) - m
    root = np.sqrt(km * km + s * s)
    return b * (rho + km / root), b * (s * s) / (root ** 3)


def svi_min_butterfly_g(p, k_grid):
    """Durrleman g(k); g>=0 everywhere <=> no butterfly arbitrage. Returns (min_g, k*)."""
    w = svi_total_variance(k_grid, p)
    wp, wpp = _svi_deriv(k_grid, p)
    g = (1.0 - k_grid * wp / (2.0 * w)) ** 2 - (wp ** 2) / 4.0 * (1.0 / w + 0.25) + wpp / 2.0
    i = int(np.argmin(g))
    return float(g[i]), float(k_grid[i])


def fit_svi(strikes, vols, forward, T):
    """Least-squares fit of raw SVI to market total variances. Returns a vol(K) callable
    carrying its params on `.svi_params` (and `.svi_forward`/`.svi_T`)."""
    from scipy.optimize import least_squares
    K = np.asarray(strikes, float)
    k = np.log(K / forward)
    w = (np.asarray(vols, float) ** 2) * T
    # drop any non-finite / non-positive nodes (a single NaN makes the bounds NaN and
    # scipy then rejects the initial guess as "outside of provided bounds")
    good = np.isfinite(k) & np.isfinite(w) & (w > 0)
    k, w = k[good], w[good]
    if len(w) < 5:
        raise ValueError(f"only {len(w)} finite variance node(s); need >=5 for SVI")
    wmax, wmin = float(np.max(w)), float(np.min(w))
    # init: flat-ish level, mild equity skew (rho<0), centred minimum
    p0 = [max(wmin * 0.5, 1e-4), 0.1 * (wmax + 1e-6), -0.3, 0.0, 0.1]
    lo = [-wmax, 1e-8, -0.999, k.min() - 0.5, 1e-4]
    hi = [2.0 * wmax + 1e-6, 10.0, 0.999, k.max() + 0.5, 2.0]
    # keep the guess strictly inside the box (trf rejects on-boundary / out-of-bounds p0)
    p0 = [min(max(v, l + 1e-9), h - 1e-9) for v, l, h in zip(p0, lo, hi)]
    res = least_squares(lambda p: svi_total_variance(k, p) - w, p0, bounds=(lo, hi),
                        method="trf", max_nfev=4000)
    p = res.x

    def vol(Kq):
        kk = np.log(np.asarray(Kq, float) / forward)
        return np.sqrt(np.maximum(svi_total_variance(kk, p), 1e-10) / T)

    vol.svi_params = tuple(p)
    vol.svi_forward = float(forward)
    vol.svi_T = float(T)
    return vol


def _flat_vol(level):
    """A constant vol(K) callable — last-resort fallback when no fit is possible."""
    lvl = float(level) if np.isfinite(level) else 0.20
    lvl = max(lvl, 1e-4)

    def vol(Kq):
        return np.full(np.shape(np.asarray(Kq, float)), lvl, dtype=float)

    vol.flat_level = lvl
    return vol


def fit_skew(strikes, vols, *, model_name="poly", degree=5, forward=None, T=None):
    """Unified skew fitter -> a vol(K) callable. model_name in {'poly','svi'}.

    Inputs are sanitised first (NaN/inf and non-positive vols dropped, degree capped to
    the node count). SVI needs forward & T and >=5 good nodes; on any failure it falls
    back to the polynomial, and the polynomial in turn falls back to a flat smile — so a
    thin/after-hours chain degrades gracefully instead of raising LinAlgError/SVD."""
    K = np.asarray(strikes, float)
    V = np.asarray(vols, float)
    good = np.isfinite(K) & np.isfinite(V) & (V > 0)
    Kg, Vg = K[good], V[good]

    if len(Kg) < 2:
        finite = V[np.isfinite(V)]
        lvl = float(np.median(finite)) if finite.size else 0.20
        print(f"[skew] only {len(Kg)} valid IV node(s); using flat smile at {lvl:.4f}.")
        return _flat_vol(lvl)

    deg = int(max(1, min(int(degree), len(Kg) - 1)))

    if model_name == "svi" and forward is not None and T is not None:
        try:
            return fit_svi(Kg, Vg, forward, T)
        except Exception as e:
            print(f"[svi] fit failed ({e}); falling back to degree-{deg} polynomial")

    try:
        return fit_skew_curve(Kg, Vg, deg)
    except Exception as e:
        lvl = float(np.median(Vg))
        print(f"[poly] fit failed ({e}); using flat smile at {lvl:.4f}.")
        return _flat_vol(lvl)


def vol_at_strike(strike, poly):
    return np.maximum(poly(strike), 1e-4)


def curve_vol(strikes, poly, grid_strikes, forward, one_sd, z_grid,
              slope_left, slope_right, wings_on):
    """Vol curve WITH wing extrapolation: polynomial inside +/-3 SD, a straight sloped
    line beyond the end grid points (slope in vol-fraction per SD; >0 turns the wings up)."""
    K = np.asarray(strikes, float)
    v = np.asarray(poly(K), float)
    if wings_on:
        kL, kR = grid_strikes[0], grid_strikes[-1]
        vL, vR = float(poly(kL)), float(poly(kR))
        z = (K - forward) / one_sd
        zL, zR = z_grid[0], z_grid[-1]
        v = np.where(K < kL, vL + slope_left * (zL - z), v)
        v = np.where(K > kR, vR + slope_right * (z - zR), v)
    return np.maximum(v, 1e-4)


def fine_strikes(grid_strikes, forward, one_sd, z_grid, wing_extra_sd, wings_on, n=320):
    """Dense strike grid for plotting/pricing, widened past the grid when wings are on."""
    extra = wing_extra_sd if wings_on else 0.0
    lo = max(forward + (z_grid[0] - extra) * one_sd, 0.02 * forward)
    hi = forward + (z_grid[-1] + extra) * one_sd
    return np.linspace(lo, hi, n)


# --- implied (risk-neutral) distribution: Breeden-Litzenberger ----------------------
def raw_density(strikes, calls, r, T):
    """UNCLIPPED risk-neutral density e^{rT} d2C/dK2. Negative regions => butterfly
    arbitrage in the fitted smile (a call butterfly with a negative price)."""
    dK = strikes[1] - strikes[0]
    return np.exp(r * T) * np.gradient(np.gradient(calls, dK), dK)


def implied_pdf(strikes, calls, r, T):
    """f(K) = e^{rT} d2C/dK2, clipped >=0 and normalised to area 1."""
    pdf = np.clip(raw_density(strikes, calls, r, T), 0.0, None)
    area = np.sum((pdf[:-1] + pdf[1:]) / 2.0 * np.diff(strikes))
    return pdf / area if area > 0 else pdf


def butterfly_arb(strikes, calls, r, T, tol_frac=1e-3):
    """Scan the raw density for negative regions (butterfly arb). Returns a dict with
    whether arb is present, the worst (most negative) density, and the % of the
    interior strike range that is negative."""
    dens = raw_density(strikes, calls, r, T)[2:-2]      # drop noisy finite-diff edges
    ks = np.asarray(strikes, float)[2:-2]
    scale = max(float(np.max(dens)), 1e-12)
    neg = dens < -tol_frac * scale
    return dict(has_arb=bool(neg.any()), min_density=float(np.min(dens)),
                frac_negative=float(neg.mean()) * 100.0,
                worst_strike=float(ks[int(np.argmin(dens))]))


def calendar_arb(bundles, vol_of, k_grid=None):
    """Check total variance w = sigma^2 * T is non-decreasing in T across tenors at shared
    log-moneyness. `bundles` is a list of objects with .t/.forward; `vol_of(bundle, K)`
    returns that tenor's IV at strike K. Returns dict(has_arb, n_pairs, worst)."""
    if k_grid is None:
        k_grid = np.linspace(-0.25, 0.25, 25)
    ordered = sorted(bundles, key=lambda b: b.t)
    worst = 0.0
    pairs = 0
    for short, long in zip(ordered[:-1], ordered[1:]):
        pairs += 1
        Ks_s, Ks_l = short.forward * np.exp(k_grid), long.forward * np.exp(k_grid)
        w_s = (np.asarray(vol_of(short, Ks_s), float) ** 2) * short.t
        w_l = (np.asarray(vol_of(long, Ks_l), float) ** 2) * long.t
        worst = min(worst, float(np.min(w_l - w_s)))     # <0 => longer T has less total var
    return dict(has_arb=bool(worst < -1e-6), n_pairs=pairs, worst=worst)


def dist_stats(x, pdf):
    """Mean / median / mode / std / skew / excess-kurtosis of a discrete density."""
    w = pdf / pdf.sum()
    mean = float((x * w).sum())
    std = float(np.sqrt(((x - mean) ** 2 * w).sum()))
    median = float(x[np.searchsorted(np.cumsum(w), 0.5)])
    mode = float(x[int(np.argmax(pdf))])
    skew = float((((x - mean) / std) ** 3 * w).sum()) if std > 0 else 0.0
    kurt = float((((x - mean) / std) ** 4 * w).sum() - 3.0) if std > 0 else 0.0
    return dict(mean=mean, median=median, mode=mode, std=std, skew=skew, kurt=kurt)


def pctile(arr, val):
    """Percentile rank (0-100) of val within arr, ignoring NaNs."""
    a = np.asarray(arr, float)
    a = a[~np.isnan(a)]
    return float((a <= val).mean() * 100.0) if len(a) else float("nan")


def hover_stats(hist_pct, cur_pct):
    """IV rank / percentile / 3-month z-score of cur_pct within a history series (vol pts)."""
    if hist_pct is None:
        return None
    s = hist_pct.dropna()
    if len(s) == 0:
        return None
    lo, hi = float(s.min()), float(s.max())
    last63 = s.iloc[-63:]
    mu, sd = float(last63.mean()), float(last63.std())
    return dict(lo=lo, hi=hi, mean=float(s.mean()), n=len(s),
                rank=(100.0 * (cur_pct - lo) / (hi - lo) if hi > lo else float("nan")),
                pctl=pctile(s.values, cur_pct),
                z3=((cur_pct - mu) / sd if sd > 0 else float("nan")),
                pctl3=pctile(last63.values, cur_pct))


# --- seeding the curve from the market chain ----------------------------------------
def market_iv_by_strike(chain, forward):
    """OTM implied-vol series indexed by strike (puts below fwd, calls above)."""
    df = chain.sort_index()
    iv = np.where(df.index <= forward, df["iv_put"], df["iv_call"])
    s = __import__("pandas").Series(iv, index=df.index, name="iv").dropna()
    if "implied_vol" in df.columns:
        s = s.combine_first(df["implied_vol"])
    return s.dropna().sort_index()


def market_skew_pct(chain, forward, atf_vol, grid_strikes, z_grid):
    """Skew% at each SD node, interpolated from market IVs. 0-SD pinned to the ATF anchor.
    If no per-strike IVs are available (thin/after-hours chain) falls back to a flat smile."""
    iv_s = market_iv_by_strike(chain, forward)
    if len(iv_s) == 0 or not np.isfinite(atf_vol) or atf_vol <= 0:
        return {z: 0.0 for z in z_grid}, iv_s
    grid_iv = np.interp(grid_strikes, iv_s.index.values, iv_s.values)
    skew = {z: (grid_iv[i] / atf_vol - 1.0) for i, z in enumerate(z_grid)}
    skew[0.0] = 0.0
    return skew, iv_s


def delta_point(poly, grid_strikes, S, T, r, q, slope_left, slope_right, wings_on,
                forward, one_sd, z_grid, target, is_call, n=400):
    """Strike & IV on the wing-aware curve at a target delta (e.g. 0.20 call, -0.20 put)."""
    Ks = np.linspace(grid_strikes[0], grid_strikes[-1], n)
    sig = curve_vol(Ks, poly, grid_strikes, forward, one_sd, z_grid, slope_left, slope_right, wings_on)
    d = np.array([bs_price_delta(S, K, T, r, q, s, is_call)[1] for K, s in zip(Ks, sig)])
    i = int(np.argmin(np.abs(d - target)))
    return float(Ks[i]), float(sig[i])


# --- VIX / VVIX empirical analytics -------------------------------------------------
def cumulative_price_distribution(closes, bin_size=2.0):
    """Cumulative distribution of a close series over fixed-width price bins.
    Returns (df[price_bin,count,prob,cum_prob], current_close, current_bin_label)."""
    closes = pd.Series(closes).dropna()
    current_close = float(closes.iloc[-1])
    lo = np.floor(closes.min() / bin_size) * bin_size
    hi = np.ceil(closes.max() / bin_size) * bin_size
    bins = np.arange(lo, hi + bin_size, bin_size)
    labels = [f"{int(b)}–{int(b + bin_size)}" for b in bins[:-1]]
    binned = pd.cut(closes, bins=bins, labels=labels, include_lowest=True)
    df = (binned.value_counts().sort_index().rename_axis("price_bin").reset_index(name="count"))
    total = df["count"].sum()
    df["prob"] = df["count"] / total if total else 0.0
    df["cum_prob"] = df["prob"].cumsum()
    current_bin = None
    for interval, label in zip(pd.IntervalIndex.from_breaks(bins), labels):
        if current_close in interval:
            current_bin = label
            break
    return df, current_close, current_bin


def vvix_vix_ratio_table(vix, vvix, ewma_fn=None, lookback=21, ewma_alpha=0.94,
                         percentile_thres=0.75, upper_thres=1.75, high_regime="upper_thres"):
    """log(VVIX/VIX) convexity-stress table: raw signal, EWMA, rolling percentile, static
    threshold, and a boolean high-convexity regime flag. `ewma_fn` is the pipeline's
    calc_ewma_zscore (injected); falls back to a plain pandas EWM if unavailable."""
    s = np.log(pd.Series(vvix).astype(float) / pd.Series(vix).astype(float))
    ratio = s.to_frame(name="log_VVIX_VIX")
    try:
        ratio["ewma"], _ = ewma_fn(ratio["log_VVIX_VIX"], ewma_alpha=ewma_alpha, return_zscore=False)
    except Exception:
        ratio["ewma"] = ratio["log_VVIX_VIX"].ewm(alpha=1.0 - ewma_alpha).mean()
    ratio["perc"] = ratio["log_VVIX_VIX"].rolling(lookback).quantile(percentile_thres)
    ratio["upper_thres"] = upper_thres
    if high_regime == "ewma":
        ratio["high_regime"] = ratio["log_VVIX_VIX"] >= ratio["ewma"]
    elif high_regime == "percentile":
        ratio["high_regime"] = ratio["log_VVIX_VIX"] >= ratio["perc"]
    else:
        ratio["high_regime"] = ratio["log_VVIX_VIX"] >= upper_thres
    return ratio
