"""Pure realised-volatility term-structure calculations.

The I/O layer normalises live or synthetic sources and passes ordinary pandas objects
here.  Nothing in this module imports a market-data vendor or performs I/O.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


ESTIMATOR_ROWS = (
    "C-C",
    "Parkinson",
    "Hodges-Tompkins",
    "Yang-Zhang",
    "EWMA_Halflife",
    "Mean Volatility",
    "Mean Intra",
    "Mean C-C",
    "HF Total RV",
    "HF Continuous Intraday RV",
)


@dataclass(frozen=True)
class HFCurves:
    """Rolling high-frequency curves; columns are completed-session lookbacks."""

    total: pd.DataFrame
    continuous: pd.DataFrame
    jump_share: pd.DataFrame
    overnight_share: pd.DataFrame
    continuous_share: pd.DataFrame


@dataclass(frozen=True)
class RVTermState:
    """Immutable payload consumed by the dashboard and inspection tools."""

    estimator_table: pd.DataFrame
    integrated_variance_table: pd.DataFrame
    hf_daily: pd.DataFrame
    hf_curves: HFCurves
    shape_history: pd.DataFrame
    summary: Mapping[str, object]
    iv_curve: pd.DataFrame
    variance_shares: pd.DataFrame
    metadata: Mapping[str, object] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    volatility_signature: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def available(self) -> bool:
        return bool(
            not self.estimator_table.empty
            or not self.iv_curve.empty
            or not self.hf_curves.total.empty
        )


def empty_hf_curves(lookbacks: Sequence[int] = ()) -> HFCurves:
    cols = [int(x) for x in lookbacks]
    blank = pd.DataFrame(columns=cols, dtype=float)
    return HFCurves(blank.copy(), blank.copy(), blank.copy(), blank.copy(), blank.copy())


def recover_daily_variances(
    rvdf: pd.DataFrame,
    *,
    source_basis: float,
    complete_flags: pd.Series | None = None,
    asof_timestamps: pd.Series | None = None,
    sample_minutes: int = 5,
) -> pd.DataFrame:
    """Recover unannualised daily variances from annualised-volatility columns."""
    if source_basis <= 0:
        raise ValueError("source_basis must be positive")
    if rvdf is None or getattr(rvdf, "empty", True):
        return pd.DataFrame(columns=[
            "total_variance", "intraday_variance", "overnight_variance",
            "bipower_variance", "jump_variance", "gk_variance",
            "is_complete_session", "asof_timestamp", "num_obs", "sample_minutes",
        ])
    required = {"rv_daily", "overnight", "bipower_var", "jump_var", "gk_daily"}
    missing = required.difference(rvdf.columns)
    if missing:
        raise ValueError(f"HF RV frame missing required columns: {sorted(missing)}")

    src = rvdf.copy()
    src.index = pd.DatetimeIndex(pd.to_datetime(src.index)).tz_localize(None).normalize()
    src = src[~src.index.duplicated(keep="last")].sort_index()

    def _variance(col):
        vol = pd.to_numeric(src[col], errors="coerce")
        return vol.pow(2) / float(source_basis)

    out = pd.DataFrame(index=src.index)
    out["total_variance"] = _variance("rv_daily")
    out["overnight_variance"] = _variance("overnight")
    out["intraday_variance"] = (
        out["total_variance"] - out["overnight_variance"]
    ).clip(lower=0.0)
    out["bipower_variance"] = _variance("bipower_var")
    out["jump_variance"] = _variance("jump_var")
    out["gk_variance"] = _variance("gk_daily")

    if complete_flags is None:
        out["is_complete_session"] = True
    else:
        flags = pd.Series(complete_flags).copy()
        flags.index = pd.DatetimeIndex(pd.to_datetime(flags.index)).tz_localize(None).normalize()
        out["is_complete_session"] = flags[~flags.index.duplicated(keep="last")].reindex(out.index)
        out["is_complete_session"] = out["is_complete_session"].fillna(False).astype(bool)

    if asof_timestamps is None:
        out["asof_timestamp"] = pd.NaT
    else:
        stamps = pd.Series(asof_timestamps).copy()
        stamps.index = pd.DatetimeIndex(pd.to_datetime(stamps.index)).tz_localize(None).normalize()
        out["asof_timestamp"] = stamps[~stamps.index.duplicated(keep="last")].reindex(out.index)

    out["num_obs"] = pd.to_numeric(src.get("M", np.nan), errors="coerce")
    out["sample_minutes"] = int(sample_minutes)
    for col in ("bns_jump_z", "bns_jump_pval", "jump_sig_flag", "vol_tremor_score"):
        if col in src:
            out[col] = src[col]
    return out


def aggregate_hf_curves(
    daily_variance: pd.DataFrame,
    lookbacks: Sequence[int],
    *,
    target_basis: float,
) -> HFCurves:
    """Aggregate completed daily variance using RMS/variance arithmetic."""
    if target_basis <= 0:
        raise ValueError("target_basis must be positive")
    lookbacks = tuple(dict.fromkeys(int(x) for x in lookbacks))
    if any(x <= 0 for x in lookbacks):
        raise ValueError("lookbacks must be positive integers")
    if daily_variance is None or getattr(daily_variance, "empty", True):
        return empty_hf_curves(lookbacks)

    required = {
        "total_variance", "intraday_variance", "overnight_variance",
        "bipower_variance", "jump_variance", "is_complete_session",
    }
    missing = required.difference(daily_variance.columns)
    if missing:
        raise ValueError(f"daily variance frame missing columns: {sorted(missing)}")
    daily = daily_variance.loc[daily_variance["is_complete_session"].fillna(False)].copy()
    daily = daily.sort_index()

    total = pd.DataFrame(index=daily.index, columns=lookbacks, dtype=float)
    continuous = total.copy()
    jump_share = total.copy()
    overnight_share = total.copy()
    continuous_share = total.copy()
    for n in lookbacks:
        total_sum = daily["total_variance"].rolling(n, min_periods=n).sum()
        intra_sum = daily["intraday_variance"].rolling(n, min_periods=n).sum()
        bv_sum = daily["bipower_variance"].rolling(n, min_periods=n).sum()
        jump_sum = daily["jump_variance"].rolling(n, min_periods=n).sum()
        overnight_sum = daily["overnight_variance"].rolling(n, min_periods=n).sum()
        total[n] = np.sqrt(total_sum * float(target_basis) / n)
        continuous[n] = np.sqrt(bv_sum * float(target_basis) / n)
        jump_share[n] = jump_sum.div(intra_sum.where(intra_sum > 0))
        overnight_share[n] = overnight_sum.div(total_sum.where(total_sum > 0))
        continuous_share[n] = bv_sum.div(total_sum.where(total_sum > 0))
    return HFCurves(total, continuous, jump_share, overnight_share, continuous_share)


def trailing_percentile(
    series: pd.Series,
    *,
    window: int = 252,
    min_periods: int = 60,
) -> pd.Series:
    """No-lookahead midpoint percentile of each value within its trailing history."""
    if window <= 0 or min_periods <= 0:
        raise ValueError("window and min_periods must be positive")
    s = pd.to_numeric(pd.Series(series), errors="coerce")

    def _rank_last(values):
        x = np.asarray(values, float)
        x = x[np.isfinite(x)]
        if not len(x):
            return np.nan
        last = x[-1]
        return 100.0 * ((x < last).sum() + 0.5 * (x == last).sum()) / len(x)

    return s.rolling(window=window, min_periods=min_periods).apply(_rank_last, raw=True)


def rv_shape_history(
    hf_total: pd.DataFrame,
    *,
    percentile_window: int = 252,
    percentile_min_periods: int = 60,
) -> pd.DataFrame:
    """HF Total RV curve shape, acceleration and no-lookahead percentiles."""
    needed = {5, 10, 20, 30}
    missing = needed.difference(hf_total.columns)
    if missing:
        raise ValueError(f"HF total curve missing lookbacks: {sorted(missing)}")
    out = pd.DataFrame(index=hf_total.index)
    out["slope_5_20"] = hf_total[5].div(hf_total[20])
    out["slope_10_30"] = hf_total[10].div(hf_total[30])
    out["curvature"] = (hf_total[5] + hf_total[10]).div(2.0 * hf_total[20])
    out["rv_acceleration"] = hf_total[5].diff(3)
    for col in ("slope_5_20", "slope_10_30", "curvature", "rv_acceleration"):
        out[col + "_pct"] = trailing_percentile(
            out[col], window=percentile_window, min_periods=percentile_min_periods
        )
    return out


def classify_regime(
    slope_5_20: float,
    slope_10_30: float,
    acceleration_percentile: float,
    *,
    compressed_threshold: float = 0.90,
    inverted_threshold: float = 1.00,
    building_percentile: float = 70.0,
) -> str:
    """Transparent qualitative label; intentionally not a trading recommendation."""
    vals = (slope_5_20, slope_10_30)
    if not all(np.isfinite(float(x)) for x in vals):
        return "RV regime unavailable"
    if all(float(x) > inverted_threshold for x in vals):
        return "Front-end RV inverted"
    if np.isfinite(float(acceleration_percentile)) and acceleration_percentile >= building_percentile:
        return "Front-end RV building"
    if all(float(x) <= compressed_threshold for x in vals):
        return "Front-end RV compressed"
    return "RV curve normalising"


def movement_summary(
    atm_iv: float,
    spot: float,
    daily_closes: pd.Series,
    *,
    trading_year: float = 252.0,
    realised_sessions: int = 5,
) -> dict[str, float]:
    """Comparable implied and realised daily-movement quantities."""
    if trading_year <= 0:
        raise ValueError("trading_year must be positive")
    iv = float(atm_iv)
    px = float(spot)
    sigma = iv / np.sqrt(float(trading_year))
    expected_abs = sigma * np.sqrt(2.0 / np.pi)
    closes = pd.to_numeric(pd.Series(daily_closes), errors="coerce").dropna()
    simple = closes.pct_change().dropna().tail(int(realised_sessions))
    realised_abs_pct = float(simple.abs().mean()) if len(simple) else np.nan
    return {
        "implied_daily_sigma_pct": sigma * 100.0,
        "implied_expected_abs_pct": expected_abs * 100.0,
        "implied_expected_abs_points": px * expected_abs,
        "realised_average_abs_pct": realised_abs_pct * 100.0,
        "realised_average_abs_points": px * realised_abs_pct,
        "movement_gap_points": px * (expected_abs - realised_abs_pct),
    }


def build_estimator_table(
    estimator_frames: Mapping[int, pd.DataFrame],
    lookbacks: Sequence[int],
    *,
    hf_curves: HFCurves | None = None,
) -> pd.DataFrame:
    """Construct the canonical current estimator table with stable row/column order."""
    lookbacks = tuple(dict.fromkeys(int(x) for x in lookbacks))
    table = pd.DataFrame(index=ESTIMATOR_ROWS, columns=lookbacks, dtype=float)
    aliases = {
        "C-C": "C-C",
        "Parkinson": "Parkinson",
        "Hodges-Tompkins": "Hodges-Tompkins",
        "Yang-Zhang": "YZ",
        "EWMA_Halflife": "EWMA_halflife",
        "Mean Volatility": "Mean",
    }
    for n in lookbacks:
        frame = estimator_frames.get(n)
        if frame is None or getattr(frame, "empty", True):
            continue
        for row, col in aliases.items():
            if col in frame.columns:
                values = pd.to_numeric(frame[col], errors="coerce").dropna()
                if len(values):
                    table.loc[row, n] = float(values.iloc[-1])
        table.loc["Mean Intra", n] = table.loc[
            ["Parkinson", "Hodges-Tompkins", "Yang-Zhang"], n
        ].mean(skipna=True)
        table.loc["Mean C-C", n] = table.loc[["C-C", "EWMA_Halflife"], n].mean(skipna=True)

    if hf_curves is not None:
        for n in lookbacks:
            if n in hf_curves.total and hf_curves.total[n].notna().any():
                table.loc["HF Total RV", n] = float(hf_curves.total[n].dropna().iloc[-1])
            if n in hf_curves.continuous and hf_curves.continuous[n].notna().any():
                table.loc["HF Continuous Intraday RV", n] = float(
                    hf_curves.continuous[n].dropna().iloc[-1]
                )
    table.index.name = "RV Rolling"
    table.columns.name = "lookback_sessions"
    return table


def integrated_variance_table(
    annualised_vol_table: pd.DataFrame,
    *,
    annualisation_basis: float,
) -> pd.DataFrame:
    """Convert annualised RV cells into variance accumulated over each lookback."""
    if annualisation_basis <= 0:
        raise ValueError("annualisation_basis must be positive")
    table = annualised_vol_table.apply(pd.to_numeric, errors="coerce").copy()
    for col in table.columns:
        try:
            sessions = float(col)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"RV lookback column is not numeric: {col!r}") from exc
        table[col] = table[col].pow(2) * sessions / float(annualisation_basis)
    table.index.name = annualised_vol_table.index.name
    table.columns.name = "lookback_sessions"
    return table


def latest_variance_shares(hf_curves: HFCurves) -> pd.DataFrame:
    """Current variance-composition diagnostics by lookback."""
    rows = []
    for n in hf_curves.total.columns:
        def _last(frame, col=n):
            s = pd.to_numeric(frame[col], errors="coerce").dropna()
            return float(s.iloc[-1]) if len(s) else np.nan

        rows.append({
            "lookback": int(n),
            "continuous_share_of_total": _last(hf_curves.continuous_share),
            "jump_share_of_intraday": _last(hf_curves.jump_share),
            "overnight_share_of_total": _last(hf_curves.overnight_share),
        })
    return pd.DataFrame(rows).set_index("lookback") if rows else pd.DataFrame()


def build_summary(
    *,
    shape_history: pd.DataFrame,
    movement: Mapping[str, float],
    hf_curves: HFCurves,
    compressed_threshold: float = 0.90,
    inverted_threshold: float = 1.00,
    building_percentile: float = 70.0,
) -> dict[str, object]:
    """Create the current dashboard summary from pure calculation outputs."""
    summary: dict[str, object] = dict(movement)
    valid = shape_history.dropna(subset=["slope_5_20", "slope_10_30", "curvature"])
    if valid.empty:
        summary.update(regime="RV regime unavailable", regime_source="HF total")
        return summary
    last = valid.iloc[-1]
    summary.update({k: float(last[k]) if pd.notna(last[k]) else np.nan for k in valid.columns})
    summary["regime"] = classify_regime(
        summary["slope_5_20"], summary["slope_10_30"],
        summary.get("rv_acceleration_pct", np.nan),
        compressed_threshold=compressed_threshold,
        inverted_threshold=inverted_threshold,
        building_percentile=building_percentile,
    )
    summary["regime_source"] = "HF total"
    for n in (5, 10, 20, 30):
        if n in hf_curves.total and hf_curves.total[n].notna().any():
            summary[f"hf_total_rv_{n}"] = float(hf_curves.total[n].dropna().iloc[-1])
    return summary
