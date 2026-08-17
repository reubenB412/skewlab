from __future__ import annotations

import numpy as np
import pandas as pd

from skewlab import rv


def _daily_variance_frame(n=220, *, complete=True):
    idx = pd.bdate_range("2025-01-02", periods=n)
    x = np.linspace(0.00005, 0.00020, n)
    return pd.DataFrame({
        "total_variance": x,
        "intraday_variance": x * 0.8,
        "overnight_variance": x * 0.2,
        "bipower_variance": x * 0.65,
        "jump_variance": x * 0.15,
        "gk_variance": x * 0.7,
        "is_complete_session": complete,
    }, index=idx)


def test_recover_and_variance_rms_not_arithmetic_mean():
    idx = pd.bdate_range("2026-01-02", periods=2)
    source_vol = pd.Series([0.10, 0.30], index=idx)
    frame = pd.DataFrame({
        "rv_daily": source_vol,
        "overnight": [0.0, 0.0],
        "bipower_var": source_vol * 0.8,
        "jump_var": source_vol * 0.2,
        "gk_daily": source_vol * 0.9,
        "M": [78, 78],
    }, index=idx)
    daily = rv.recover_daily_variances(frame, source_basis=365)
    curves = rv.aggregate_hf_curves(daily, [2], target_basis=365)
    got = curves.total[2].dropna().iloc[-1]
    expected = np.sqrt((0.10**2 + 0.30**2) / 2)
    assert np.isclose(got, expected)
    assert not np.isclose(got, np.mean([0.10, 0.30]))


def test_source_to_target_rebasing():
    idx = pd.bdate_range("2026-01-02", periods=5)
    frame = pd.DataFrame({
        "rv_daily": 0.20, "overnight": 0.0, "bipower_var": 0.20,
        "jump_var": 0.0, "gk_daily": 0.20,
    }, index=idx)
    daily = rv.recover_daily_variances(frame, source_basis=365)
    curves = rv.aggregate_hf_curves(daily, [5], target_basis=252)
    assert np.isclose(curves.total[5].dropna().iloc[-1], 0.20 * np.sqrt(252 / 365))


def test_partial_session_is_excluded():
    daily = _daily_variance_frame(6)
    daily.iloc[-1, daily.columns.get_loc("total_variance")] = 1.0
    daily.iloc[-1, daily.columns.get_loc("is_complete_session")] = False
    curves = rv.aggregate_hf_curves(daily, [5], target_basis=252)
    expected = np.sqrt(daily.iloc[:5]["total_variance"].mean() * 252)
    assert curves.total.index[-1] == daily.index[-2]
    assert np.isclose(curves.total[5].dropna().iloc[-1], expected)


def test_variance_shares_use_documented_denominators():
    curves = rv.aggregate_hf_curves(_daily_variance_frame(5), [5], target_basis=252)
    assert np.isclose(curves.jump_share[5].iloc[-1], 0.15 / 0.8)
    assert np.isclose(curves.overnight_share[5].iloc[-1], 0.20)
    assert np.isclose(curves.continuous_share[5].iloc[-1], 0.65)


def test_shape_percentiles_have_no_future_leakage():
    daily = _daily_variance_frame(220)
    base_curves = rv.aggregate_hf_curves(daily, [5, 10, 20, 30], target_basis=252)
    base = rv.rv_shape_history(base_curves.total, percentile_window=80, percentile_min_periods=20)
    extra = _daily_variance_frame(1).set_axis([daily.index[-1] + pd.offsets.BDay()])
    extended = pd.concat([daily, extra])
    extended.iloc[-1, extended.columns.get_loc("total_variance")] = 0.5
    future_curves = rv.aggregate_hf_curves(extended, [5, 10, 20, 30], target_basis=252)
    future = rv.rv_shape_history(future_curves.total, percentile_window=80, percentile_min_periods=20)
    pd.testing.assert_series_equal(base.loc[daily.index[-1]], future.loc[daily.index[-1]],
                                   check_names=False)


def test_regime_boundaries_and_precedence():
    assert rv.classify_regime(0.90, 0.90, 69.9) == "Front-end RV compressed"
    assert rv.classify_regime(0.80, 0.80, 70.0) == "Front-end RV building"
    assert rv.classify_regime(1.01, 1.01, 99.0) == "Front-end RV inverted"
    assert rv.classify_regime(1.00, 1.00, 20.0) == "RV curve normalising"


def test_daily_movement_conversions():
    closes = pd.Series([100.0, 101.0, 99.99, 100.9899, 99.980001])
    out = rv.movement_summary(0.20, 100.0, closes, trading_year=252, realised_sessions=3)
    assert np.isclose(out["implied_daily_sigma_pct"], 20 / np.sqrt(252))
    assert np.isclose(out["implied_expected_abs_pct"],
                      20 / np.sqrt(252) * np.sqrt(2 / np.pi))
    expected_abs_return = closes.pct_change().dropna().tail(3).abs().mean()
    assert np.isclose(out["realised_average_abs_points"], 100 * expected_abs_return)


def test_estimator_table_order_columns_and_missing_values():
    idx = pd.bdate_range("2026-01-02", periods=3)
    frame = pd.DataFrame({
        "C-C": [0.1, 0.11, 0.12], "Parkinson": [0.09, 0.10, 0.11],
        "Hodges-Tompkins": [0.10, 0.11, 0.12], "YZ": [0.11, 0.12, 0.13],
        "EWMA_halflife": [0.12, 0.13, 0.14], "Mean": [0.105, 0.115, 0.125],
    }, index=idx)
    table = rv.build_estimator_table({5: frame}, [5, 10])
    assert tuple(table.index) == rv.ESTIMATOR_ROWS
    assert list(table.columns) == [5, 10]
    assert np.isclose(table.loc["Mean Intra", 5], np.mean([0.11, 0.12, 0.13]))
    assert table[10].isna().all()
    integrated = rv.integrated_variance_table(table, annualisation_basis=252)
    assert np.isclose(integrated.loc["C-C", 5], table.loc["C-C", 5] ** 2 * 5 / 252)


def test_compressed_sample_summary():
    idx = pd.bdate_range("2025-01-02", periods=100)
    total = pd.DataFrame({
        5: np.linspace(0.10, 0.061, 100), 10: np.linspace(0.11, 0.076, 100),
        20: np.full(100, 0.10), 30: np.full(100, 0.10),
    }, index=idx)
    shape = rv.rv_shape_history(total, percentile_window=80, percentile_min_periods=20)
    empty = rv.empty_hf_curves([5, 10, 20, 30])
    curves = rv.HFCurves(total, total.copy(), empty.jump_share, empty.overnight_share,
                         empty.continuous_share)
    summary = rv.build_summary(shape_history=shape, movement={}, hf_curves=curves)
    assert np.isclose(summary["slope_5_20"], 0.61)
    assert np.isclose(summary["slope_10_30"], 0.76)
    assert summary["regime"] == "Front-end RV compressed"
