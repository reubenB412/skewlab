"""Net-liquidity projection: continuous compounding + annuity-due deposits."""
import numpy as np

from skewlab import model


def test_pure_continuous_compounding():
    df = model.net_liquidity_projection(100_000, 0.10, 10, contribution=0)
    assert len(df) == 21                                  # t0 + 20 half-years
    assert abs(df.net_liq.iloc[0] - 100_000) < 1e-6       # t=0 -> principal
    assert abs(df.net_liq.iloc[-1] - 100_000 * np.exp(1.0)) < 1e-2   # 10y -> P*e^(r*10)
    assert df.added.abs().sum() == 0


def test_annuity_due_matches_closed_form():
    P, r, yrs, C, dt = 100_000.0, 0.08, 10.0, 10_000.0, 1.0
    df = model.net_liquidity_projection(P, r, yrs, contribution=C, contrib_freq_years=dt)
    N = int(round(yrs / dt))                              # 10 deposits at t = 0..9 (annuity-due)
    fv_principal = P * np.exp(r * yrs)
    fv_deposits = C * np.exp(r * dt) * (np.exp(r * dt * N) - 1) / (np.exp(r * dt) - 1)
    assert abs(df.net_liq.iloc[-1] - (fv_principal + fv_deposits)) < 1e-2
    assert abs(df.net_liq.iloc[0] - (P + C)) < 1e-6       # first deposit lands at t=0
    assert abs(df.added.iloc[-1] - N * C) < 1e-6
    assert abs(df.gain.iloc[-1] - (df.net_liq.iloc[-1] - P - N * C)) < 1e-2


def test_six_month_deposit_frequency():
    df = model.net_liquidity_projection(100_000, 0.10, 5, contribution=5_000,
                                        contrib_freq_years=0.5)
    n_dep = int(round(5 / 0.5))                           # deposits at 0, .5, ..., 4.5
    assert abs(df.added.iloc[-1] - n_dep * 5_000) < 1e-6
    assert abs(df.net_liq.iloc[0] - 105_000) < 1e-6       # deposit already present at t0
