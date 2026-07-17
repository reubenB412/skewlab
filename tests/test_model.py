"""Pure quant-engine tests: put-call parity, risk-neutral density, no-arbitrage, greeks."""
import numpy as np

from skewlab import model


def test_put_call_parity():
    S, K, T, r, q, sig = 100.0, 105.0, 0.5, 0.03, 0.01, 0.20
    c = model.bs_all(S, K, T, r, q, sig, True)[0]
    p = model.bs_all(S, K, T, r, q, sig, False)[0]
    assert abs((c - p) - (S * np.exp(-q * T) - K * np.exp(-r * T))) < 1e-8


def test_breeden_litzenberger_density_integrates_to_one():
    # Constant-vol (lognormal) sheet -> the recovered risk-neutral density must integrate ~1.
    S, T, r, sig = 100.0, 0.5, 0.02, 0.20
    K = np.linspace(30.0, 220.0, 2000)
    calls = model.bs_call_vec(K, np.full_like(K, sig), S, T, r, 0.0)
    pdf = model.implied_pdf(K, calls, r, T)
    area = float(np.sum((pdf[:-1] + pdf[1:]) / 2.0 * np.diff(K)))   # trapezoid
    assert 0.97 < area < 1.03


def test_call_delta_monotonic_in_strike():
    Ks = np.linspace(80.0, 130.0, 60)
    dc = [model.bs_price_delta(100.0, K, 0.5, 0.02, 0.0, 0.20, True)[1] for K in Ks]
    # call delta is (weakly) decreasing as strike rises
    assert all(a >= b - 1e-9 for a, b in zip(dc[:-1], dc[1:]))
    assert dc[0] > dc[-1]


def test_svi_fit_is_arbitrage_free_on_a_calm_smile():
    F, T = 100.0, 0.08
    k = np.array([-0.20, -0.13, -0.06, 0.0, 0.06, 0.13, 0.20])
    Ks = F * np.exp(k)
    vols = np.array([0.26, 0.22, 0.19, 0.17, 0.155, 0.150, 0.152])   # put-skewed
    fit = model.fit_svi(Ks, vols, F, T)
    min_g, _ = model.svi_min_butterfly_g(fit.svi_params, np.linspace(-0.5, 0.5, 400))
    assert min_g > -1e-6           # Durrleman g >= 0 => no butterfly arbitrage


def test_rv_atmf_straddle_value_and_guards():
    F, sig, dte = 700.0, 0.20, 30
    td = round(dte * 5 / 7)
    expected = 0.7978845608 * F * sig * np.sqrt(td / 252.0)
    assert abs(model.rv_atmf_straddle(F, sig, dte, 252.0) - expected) < 1e-6
    assert np.isnan(model.rv_atmf_straddle(F, 0.0, dte))     # non-positive vol
    assert np.isnan(model.rv_atmf_straddle(0.0, sig, dte))   # bad forward
    assert np.isnan(model.rv_atmf_straddle(F, None, dte))    # None
