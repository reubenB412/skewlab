# Methodology

The maths behind skewlab. All pricing is Black–Scholes with a continuous dividend yield
`q`; time to expiry `T` is in years (calendar/365 unless noted).

## 1. Forward and at-the-forward vol

The forward is identified from put–call parity: the ATM-forward strike is the one that
minimises `|mid_call − mid_put|`. With rate `r` and dividend `q`,

```
F = S · e^(r−q)T
```

The at-the-forward vol (ATF) is anchored on the median implied vol of the strikes nearest
`F`, which is robust to a single bad ATM quote. The one-standard-deviation move is
`σ₁ = F · σ_ATF · √T`, and the skew is seeded on a grid of `z ∈ {−3…+3}` standard deviations
at strikes `F + z·σ₁`.

## 2. SVI smile (arbitrage-controllable)

Total implied variance `w(k) = σ_BS(k)² · T` as a function of log-moneyness `k = ln(K/F)` is
modelled with Gatheral's raw SVI:

```
w(k) = a + b · ( ρ (k − m) + √((k − m)² + s²) )
```

with parameters `(a, b, ρ, m, s)`. This gives linear wings, a single smooth minimum, and a
**closed-form butterfly test** via the Durrleman function `g(k)`; `g(k) ≥ 0` everywhere is
equivalent to no butterfly arbitrage. A degree-`n` polynomial fit is retained as a legacy
option, but it can wiggle between nodes and imply negative density — which the no-arbitrage
card then flags.

## 3. Risk-neutral density (Breeden–Litzenberger)

The risk-neutral density is the discounted second derivative of the call price with respect
to strike:

```
p(K) = e^(rT) · ∂²C/∂K²
```

skewlab evaluates this on a fine strike grid from the fitted smile. A correct density
integrates to 1 (a unit test asserts this) and is non-negative — the **butterfly** check is
exactly `p(K) ≥ 0`. The **calendar** check requires total variance to be non-decreasing in
maturity across the fetched tenors.

## 4. RV vs IV — realized-implied fair value

The variance-risk-premium panel turns a backward-looking realized vol into a forward-looking
*fair* option value and compares it to the market.

- **Fair vol** = `σ_RV`, the composite realized vol as of the most recent close.
- **Fair ATM-forward straddle**:

```
straddle ≈ √(2/π) · F · σ_RV · √τ ≈ 0.7979 · F · σ_RV · √τ
```

This is the leading (undiscounted, first-order) term of the exact ATMF straddle
`2·F·e^(−rT)·(2Φ(σ√τ/2) − 1)`; equivalently `0.7979 · σ₁`.

**Day-count consistency.** A realized vol from daily returns is annualised on *trading* days
(~252/yr), so `τ` here is measured in trading days too — `τ = (trading days to expiry)/252`,
**not** calendar `dte/365`. Pricing a 252-basis vol against a 365 calendar `τ` would bias the
straddle high by `√(365/252) − 1 ≈ +20%` on `τ` (~+9% on the straddle).

The panel then shows the market at the day's **open** and **now** against this fair value, so
the variance-risk premium `(σ_IV − σ_RV)` and the intraday drift `(σ_now − σ_open)` read at a
glance.

## 5. Composite realized-vol estimator

The realized-vol benchmark blends several estimators to trade off bias and sampling variance:
close-to-close, Parkinson, Hodges–Tompkins, Yang–Zhang, optional Garman–Klass /
Rogers–Satchell range estimators, an EWMA counterweight, and an optional GJR-GARCH(1,1,1)
conditional-vol column. The composite `Mean` uses fixed efficiency weights (inverse sampling
variance), which down-weight the noisy close-to-close estimator and up-weight the
range-based ones. The lookback in trading days is matched to the option's calendar horizon
via `trading_days_for_dte(dte) = round(dte · 5/7)`.
