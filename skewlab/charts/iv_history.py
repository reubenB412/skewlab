"""skewlab.charts.iv_history — 2x2 vol-trader regime panel (gated on use_iv_history)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import model, theme


def _delta_bucket_ivs(snap, cs):
    Ks = np.linspace(snap.grid_strikes[0], snap.grid_strikes[-1], 200)
    sig = snap.curve_vol(Ks, cs)
    dC = np.array([model.bs_price_delta(snap.spot, K, snap.t, snap.r, snap.q, s, True)[1]
                   for K, s in zip(Ks, sig)])
    dP = np.array([model.bs_price_delta(snap.spot, K, snap.t, snap.r, snap.q, s, False)[1]
                   for K, s in zip(Ks, sig)])
    pick = lambda d, tgt: float(sig[int(np.argmin(np.abs(d - tgt)))])
    return dict(c25=pick(dC, 0.25), p25=pick(dP, -0.25), c10=pick(dC, 0.10), p10=pick(dP, -0.10))


def make(snap, cs):
    if snap.iv_history is None or getattr(snap.iv_history, "empty", True):
        return None
    atm = (snap.iv_atm.dropna() * 100.0) if snap.iv_atm is not None else pd.Series(dtype=float)
    if len(atm) == 0:
        f = go.Figure()
        f.add_annotation(text="IV history loaded but the ATM series is empty — see the [iv-history] "
                              "line at startup (no rows returned for this symbol/date range).",
                         showarrow=False, font=dict(size=13))
        f.update_layout(title=f"{snap.symbol} IV history & regime", template=theme.TEMPLATE, height=320)
        return f
    idx = pd.DatetimeIndex(atm.index)
    cols = snap.iv_history.columns
    have25 = {"25d_put", "25d_call"}.issubset(cols)
    have10 = {"10d_put", "10d_call"}.issubset(cols)
    rr25 = (snap.iv_history["25d_put"] - snap.iv_history["25d_call"]).reindex(atm.index) * 100.0 if have25 else None
    rr10 = (snap.iv_history["10d_put"] - snap.iv_history["10d_call"]).reindex(atm.index) * 100.0 if have10 else None
    rv = (snap.iv_rv.reindex(atm.index) * 100.0) if snap.iv_rv is not None else None

    today_ts = pd.to_datetime(snap.date)
    today_atm = snap.atf * 100.0
    try:
        db = _delta_bucket_ivs(snap, cs)
        today_rr = (db["p25"] - db["c25"]) * 100.0
    except Exception:
        today_rr = float("nan")

    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.14, horizontal_spacing=0.09,
                        subplot_titles=("ATM implied vs realized (carry / VRP)", "ATM vol regime",
                                        "Skew over time (25Δ / 10Δ risk reversal)", "Vol vs skew regime"))
    star = dict(color="black", size=11, symbol="star")

    cur_vrp = float("nan")
    if rv is not None and rv.dropna().size:
        fig.add_trace(go.Scatter(x=idx, y=rv.values, name="realized", line=dict(color="gray", width=1.5)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=idx, y=atm.values, name="implied (ATM)",
                                 line=dict(color="#636efa", width=2), fill="tonexty",
                                 fillcolor="rgba(44,160,44,0.12)"), row=1, col=1)
        cur_vrp = today_atm - float(rv.dropna().iloc[-1])
    else:
        fig.add_trace(go.Scatter(x=idx, y=atm.values, name="implied (ATM)",
                                 line=dict(color="#636efa", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[today_ts], y=[today_atm], mode="markers", name="today", marker=star),
                  row=1, col=1)

    fig.add_hrect(y0=float(atm.min()), y1=float(atm.max()), fillcolor="rgba(99,110,250,0.06)",
                  line_width=0, row=1, col=2)
    fig.add_trace(go.Scatter(x=idx, y=atm.values, name="ATM vol", showlegend=False,
                             line=dict(color="#636efa", width=1.5)), row=1, col=2)
    fig.add_hline(y=float(atm.median()), line=dict(color="gray", dash="dash"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[today_ts], y=[today_atm], mode="markers", showlegend=False, marker=star),
                  row=1, col=2)
    p_atm = model.pctile(atm.values, today_atm)

    p_rr = float("nan")
    if rr25 is not None:
        fig.add_trace(go.Scatter(x=idx, y=rr25.values, name="25Δ RR", line=dict(color="#d62728", width=1.8)),
                      row=2, col=1)
        fig.add_hline(y=float(np.nanmedian(rr25.values)), line=dict(color="gray", dash="dash"), row=2, col=1)
        fig.add_trace(go.Scatter(x=[today_ts], y=[today_rr], mode="markers", showlegend=False, marker=star),
                      row=2, col=1)
        p_rr = model.pctile(rr25.values, today_rr)
    if rr10 is not None:
        fig.add_trace(go.Scatter(x=idx, y=rr10.values, name="10Δ RR",
                                 line=dict(color="#ff7f0e", width=1.2, dash="dot")), row=2, col=1)

    resid = float("nan")
    if rr25 is not None:
        x, y = atm.values, rr25.reindex(atm.index).values
        m = ~(np.isnan(x) | np.isnan(y))
        fig.add_trace(go.Scatter(x=x[m], y=y[m], mode="markers", name="days",
                                 marker=dict(size=5, color=np.arange(int(m.sum())),
                                             colorscale="Viridis", showscale=False)), row=2, col=2)
        if m.sum() > 2:
            b1, b0 = np.polyfit(x[m], y[m], 1)
            xs = np.linspace(float(np.nanmin(x[m])), float(np.nanmax(x[m])), 50)
            fig.add_trace(go.Scatter(x=xs, y=b1 * xs + b0, mode="lines", showlegend=False,
                                     line=dict(color="gray", dash="dash")), row=2, col=2)
            resid = today_rr - (b1 * today_atm + b0)
        fig.add_trace(go.Scatter(x=[today_atm], y=[today_rr], mode="markers", showlegend=False, marker=star),
                      row=2, col=2)
    fig.update_xaxes(title_text="ATM vol (%)", row=2, col=2)
    fig.update_yaxes(title_text="25Δ RR (pts)", row=2, col=2)

    bits = [f"ATM {today_atm:.1f}% ({p_atm:.0f}pct)"]
    if not np.isnan(cur_vrp):
        bits.append(f"VRP {cur_vrp:+.1f} ({'sell' if cur_vrp > 0 else 'buy'}-vol carry)")
    if rr25 is not None:
        bits.append(f"25Δ RR {today_rr:+.1f} ({p_rr:.0f}pct)")
    if not np.isnan(resid):
        bits.append(f"skew {'rich' if resid > 0 else 'cheap'} for vol ({resid:+.1f} vs fit)")
    fig.update_layout(title=f"{snap.symbol} IV history & regime — " + " | ".join(bits),
                      template=theme.TEMPLATE, height=730, margin=dict(t=60, b=70),
                      legend=dict(orientation="h", yanchor="top", y=-0.06, x=0.5, xanchor="center",
                                  font=dict(size=10)))
    return fig
