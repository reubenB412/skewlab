"""skewlab.charts.position — net Greeks roll-up + the position bar chart.

`position_risk` and `pnl_decomp` are pure helpers reused by analysis.py and pnl.py.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .. import model, theme

DC = 365.0  # day-count for the 1-day theta bump


def _vol(snap, cs, K):
    return float(snap.curve_vol(np.array([K]), cs)[0])


def position_risk(snap, cs):
    """Net position-weighted Greeks today (contracts; 100x multiplier)."""
    net = dict(delta=float(snap.shares or 0), gamma=0.0, vega_pt=0.0, theta=0.0, extrinsic=0.0)
    rows = []
    for K, typ, u in (snap.positions or []):
        is_call = str(typ).upper().startswith("C")
        sig = _vol(snap, cs, K)
        price, delta, gamma, vega, theta_day = model.bs_all(
            snap.spot, K, snap.t, snap.r, snap.q, sig, is_call)  # closed-form Greeks
        intrinsic = max(snap.spot - K, 0.0) if is_call else max(K - snap.spot, 0.0)
        m = u * 100.0
        net["delta"] += m * delta
        net["gamma"] += m * gamma           # shares per $1 (per contract: 100·gamma)
        net["vega_pt"] += m * vega * 0.01    # $ per 1 vol point
        net["theta"] += m * theta_day        # $ per calendar day
        net["extrinsic"] += m * (price - intrinsic)
        rows.append((K, "C" if is_call else "P", u))
    net["rows"] = rows
    return net


def pnl_decomp(snap, cs):
    """T vs previous-obs P&L, attributed via previous-date Greeks."""
    if snap.chain_prev is None or snap.prev_poly is None:
        return None
    from ..data import _core
    Sp, rp, qp, tp, dtep, atfp, fwdp = _core(snap.chain_prev, snap.cfg.day_count)
    dS = snap.spot - Sp
    days = (tp - snap.t) * DC
    b = dict(share=float(snap.shares or 0) * dS, delta=0.0, gamma=0.0, theta=0.0, vega=0.0, actual_opt=0.0)
    pv = lambda K: float(model.curve_vol(np.array([K]), snap.prev_poly, snap.grid_strikes, snap.forward,
                                         snap.one_sd, snap.z_grid, cs.slope_left, cs.slope_right,
                                         cs.wings_on)[0])
    for K, typ, u in (snap.positions or []):
        is_call = str(typ).upper().startswith("C")
        m = u * 100.0
        sigT = _vol(snap, cs, K)
        sigP = pv(K)
        pT, _, _, _, _ = model.bs_all(snap.spot, K, snap.t, snap.r, snap.q, sigT, is_call)
        # previous-date analytic Greeks drive the first-order attribution
        pP, dP, gammaP, vegaP, thetaP_day = model.bs_all(Sp, K, tp, rp, qp, sigP, is_call)
        b["actual_opt"] += m * (pT - pP)
        b["delta"] += m * dP * dS
        b["gamma"] += m * 0.5 * gammaP * dS * dS
        b["theta"] += m * thetaP_day * days
        b["vega"] += m * vegaP * (sigT - sigP)
    b["residual"] = b["actual_opt"] - (b["delta"] + b["gamma"] + b["theta"] + b["vega"])
    b["realized_vol"] = b["gamma"] + b["theta"]
    b["total"] = b["share"] + b["actual_opt"]
    b["days"], b["dS"] = days, dS
    return b


def make(snap, cs, title_suffix=""):
    net = position_risk(snap, cs)
    rows = net["rows"]
    fig = go.Figure()
    if not rows:
        fig.add_annotation(text="No position legs defined (set cfg.positions / cfg.shares)",
                           showarrow=False, font=dict(size=14))
        fig.update_layout(title=f"{snap.symbol} position & net Greeks {snap.date}",
                          template=theme.TEMPLATE, height=440)
        return fig
    Ks = [r[0] for r in rows]
    us = [r[2] for r in rows]
    typ = [r[1] for r in rows]
    colors = ["#2ca02c" if u >= 0 else "#d62728" for u in us]
    labels = [f"{t} {'+' if u > 0 else ''}{u}" for t, u in zip(typ, us)]
    fig.add_trace(go.Bar(x=Ks, y=us, marker_color=colors, text=labels, textposition="outside"))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.add_vline(x=snap.forward, line=dict(color="green", dash="dot"),
                  annotation_text="forward", annotation_position="top")
    txt = (f"net Δ {net['delta']:,.0f} sh<br>Γ {net['gamma']:,.1f} sh/$<br>"
           f"Θ {net['theta']:,.0f} $/day<br>Vega {net['vega_pt']:,.0f} $/volpt<br>"
           f"time prem {net['extrinsic']:,.0f} $")
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99, align="left", showarrow=False,
                       text=txt, bgcolor="rgba(255,255,255,0.78)", bordercolor="lightgray",
                       borderwidth=1, font=dict(size=11))
    fig.update_layout(title=f"{snap.symbol} position & net Greeks {snap.date}{title_suffix}",
                      xaxis_title="strike", yaxis_title="net units (+long / -short)",
                      template=theme.TEMPLATE, height=440, margin=dict(t=60, b=60), showlegend=False)
    return fig
