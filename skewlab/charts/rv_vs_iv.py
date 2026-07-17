"""skewlab.charts.rv_vs_iv — realized-implied fair value vs the market (dumbbell).

Two stacked number-lines. Top: implied vol (pts). Bottom: ATM-forward straddle ($).
On each, the RV benchmark (fair value from the most-recent-close composite realized vol)
is a diamond; the market at the day's OPEN is a circle and NOW is a star, joined by a
dotted line (the intraday drift). The thick grey bar spans RV -> now (the discrepancy /
variance-risk premium the market is charging over realized).

Pure: make(snap, cs) -> Figure or None. Does not react to the sliders (reacts=False).
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import theme

RV_C, OPEN_C, NOW_C, GAP_C = "#0ea5a4", "#64748b", "#2f6feb", "#cbd5e1"


def _fin(x):
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def make(snap, cs, **kw):
    rv_iv = getattr(snap, "rv_iv", None)
    if not _fin(rv_iv):
        return None
    rv_iv = float(rv_iv) * 100.0
    now_iv = float(snap.atf) * 100.0
    open_iv = getattr(snap, "open_atf", None)
    open_iv = float(open_iv) * 100.0 if _fin(open_iv) else None

    rv_str = getattr(snap, "rv_straddle", None)
    now_str = getattr(snap, "now_straddle", None)
    open_str = getattr(snap, "open_straddle", None)
    open_ts = getattr(snap, "open_capture_ts", None)
    now_ts = getattr(snap, "now_capture_ts", None) or "now"
    asof = getattr(snap, "rv_asof", None)
    lb = getattr(snap, "rv_lookback", None)

    vrp = now_iv - rv_iv
    drift = (now_iv - open_iv) if open_iv is not None else None
    t1 = (f"Implied vol (pts) — fair RV {rv_iv:.1f}% ({lb}td @ {asof}) · "
          f"VRP now {vrp:+.1f}" + (f" · drift {drift:+.1f}" if drift is not None else ""))
    sgap = (now_str - rv_str) if (_fin(now_str) and _fin(rv_str)) else None
    t2 = ("ATM-forward straddle ($)"
          + (f" — fair {rv_str:.2f} · now {now_str:.2f} ({sgap:+.2f})"
             if (sgap is not None) else ""))

    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.30, subplot_titles=(t1, t2))
    seen = set()

    def _marker(row, val, base, color, sym, size, nd, unit):
        show = base not in seen
        seen.add(base)
        fig.add_trace(go.Scatter(
            x=[val], y=[0], mode="markers+text",
            marker=dict(color=color, size=size, symbol=sym, line=dict(color="white", width=1.2)),
            text=[f"{val:.{nd}f}"], textposition="top center", textfont=dict(size=11),
            name=base, legendgroup=base, showlegend=show,
            hovertemplate=f"{base}: %{{x:.{nd}f}} {unit}<extra></extra>"), row=row, col=1)

    def panel(row, rv_v, open_v, now_v, unit, nd):
        if not (_fin(rv_v) or _fin(now_v)):
            fig.add_annotation(text="straddle unavailable", xref=f"x{row} domain",
                               yref=f"y{row} domain", x=0.5, y=0.5, showarrow=False,
                               font=dict(color="#999", size=12))
            fig.update_yaxes(range=[-1, 1], showticklabels=False, showgrid=False,
                             zeroline=False, row=row, col=1)
            return
        if _fin(rv_v) and _fin(now_v):        # discrepancy span RV -> now
            fig.add_trace(go.Scatter(x=[rv_v, now_v], y=[0, 0], mode="lines",
                          line=dict(color=GAP_C, width=7), showlegend=False,
                          hoverinfo="skip"), row=row, col=1)
        if _fin(open_v) and _fin(now_v):      # intraday move open -> now
            fig.add_trace(go.Scatter(x=[open_v, now_v], y=[0, 0], mode="lines",
                          line=dict(color=NOW_C, width=2, dash="dot"), showlegend=False,
                          hoverinfo="skip"), row=row, col=1)
        if _fin(rv_v):
            _marker(row, rv_v, "RV fair", RV_C, "diamond", 15, nd, unit)
        if _fin(open_v):
            _marker(row, open_v, "open", OPEN_C, "circle", 13, nd, unit)
        if _fin(now_v):
            _marker(row, now_v, "now", NOW_C, "star", 17, nd, unit)
        vals = [v for v in (rv_v, open_v, now_v) if _fin(v)]
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.35, hi * 0.02, 0.1)
        fig.update_xaxes(range=[lo - pad, hi + pad], title_text=unit, row=row, col=1)
        fig.update_yaxes(range=[-1, 1.2], showticklabels=False, showgrid=False,
                         zeroline=False, row=row, col=1)

    panel(1, rv_iv, open_iv, now_iv, "vol pts", 1)
    panel(2, rv_str, open_str, now_str, "$", 2)

    fig.update_layout(template=theme.TEMPLATE, height=430, margin=dict(t=70, b=48, r=150),
                      title=f"{snap.symbol} — realized-implied fair value vs market · {snap.date}",
                      legend=theme.LEGEND_SIDE)
    return fig
