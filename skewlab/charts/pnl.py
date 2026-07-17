"""skewlab.charts.pnl — T/T-1 P&L decomposition waterfall."""
from __future__ import annotations

import plotly.graph_objects as go

from .. import theme
from . import position


def make(snap, cs, title_suffix=""):
    fig = go.Figure()
    b = position.pnl_decomp(snap, cs)
    if b is None:
        fig.add_annotation(text="Set prev_date (or pin-lookback) for the P&L decomposition",
                           showarrow=False, font=dict(size=14))
        fig.update_layout(title=f"{snap.symbol} P&L decomposition {snap.date}",
                          template=theme.TEMPLATE, height=440)
        return fig
    labels = ["Share", "Option Δ", "Gamma", "Theta", "Implied vol (Vega)", "Residual", "Total"]
    vals = [b["share"], b["delta"], b["gamma"], b["theta"], b["vega"], b["residual"], b["total"]]
    measure = ["relative"] * 6 + ["total"]
    fig.add_trace(go.Waterfall(
        x=labels, y=vals, measure=measure, connector=dict(line=dict(color="lightgray")),
        decreasing=dict(marker=dict(color="#d62728")), increasing=dict(marker=dict(color="#2ca02c")),
        totals=dict(marker=dict(color="#1f77b4"))))
    sub = (f"Δ{b['days']:.0f}d · spot {b['dS']:+.2f} · realized-vol P&L (Γ+Θ) {b['realized_vol']:+,.0f} · "
           f"implied-vol P&L {b['vega']:+,.0f}")
    fig.update_layout(title=f"{snap.symbol} P&L decomposition {snap.date} vs {snap.prev_label}{title_suffix}",
                      xaxis_title=sub, yaxis_title="P&L ($)",
                      template=theme.TEMPLATE, height=460, margin=dict(t=60, b=80), showlegend=False)
    return fig
