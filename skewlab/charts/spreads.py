"""skewlab.charts.spreads — 1-SD-wide vertical spread prices across strikes."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .. import model, theme


def make(snap, cs, title_suffix=""):
    fine = snap.fine_strikes(cs.wings_on, n=240)
    w = snap.one_sd
    vols = snap.curve_vol(fine, cs)
    calls = model.bs_call_vec(fine, vols, snap.spot, snap.t, snap.r, snap.q)
    puts = calls - snap.spot * np.exp(-snap.q * snap.t) + fine * np.exp(-snap.r * snap.t)
    call_spread = calls - np.interp(fine + w, fine, calls)
    put_spread = puts - np.interp(fine - w, fine, puts)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fine, y=call_spread, mode="lines", name="call spread (K, K+1SD)",
                             line=dict(width=2.5, color="#1f77b4")))
    fig.add_trace(go.Scatter(x=fine, y=put_spread, mode="lines", name="put spread (K, K-1SD)",
                             line=dict(width=2.5, color="#d62728")))
    fig.add_vline(x=snap.forward, line=dict(color="green", dash="dot"),
                  annotation_text="forward", annotation_position="top")
    fig.update_layout(title=f"{snap.symbol} vertical spread prices (1 SD wide) {snap.date}{title_suffix}",
                      xaxis_title="lower strike", yaxis_title="spread price ($)",
                      template=theme.TEMPLATE, height=440, margin=dict(t=56, b=52, r=180),
                      legend=theme.LEGEND_SIDE)
    return fig
