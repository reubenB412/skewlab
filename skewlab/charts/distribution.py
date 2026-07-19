"""skewlab.charts.distribution — Breeden-Litzenberger implied density vs flat-sheet."""
from __future__ import annotations

import dataclasses

import numpy as np
import plotly.graph_objects as go

from .. import model, theme


def make(snap, cs, title_suffix=""):
    # Density is computed from the SMOOTH SVI fit (wings off), regardless of the skew-curve
    # wing display. The linear tail extrapolation is a display/pricing convenience; its slope
    # kink at the grid ends would otherwise inject non-physical spikes into d2C/dK2 (and hence
    # the risk-neutral density). The fitted smile is the right source for the implied measure.
    cs_smooth = dataclasses.replace(cs, wings_on=False)
    fine = snap.fine_strikes(False)
    calls_skew = model.bs_call_vec(fine, snap.curve_vol(fine, cs_smooth), snap.spot, snap.t, snap.r, snap.q)
    calls_flat = model.bs_call_vec(fine, np.full_like(fine, cs.atf), snap.spot, snap.t, snap.r, snap.q)
    x_pdf = fine[2:-2]
    pdf_skew = model.implied_pdf(fine, calls_skew, snap.r, snap.t)[2:-2]
    pdf_flat = model.implied_pdf(fine, calls_flat, snap.r, snap.t)[2:-2]
    st = model.dist_stats(x_pdf, pdf_skew)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=snap.mkt_pdf_x, y=snap.mkt_pdf_y, mode="lines", name="market (original)",
                             line=dict(width=2, color="rgba(120,120,120,0.6)", dash="dot")))
    fig.add_trace(go.Scatter(x=x_pdf, y=pdf_skew, mode="lines", name="implied (skew)",
                             line=dict(width=2.5, color="#636efa"), fill="tozeroy",
                             fillcolor="rgba(99,110,250,0.18)"))
    fig.add_trace(go.Scatter(x=x_pdf, y=pdf_flat, mode="lines", name="flat-sheet (lognormal)",
                             line=dict(width=2, color="black", dash="dash")))
    fig.add_vline(x=snap.forward, line=dict(color="green", dash="dot"),
                  annotation_text="forward", annotation_position="top")
    stats_txt = (f"mean {st['mean']:.1f}<br>median {st['median']:.1f}<br>mode {st['mode']:.1f}"
                 f"<br>std {st['std']:.1f}<br>skew {st['skew']:+.2f}<br>kurt {st['kurt']:+.2f}")
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99, align="left", showarrow=False,
                       text=stats_txt, bgcolor="rgba(255,255,255,0.7)", bordercolor="lightgray",
                       borderwidth=1, font=dict(size=11))
    fig.update_layout(title=f"{snap.symbol} implied distribution {snap.date}{title_suffix}",
                      xaxis_title="terminal price", yaxis_title="density",
                      template=theme.TEMPLATE, height=460, margin=dict(t=56, b=52, r=170),
                      legend=theme.LEGEND_SIDE)
    pad = 0.05
    fig.update_xaxes(range=[snap.forward + (snap.z_grid[0] + pad) * snap.one_sd,
                            snap.forward + (snap.z_grid[-1] - pad) * snap.one_sd])
    return fig
