"""skewlab.charts.strike_vol_change — per-strike vol move, today vs the previous obs."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .. import model, theme


def make(snap, cs, title_suffix=""):
    fig = go.Figure()
    if snap.prev_poly is None:
        fig.add_annotation(text="Set prev_date (or pin-lookback) to see strike vol changes",
                           showarrow=False, font=dict(size=14))
        fig.update_layout(title=f"{snap.symbol} Strike Vol Changes {snap.date}",
                          template=theme.TEMPLATE, height=440)
        return fig
    strikes = snap.fine_strikes(cs.wings_on, n=snap.cfg.n_strikes)
    today = snap.curve_vol(strikes, cs)
    prev = model.curve_vol(strikes, snap.prev_poly, snap.grid_strikes, snap.forward, snap.one_sd,
                           snap.z_grid, cs.slope_left, cs.slope_right, cs.wings_on)
    dvol = (today - prev) * 100
    colors = np.where(dvol >= 0, "#2ca02c", "#d62728")
    fig.add_trace(go.Bar(x=strikes, y=dvol, marker_color=colors, name="Δ strike vol"))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    fig.add_vline(x=snap.forward, line=dict(color="green", dash="dot"),
                  annotation_text="forward", annotation_position="top")
    fig.update_layout(title=f"{snap.symbol} Strike Vol Changes {snap.date} vs {snap.prev_label}{title_suffix}",
                      xaxis_title="strike", yaxis_title="Δ implied vol (vol pts)",
                      template=theme.TEMPLATE, height=440, margin=dict(t=60, b=60), showlegend=False)
    return fig
