"""skewlab.charts.vvix_vix_ratio — log(VVIX/VIX) convexity-stress regime over time.

High log(VVIX/VIX) = the market is paying up for vol-of-vol relative to vol itself, a
classic crash-convexity / tail-bid signal (favours selling VXX calls, buying SPX puts).
The high-convexity regime (signal above threshold) is shaded. Independent of the skew
sliders, so it ignores `cs`.
"""
from __future__ import annotations

import plotly.graph_objects as go

from .. import theme


def make(snap, cs):
    ratio = snap.vix_vvix_ratio
    if ratio is None or getattr(ratio, "empty", True):
        return None
    cfg = snap.cfg
    idx = ratio.index
    upper = float(cfg.ratio_upper_thres)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=ratio["log_VVIX_VIX"], mode="lines", name="log(VVIX / VIX)",
                             line=dict(width=1.5, color="#2f6feb"), opacity=0.8))
    if "ewma" in ratio:
        fig.add_trace(go.Scatter(x=idx, y=ratio["ewma"], mode="lines", name="EWMA (regime)",
                                 line=dict(width=2, color="#f59e0b")))
    if "perc" in ratio:
        fig.add_trace(go.Scatter(x=idx, y=ratio["perc"], mode="lines",
                                 name=f"rolling {int(cfg.ratio_percentile_thres*100)}th pctile",
                                 line=dict(width=1, dash="dot", color="#0ea5a4")))
    fig.add_hline(y=upper, line=dict(width=1.5, dash="dash", color="#ef553b"),
                  annotation_text=f"threshold {upper:.2f}", annotation_position="top left")

    # shade the high-convexity-stress regime (signal at/above the threshold band)
    if "high_regime" in ratio:
        shaded = ratio["log_VVIX_VIX"].where(ratio["high_regime"])
        fig.add_trace(go.Scatter(x=idx, y=[upper] * len(idx), mode="none", showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=idx, y=shaded, mode="none", fill="tonexty",
                                 fillcolor="rgba(239,85,59,0.18)", name="high-convexity stress"))

    fig.update_layout(
        title="VVIX / VIX ratio — high values favour selling VXX calls & buying SPX puts",
        xaxis_title="date", yaxis_title="log(VVIX / VIX)", template=theme.TEMPLATE, height=440,
        margin=dict(t=56, b=52, r=24),
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01, font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.6)"))
    return fig
