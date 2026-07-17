"""skewlab.charts.vix_distribution — ^VIX & ^VVIX empirical close distributions.

For each index: a probability histogram (bars, % of days in each price bin) plus the
cumulative distribution (line, right axis). The bin holding the latest close is outlined
so you can read off "how extended is vol right now" at a glance. Independent of the skew
sliders, so it ignores `cs`.
"""
from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .. import theme

_HILITE = "#f2c200"


def _panel(fig, dist, label, row):
    df, cur_close, cur_bin = dist
    edge_c = [_HILITE if b == cur_bin else "rgba(0,0,0,0)" for b in df["price_bin"]]
    edge_w = [3 if b == cur_bin else 0 for b in df["price_bin"]]
    # histogram (probability per bin)
    fig.add_trace(go.Bar(
        x=df["price_bin"], y=df["prob"], customdata=df["count"], name=f"{label} histogram",
        marker=dict(color="#2f6feb", line=dict(color=edge_c, width=edge_w)), opacity=0.9,
        hovertemplate="<b>%{x}</b><br>share %{y:.1%}<br>days %{customdata}<extra></extra>"),
        row=row, col=1, secondary_y=False)
    # cumulative distribution (line, right axis)
    fig.add_trace(go.Scatter(
        x=df["price_bin"], y=df["cum_prob"], mode="lines", name=f"{label} cumulative",
        line=dict(color="#ef553b", width=2),
        hovertemplate="<b>%{x}</b><br>cum prob %{y:.1%}<extra></extra>"),
        row=row, col=1, secondary_y=True)
    if cur_bin is not None:
        fig.add_annotation(x=cur_bin, y=1.0, yref=f"y{2*row} domain" if row > 1 else "y2 domain",
                           text=f"last {cur_close:.1f}", showarrow=True, arrowhead=2, ax=0, ay=-22,
                           font=dict(size=10, color="#9a7a00"), row=row, col=1, secondary_y=True)
    fig.update_yaxes(title_text="share of days", tickformat=".0%", row=row, col=1, secondary_y=False)
    fig.update_yaxes(title_text="cumulative", tickformat=".0%", range=[0, 1.02],
                     row=row, col=1, secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text=f"{label} close range", tickangle=45, row=row, col=1)


def _figure(panels, title):
    """panels: list of (dist_tuple, label). Returns a 1-col, secondary-y subplot, or None."""
    panels = [(d, lbl) for d, lbl in panels if d is not None]
    if not panels:
        return None
    fig = make_subplots(rows=len(panels), cols=1, vertical_spacing=0.18,
                        specs=[[{"secondary_y": True}] for _ in panels],
                        subplot_titles=[f"{lbl} — price distribution" for _, lbl in panels])
    for i, (dist, lbl) in enumerate(panels, start=1):
        _panel(fig, dist, lbl, i)
    fig.update_layout(title=title, template=theme.TEMPLATE, height=360 * len(panels), bargap=0.06,
                      margin=dict(t=64, b=56, r=64),
                      legend=dict(orientation="h", yanchor="top", y=-0.08, x=0.5, xanchor="center",
                                  font=dict(size=10)))
    return fig


def make(snap, cs):
    """Full-history VIX & VVIX close distributions."""
    return _figure([(snap.vix_dist, "VIX"), (snap.vvix_dist, "VVIX")],
                   "VIX / VVIX close distribution (full history) — histogram + cumulative")


def make_since(snap, cs):
    """Same, but only since cfg.vix_dist_since (hidden when that date is None)."""
    if not snap.since_when:
        return None
    return _figure([(snap.vix_dist_since, "VIX"), (snap.vvix_dist_since, "VVIX")],
                   f"VIX / VVIX close distribution since {snap.since_when} — histogram + cumulative")
