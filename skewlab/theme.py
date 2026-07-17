"""skewlab.theme — shared plotly styling (one registration, reused by every chart)."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
TEMPLATE = "moontower"
LEGEND_SIDE = dict(orientation="v", yanchor="top", y=0.99, x=1.012, xanchor="left",
                   font=dict(size=10), bgcolor="rgba(255,255,255,0)")
TERM_COLORS = {10: "#9467bd", 90: "#17becf"}

if TEMPLATE not in pio.templates:
    pio.templates[TEMPLATE] = go.layout.Template(layout=dict(
        font=dict(family=FONT, size=12, color="#374151"),
        title=dict(font=dict(family=FONT, size=15, color="#111827"), x=0.012, xanchor="left"),
        paper_bgcolor="white", plot_bgcolor="white",
        colorway=["#2f6feb", "#ef553b", "#0ea5a4", "#f59e0b", "#8b5cf6", "#64748b"],
        xaxis=dict(gridcolor="#eef1f5", zerolinecolor="#e5e7eb", linecolor="#e5e7eb",
                   ticks="outside", tickcolor="#e5e7eb", ticklen=4),
        yaxis=dict(gridcolor="#eef1f5", zerolinecolor="#e5e7eb", linecolor="#e5e7eb",
                   ticks="outside", tickcolor="#e5e7eb", ticklen=4),
        legend=dict(font=dict(size=11)),
        margin=dict(t=54, b=46, l=58, r=24),
    ))
