"""Pure RV-lookback versus IV-maturity term-structure chart."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .. import theme


def make(snap, cs=None):
    """Build the non-reactive term-structure figure from ``snap.rv_term``."""
    state = getattr(snap, "rv_term", None)
    if state is None or not state.available:
        return None
    fig = go.Figure()
    table = state.estimator_table
    styles = [
        ("Mean Volatility", "Mean RV", "#111827", 4, "solid"),
        ("Mean C-C", "Mean C-C", "#dc2626", 2, "dash"),
        ("Mean Intra", "Mean Intra", "#d97706", 2, "dash"),
        ("HF Total RV", "HF Total RV", "#0f9f9a", 3.5, "solid"),
    ]
    for row, label, color, width, dash in styles:
        if row not in table.index:
            continue
        y = pd.to_numeric(table.loc[row], errors="coerce") * 100.0
        if not y.notna().any():
            continue
        integrated = pd.to_numeric(
            state.integrated_variance_table.loc[row], errors="coerce"
        )
        custom = np.column_stack([np.asarray(table.columns, float), integrated.values])
        fig.add_trace(go.Scatter(
            x=np.asarray(table.columns, float), y=y.values, mode="lines+markers",
            name=label, connectgaps=False,
            line={"color": color, "width": width, "dash": dash},
            marker={"size": 7}, customdata=custom,
            hovertemplate=(f"{label}<br>RV lookback %{{customdata[0]:.0f}} completed sessions"
                           "<br>%{y:.2f}%<br>integrated variance %{customdata[1]:.5f}"
                           "<extra></extra>"),
        ))

    iv = state.iv_curve.copy()
    if not iv.empty and "atf_iv" in iv:
        iv = iv.reset_index().sort_values("requested_tenor")
        x = pd.to_numeric(iv["actual_dte"], errors="coerce")
        y = pd.to_numeric(iv["atf_iv"], errors="coerce") * 100.0
        custom = np.column_stack([
            pd.to_numeric(iv["requested_tenor"], errors="coerce"),
            pd.to_numeric(iv["integrated_variance"], errors="coerce"),
        ])
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name="Implied Vol",
            connectgaps=False, line={"color": "#3478d4", "width": 4},
            marker={"size": 8, "symbol": "diamond"}, customdata=custom,
            hovertemplate=("Implied Vol<br>actual maturity %{x:.0f} calendar days"
                           "<br>requested %{customdata[0]:.0f}D"
                           "<br>%{y:.2f}%<br>integrated variance %{customdata[1]:.5f}"
                           "<extra></extra>"),
        ))

    target = state.metadata.get("target_basis", 252)
    fig.update_layout(
        title={
            "text": (f"RV versus IV term structure — {snap.symbol}"
                     f"<br><sup>RV: backward-looking {target:g}-session basis · "
                     "IV: forward market quote on calendar/365 variance time</sup>"),
            "x": 0.01,
        },
        template=theme.TEMPLATE,
        height=455,
        hovermode="x unified",
        xaxis_title="RV lookback / IV maturity (days)",
        yaxis_title="Annualised volatility (%)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "right", "x": 1},
        margin={"l": 65, "r": 25, "t": 100, "b": 60},
    )
    return fig
