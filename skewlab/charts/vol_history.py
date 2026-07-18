"""skewlab.charts.vol_history — the IV-history vs realized views (in the IV-history section).

Two pure charts, both `reacts=False` (they don't depend on the sliders):

  make(snap, cs)             implied-vol history buckets (ATM, 25Δ/10Δ put+call) overlaid
                             on the composite realized-vol line — the daily "IV vs RV" story.
  make_estimators(snap, cs)  the composite realized-vol ESTIMATOR STACK: every estimator
                             column (C-C, Parkinson, YZ, EWMA, GARCH, …) with the blended
                             Mean emphasised — the plot `get_composite_realised_volatility`
                             draws on its own.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .. import theme

# (column, legend label, colour, dash, width) for the IV-history overlay
_IV_SERIES = [
    ("atm_iv",   "ATM IV",   "#2f6feb", "solid", 2.4),
    ("25d_put",  "25Δ put",  "#ef553b", "dash",  1.4),
    ("25d_call", "25Δ call", "#0ea5a4", "dash",  1.4),
    ("10d_put",  "10Δ put",  "#f59e0b", "dot",   1.3),
    ("10d_call", "10Δ call", "#8b5cf6", "dot",   1.3),
]


def has_history(snap):
    """True if there's any IV-history or realized-vol series to plot."""
    a = getattr(snap, "iv_atm", None)
    r = getattr(snap, "iv_rv", None)
    return ((a is not None and len(pd.Series(a).dropna()) > 0)
            or (r is not None and len(pd.Series(r).dropna()) > 0))


def has_estimators(snap):
    rv = getattr(snap, "rv_estimators", None)
    return rv is not None and not getattr(rv, "empty", True)


def _clip_start(df, start):
    """Return df restricted to index >= start (a date string / Timestamp), if given."""
    if start is None or df is None or df.empty:
        return df
    try:
        return df.loc[pd.to_datetime(start):]
    except Exception:
        return df


def make(snap, cs, start=None, **kw):
    from .. import inspect as _inspect          # reuse the same joined frame as DATA["vol_history"]
    vh = _inspect.vol_history_frame(snap)
    if vh is None or vh.empty:
        return None
    vh = _clip_start(vh.sort_index(), start)
    if vh is None or vh.empty:
        return None
    try:
        vh = vh.interpolate(method="time")       # bridge small gaps like the play-area plot
    except Exception:
        pass

    fig = go.Figure()
    for col, name, color, dash, w in _IV_SERIES:
        if col in vh.columns and vh[col].notna().any():
            s = vh[col].dropna()
            fig.add_trace(go.Scatter(x=s.index, y=s.values * 100.0, name=name, mode="lines",
                                     line=dict(color=color, width=w, dash=dash)))
    if "realized_vol" in vh.columns and vh["realized_vol"].notna().any():
        s = vh["realized_vol"].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values * 100.0, name="composite RV (Mean)",
                                 mode="lines", line=dict(color="#111827", width=2.6)))
    if not fig.data:
        return None
    fig.update_layout(title=f"{snap.symbol} — implied-vol history vs composite realized",
                      xaxis_title="date", yaxis_title="annualised vol (%)",
                      template=theme.TEMPLATE, height=430, margin=dict(t=54, b=46, r=150),
                      legend=theme.LEGEND_SIDE, hovermode="x unified")
    return fig


def make_estimators(snap, cs, start=None, **kw):
    rv = getattr(snap, "rv_estimators", None)
    if rv is None or getattr(rv, "empty", True):
        return None
    rv = rv.copy()
    rv.index = pd.DatetimeIndex(rv.index)
    rv = _clip_start(rv.sort_index(), start)
    if rv is None or rv.empty:
        return None
    others = [c for c in rv.columns if c != "Mean"]

    fig = go.Figure()
    for col in others:
        s = pd.to_numeric(rv[col], errors="coerce").dropna()
        if len(s):
            fig.add_trace(go.Scatter(x=s.index, y=s.values * 100.0, name=str(col), mode="lines",
                                     line=dict(width=1.1), opacity=0.72))
    if "Mean" in rv.columns:
        s = pd.to_numeric(rv["Mean"], errors="coerce").dropna()
        if len(s):
            fig.add_trace(go.Scatter(x=s.index, y=s.values * 100.0, name="Mean (composite)",
                                     mode="lines", line=dict(color="#111827", width=2.8)))
    if not fig.data:
        return None
    fig.update_layout(title=f"{snap.symbol} — realized-vol estimator stack "
                            f"({snap.rv_lookback or ''}td lookback)",
                      xaxis_title="date", yaxis_title="annualised vol (%)",
                      template=theme.TEMPLATE, height=400, margin=dict(t=54, b=46, r=150),
                      legend=theme.LEGEND_SIDE, hovermode="x unified")
    return fig
