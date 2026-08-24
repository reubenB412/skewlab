"""Pure RV-lookback versus ATM-IV-maturity term-structure chart."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .. import theme


def _iv_points(snap, state):
    """Return aligned auxiliary tenors plus the trusted main-snapshot ATF anchor."""
    cols = [
        "requested_tenor", "actual_dte", "expiry", "observation_date",
        "is_aligned", "atf_iv", "calendar_year_fraction", "integrated_variance",
        "is_main_snapshot",
    ]
    iv = state.iv_curve.copy()
    if not iv.empty:
        iv = iv.reset_index()
        if "requested_tenor" not in iv:
            iv["requested_tenor"] = np.nan
        aligned = iv.get("is_aligned", pd.Series(True, index=iv.index)).fillna(False).astype(bool)
        iv = iv.loc[
            aligned
            & pd.to_numeric(iv.get("actual_dte"), errors="coerce").notna()
            & pd.to_numeric(iv.get("atf_iv"), errors="coerce").notna()
        ].copy()
        iv["is_main_snapshot"] = False
    else:
        iv = pd.DataFrame(columns=cols)

    main_dte = float(snap.dte)
    main_iv = float(snap.atf)
    main = pd.DataFrame([{
        "requested_tenor": float(getattr(snap.cfg, "target_dte", main_dte)),
        "actual_dte": main_dte,
        "expiry": pd.Timestamp(snap.date) + pd.Timedelta(days=main_dte),
        "observation_date": pd.Timestamp(snap.date),
        "is_aligned": True,
        "atf_iv": main_iv,
        "calendar_year_fraction": main_dte / float(snap.cfg.day_count),
        "integrated_variance": main_iv ** 2 * main_dte / float(snap.cfg.day_count),
        "is_main_snapshot": True,
    }])
    iv = pd.concat([iv, main], ignore_index=True, sort=False)
    iv["actual_dte"] = pd.to_numeric(iv["actual_dte"], errors="coerce")
    # Prefer the trusted main snapshot if an auxiliary request resolves to its actual DTE.
    return (
        iv.sort_values(["actual_dte", "is_main_snapshot"])
        .drop_duplicates("actual_dte", keep="last")
        .sort_values("actual_dte")
    )


def make(snap, cs=None):
    """Build the non-reactive term-structure figure from ``snap.rv_term``."""
    state = getattr(snap, "rv_term", None)
    if state is None or not state.available:
        return None
    fig = go.Figure()
    table = state.estimator_table
    styles = [
        ("Mean Volatility", "Mean RV", "#111827", 3.0, "solid"),
        ("HF Total RV", "HF Total RV", "#0f9f9a", 2.7, "solid"),
        ("Mean C-C", "Mean close-to-close", "#dc2626", 1.8, "dash"),
        ("Mean Intra", "Mean intraday", "#d97706", 1.8, "dash"),
    ]
    for rank, (row, label, color, width, dash) in enumerate(styles, start=10):
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
            marker={"size": 6}, legendrank=rank,
            customdata=custom,
            hovertemplate=(f"{label}<br>RV lookback %{{customdata[0]:.0f}} completed sessions"
                           "<br>%{y:.2f}%<br>integrated variance %{customdata[1]:.5f}"
                           "<extra></extra>"),
        ))

    iv = _iv_points(snap, state)
    if not iv.empty:
        x = pd.to_numeric(iv["actual_dte"], errors="coerce")
        y = pd.to_numeric(iv["atf_iv"], errors="coerce") * 100.0
        obs = pd.to_datetime(iv["observation_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        custom = np.column_stack([
            pd.to_numeric(iv["requested_tenor"], errors="coerce"),
            pd.to_numeric(iv["integrated_variance"], errors="coerce"),
            obs.fillna("unknown"),
            iv["is_main_snapshot"].map(
                lambda value: "main snapshot" if bool(value) else "tenor chain"
            ),
        ])
        labels = [""] * len(iv)
        labels[-1] = "ATM IV"
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers+text",
            name="ATM implied volatility (forward)", connectgaps=False,
            line={"color": "#2f6feb", "width": 4.5},
            marker={"size": 10, "symbol": "diamond", "color": "#2f6feb",
                    "line": {"color": "white", "width": 1.2}},
            text=labels, textposition="top right",
            textfont={"color": "#2f6feb", "size": 11},
            cliponaxis=False, customdata=custom, legendrank=1,
            hovertemplate=("<b>ATM implied volatility</b>"
                           "<br>actual maturity %{x:.0f} calendar days"
                           "<br>requested %{customdata[0]:.0f}D"
                           "<br>observation %{customdata[2]}"
                           "<br>source %{customdata[3]}"
                           "<br><b>%{y:.2f}%</b>"
                           "<br>integrated variance %{customdata[1]:.5f}"
                           "<extra></extra>"),
        ))

    target = state.metadata.get("target_basis", 252)
    fig.update_layout(
        title=f"{snap.symbol} RV vs ATM IV term structure · {snap.date}",
        template=theme.TEMPLATE,
        height=460,
        hovermode="x unified",
        xaxis_title="RV lookback / IV maturity (days)",
        yaxis_title="annualised volatility (%)",
        legend=theme.LEGEND_SIDE,
        margin={"l": 65, "r": 205, "t": 58, "b": 62},
    )
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.01, showarrow=False, align="left",
        text=(f"RV uses completed trailing sessions on a {float(target):g}-session basis; "
              f"blue diamonds are forward ATM IV on calendar/{float(snap.cfg.day_count):g}."),
        font={"size": 10, "color": "#64748b"}, bgcolor="rgba(255,255,255,0.78)",
    )
    return fig
