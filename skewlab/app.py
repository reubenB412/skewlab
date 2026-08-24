"""skewlab.app — the Dash dashboard, built generically from the chart registry.

`build_app(snap)` wires sliders + scenario presets + Apply/Reset to every chart in
`charts.active(snap)`. Reacting charts redraw on each Apply; non-reacting ones (IV
history) render once. `serve(snap, ...)` runs it; `write_static_html(snap, outdir)` is
the no-Dash fallback.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import analysis, theme
from . import charts as charts_pkg
from .data import CurveState


def _opd_format_display_model(df, pct_cols=None, axis=0):
    """Build the local Dash display model matching ``opd._format_display`` semantics.

    Pandas Styler emits selector-based CSS in a ``<style>`` block, which is not reliable when
    embedded through Dash Markdown. Compute Styler's cell context and carry the resulting colours
    into the native Dash table as inline styles instead.
    """
    frame = pd.DataFrame(df).copy()
    pct_cols = [col for col in (pct_cols or []) if col in frame.columns]
    numeric_cols = frame.select_dtypes(include=[np.number]).columns.tolist()
    date_cols = frame.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    fmt = {}
    for col in frame.columns:
        if col in pct_cols and col in numeric_cols:
            fmt[col] = "{:.2%}"
        elif col in numeric_cols:
            fmt[col] = "{:.2f}"
        elif col in date_cols:
            fmt[col] = lambda x: x.strftime("%Y-%m-%d") if pd.notnull(x) else ""
        else:
            fmt[col] = "{}"

    def red_if_negative(value):
        return "color: red" if isinstance(value, (int, float)) and value < 0 else ""

    styled = (
        frame.style
        .format(fmt, na_rep="")
        .background_gradient(cmap="YlGnBu", axis=axis, subset=numeric_cols)
        .applymap(red_if_negative)
    )
    styled._compute()

    rows = []
    for row_no, (index, row) in enumerate(frame.iterrows()):
        cells = []
        for col_no, col in enumerate(frame.columns):
            value = row[col]
            if pd.isna(value):
                text = ""
                cell_style = {"backgroundColor": "#fff", "color": "#374151"}
            else:
                if col in pct_cols and col in numeric_cols:
                    text = f"{float(value):.2%}"
                elif col in numeric_cols:
                    text = f"{float(value):.2f}"
                elif col in date_cols:
                    text = pd.Timestamp(value).strftime("%Y-%m-%d")
                else:
                    text = str(value)
                props = dict(styled.ctx.get((row_no, col_no), ()))
                cell_style = {
                    "backgroundColor": props.get("background-color", "#fff"),
                    "color": (
                        "red"
                        if col in numeric_cols and float(value) < 0
                        else props.get("color", "#374151")
                    ),
                }
            cells.append({"text": text, "style": cell_style})
        rows.append({"index": str(index), "cells": cells})
    return {
        "column_name": str(frame.columns.name or ""),
        "index_name": str(frame.index.name or ""),
        "columns": [str(col) for col in frame.columns],
        "rows": rows,
    }


def _opd_format_display_table(html, df, pct_cols=None, axis=0):
    """Render the Styler-derived model as semantic Dash HTML with inline cell colours."""
    model = _opd_format_display_model(df, pct_cols=pct_cols, axis=axis)
    header = html.Thead([
        html.Tr([
            html.Th(model["column_name"], scope="col", className="rv-index-head"),
            *[html.Th(col, scope="col") for col in model["columns"]],
        ]),
        html.Tr([
            html.Th(model["index_name"], scope="col", className="rv-index-name"),
            *[html.Th("") for _ in model["columns"]],
        ]),
    ])
    body = html.Tbody([
        html.Tr([
            html.Th(row["index"], scope="row", className="rv-row-head"),
            *[html.Td(cell["text"], style=cell["style"]) for cell in row["cells"]],
        ])
        for row in model["rows"]
    ])
    return html.Div(html.Table([header, body]), className="rv-styler")


def _figures(snap, cs):
    """Build every active figure for a control state, keyed by chart key. Robust per-chart."""
    out = {}
    for c in charts_pkg.active(snap):
        try:
            fig = c.make(snap, cs)
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"{c.title} error: {e}", showarrow=False, font=dict(size=12))
            fig.update_layout(title=f"{snap.symbol} {c.title}", template=theme.TEMPLATE, height=320)
        if fig is not None:
            out[c.key] = fig
    return out


def build_app(snap):
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output, State

    cfg = snap.cfg
    HALF = cfg.half_iv_slider
    z_grid = list(snap.z_grid)
    SEED = {z: round(snap.grid_vols[i] * 100, 2) for i, z in enumerate(z_grid)}
    keys = [c.key for c in charts_pkg.active(snap)]
    titles = {c.key: c.title for c in charts_pkg.active(snap)}

    iv_id = lambda z: f"iv_{int(z)}"
    _grouplbl = {"fontSize": "11px", "fontWeight": 700, "letterSpacing": ".05em",
                 "textTransform": "uppercase", "color": "#64748b", "margin": "16px 0 6px"}
    _slbl = {"fontSize": "12px", "fontWeight": 600, "color": "#334155", "marginTop": "10px"}

    ctrl = [
        html.Div("Controls", style={"fontWeight": 800, "fontSize": "15px", "color": "#0f172a",
                                    "marginBottom": "12px"}),
        html.Div("Scenario preset", style={**_grouplbl, "marginTop": "0"}),
        dcc.Dropdown(id="scenario", clearable=False, value="market",
                     options=[{"label": ("Market (live sliders)" if k == "market" else k), "value": k}
                              for k in cfg.scenarios],
                     style={"fontSize": "12px", "marginBottom": "4px"}),
        html.Div("pick a regime to overlay vs today's market shadow; set back to Market to use "
                 "the sliders.", style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "10px"}),
        html.Div([html.Button("Apply changes", id="apply", n_clicks=0, className="mt-btn",
                              style={"marginRight": "8px", "background": "#2f6feb", "color": "white"}),
                  html.Button("Reset", id="reset", n_clicks=0, className="mt-btn",
                              style={"background": "#eef2f7", "color": "#334155"})],
                 style={"marginBottom": "4px"}),
        html.Div("IV by standard-deviation node (%)", style=_grouplbl),
    ]
    for z in z_grid:
        lbl = "ATF (0 SD)" if z == 0.0 else f"{int(z):+d} SD"
        s = SEED[z]
        ctrl += [html.Div(lbl, style=_slbl),
                 dcc.Slider(id=iv_id(z), min=round(max(0.5, s - HALF), 2), max=round(s + HALF, 2),
                            step=0.1, value=s, marks=None,
                            tooltip={"placement": "bottom", "always_visible": True})]
    # Wing controls apply to the POLYNOMIAL model only — SVI carries its own arbitrage-free
    # wings, so under SVI these are disabled (and 'wings on' is held True purely so the plot
    # range still extends past +/-3 SD to show SVI's own wings).
    _is_svi = (getattr(cfg, "skew_model", "svi") == "svi")
    _dis_lbl = {**_slbl, "color": "#94a3b8"} if _is_svi else _slbl
    ctrl += [html.Div("Wing extrapolation", style=_grouplbl)]
    if _is_svi:
        ctrl += [html.Div("SVI extrapolates its own arbitrage-free wings; the manual wing "
                          "controls apply to the polynomial model only.",
                          style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "6px"})]
    ctrl += [
        dcc.Checklist(id="wings",
                      options=[{"label": " wings on", "value": "on", "disabled": _is_svi}],
                      value=(["on"] if (cfg.wings_on or _is_svi) else []),
                      style={"fontSize": "13px", "color": ("#94a3b8" if _is_svi else "#334155")}),
        html.Div("slope L (vol pts / SD)", style=_dis_lbl),
        dcc.Slider(id="slope_l", min=-10, max=25, step=0.5, value=cfg.slope_left * 100, marks=None,
                   disabled=_is_svi, tooltip={"placement": "bottom", "always_visible": True}),
        html.Div("slope R (vol pts / SD)", style=_dis_lbl),
        dcc.Slider(id="slope_r", min=-10, max=25, step=0.5, value=cfg.slope_right * 100, marks=None,
                   disabled=_is_svi, tooltip={"placement": "bottom", "always_visible": True}),
    ]
    controls = html.Div(ctrl, className="mt-card", style={
        "flex": "0 0 280px", "padding": "18px", "position": "sticky", "top": "16px",
        "margin": "16px 0 16px 16px", "maxHeight": "calc(100vh - 32px)", "overflowY": "auto"})

    def _graph_card(gid):
        return html.Div(dcc.Graph(id=f"g_{gid}", config={"displaylogo": False,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}),
                        className="mt-card", style={"marginBottom": "14px", "padding": "6px 8px 8px"})

    # --- vol-history section: IV-history-vs-realized + RV estimator stack, own start-date ---
    _gcfg = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

    # --- realised-vol regime + estimator/IV term structure (non-reactive) ---
    def _finite(value):
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None

    def _ordinal(value):
        value = _finite(value)
        if value is None:
            return "percentile unavailable"
        n = round(value)
        suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix} percentile"

    def _percentile_rail(value):
        value = _finite(value)
        if value is None:
            return html.Div(style={"height": "6px", "background": "#e2e8f0", "borderRadius": "6px"})
        left = min(max(value, 0.0), 100.0)
        return html.Div([
            html.Div(style={"position": "absolute", "left": f"calc({left:.1f}% - 4px)",
                            "top": "-3px", "width": "8px", "height": "12px",
                            "borderRadius": "5px", "background": "#d97706",
                            "boxShadow": "0 0 0 2px #fff"}),
        ], style={"height": "6px", "background": "linear-gradient(90deg,#dbeafe,#99f6e4)",
                  "borderRadius": "6px", "position": "relative", "margin": "9px 0 7px"})

    def _shape_metric(label, value, percentile, *, vol_points=False, reference=None):
        value = _finite(value)
        shown = "—" if value is None else (f"{value * 100:+.2f} vol pts" if vol_points else f"{value:.2f}")
        detail = _ordinal(percentile) + (f" · {reference}" if reference else "")
        return html.Div([
            html.Div(label, style={"fontSize": "11px", "fontWeight": 700,
                                   "textTransform": "uppercase", "letterSpacing": ".035em",
                                   "color": "#64748b"}),
            html.Div(shown, style={"fontSize": "22px", "fontWeight": 800,
                                   "color": "#0f172a", "marginTop": "3px"}),
            _percentile_rail(percentile),
            html.Div(detail, style={"fontSize": "11px", "color": "#64748b"}),
        ], style={"border": "1px solid #e8edf3", "borderRadius": "10px",
                  "padding": "12px", "minWidth": "0"})

    def _movement_bar(label, pct, points, color, scale):
        pct, points = _finite(pct), _finite(points)
        width = 0.0 if points is None or scale <= 0 else min(100.0, 100.0 * points / scale)
        value = "—" if pct is None else f"{pct:.2f}%"
        if points is not None:
            value += f" · {points:.2f} points"
        return html.Div([
            html.Div([html.Span(label), html.Span(value, style={"fontWeight": 700})],
                     style={"display": "flex", "justifyContent": "space-between", "gap": "12px",
                            "fontSize": "11.5px", "color": "#64748b", "marginBottom": "5px"}),
            html.Div(html.Div(style={"height": "100%", "width": f"{width:.1f}%",
                                     "background": color, "borderRadius": "6px"}),
                     style={"height": "8px", "background": "#edf1f5", "borderRadius": "6px"}),
        ], style={"marginBottom": "10px"})

    rv_term_section = None
    rv_state = getattr(snap, "rv_term", None)
    if rv_state is not None and rv_state.available:
        sm = dict(rv_state.summary)
        implied_pts = _finite(sm.get("implied_expected_abs_points"))
        realised_pts = _finite(sm.get("realised_average_abs_points"))
        move_scale = max([x for x in (implied_pts, realised_pts) if x is not None] + [1e-12])
        sigma_pct = _finite(sm.get("implied_daily_sigma_pct"))
        gap_pts = _finite(sm.get("movement_gap_points"))
        gap_text = ("Implied-versus-realised movement gap unavailable." if gap_pts is None else
                    f"Implied expected absolute movement is {abs(gap_pts):.2f} points "
                    f"{'above' if gap_pts >= 0 else 'below'} the recent 5-session realised average.")
        movement = html.Div([
            html.Div("Daily movement comparison", style={"fontWeight": 800, "fontSize": "13px",
                                                           "marginBottom": "8px"}),
            html.Div([html.Span("Implied daily 1-sigma move", style={"color": "#64748b"}),
                      html.Span("—" if sigma_pct is None else f"{sigma_pct:.2f}%",
                                style={"fontWeight": 800, "color": "#2563eb"})],
                     style={"display": "flex", "justifyContent": "space-between", "fontSize": "12px",
                            "padding": "8px 10px", "background": "#eff6ff",
                            "borderRadius": "8px", "marginBottom": "10px"}),
            _movement_bar("Implied expected absolute daily move",
                          sm.get("implied_expected_abs_pct"), implied_pts, "#3478d4", move_scale),
            _movement_bar("Realised average absolute move · last 5 sessions",
                          sm.get("realised_average_abs_pct"), realised_pts, "#0f9f9a", move_scale),
            html.Div(gap_text, style={"fontSize": "11px", "color": "#64748b"}),
        ], style={"border": "1px solid #e8edf3", "borderRadius": "10px",
                  "padding": "13px", "marginBottom": "12px"})

        metrics = html.Div([
            _shape_metric("5d / 20d RV slope", sm.get("slope_5_20"), sm.get("slope_5_20_pct")),
            _shape_metric("10d / 30d RV slope", sm.get("slope_10_30"), sm.get("slope_10_30_pct")),
            _shape_metric("RV curvature", sm.get("curvature"), sm.get("curvature_pct"),
                          reference="stress ref > 1.10"),
            _shape_metric("RV acceleration · 3 sessions", sm.get("rv_acceleration"),
                          sm.get("rv_acceleration_pct"), vol_points=True),
        ], className="rv-metric-grid")

        table = rv_state.estimator_table.copy()
        rv_table = _opd_format_display_table(
            html, table, pct_cols=list(table.columns), axis=1
        )

        warnings = [html.Div(f"⚠ {w}") for w in rv_state.warnings]
        warning_box = (html.Div(warnings, style={"fontSize": "11px", "color": "#92400e",
                                                "background": "#fffbeb",
                                                "border": "1px solid #fde68a",
                                                "padding": "8px 10px",
                                                "borderRadius": "8px", "marginTop": "10px"})
                       if warnings else None)
        meta = rv_state.metadata
        source_line = (
            f"{meta.get('source', 'backend')} · {meta.get('sample_minutes', 5)}-minute variance · "
            f"source basis {meta.get('source_basis', '—')} → table basis {meta.get('target_basis', '—')} · "
            f"{meta.get('percentile_method', 'trailing history')}. Settled sessions only."
        )
        rv_fig = charts_pkg.rv_term_structure.make(snap, CurveState.market(snap))
        summary_card = html.Div([
            html.Div([
                html.Div("Realised-vol regime & term structure",
                         style={"fontWeight": 800, "fontSize": "15px", "color": "#0f172a"}),
                html.Div(f"{sm.get('regime', 'RV regime unavailable')} · "
                         f"{sm.get('regime_source', 'HF total')}",
                         style={"fontWeight": 700, "fontSize": "12px", "color": "#d97706"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "12px",
                      "alignItems": "center", "justifyContent": "space-between",
                      "flexWrap": "wrap"}),
            movement, metrics,
            html.Div(source_line, style={"fontSize": "10.5px", "color": "#64748b",
                                         "marginTop": "10px"}),
            warning_box,
        ], className="mt-card", style={"marginBottom": "14px", "padding": "18px"})
        table_card = html.Div([
            html.Div("Current RV estimator term structure",
                     style={"fontWeight": 800, "fontSize": "15px", "color": "#0f172a",
                            "marginBottom": "10px"}),
            rv_table,
        ], className="mt-card", style={"marginBottom": "14px", "padding": "18px",
                                         "overflow": "hidden"})
        chart_card = html.Div(
            dcc.Graph(figure=rv_fig, config=_gcfg) if rv_fig is not None else
            html.Div("Term-structure chart unavailable.",
                     style={"color": "#64748b", "padding": "20px"}),
            className="mt-card", style={"marginBottom": "14px", "padding": "6px 8px 8px"},
        )
        rv_term_section = html.Div([summary_card, table_card, chart_card])

    _VH_ON = charts_pkg.vol_history.has_history(snap) or charts_pkg.vol_history.has_estimators(snap)
    volhist_section = None
    if _VH_ON:
        volhist_section = html.Div([
            html.Div("Vol history — implied vs realized", style={
                "fontWeight": 800, "fontSize": "15px", "color": "#0f172a", "marginBottom": "4px"}),
            html.Div("Implied-vol history buckets vs the composite realized-vol Mean, plus the "
                     "realized-vol estimator stack. Set the x-axis start date and refresh.",
                     style={"fontSize": "11.5px", "color": "#94a3b8", "marginBottom": "12px"}),
            html.Div([
                html.Div([html.Div("X-axis start date", style={**_slbl, "marginTop": "0"}),
                          dcc.DatePickerSingle(id="vh_start", date="2026-01-01",
                                               display_format="YYYY-MM-DD")],
                         style={"marginRight": "12px"}),
                html.Button("Refresh", id="vh_refresh", n_clicks=0, className="mt-btn",
                            style={"background": "#0891b2", "color": "white", "alignSelf": "flex-end"}),
            ], style={"display": "flex", "gap": "10px", "alignItems": "flex-end", "marginBottom": "12px"}),
            dcc.Loading(dcc.Graph(id="g_iv_vs_rv", config=_gcfg), type="dot"),
            dcc.Loading(dcc.Graph(id="g_rv_estimators", config=_gcfg), type="dot"),
        ], className="mt-card", style={"marginTop": "4px", "padding": "18px", "marginBottom": "14px"})

    ordered_groups = [
        ("curve", "strike_vol_change", "distribution"),
        ("rv_vs_iv",),
        ("iv_history",),
        ("vix_distribution", "vix_distribution_since", "vvix_vix_ratio"),
        ("position", "pnl"),
    ]
    placed = {k for group in ordered_groups for k in group}
    main_children = [html.Div(id="analysis", style={"marginBottom": "14px"})]
    main_children += [_graph_card(k) for k in ordered_groups[0] if k in keys]
    if rv_term_section is not None:
        main_children.append(rv_term_section)
    main_children += [_graph_card(k) for k in ordered_groups[1] if k in keys]
    main_children += [_graph_card(k) for k in ordered_groups[2] if k in keys]
    if volhist_section is not None:
        main_children.append(volhist_section)
    main_children += [_graph_card(k) for group in ordered_groups[3:] for k in group if k in keys]
    main_children += [_graph_card(k) for k in keys if k not in placed]
    main = html.Div(main_children, style={"flex": "1", "padding": "16px", "minWidth": "0"})

    topbar = html.Div([
        html.Div([html.Span("Moontower", style={"fontWeight": 800, "fontSize": "17px", "color": "#fff"}),
                  html.Span("  ·  skew & vol dashboard", style={"color": "#94a3b8", "fontSize": "14px"})]),
        html.Div(f"{snap.symbol}  ·  {snap.date}  ·  {snap.dte:.0f} DTE  ·  spot {snap.spot:,.2f}  ·  "
                 f"fwd {snap.forward:,.2f}",
                 style={"color": "#cbd5e1", "fontSize": "13px", "fontWeight": 600}),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
              "background": "#0f172a", "padding": "13px 24px"})

    # suppress_callback_exceptions: some controls (e.g. the SVI-disabled wing sliders) and
    # the vol-history section are created conditionally, so their callbacks may reference
    # components not present in every layout.
    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.title = f"{snap.symbol} skew dashboard"
    app.index_string = """<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  body { font-family: Inter, -apple-system, 'Segoe UI', Roboto, sans-serif; margin:0; background:#eef1f5; color:#1f2937; }
  ::-webkit-scrollbar { width:9px; height:9px; }
  ::-webkit-scrollbar-thumb { background:#c7cfdb; border-radius:6px; }
  ::-webkit-scrollbar-thumb:hover { background:#aab4c4; }
  .mt-btn { cursor:pointer; border:none; border-radius:8px; padding:9px 14px; font-weight:600;
            font-size:13px; transition:filter .15s, box-shadow .15s; box-shadow:0 1px 2px rgba(16,24,40,.08); }
  .mt-btn:hover { filter:brightness(.95); }
  .mt-card { background:#fff; border-radius:13px; border:1px solid #eceff3;
             box-shadow:0 1px 3px rgba(16,24,40,.07), 0 1px 2px rgba(16,24,40,.04); }
  .rc-slider-track { background:#2f6feb !important; }
  .rc-slider-handle { border-color:#2f6feb !important; opacity:1 !important; }
  .rc-slider-handle:hover { border-color:#2f6feb !important; }
  .rv-metric-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:9px; }
  .rv-styler { width:100%; overflow-x:auto; }
  .rv-styler table { width:100%; min-width:860px; border-collapse:collapse; font-size:11.5px;
                     table-layout:fixed; }
  .rv-styler th, .rv-styler td { padding:7px 10px; border-bottom:1px solid #edf1f5;
                                 text-align:right; white-space:nowrap; }
  .rv-styler thead th { color:#334155; background:#f8fafc; font-weight:700; }
  .rv-styler th:first-child { width:230px; min-width:230px; }
  .rv-styler .rv-index-head, .rv-styler .rv-index-name { text-align:center; }
  .rv-styler tbody th { color:#334155; background:#fff; font-weight:600; text-align:left;
                        position:sticky; left:0; z-index:2; }
  @media (max-width: 1050px) { .rv-metric-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
  @media (max-width: 660px) { .rv-metric-grid { grid-template-columns:1fr; } }
</style></head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""
    app.layout = html.Div([topbar, html.Div([controls, main],
                          style={"display": "flex", "alignItems": "flex-start",
                                 "maxWidth": "1520px", "margin": "0 auto"})])

    iv_states = [State(iv_id(z), "value") for z in z_grid]
    outputs = ([Output("analysis", "children")]
               + [Output(f"g_{k}", "figure") for k in keys]
               + [Output(iv_id(z), "value") for z in z_grid]
               + [Output("wings", "value"), Output("slope_l", "value"), Output("slope_r", "value"),
                  Output("scenario", "value")])

    @app.callback(
        outputs,
        [Input("apply", "n_clicks"), Input("reset", "n_clicks"), Input("scenario", "value")],
        iv_states + [State("wings", "value"), State("slope_l", "value"), State("slope_r", "value")],
    )
    def _render(apply_clicks, reset_clicks, scenario, *state):
        n = len(z_grid)
        try:
            _trg = dash.callback_context.triggered
            trig = _trg[0]["prop_id"].split(".")[0] if _trg else ""
        except Exception:
            trig = ""
        is_reset = (trig == "reset")
        scen = "market" if is_reset else (scenario or "market")

        if scen != "market":
            cs = CurveState.from_scenario(snap, scen)
            slider_out = [dash.no_update] * (n + 3)
            scen_out = dash.no_update
        else:
            if is_reset or (not apply_clicks and not reset_clicks and not scenario):
                iv_vals = [SEED[z] for z in z_grid]
                wings_val = (["on"] if (cfg.wings_on or _is_svi) else [])
                sl_val, sr_val = cfg.slope_left * 100, cfg.slope_right * 100
                slider_out = iv_vals + [wings_val, sl_val, sr_val]
            else:
                iv_vals = list(state[:n])
                wings_val, sl_val, sr_val = state[n], state[n + 1], state[n + 2]
                slider_out = [dash.no_update] * (n + 3)

            def _num(v, fallback):
                """Coerce a slider value to a finite float; fall back if None/blank/NaN."""
                try:
                    f = float(v)
                    return f if np.isfinite(f) else float(fallback)
                except (TypeError, ValueError):
                    return float(fallback)

            grid_vols = np.array([_num(v, SEED[z_grid[i]]) / 100.0 for i, v in enumerate(iv_vals)])
            sl = _num(sl_val, cfg.slope_left * 100) / 100.0
            sr = _num(sr_val, cfg.slope_right * 100) / 100.0
            cs = CurveState.from_grid(snap, grid_vols, sl, sr, "on" in (wings_val or []))
            scen_out = "market" if is_reset else dash.no_update

        figs = _figures(snap, cs)
        try:
            an = analysis.render_html(snap, cs)
        except Exception as e:
            from dash import html as _h
            an = _h.Pre(f"(analysis unavailable: {e})")
        fig_out = [figs.get(k, go.Figure()) for k in keys]
        return [an] + fig_out + slider_out + [scen_out]

    if _VH_ON:
        @app.callback(
            [Output("g_iv_vs_rv", "figure"), Output("g_rv_estimators", "figure")],
            [Input("vh_start", "date"), Input("vh_refresh", "n_clicks")],
        )
        def _volhist(start_date, _clicks):
            cs0 = CurveState.market(snap)
            f1 = charts_pkg.vol_history.make(snap, cs0, start=start_date) or go.Figure()
            f2 = charts_pkg.vol_history.make_estimators(snap, cs0, start=start_date) or go.Figure()
            return f1, f2

    return app


def serve(snap, port=8050, open_browser=True):
    try:
        build = build_app(snap)
    except Exception as e:
        import traceback
        print(f"[dashboard] Dash unavailable ({type(e).__name__}: {e}).")
        print('[dashboard] If it is a Flask/Werkzeug clash: pip install "dash>=2.11,<3" "flask<3.1" "werkzeug<3.1"')
        traceback.print_exc()
        return write_static_html(snap)
    url = f"http://127.0.0.1:{port}"
    print(f"[dashboard] serving at {url}  (interrupt to stop)")
    if open_browser:
        import webbrowser, threading
        threading.Timer(1.25, lambda: webbrowser.open_new(url)).start()
    build.run(port=port, debug=False)


def write_static_html(snap, outdir="."):
    """Fallback: one standalone HTML per chart + the analysis text."""
    import os
    os.makedirs(outdir, exist_ok=True)
    cs = CurveState.market(snap)
    written = []
    for c in charts_pkg.active(snap):
        fig = c.make(snap, cs)
        if fig is None:
            continue
        path = os.path.join(outdir, f"skewlab_{c.key}.html")
        fig.write_html(path)
        written.append(path)
    rv_fig = charts_pkg.rv_term_structure.make(snap, cs)
    if rv_fig is not None:
        path = os.path.join(outdir, "skewlab_rv_term_structure.html")
        rv_fig.write_html(path)
        written.append(path)
    apath = os.path.join(outdir, "skewlab_analysis.txt")
    with open(apath, "w") as fh:
        fh.write(analysis.render_text(snap, cs))
    written.append(apath)
    print(f"[dashboard] wrote {len(written)} static files to {outdir}")
    return written
