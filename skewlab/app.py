"""skewlab.app — the Dash dashboard, built generically from the chart registry.

`build_app(snap)` wires sliders + scenario presets + Apply/Reset to every chart in
`charts.active(snap)`. Reacting charts redraw on each Apply; non-reacting ones (IV
history) render once. `serve(snap, ...)` runs it; `write_static_html(snap, outdir)` is
the no-Dash fallback.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from . import analysis, model, theme
from . import charts as charts_pkg
from .data import CurveState


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
    ctrl += [
        html.Div("Wing extrapolation", style=_grouplbl),
        dcc.Checklist(id="wings", options=[{"label": " wings on", "value": "on"}],
                      value=(["on"] if cfg.wings_on else []),
                      style={"fontSize": "13px", "color": "#334155"}),
        html.Div("slope L (vol pts / SD)", style=_slbl),
        dcc.Slider(id="slope_l", min=-10, max=25, step=0.5, value=cfg.slope_left * 100, marks=None,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Div("slope R (vol pts / SD)", style=_slbl),
        dcc.Slider(id="slope_r", min=-10, max=25, step=0.5, value=cfg.slope_right * 100, marks=None,
                   tooltip={"placement": "bottom", "always_visible": True}),
    ]
    controls = html.Div(ctrl, className="mt-card", style={
        "flex": "0 0 280px", "padding": "18px", "position": "sticky", "top": "16px",
        "margin": "16px 0 16px 16px", "maxHeight": "calc(100vh - 32px)", "overflowY": "auto"})

    def _graph_card(gid):
        return html.Div(dcc.Graph(id=f"g_{gid}", config={"displaylogo": False,
                        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}),
                        className="mt-card", style={"marginBottom": "14px", "padding": "6px 8px 8px"})

    # --- net-liquidity growth calculator (continuous compounding + annuity-due deposits) ---
    def _nl_input(id_, label, value, **kw):
        return html.Div([html.Div(label, style={**_slbl, "marginTop": "0"}),
                         dcc.Input(id=id_, type="number", value=value, debounce=True,
                                   style={"width": "100%", "padding": "7px 9px", "borderRadius": "8px",
                                          "border": "1px solid #d7dde5", "fontSize": "13px",
                                          "boxSizing": "border-box"}, **kw)],
                        style={"flex": "1 1 120px", "minWidth": "108px"})

    netliq_panel = html.Div([
        html.Div("Net-liquidity growth — continuous compounding", style={
            "fontWeight": 800, "fontSize": "15px", "color": "#0f172a", "marginBottom": "4px"}),
        html.Div("value(t) = P·e^(r·t) + level deposits made annuity-due (start of each period), "
                 "each compounding continuously. No withdrawals; sampled every 6 months.",
                 style={"fontSize": "11.5px", "color": "#94a3b8", "marginBottom": "13px"}),
        html.Div([
            _nl_input("nl_principal", "Principal ($)", 100000, min=0, step=1000),
            _nl_input("nl_rate", "Cont. rate (%/yr)", 10, step=0.25),
            _nl_input("nl_years", "Horizon (years)", 10, min=0.5, step=0.5),
            _nl_input("nl_contrib", "Deposit ($)", 0, min=0, step=500),
            html.Div([html.Div("Deposit frequency", style={**_slbl, "marginTop": "0"}),
                      dcc.Dropdown(id="nl_freq", clearable=False, value="1.0",
                                   options=[{"label": "Every 6 months", "value": "0.5"},
                                            {"label": "Every 1 year", "value": "1.0"}],
                                   style={"fontSize": "12px"})],
                     style={"flex": "1 1 150px", "minWidth": "140px"}),
        ], style={"display": "flex", "gap": "10px", "flexWrap": "wrap",
                  "alignItems": "flex-end", "marginBottom": "13px"}),
        html.Button("Recalculate", id="nl_run", n_clicks=0, className="mt-btn",
                    style={"background": "#0891b2", "color": "white", "marginBottom": "13px"}),
        dcc.Loading(html.Div(id="netliq_out"), type="dot"),
    ], className="mt-card", style={"marginTop": "4px", "padding": "18px"})

    main = html.Div([html.Div(id="analysis", style={"marginBottom": "14px"})]
                    + [_graph_card(k) for k in keys]
                    + [netliq_panel],
                    style={"flex": "1", "padding": "16px", "minWidth": "0"})

    topbar = html.Div([
        html.Div([html.Span("Moontower", style={"fontWeight": 800, "fontSize": "17px", "color": "#fff"}),
                  html.Span("  ·  skew & vol dashboard", style={"color": "#94a3b8", "fontSize": "14px"})]),
        html.Div(f"{snap.symbol}  ·  {snap.date}  ·  {snap.dte:.0f} DTE  ·  spot {snap.spot:,.2f}  ·  "
                 f"fwd {snap.forward:,.2f}",
                 style={"color": "#cbd5e1", "fontSize": "13px", "fontWeight": 600}),
    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
              "background": "#0f172a", "padding": "13px 24px"})

    app = dash.Dash(__name__)
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
                wings_val = (["on"] if cfg.wings_on else [])
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

    @app.callback(
        Output("netliq_out", "children"),
        Input("nl_run", "n_clicks"),
        [State("nl_principal", "value"), State("nl_rate", "value"), State("nl_years", "value"),
         State("nl_contrib", "value"), State("nl_freq", "value")],
    )
    def _netliq(_clicks, principal, rate_pct, years, contrib, freq):
        from dash import dash_table, html as _h

        def _f(v, d):
            try:
                x = float(v)
                return x if np.isfinite(x) else float(d)
            except (TypeError, ValueError):
                return float(d)

        P = max(_f(principal, 100000.0), 0.0)
        r = _f(rate_pct, 10.0) / 100.0
        yrs = max(_f(years, 10.0), 0.5)
        C = max(_f(contrib, 0.0), 0.0)
        dt = _f(freq, 1.0)
        try:
            df = model.net_liquidity_projection(P, r, yrs, contribution=C, contrib_freq_years=dt)
        except Exception as e:
            return _h.Pre(f"(calculation error: {e})")

        disp = df.copy()
        disp["date"] = [d.strftime("%Y-%m-%d") for d in df["date"]]
        disp["years"] = df["years"].map(lambda x: f"{x:.1f}")
        for col in ("added", "net_liq", "gain"):
            disp[col] = df[col].map(lambda x: f"${x:,.2f}")
        labels = {"period": "period", "years": "years", "date": "date",
                  "added": "deposited", "net_liq": "net liq", "gain": "growth"}
        cols = [{"name": labels[c], "id": c}
                for c in ["period", "years", "date", "added", "net_liq", "gain"]]
        table = dash_table.DataTable(
            columns=cols, data=disp.to_dict("records"),
            style_table={"maxHeight": "440px", "overflowY": "auto", "border": "1px solid #eceff3",
                         "borderRadius": "10px"},
            style_cell={"fontFamily": "Inter", "fontSize": "12.5px", "padding": "6px 12px",
                        "textAlign": "right", "border": "none",
                        "borderBottom": "1px solid #f1f5f9"},
            style_header={"fontWeight": "700", "background": "#f1f5f9", "textAlign": "right",
                          "position": "sticky", "top": 0},
            style_cell_conditional=[{"if": {"column_id": c}, "textAlign": "left"}
                                    for c in ("period", "years", "date")],
            style_data_conditional=[{"if": {"row_index": "odd"}, "backgroundColor": "#fafbfc"},
                                    {"if": {"filter_query": "{period} = " + str(int(df['period'].iloc[-1]))},
                                     "fontWeight": "700"}],
        )
        final = float(df["net_liq"].iloc[-1])
        tot_add = float(df["added"].iloc[-1])
        grow = float(df["gain"].iloc[-1])
        eff = (np.exp(r) - 1.0) * 100.0
        summary = _h.Div([
            _h.Span(f"Final: ${final:,.2f}", style={"fontWeight": 800, "fontSize": "14px",
                    "color": "#0f172a", "marginRight": "14px"}),
            _h.Span(f"principal ${P:,.0f} + deposited ${tot_add:,.0f} + growth ${grow:,.0f}  ·  "
                    f"~{eff:.2f}% effective/yr",
                    style={"color": "#475569", "fontSize": "12.5px"}),
        ], style={"margin": "2px 0 12px"})
        return [summary, table]

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
    apath = os.path.join(outdir, "skewlab_analysis.txt")
    with open(apath, "w") as fh:
        fh.write(analysis.render_text(snap, cs))
    written.append(apath)
    print(f"[dashboard] wrote {len(written)} static files to {outdir}")
    return written
