"""skewlab.charts.curve — the fitted skew curve with delta markers and term overlays."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .. import model, theme


def _mark_wings(fig, snap):
    for k in (snap.grid_strikes[0], snap.grid_strikes[-1]):
        fig.add_vline(x=k, line=dict(color="lightgray", dash="dot", width=1))


def _hist_pct(snap, col):
    if snap.iv_history is None or col not in getattr(snap.iv_history, "columns", []):
        return None
    return snap.iv_history[col] * 100.0


def _add_delta_markers(fig, snap, cs):
    """20Δ strangle to SELL (carry) and 10Δ tails to BUY, with rich hover stats.
    History uses the 25Δ buckets as a proxy for 20Δ."""
    rv_last = (float(snap.iv_rv.dropna().iloc[-1]) * 100.0
               if (snap.iv_rv is not None and snap.iv_rv.dropna().size) else None)

    def line(label, iv, K, stats, mode):
        head = f"<b>{label}</b>  ·  strike {K:.1f}  ·  IV {iv * 100:.1f}%"
        if stats is None:
            return head + "<extra></extra>"
        rich = "rich" if stats['pctl'] >= 65 else "cheap" if stats['pctl'] <= 35 else "mid"
        body = (f"<br>IV rank {stats['rank']:.0f}  ·  IV pctile {stats['pctl']:.0f} ({rich})"
                f"<br>3m z {stats['z3']:+.2f}  ·  3m pctile {stats['pctl3']:.0f}"
                f"<br>range [{stats['lo']:.1f}–{stats['hi']:.1f}], mean {stats['mean']:.1f}  ({stats['n']} obs)")
        if mode == "sell" and rv_last is not None:
            body += f"<br>vs realized: VRP {iv * 100 - rv_last:+.1f} pts  ({'rich→sell' if iv*100>rv_last else 'cheap'})"
        body += ("<br><i>sell carry: want high rank/pctile/z</i>" if mode == "sell"
                 else "<br><i>buy tail: want low rank/pctile/z</i>")
        return head + body + "<extra></extra>"

    dp = lambda tgt, call: model.delta_point(cs.poly, snap.grid_strikes, snap.spot, snap.t, snap.r,
                                              snap.q, cs.slope_left, cs.slope_right, cs.wings_on,
                                              snap.forward, snap.one_sd, snap.z_grid, tgt, call)
    pk, pv = dp(-0.20, False)
    ck, cv = dp(0.20, True)
    ps = model.hover_stats(_hist_pct(snap, "25d_put"), pv * 100)
    csl = model.hover_stats(_hist_pct(snap, "25d_call"), cv * 100)
    fig.add_trace(go.Scatter(
        x=[pk, ck], y=[pv * 100, cv * 100], mode="markers", name="sell 20Δ (carry)",
        marker=dict(size=14, symbol="triangle-down", color="#b21f2d", line=dict(width=1.2, color="white")),
        hovertemplate=[line("20Δ put — sell (25Δ hist proxy)", pv, pk, ps, "sell"),
                       line("20Δ call — sell (25Δ hist proxy)", cv, ck, csl, "sell")]))

    pk2, pv2 = dp(-0.10, False)
    ck2, cv2 = dp(0.10, True)
    ps2 = model.hover_stats(_hist_pct(snap, "10d_put"), pv2 * 100)
    cs2 = model.hover_stats(_hist_pct(snap, "10d_call"), cv2 * 100)
    fig.add_trace(go.Scatter(
        x=[pk2, ck2], y=[pv2 * 100, cv2 * 100], mode="markers", name="buy 10Δ (tails)",
        marker=dict(size=14, symbol="triangle-up", color="#0ea5a4", line=dict(width=1.2, color="white")),
        hovertemplate=[line("10Δ put — buy tail", pv2, pk2, ps2, "buy"),
                       line("10Δ call — buy tail", cv2, ck2, cs2, "buy")]))


def _delta_point_b(b, target, is_call):
    Ks = np.linspace(b.grid_strikes[0], b.grid_strikes[-1], 400)
    sig = np.maximum(b.poly(Ks), 1e-4)
    d = np.array([model.bs_price_delta(b.spot, K, b.t, b.r, b.q, s, is_call)[1]
                  for K, s in zip(Ks, sig)])
    i = int(np.argmin(np.abs(d - target)))
    return float(Ks[i]), float(sig[i])


def _term_bucket_series(snap, b, col_h):
    # Only use this tenor's OWN history (built when term_iv_history=True). We deliberately
    # do NOT fall back to the 30-DTE panel as a proxy — a longer/shorter tenor ranks
    # differently, so a proxy rank is misleading. No per-tenor history -> no hover stats.
    h = b.hist
    if h is not None and col_h in getattr(h, "columns", []):
        s = (h[col_h] * 100.0).dropna()
        if len(s):
            return s, False
    return None, False


def _term_hover(snap, label, iv, K, stats, atf_term, proxy=False):
    head = (f"<b>{label}</b>  ·  strike {K:.1f}  ·  IV {iv * 100:.1f}%"
            f"<br>skew vs ATF {(iv - atf_term) * 100:+.1f} pts")
    if stats is None:
        # No per-tenor history -> show the live point only, no (misleading) proxy rank.
        hint = ("set use_iv_history = True" if not snap.cfg.use_iv_history
                else "set term_iv_history = True for this tenor's IV rank/percentile")
        return head + f"<br><i>{hint}</i><extra></extra>"
    rich = "rich" if stats['pctl'] >= 65 else "cheap" if stats['pctl'] <= 35 else "mid"
    return (head + f"<br>IV rank {stats['rank']:.0f}  ·  pctile {stats['pctl']:.0f} ({rich})"
            f"<br>3m z {stats['z3']:+.2f}  ·  3m pctile {stats['pctl3']:.0f}"
            f"<br>range [{stats['lo']:.1f}–{stats['hi']:.1f}], mean {stats['mean']:.1f}<extra></extra>")


def _add_term_curves(fig, snap):
    for b in snap.term_bundles:
        col = theme.TERM_COLORS.get(b.tenor, "#888888")
        grp = f"term{b.tenor}"
        exp = b.expiry.strftime("%d %b %Y")
        lbl_curve = f"{b.dte:.0f}DTE{' monthly' if b.monthly else ''} · {exp}"
        fine = np.linspace(b.grid_strikes[0], b.grid_strikes[-1], 200)
        # term overlays start OFF (legend-only) — click the tenor in the legend to show it
        fig.add_trace(go.Scatter(x=fine, y=np.maximum(b.poly(fine), 1e-4) * 100, mode="lines",
                                 name=lbl_curve, legendgroup=grp, visible="legendonly",
                                 line=dict(width=2, dash="dot", color=col)))
        marks = [(-0.25, "25d_put", "25Δ put", False, "circle"),
                 (-0.10, "10d_put", "10Δ put", False, "circle"),
                 (0.25, "25d_call", "25Δ call", True, "square"),
                 (0.10, "10d_call", "10Δ call", True, "square")]
        for tgt, col_h, lbl, is_call, sym in marks:
            K, iv = _delta_point_b(b, tgt, is_call)
            hp, proxy = _term_bucket_series(snap, b, col_h)
            stats = model.hover_stats(hp, iv * 100)
            tr = dict(x=[K], y=[iv * 100], mode="markers", showlegend=False, legendgroup=grp,
                      visible="legendonly",
                      marker=dict(size=9, symbol=sym, color=col, line=dict(width=1, color="white")))
            if stats is None:
                # no per-tenor history (term_iv_history off) -> no hover at all, no proxy
                tr["hoverinfo"] = "skip"
            else:
                tr["hovertemplate"] = _term_hover(snap, f"{b.dte:.0f}DTE ({exp}) {lbl}", iv, K,
                                                  stats, b.atf, proxy=proxy)
            fig.add_trace(go.Scatter(**tr))


def make(snap, cs, title_suffix=""):
    fine = snap.fine_strikes(cs.wings_on)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=snap.mkt_curve_x, y=snap.mkt_curve_y, mode="lines",
                             name="market (original)", visible="legendonly",
                             line=dict(width=2, color="rgba(120,120,120,0.55)", dash="dot")))
    is_svi = hasattr(cs.poly, "svi_params")
    model_tag = "SVI" if is_svi else f"poly·{snap.cfg.poly_degree}"
    fig.add_trace(go.Scatter(x=fine, y=snap.curve_vol(fine, cs) * 100, mode="lines",
                             name=f"fitted skew curve ({model_tag})", line=dict(width=3, color="#636efa")))
    fig.add_trace(go.Scatter(x=snap.grid_strikes, y=np.asarray(cs.grid_vols) * 100, mode="markers",
                             name="grid points", visible="legendonly",
                             marker=dict(size=10, symbol="circle-open", line=dict(width=2), color="#EF553B")))
    fig.add_trace(go.Scatter(x=[snap.forward], y=[cs.atf * 100], mode="markers", name="at-the-forward",
                             marker=dict(size=12, color="green", symbol="diamond")))
    fig.add_trace(go.Scatter(x=snap.market_iv.index, y=snap.market_iv.values * 100, mode="markers",
                             name="market IVs", visible="legendonly",
                             marker=dict(size=5, color="lightgray")))
    if snap.prev_poly is not None:
        prev_y = model.curve_vol(fine, snap.prev_poly, snap.grid_strikes, snap.forward, snap.one_sd,
                                 snap.z_grid, cs.slope_left, cs.slope_right, cs.wings_on) * 100
        fig.add_trace(go.Scatter(x=fine, y=prev_y, mode="lines", name=f"prev ({snap.prev_label})",
                                 line=dict(width=2, dash="dash", color="orange")))
        if getattr(snap, "prev_atf", None) is not None and snap.prev_forward is not None:
            fig.add_trace(go.Scatter(
                x=[snap.prev_forward], y=[snap.prev_atf * 100], mode="markers",
                name="prev at-the-forward",
                marker=dict(size=12, color="orange", symbol="diamond-open", line=dict(width=2, color="orange")),
                hovertemplate=(f"<b>prev ATF</b> · fwd {snap.prev_forward:.1f} · "
                               f"IV {snap.prev_atf * 100:.1f}%<extra></extra>")))
    if cs.wings_on:
        _mark_wings(fig, snap)
    _add_delta_markers(fig, snap, cs)
    if snap.cfg.show_term_curves and snap.term_bundles:
        _add_term_curves(fig, snap)
    if is_svi:
        a, b, rho, mm, s = cs.poly.svi_params
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.02, align="left", showarrow=False,
                           text=(f"SVI  a={a:.3f}  b={b:.3f}  ρ={rho:+.2f}  m={mm:+.3f}  σ={s:.3f}"),
                           font=dict(size=10, color="#636efa"), bgcolor="rgba(255,255,255,0.7)",
                           bordercolor="#dfe3e8", borderwidth=1)
    fig.update_layout(title=f"{snap.symbol} skew curve {snap.date} · {model_tag} fit{title_suffix}",
                      xaxis_title="strike", yaxis_title="implied vol (%)",
                      template=theme.TEMPLATE, height=520, margin=dict(t=56, b=52, r=170),
                      legend=theme.LEGEND_SIDE)
    return fig
