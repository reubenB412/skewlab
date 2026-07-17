"""skewlab.analysis — data-driven narrative (metrics + plain-text + styled HTML cards).

`metrics(snap, cs)` computes everything once; `render_text` and `render_html` consume it.
"""
from __future__ import annotations

import numpy as np
from . import model


def _fmt(x, nd=2):
    """Format a possibly-None / NaN number, else 'n/a'."""
    try:
        v = float(x)
        return f"{v:.{nd}f}" if np.isfinite(v) else "n/a"
    except (TypeError, ValueError):
        return "n/a"


def _delta_bucket_ivs(snap, cs):
    Ks = np.linspace(snap.grid_strikes[0], snap.grid_strikes[-1], 200)
    sig = snap.curve_vol(Ks, cs)
    dC = np.array([model.bs_price_delta(snap.spot, K, snap.t, snap.r, snap.q, s, True)[1]
                   for K, s in zip(Ks, sig)])
    dP = np.array([model.bs_price_delta(snap.spot, K, snap.t, snap.r, snap.q, s, False)[1]
                   for K, s in zip(Ks, sig)])
    pick = lambda d, tgt: float(sig[int(np.argmin(np.abs(d - tgt)))])
    return dict(c25=pick(dC, 0.25), p25=pick(dP, -0.25), c10=pick(dC, 0.10), p10=pick(dP, -0.10))


def metrics(snap, cs):
    iz = list(snap.z_grid)
    K_dn, K_up = snap.grid_strikes[iz.index(-1.0)], snap.grid_strikes[iz.index(1.0)]
    v_atf = float(snap.curve_vol(np.array([snap.forward]), cs)[0])
    v_dn = float(snap.curve_vol(np.array([K_dn]), cs)[0])
    v_up = float(snap.curve_vol(np.array([K_up]), cs)[0])
    rr = (v_dn - v_up) * 100.0
    fly = ((v_dn + v_up) / 2.0 - v_atf) * 100.0
    fine = snap.fine_strikes(cs.wings_on)
    calls = model.bs_call_vec(fine, snap.curve_vol(fine, cs), snap.spot, snap.t, snap.r, snap.q)
    pdf = model.implied_pdf(fine, calls, snap.r, snap.t)
    xc, pc = fine[2:-2], pdf[2:-2]
    st = model.dist_stats(xc, pc)
    w = pc / pc.sum() if pc.sum() > 0 else pc
    p_above = float(w[xc > snap.forward].sum()) * 100.0
    skew_word = ("steep" if rr >= 4 else "moderate" if rr >= 2 else "flat" if rr > -2
                 else "inverted (call-skewed)")
    shape = ("more likely to grind up, with a long/fat downside tail (negative skew)" if st['skew'] < -0.2
             else "likely to drift down, with a long upside tail (positive/bubble skew)" if st['skew'] > 0.2
             else "roughly symmetric")
    # --- no-arbitrage diagnostics (on the live fitted smile) ---
    bf = model.butterfly_arb(fine, calls, snap.r, snap.t)
    cal = None
    if snap.term_bundles and len(snap.term_bundles) >= 2:
        cal = model.calendar_arb(snap.term_bundles, lambda b, K: np.maximum(b.poly(K), 1e-4))
    noarb = dict(butterfly=bf, calendar=cal,
                 ok=(not bf["has_arb"]) and (cal is None or not cal["has_arb"]))

    m = dict(atf=cs.atf, v_dn=v_dn, v_up=v_up, rr=rr, fly=fly, st=st, p_above=p_above,
             skew_word=skew_word, shape=shape, smile_bid=(fly > 0.7), change=None, regime=None,
             position=None, implications=[], noarb=noarb)

    if snap.prev_poly is not None:
        pvol = lambda K: float(model.curve_vol(np.array([K]), snap.prev_poly, snap.grid_strikes,
                                               snap.forward, snap.one_sd, snap.z_grid,
                                               cs.slope_left, cs.slope_right, cs.wings_on)[0])
        v_atf_p, v_dn_p, v_up_p = pvol(snap.forward), pvol(K_dn), pvol(K_up)
        rr_p = (v_dn_p - v_up_p) * 100.0
        d_atf, d_rr = (v_atf - v_atf_p) * 100.0, rr - rr_p
        finep = snap.fine_strikes(cs.wings_on)
        callsp = model.bs_call_vec(finep, model.curve_vol(finep, snap.prev_poly, snap.grid_strikes,
                                   snap.forward, snap.one_sd, snap.z_grid, cs.slope_left,
                                   cs.slope_right, cs.wings_on), snap.spot, snap.t, snap.r, snap.q)
        st_p = model.dist_stats(finep[2:-2], model.implied_pdf(finep, callsp, snap.r, snap.t)[2:-2])
        note = ("Vol up + skew steeper = risk-off: downside being bid, tails fattening."
                if (d_atf > 0.3 and d_rr > 0.5) else
                "Vol down + skew flatter = risk-on / vol compression: downside hedges bleeding."
                if (d_atf < -0.3 and d_rr < -0.5) else
                "Vol compressed with skew roughly intact -- premium sellers favored." if d_atf < -0.3 else
                "Vol expanded -- long-vega/long-gamma rewarded, short premium hurt." if d_atf > 0.3 else "")
        m['change'] = dict(v_atf=v_atf, v_atf_p=v_atf_p, d_atf=d_atf, rr=rr, rr_p=rr_p, d_rr=d_rr,
                           st_skew=st['skew'], st_p_skew=st_p['skew'], std=st['std'], std_p=st_p['std'],
                           vol_word=("rose" if d_atf > 0.3 else "fell" if d_atf < -0.3 else "was ~unchanged"),
                           skew_chg=("steepened" if d_rr > 0.5 else "flattened" if d_rr < -0.5 else "held"),
                           note=note)

    if snap.iv_history is not None and not snap.iv_history.empty:
        atm_h = snap.iv_atm.dropna() * 100.0
        p_atm = model.pctile(atm_h.values, cs.atf * 100.0)
        reg = dict(p_atm=p_atm, atm_lo=float(atm_h.min()), atm_hi=float(atm_h.max()),
                   atm_mean=float(atm_h.mean()), start=str(atm_h.index.min().date()),
                   end=str(atm_h.index.max().date()), n=len(atm_h), rr25=None,
                   vol_rich=("rich" if p_atm > 70 else "cheap" if p_atm < 30 else "mid-range"))
        if {"25d_put", "25d_call"}.issubset(snap.iv_history.columns):
            rr_h = (snap.iv_history["25d_put"] - snap.iv_history["25d_call"]).dropna() * 100.0
            db = _delta_bucket_ivs(snap, cs)
            rr25 = (db["p25"] - db["c25"]) * 100.0
            p_rr = model.pctile(rr_h.values, rr25)
            reg.update(rr25=rr25, p_rr=p_rr, rr_lo=float(rr_h.min()), rr_hi=float(rr_h.max()),
                       skew_rich=("rich" if p_rr > 70 else "cheap" if p_rr < 30 else "around normal"))
        m['regime'] = reg

    if snap.has_positions:
        from .charts import position as _pos
        net = _pos.position_risk(snap, cs)
        m['position'] = dict(net=net, pnl=_pos.pnl_decomp(snap, cs),
                             gpos=("long" if net['gamma'] > 0 else "short" if net['gamma'] < 0 else "flat"),
                             vpos=("long" if net['vega_pt'] > 0 else "short" if net['vega_pt'] < 0 else "flat"),
                             dpos=("long" if net['delta'] > 0 else "short" if net['delta'] < 0 else "flat"))

    # --- RV vs IV: realized-implied fair value vs the market now / at the open ---
    m['rv'] = None
    rv_iv = getattr(snap, "rv_iv", None)
    if rv_iv is not None and np.isfinite(rv_iv):
        now_iv = float(snap.atf)
        rv_str = getattr(snap, "rv_straddle", None)
        now_str = getattr(snap, "now_straddle", None)
        open_iv = getattr(snap, "open_atf", None)
        open_str = getattr(snap, "open_straddle", None)
        _fin = lambda x: x is not None and np.isfinite(float(x))
        has_open = _fin(open_iv)
        rvm = dict(
            rv_iv=rv_iv * 100.0, rv_straddle=rv_str, rv_asof=getattr(snap, "rv_asof", None),
            rv_lookback=getattr(snap, "rv_lookback", None),
            now_iv=now_iv * 100.0, now_straddle=now_str, now_ts=getattr(snap, "now_capture_ts", None),
            open_iv=(float(open_iv) * 100.0 if has_open else None), open_straddle=open_str,
            open_ts=getattr(snap, "open_capture_ts", None),
            vrp_now=(now_iv - rv_iv) * 100.0,
            vrp_open=((float(open_iv) - rv_iv) * 100.0 if has_open else None),
            drift_iv=((now_iv - float(open_iv)) * 100.0 if has_open else None),
            straddle_gap=(float(now_str) - float(rv_str) if (_fin(now_str) and _fin(rv_str)) else None),
            straddle_drift=(float(now_str) - float(open_str)
                            if (has_open and _fin(now_str) and _fin(open_str)) else None),
        )
        rvm["rich"] = rvm["vrp_now"] > 0
        m['rv'] = rvm

    imps = []
    if rr >= 3:
        imps.append("Steep put skew: OTM puts rich vs calls on vol; put spreads/ratios finance the rich "
                    "tail, OTM calls are cheap vol (inexpensive upside convexity).")
    elif rr <= -1:
        imps.append("Call-skewed: upside calls rich (squeeze/bubble); call spreads cheap; profile leans "
                    "drift-down-with-upside-tail.")
    else:
        imps.append("Flat-ish skew: little directional skew edge.")
    if fly > 1.0:
        imps.append("Pronounced smile: both wings rich -- selling tails (defined-risk) harvests premium.")
    if st['skew'] < -0.3:
        imps.append("Negative-skew distribution: high-probability up-grind; a call spread/butterfly near "
                    "the mode expresses it, OTM puts buy the fat-tail magnitude the market implies.")
    m['implications'] = imps
    return m


def render_text(snap, cs):
    m = metrics(snap, cs)
    st = m['st']
    L = ["=== BESPOKE ANALYSIS ===", f"WHAT'S GOING ON — {snap.symbol} {snap.date}, {snap.dte:.0f}DTE",
         (f"  ATF vol {m['atf']*100:.1f}%. Skew {m['skew_word']}: 1-SD RR {m['rr']:+.1f} pts "
          f"(down {m['v_dn']*100:.1f}% vs up {m['v_up']*100:.1f}%). smile {m['fly']:+.1f}"
          + (" -- both wings bid." if m['smile_bid'] else ".")),
         (f"  Dist: median {st['median']:.0f}, mode {st['mode']:.0f}, std {st['std']:.0f}, "
          f"skew {st['skew']:+.2f}, kurt {st['kurt']:+.2f}. ~{m['p_above']:.0f}% above fwd -- {m['shape']}.")]
    c = m['change']
    if c:
        L.append(f"HOW IT'S CHANGED -- vs {snap.prev_label}")
        L.append(f"  ATF {c['vol_word']} {c['d_atf']:+.1f} pts; skew {c['skew_chg']} "
                 f"({c['rr_p']:+.1f}->{c['rr']:+.1f}); dist skew {c['st_p_skew']:+.2f}->{c['st_skew']:+.2f}."
                 + (f"  {c['note']}" if c['note'] else ""))
    r = m['regime']
    if r:
        L.append(f"REGIME -- {r['start']}..{r['end']} ({r['n']} obs)")
        L.append(f"  ATF vol {snap.atf*100:.1f}% at {r['p_atm']:.0f}th pctile [{r['atm_lo']:.1f}-"
                 f"{r['atm_hi']:.1f}%] -- vol {r['vol_rich']}.")
        if r['rr25'] is not None:
            L.append(f"  25d RR {r['rr25']:+.1f} at {r['p_rr']:.0f}th pctile -- skew {r['skew_rich']}.")
    rv = m['rv']
    if rv:
        L.append("IV vs REALIZED (fair value from most-recent-close composite RV)")
        L.append(f"  RV {rv['rv_iv']:.1f}% ({rv['rv_lookback']}td @ {rv['rv_asof']}) -> fair ATMF "
                 f"straddle {_fmt(rv['rv_straddle'])}.  Now: IV {rv['now_iv']:.1f}% / straddle "
                 f"{_fmt(rv['now_straddle'])}; VRP {rv['vrp_now']:+.1f} pts "
                 f"({'IV over realized' if rv['rich'] else 'IV under realized'}"
                 + (f", straddle {_fmt(rv['straddle_gap'])} vs fair" if rv['straddle_gap'] is not None else "")
                 + ").")
        if rv['open_iv'] is not None:
            L.append(f"  Open: IV {rv['open_iv']:.1f}% / straddle {_fmt(rv['open_straddle'])}; "
                     f"intraday drift {rv['drift_iv']:+.1f} pts vol"
                     + (f", {_fmt(rv['straddle_drift'])} straddle" if rv['straddle_drift'] is not None else "")
                     + f" (open {rv['open_ts']} -> now {rv['now_ts'] or '?'}).")

    p = m['position']
    if p:
        net = p['net']
        L.append("POSITION")
        L.append(f"  net {p['dpos']} {abs(net['delta']):,.0f}Δ, {p['gpos']} gamma, {p['vpos']} vega, "
                 f"theta {net['theta']:+,.0f}/day.")
    na = m['noarb']
    L.append("NO-ARBITRAGE CHECK")
    if na['ok']:
        L.append(f"  clean: smile density >= 0 (min {na['butterfly']['min_density']:.2e})"
                 + (f", total variance non-decreasing across {na['calendar']['n_pairs']} tenor pair(s)."
                    if na['calendar'] else "."))
    else:
        bf = na['butterfly']
        if bf['has_arb']:
            L.append(f"  ⚠ BUTTERFLY arb: density < 0 over ~{bf['frac_negative']:.0f}% of strikes "
                     f"(worst near {bf['worst_strike']:.0f}). The fitted smile implies a negative "
                     f"butterfly price — tighten the fit (SVI) or smooth the nodes.")
        if na['calendar'] and na['calendar']['has_arb']:
            L.append(f"  ⚠ CALENDAR arb: total variance dips with maturity (worst {na['calendar']['worst']:.4f}). "
                     f"A near tenor is pricing more variance than a longer one.")
    L.append("TRADING IMPLICATIONS (educational, not advice)")
    for imp in m['implications']:
        L.append("  - " + imp)
    return "\n".join(L)


def render_html(snap, cs):
    """Styled Dash cards (imported lazily so non-Dash use doesn't require dash)."""
    from dash import html
    m = metrics(snap, cs)
    st = m['st']
    TONE = {"good": ("#e7f6ec", "#1e7e34"), "bad": ("#fdecea", "#b21f2d"),
            "warn": ("#fff4e5", "#a15c00"), "info": ("#e8f0fe", "#1a4fce"), "neutral": ("#eef0f3", "#444")}

    def pill(text, tone="neutral"):
        bg, fg = TONE[tone]
        return html.Span(text, style={"background": bg, "color": fg, "padding": "2px 9px",
                         "borderRadius": "999px", "fontSize": "11.5px", "fontWeight": 600,
                         "marginRight": "6px", "marginBottom": "5px", "display": "inline-block"})

    def card(title, accent, body, badges=None):
        kids = [html.Div(title, style={"fontWeight": 700, "fontSize": "12px", "letterSpacing": ".03em",
                "color": "#333", "marginBottom": "7px", "textTransform": "uppercase"})]
        if badges:
            kids.append(html.Div(badges, style={"marginBottom": "7px"}))
        kids.append(html.Div(body, style={"fontSize": "13px", "color": "#333", "lineHeight": "1.55"}))
        return html.Div(kids, style={"borderLeft": f"4px solid {accent}", "background": "#fff",
                        "borderRadius": "8px", "padding": "12px 14px", "marginBottom": "10px",
                        "boxShadow": "0 1px 3px rgba(0,0,0,0.07)"})

    cards = [html.Div([html.Span(f"{snap.symbol} ", style={"fontSize": "20px", "fontWeight": 800}),
             html.Span(f"skew analysis · {snap.date} · {snap.dte:.0f} DTE",
                       style={"color": "#777", "fontSize": "13px"})], style={"marginBottom": "10px"})]
    badges = [pill(f"ATF vol {m['atf']*100:.1f}%", "info"),
              pill(f"skew {m['skew_word']} · {m['rr']:+.1f} RR", "warn" if "inverted" in m['skew_word'] else "info"),
              pill(f"smile {m['fly']:+.1f}", "neutral"),
              pill(f"{m['p_above']:.0f}% above fwd", "good" if m['p_above'] >= 50 else "bad")]
    body = [html.Div(f"Downside {m['v_dn']*100:.1f}% vs upside {m['v_up']*100:.1f}%."
                     + (" Both wings bid (fat tails)." if m['smile_bid'] else "")),
            html.Div(f"Distribution: median {st['median']:.0f}, mode {st['mode']:.0f}, std {st['std']:.0f}, "
                     f"skew {st['skew']:+.2f}, kurt {st['kurt']:+.2f} — {m['shape']}.", style={"marginTop": "4px"})]
    cards.append(card("What's going on", "#2f6feb", body, badges))

    c = m['change']
    if c:
        tone_v = "bad" if c['d_atf'] > 0.3 else "good" if c['d_atf'] < -0.3 else "neutral"
        badges = [pill(f"ATF {c['vol_word']} {c['d_atf']:+.1f}", tone_v),
                  pill(f"skew {c['skew_chg']} {c['d_rr']:+.1f}", "neutral")]
        body = [html.Div(f"{c['v_atf_p']*100:.1f}% → {c['v_atf']*100:.1f}% vol; RR {c['rr_p']:+.1f} → "
                         f"{c['rr']:+.1f}; dist skew {c['st_p_skew']:+.2f} → {c['st_skew']:+.2f}.")]
        if c['note']:
            body.append(html.Div(c['note'], style={"marginTop": "4px", "fontStyle": "italic", "color": "#555"}))
        cards.append(card(f"How it's changed · vs {snap.prev_label}", "#8b5cf6", body, badges))

    r = m['regime']
    if r:
        tv = "bad" if r['vol_rich'] == "rich" else "good" if r['vol_rich'] == "cheap" else "neutral"
        badges = [pill(f"vol {r['p_atm']:.0f}th pctile · {r['vol_rich']}", tv)]
        if r['rr25'] is not None:
            badges.append(pill(f"skew {r['p_rr']:.0f}th pctile · {r['skew_rich']}", "neutral"))
        body = [html.Div(f"ATF vol {snap.atf*100:.1f}% in range [{r['atm_lo']:.1f}–{r['atm_hi']:.1f}%], "
                         f"mean {r['atm_mean']:.1f}%.")]
        if r['rr25'] is not None:
            body.append(html.Div(f"25Δ RR {r['rr25']:+.1f} in range [{r['rr_lo']:+.1f}..{r['rr_hi']:+.1f}].",
                                 style={"marginTop": "4px"}))
        cards.append(card(f"Regime · {r['start']} → {r['end']} ({r['n']} obs)", "#0ea5a4", body, badges))

    p = m['position']
    if p:
        net = p['net']
        gt = {"long": "good", "short": "bad", "flat": "neutral"}
        badges = [pill(f"Δ {net['delta']:+,.0f}", "neutral"), pill(f"Γ {p['gpos']}", gt[p['gpos']]),
                  pill(f"Vega {p['vpos']}", gt[p['vpos']]), pill(f"Θ {net['theta']:+,.0f}/day", "neutral")]
        body = [html.Div(f"{p['gpos']} realized vol, {p['vpos']} implied vol.")]
        b = p['pnl']
        if b:
            drv = max([("realized vol", b['realized_vol']), ("implied vol", b['vega']),
                       ("delta", b['delta'] + b['share'])], key=lambda kv: abs(kv[1]))
            body.append(html.Div([pill(f"total {b['total']:+,.0f}", "good" if b['total'] >= 0 else "bad"),
                        pill(f"realized {b['realized_vol']:+,.0f}", "neutral"),
                        pill(f"implied {b['vega']:+,.0f}", "neutral")], style={"marginTop": "6px"}))
            body.append(html.Div(f"Over {b['days']:.0f}d (spot {b['dS']:+.1f}); driver: {drv[0]}.",
                                 style={"marginTop": "2px", "color": "#555"}))
        cards.append(card("Position", "#f59e0b", body, badges))

    na = m['noarb']
    if na['ok']:
        nbadges = [pill("butterfly ✓ density ≥ 0", "good")]
        if na['calendar']:
            nbadges.append(pill(f"calendar ✓ {na['calendar']['n_pairs']} pair(s)", "good"))
        nbody = [html.Div("The fitted smile is arbitrage-clean: the risk-neutral density is "
                          "non-negative everywhere"
                          + (" and total variance rises with maturity." if na['calendar'] else "."))]
        accent = "#16a34a"
    else:
        nbadges, nbody = [], []
        bf = na['butterfly']
        if bf['has_arb']:
            nbadges.append(pill(f"butterfly ✗ {bf['frac_negative']:.0f}% neg", "bad"))
            nbody.append(html.Div(f"Negative density over ~{bf['frac_negative']:.0f}% of strikes "
                         f"(worst near {bf['worst_strike']:.0f}) — implies a negative butterfly price. "
                         f"Prefer the SVI fit or smooth the IV nodes."))
        if na['calendar'] and na['calendar']['has_arb']:
            nbadges.append(pill("calendar ✗", "bad"))
            nbody.append(html.Div(f"Total variance dips with maturity (worst {na['calendar']['worst']:.4f}).",
                                  style={"marginTop": "4px"}))
        accent = "#b21f2d"
    cards.append(card("No-arbitrage check", accent, nbody, nbadges))

    rv = m['rv']
    if rv:
        tone_vrp = "bad" if rv['vrp_now'] > 0 else "good"     # IV rich vs realized = expensive
        rbadges = [pill(f"RV {rv['rv_iv']:.1f}% · {rv['rv_lookback']}td", "info"),
                   pill(f"VRP now {rv['vrp_now']:+.1f}", tone_vrp)]
        if rv['open_iv'] is not None:
            rbadges.append(pill(f"drift {rv['drift_iv']:+.1f}", "neutral"))
        rbody = [html.Div(f"Fair (RV @ {rv['rv_asof']}): IV {rv['rv_iv']:.1f}% · straddle "
                          f"{_fmt(rv['rv_straddle'])}."),
                 html.Div(f"Now: IV {rv['now_iv']:.1f}% · straddle {_fmt(rv['now_straddle'])}"
                          + (f" ({_fmt(rv['straddle_gap'])} vs fair)" if rv['straddle_gap'] is not None else "")
                          + f". VRP {rv['vrp_now']:+.1f} pts — "
                          + ("IV richer than realized." if rv['rich'] else "IV cheaper than realized."),
                          style={"marginTop": "4px"})]
        if rv['open_iv'] is not None:
            rbody.append(html.Div(f"Open ({rv['open_ts']}): IV {rv['open_iv']:.1f}% · straddle "
                                  f"{_fmt(rv['open_straddle'])}. Intraday drift {rv['drift_iv']:+.1f} pts"
                                  + (f", {_fmt(rv['straddle_drift'])} straddle" if rv['straddle_drift'] is not None else "")
                                  + ".", style={"marginTop": "4px", "color": "#555"}))
        cards.append(card("RV vs IV · realized-implied fair value", "#0891b2", rbody, rbadges))

    cards.append(card("Trading implications · educational, not advice", "#6b7280",
                 html.Ul([html.Li(t, style={"marginBottom": "5px"}) for t in m['implications']],
                         style={"margin": "0", "paddingLeft": "18px", "fontSize": "13px"})))
    return cards
