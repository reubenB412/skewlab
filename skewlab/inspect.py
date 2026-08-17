"""skewlab.inspect — peek at every dataframe / array a run produced.

After ``snap = main(cfg)``, the immutable ``Snapshot`` holds the whole run context.
``collect_run_data(snap)`` flattens it into a ``{name: object}`` dict and
``describe_run_data(dict)`` pretty-prints a summary. Handy in the interactive window:

    from skewlab.inspect import collect_run_data, describe_run_data
    DATA = collect_run_data(snap)
    describe_run_data(DATA)
    DATA["chain"]        # the cleaned option chain the grid was actually built from
    DATA["grid"]         # z-node / strike / fitted grid-vol% / skew% table
    DATA["market_iv"]    # per-strike market IVs the SD grid was seeded from
    DATA["scalars"]      # spot / forward / ATF vol% / one_sd / r / q / dte ...
    DATA["term"][10]     # per-tenor term-structure bundle (strikes, vols, expiry)

This is the first place to look when the skew curve looks wrong: compare
``DATA["scalars"]["ATF_vol_%"]`` and ``DATA["grid"]`` against ``DATA["market_iv"]``
and ``DATA["chain"]`` around the forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _today_rr25(snap):
    """Today's live 25Δ risk reversal (put IV − call IV, in vol pts) from the fitted
    market curve — the same calc the iv_history panel uses for the 'today' star."""
    try:
        from . import model
        from .data import CurveState
        cs = CurveState.market(snap)
        Ks = np.linspace(snap.grid_strikes[0], snap.grid_strikes[-1], 200)
        sig = snap.curve_vol(Ks, cs)
        dC = np.array([model.bs_price_delta(snap.spot, K, snap.t, snap.r, snap.q, s, True)[1]
                       for K, s in zip(Ks, sig)])
        dP = np.array([model.bs_price_delta(snap.spot, K, snap.t, snap.r, snap.q, s, False)[1]
                       for K, s in zip(Ks, sig)])
        c25 = float(sig[int(np.argmin(np.abs(dC - 0.25)))])
        p25 = float(sig[int(np.argmin(np.abs(dP + 0.25)))])
        return (p25 - c25) * 100.0
    except Exception:
        return float("nan")


def iv_history_frame(snap, include_today=True):
    """Combine the four 'IV history & regime' subplots into ONE time-indexed DataFrame.

    Columns (all in vol points / %):
        atm_implied_%  — ATM implied vol            (top-left blue, top-right line)
        realized_%     — realized vol               (top-left gray)
        vrp_pts        — implied − realized          (top-left green fill / carry)
        rr25_pts       — 25Δ risk reversal (put−call)(bottom-left red, bottom-right y)
        rr10_pts       — 10Δ risk reversal          (bottom-left orange dotted)
        seq            — 0..n-1 plot order (the bottom-right scatter colour)
        is_today       — True on the trailing live row (the star)

    The bottom-right 'Vol vs skew regime' scatter is just rr25_pts (y) vs atm_implied_%
    (x). Returns an empty frame if no IV history was built for this symbol.
    """
    if snap.iv_atm is None or len(snap.iv_atm.dropna()) == 0:
        return pd.DataFrame()
    atm = snap.iv_atm.dropna() * 100.0
    df = pd.DataFrame(index=pd.DatetimeIndex(atm.index))
    df.index.name = "date"
    df["atm_implied_%"] = atm.values

    if snap.iv_rv is not None:
        df["realized_%"] = (snap.iv_rv.reindex(atm.index) * 100.0).values
        df["vrp_pts"] = df["atm_implied_%"] - df["realized_%"]

    cols = snap.iv_history.columns if snap.iv_history is not None else []
    if {"25d_put", "25d_call"}.issubset(cols):
        df["rr25_pts"] = ((snap.iv_history["25d_put"] - snap.iv_history["25d_call"])
                          .reindex(atm.index) * 100.0).values
    if {"10d_put", "10d_call"}.issubset(cols):
        df["rr10_pts"] = ((snap.iv_history["10d_put"] - snap.iv_history["10d_call"])
                          .reindex(atm.index) * 100.0).values

    df["seq"] = np.arange(len(df))
    df["is_today"] = False

    if include_today:
        row = {c: np.nan for c in df.columns}
        row["atm_implied_%"] = snap.atf * 100.0
        if "rr25_pts" in df.columns:
            row["rr25_pts"] = _today_rr25(snap)
        if "realized_%" in df.columns and df["realized_%"].dropna().size:
            row["realized_%"] = float(df["realized_%"].dropna().iloc[-1])
            row["vrp_pts"] = row["atm_implied_%"] - row["realized_%"]
        row["seq"] = len(df)
        row["is_today"] = True
        df.loc[pd.to_datetime(snap.date)] = row

    return df


def vol_history_frame(snap):
    """The full daily vol-history timeline: outer-join the date-indexed IV sources on date.

    Joins (each keeps its own columns):
        iv_atm      -> 'atm_iv'        (daily ATM implied vol, decimals)
        iv_rv       -> 'realized_vol'  (daily realized vol, decimals)
        iv_history  -> raw bucket cols ('10d_put','25d_put','atm','25d_call','10d_call')

    This is the RAW merge of the underlying series on one timeline. (For the charts'
    curated %/points version — with VRP, risk reversals and today's live row — use
    `iv_history_frame`.) Returns an empty frame if no IV history was built.
    """
    frames = []
    if snap.iv_atm is not None and len(snap.iv_atm):
        frames.append(snap.iv_atm.rename("atm_iv").to_frame())
    if snap.iv_rv is not None and len(snap.iv_rv):
        frames.append(snap.iv_rv.rename("realized_vol").to_frame())
    if snap.iv_history is not None and not getattr(snap.iv_history, "empty", True):
        frames.append(snap.iv_history.copy())

    if not frames:
        return pd.DataFrame()
    # normalise each index to a tz-naive DatetimeIndex so the join aligns on calendar day
    for fr in frames:
        fr.index = pd.DatetimeIndex(fr.index).tz_localize(None).normalize()
    out = pd.concat(frames, axis=1)               # outer join on the date index
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out.index.name = "date"
    return out


def plot_vol_history(data):
    """Quick line plot of the vol_history frame — a safe no-op if it's empty (e.g. a
    symbol with no EOD options coverage, so no IV history was built). Accepts the DATA
    dict or a Snapshot."""
    df = data["vol_history"] if isinstance(data, dict) else vol_history_frame(data)
    num = df.select_dtypes("number").dropna(how="all", axis=1).dropna(how="all") if len(df) else df
    if num is None or num.empty:
        print("[inspect] vol_history is empty (no IV history for this symbol/date) — nothing to plot.")
        return None
    return num.plot(title="vol history")


def diagnose_market_iv(snap, band_pct=0.06, jump_pts=1.0, verbose=True):
    """Investigate the grey 'market IVs' scatter (``snap.market_iv``) and find what causes a
    visual disconnect.

    The plotted series is the OTM branch: put IV at strikes <= forward, call IV above, with
    ``implied_vol`` backfilling any gaps (see model.market_iv_by_strike). A disconnect is
    almost always one of:
      • a PUT/CALL IV BASIS at the forward crossover -> a vertical step where the branch flips
      • ``implied_vol`` BACKFILL sitting off the put/call branch -> a parallel offset strand
      • DUPLICATE strikes -> two points at one strike
      • a SYSTEMATIC OFFSET vs the fitted curve -> a scaling / day-count (T) mismatch

    Returns a per-strike DataFrame; prints a summary when ``verbose``. Columns (vol %):
    iv_put_%, iv_call_%, implied_vol_%, market_iv_% (what's plotted), source, side,
    putcall_basis_pts (iv_put-iv_call), fitted_% (the fitted curve), resid_pts
    (market_iv - fitted), jump_pts (strike-to-strike |Δ| of the plotted series).
    """
    ch = getattr(snap, "chain", None)
    if ch is None or len(ch) == 0:
        print("[iv-diag] no chain on this snapshot (nothing to investigate).")
        return pd.DataFrame()
    F = float(snap.forward)
    df = ch.sort_index()
    K = df.index.values.astype(float)
    ivp = pd.to_numeric(df.get("iv_put"), errors="coerce")
    ivc = pd.to_numeric(df.get("iv_call"), errors="coerce")
    ivb = pd.to_numeric(df.get("implied_vol"), errors="coerce")

    branch = np.where(K <= F, ivp.values, ivc.values)          # the where() OTM pick
    source = np.where(K <= F, "put", "call").astype(object)
    plotted = pd.Series(branch, index=df.index)
    backfilled = plotted.isna() & ivb.notna()                   # combine_first(implied_vol)
    plotted = plotted.where(~backfilled, ivb)
    source = np.where(backfilled.values, "implied_vol", source)

    # fitted market curve at each strike (what the blue line would say here)
    fitted = np.full(len(df), np.nan)
    try:
        from .data import CurveState
        fitted = np.asarray(snap.curve_vol(K, CurveState.market(snap)), float)
    except Exception as e:
        if verbose:
            print(f"[iv-diag] (fitted-curve residual unavailable: {e})")

    out = pd.DataFrame({
        "iv_put_%": ivp.values * 100.0, "iv_call_%": ivc.values * 100.0,
        "implied_vol_%": ivb.values * 100.0, "market_iv_%": plotted.values * 100.0,
        "source": source, "side": np.where(K <= F, "<=F", ">F"),
        "putcall_basis_pts": (ivp.values - ivc.values) * 100.0,
        "fitted_%": fitted * 100.0,
        "resid_pts": (plotted.values - fitted) * 100.0,
    }, index=df.index)
    out.index.name = "strike"
    out["jump_pts"] = out["market_iv_%"].diff().abs()

    dup = pd.Index(df.index[df.index.duplicated(keep=False)]).unique()
    band = out[(out.index >= F * (1 - band_pct)) & (out.index <= F * (1 + band_pct))]
    below, above = out[out.index <= F], out[out.index > F]
    step = (float(above["market_iv_%"].iloc[0] - below["market_iv_%"].iloc[-1])
            if len(below) and len(above) else float("nan"))
    big = out[out["jump_pts"] > jump_pts]

    if verbose:
        print("\n" + "=" * 74)
        print(f"[iv-diag] MARKET-IV DISCONNECT — {snap.symbol} {snap.date}  (forward {F:.2f})")
        print("=" * 74)
        print(f"  strikes: {len(df)}   plotted market-IV points: {int(out['market_iv_%'].notna().sum())}")
        if len(band):
            print(f"  put/call basis within ±{band_pct*100:.0f}% of fwd: mean {band['putcall_basis_pts'].mean():+.2f} pts, "
                  f"max |{band['putcall_basis_pts'].abs().max():.2f}| pts")
        print(f"  STEP across the forward crossover (put→call): {step:+.2f} vol pts"
              + ("   <-- the disconnect" if np.isfinite(step) and abs(step) >= jump_pts else ""))
        n_bf = int((out["source"] == "implied_vol").sum())
        print(f"  implied_vol-backfilled strikes: {n_bf}"
              + ("   (can sit off the put/call branch → a parallel offset strand)" if n_bf else ""))
        print(f"  duplicate strikes: {len(dup)}"
              + (f" → {list(dup)[:10]}" if len(dup) else ""))
        rr = out["resid_pts"].dropna()
        if len(rr):
            print(f"  residual vs fitted curve: mean {rr.mean():+.2f} pts, std {rr.std():.2f} "
                  f"(a large SYSTEMATIC mean ⇒ scaling / day-count T mismatch)")
        if len(big):
            print(f"  strike-to-strike jumps > {jump_pts} pts ({len(big)}) — likely disconnect points:")
            with pd.option_context("display.max_rows", 15, "display.width", 160):
                print(big[["market_iv_%", "source", "side", "putcall_basis_pts", "jump_pts"]].head(15).to_string())
        else:
            print(f"  no strike-to-strike jumps > {jump_pts} pts.")
        print("=" * 74)
    return out


def plot_market_iv_diagnosis(snap, band_pct=0.06):
    """Visualise the market-IV disconnect: iv_put (red) vs iv_call (blue) per strike, the
    actually-plotted OTM series (grey), and the fitted curve (line), with the forward marked.
    Makes a put/call basis, a backfill strand, or a systematic offset obvious at a glance."""
    import matplotlib.pyplot as plt
    d = diagnose_market_iv(snap, band_pct=band_pct, verbose=False)
    if d is None or d.empty:
        print("[iv-diag] nothing to plot.")
        return None
    F = float(snap.forward)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(d.index, d["iv_put_%"], s=14, c="#ef553b", alpha=0.7, label="iv_put")
    ax.scatter(d.index, d["iv_call_%"], s=14, c="#2f6feb", alpha=0.7, label="iv_call")
    ax.scatter(d.index, d["market_iv_%"], s=26, facecolors="none", edgecolors="#555",
               label="market_iv (plotted, OTM branch)")
    bf = d[d["source"] == "implied_vol"]
    if len(bf):
        ax.scatter(bf.index, bf["market_iv_%"], s=42, c="#f59e0b", marker="x",
                   label="implied_vol backfill")
    if d["fitted_%"].notna().any():
        ax.plot(d.index, d["fitted_%"], color="#636efa", lw=2, label="fitted curve")
    ax.axvline(F, color="green", ls=":", lw=1.2, label=f"forward {F:.1f}")
    ax.set_title(f"{snap.symbol} {snap.date} — market-IV provenance (put vs call vs plotted)")
    ax.set_xlabel("strike"); ax.set_ylabel("implied vol (%)")
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout(); plt.show()
    return d


def rv_compare_frame(snap):
    """The RV-vs-IV scorecard as a 3-row DataFrame: the RV-implied fair value (from the
    most-recent-close composite realized vol) vs the market at the day's OPEN and NOW.

    Columns: iv_% (ATM implied vol), straddle ($ ATM-forward), when (timestamp/date).
    Empty frame if the RV comparison wasn't computed (show_rv_compare off, or no RV)."""
    rv_iv = getattr(snap, "rv_iv", None)
    if rv_iv is None or not np.isfinite(rv_iv):
        return pd.DataFrame()
    rows, idx = [], []
    rows.append({"iv_%": rv_iv * 100.0, "straddle": getattr(snap, "rv_straddle", np.nan),
                 "when": getattr(snap, "rv_asof", None)})
    idx.append("RV_fair")
    open_atf = getattr(snap, "open_atf", None)
    if open_atf is not None and np.isfinite(open_atf):
        rows.append({"iv_%": float(open_atf) * 100.0, "straddle": getattr(snap, "open_straddle", np.nan),
                     "when": getattr(snap, "open_capture_ts", None)})
        idx.append("open")
    rows.append({"iv_%": float(snap.atf) * 100.0, "straddle": getattr(snap, "now_straddle", np.nan),
                 "when": getattr(snap, "now_capture_ts", None)})
    idx.append("now")
    df = pd.DataFrame(rows, index=idx)
    df.index.name = f"{snap.symbol} rv{getattr(snap, 'rv_lookback', '')}"
    return df


def collect_run_data(snap):
    """Gather every dataframe / series / array / scalar the run produced into a dict,
    so it can be printed or inspected piece-by-piece in the interactive window."""
    z = np.asarray(snap.z_grid, float)
    grid = pd.DataFrame({
        "z_sd":        z,
        "strike":      np.asarray(snap.grid_strikes, float),
        "grid_vol_%":  np.asarray(snap.grid_vols, float) * 100.0,
        "skew_%":      [round(100.0 * snap.skew_pct.get(zz, float("nan")), 2) for zz in z],
    })
    scalars = pd.Series({
        "symbol":      snap.symbol,
        "date":        snap.date,
        "requested":   snap.requested_date,
        "spot":        snap.spot,
        "forward":     snap.forward,
        "ATF_vol_%":   snap.atf * 100.0,
        "one_sd":      snap.one_sd,
        "dte":         snap.dte,
        "T_years":     snap.t,
        "r":           snap.r,
        "q":           snap.q,
        "skew_model":  snap.cfg.skew_model,
        "n_positions": len(snap.positions or []),
        "shares":      snap.shares,
    })
    term = {}
    for b in (snap.term_bundles or []):
        term[b.tenor] = pd.DataFrame({
            "strike":     np.asarray(b.grid_strikes, float),
            "grid_vol_%": np.asarray([b.poly(k) for k in b.grid_strikes], float) * 100.0,
        }).assign(expiry=str(getattr(b, "expiry", "")), dte=getattr(b, "dte", None),
                  ATF_vol_pct=getattr(b, "atf", float("nan")) * 100.0)

    rv_term = getattr(snap, "rv_term", None)
    rt_table = pd.DataFrame() if rv_term is None else rv_term.estimator_table
    rt_integrated = pd.DataFrame() if rv_term is None else rv_term.integrated_variance_table
    rt_iv = pd.DataFrame() if rv_term is None else rv_term.iv_curve
    rt_daily = pd.DataFrame() if rv_term is None else rv_term.hf_daily
    rt_shape = pd.DataFrame() if rv_term is None else rv_term.shape_history
    rt_shares = pd.DataFrame() if rv_term is None else rv_term.variance_shares
    rt_signature = pd.DataFrame() if rv_term is None else rv_term.volatility_signature
    rt_summary = {} if rv_term is None else dict(rv_term.summary)
    rt_meta = {} if rv_term is None else {
        **dict(rv_term.metadata), "warnings": " | ".join(rv_term.warnings) or "none"}

    return {
        "scalars":        scalars,
        "grid":           grid,
        "market_iv":      snap.market_iv,
        "chain":          getattr(snap, "chain", None),
        "chain_prev":     snap.chain_prev,
        "mkt_curve":      pd.DataFrame({"strike": snap.mkt_curve_x, "vol_%": snap.mkt_curve_y}),
        "mkt_pdf":        pd.DataFrame({"strike": snap.mkt_pdf_x, "density": snap.mkt_pdf_y}),
        # daily vol history (iv_atm + iv_rv + iv_history joined on date — replaces the
        # three separate raw keys); iv_panel is the curated %/pts version for the charts.
        "vol_history":    vol_history_frame(snap),
        "iv_panel":       iv_history_frame(snap),
        "vix_vvix_ratio": snap.vix_vvix_ratio,
        "term":           term,
        "rv_compare":     rv_compare_frame(snap),
        "rv_estimators":  getattr(snap, "rv_estimators", None),
        "rv_term_summary": rt_summary,
        "rv_term_table":   rt_table,
        "rv_term_integrated_variance": rt_integrated,
        "rv_term_iv":      rt_iv,
        "rv_term_hf_daily": rt_daily,
        "rv_term_shape":   rt_shape,
        "rv_term_shares":  rt_shares,
        "rv_term_signature": rt_signature,
        "rv_term_metadata": rt_meta,
        "positions":      snap.positions,
    }


def _llm_csv(title, obj, max_rows=None):
    """Render a Series/DataFrame as a compact fenced CSV block for an LLM prompt."""
    if obj is None:
        return f"### {title}\n\nUnavailable."
    if isinstance(obj, pd.Series):
        obj = obj.rename(obj.name or "value").to_frame()
    if not isinstance(obj, pd.DataFrame) or obj.empty:
        return f"### {title}\n\nUnavailable."
    out = obj.tail(int(max_rows)) if max_rows is not None else obj
    return f"### {title}\n\n```csv\n{out.to_csv(float_format='%.6g').strip()}\n```"


def _near_forward_quotes(snap, rows=15):
    """Return a small quote sample around the forward instead of the full raw chain."""
    chain = getattr(snap, "chain", None)
    if chain is None or getattr(chain, "empty", True):
        return pd.DataFrame()
    df = chain.copy()
    strikes = pd.to_numeric(pd.Series(df.index, index=df.index), errors="coerce")
    df = df.assign(_distance=(strikes - float(snap.forward)).abs().values)
    df = df.sort_values("_distance").head(max(int(rows), 1)).sort_index()
    wanted = [
        "expiration", "dte", "mid_call", "mid_put", "straddle", "implied_vol",
        "iv_put", "iv_call", "delta_p", "delta_c", "oi_put", "oi_call",
        "v_put", "v_call", "S", "R", "Q", "_distance",
    ]
    out = df[[c for c in wanted if c in df.columns]].copy()
    for col in ("implied_vol", "iv_put", "iv_call", "R", "Q"):
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce") * 100.0
            out = out.rename(columns={col: col + "_%"})
    out.index.name = "strike"
    return out


def print_llm_context(snap, history_rows=20, chain_rows=15):
    """Print a compact, paste-ready Markdown/CSV briefing from the canonical snapshot.

    This export is independent of Dash and contains no vendor credentials or private
    data access. Large frames are reduced to useful recent windows and near-forward
    quotes. Missing sections are labelled rather than fabricated.
    """
    from . import analysis
    from .data import CurveState

    history_rows = max(int(history_rows), 1)
    chain_rows = max(int(chain_rows), 1)
    cs = CurveState.market(snap)
    metrics = analysis.metrics(snap, cs)
    data = collect_run_data(snap)
    st = metrics["st"]
    current = pd.Series({
        "symbol": snap.symbol,
        "observation_date": snap.date,
        "requested_date": snap.requested_date,
        "expiry_dte": snap.dte,
        "spot": snap.spot,
        "forward": snap.forward,
        "one_sd_points": snap.one_sd,
        "atf_iv_%": metrics["atf"] * 100.0,
        "down_1sd_iv_%": metrics["v_dn"] * 100.0,
        "up_1sd_iv_%": metrics["v_up"] * 100.0,
        "one_sd_rr_put_minus_call_pts": metrics["rr"],
        "one_sd_fly_pts": metrics["fly"],
        "probability_above_forward_%": metrics["p_above"],
        "distribution_mean": st["mean"],
        "distribution_median": st["median"],
        "distribution_mode": st["mode"],
        "distribution_std_points": st["std"],
        "distribution_skew": st["skew"],
        "distribution_excess_kurtosis": st["kurt"],
        "risk_free_rate_%": snap.r * 100.0,
        "dividend_yield_%": snap.q * 100.0,
        "skew_model": snap.cfg.skew_model,
    }, name="value")

    sections = [
        f"# SKEWLAB LLM CONTEXT — {snap.symbol} — {snap.date}",
        """## Analysis request

Use the evidence below to explain the current volatility and skew state; changes versus
the previous observation; implied-versus-realized relative value; the distribution, term
structure and convexity indicators; and conditional structures, risks, and invalidation
conditions. Separate observed facts from inference. Do not invent missing data, catalysts,
prices, fills, or forecasts. Flag stale or missing data and arbitrage warnings first. This
is scenario analysis, not personalized financial advice.""",
        "## Existing skewlab narrative\n\n```text\n" + analysis.render_text(snap, cs) + "\n```",
        _llm_csv("Current market and fitted-distribution metrics", current),
    ]

    notes = list(getattr(snap, "data_notes", None) or [])
    complete = data["rv_term_hf_daily"].get(
        "is_complete_session", pd.Series(dtype=bool))
    iv_values = data["rv_term_iv"].get("atf_iv", pd.Series(dtype=float))
    coverage = pd.Series({
        "data_notes": " | ".join(notes) if notes else "none",
        "current_chain_rows": 0 if data["chain"] is None else len(data["chain"]),
        "previous_chain_rows": 0 if data["chain_prev"] is None else len(data["chain_prev"]),
        "iv_history_rows": len(data["iv_panel"]),
        "rv_estimator_rows": 0 if data["rv_estimators"] is None else len(data["rv_estimators"]),
        "rv_term_hf_completed_sessions": int(complete.sum()),
        "rv_term_iv_tenors": int(iv_values.notna().sum()),
        "term_tenors_available": len(snap.term_bundles or []),
        "has_previous_fit": snap.prev_poly is not None,
        "has_positions": snap.has_positions,
    }, name="value")
    sections.append(_llm_csv("Data coverage and caveats", coverage))

    change = metrics.get("change")
    if change:
        sections.append(_llm_csv("Change versus previous observation", pd.Series({
            "previous_label": snap.prev_label,
            "previous_atf_iv_%": change["v_atf_p"] * 100.0,
            "current_atf_iv_%": change["v_atf"] * 100.0,
            "atf_change_pts": change["d_atf"],
            "previous_1sd_rr_pts": change["rr_p"],
            "current_1sd_rr_pts": change["rr"],
            "rr_change_pts": change["d_rr"],
            "previous_distribution_skew": change["st_p_skew"],
            "current_distribution_skew": change["st_skew"],
            "previous_distribution_std": change["std_p"],
            "current_distribution_std": change["std"],
            "rule_based_note": change["note"] or "none",
        }, name="value")))

    for key, title in (
        ("regime", "Historical IV and skew regime"),
        ("rv", "Implied versus realized and open-to-now"),
    ):
        value = metrics.get(key)
        if value:
            sections.append(_llm_csv(title, pd.Series(value, name="value")))

    if data["rv_term_summary"]:
        sections.extend([
            _llm_csv("HF realised-vol regime and daily-movement summary",
                     pd.Series(data["rv_term_summary"], name="value")),
            _llm_csv("RV term-structure source metadata and warnings",
                     pd.Series(data["rv_term_metadata"], name="value")),
            _llm_csv("Current RV estimator term structure", data["rv_term_table"]),
            _llm_csv("Backward RV integrated variance", data["rv_term_integrated_variance"]),
            _llm_csv("Forward ATM IV curve", data["rv_term_iv"]),
            _llm_csv("Current HF variance-composition shares", data["rv_term_shares"]),
        ])
        if not data["rv_term_signature"].empty:
            sections.append(_llm_csv("HF volatility-signature diagnostic",
                                     data["rv_term_signature"]))
        sections.append(_llm_csv(
            f"Recent HF RV curve-shape history (last {history_rows})",
            data["rv_term_shape"], max_rows=history_rows))

    noarb = metrics["noarb"]
    noarb_rows = {"overall_clean": noarb["ok"]}
    noarb_rows.update({f"butterfly_{k}": v for k, v in noarb["butterfly"].items()})
    if noarb["calendar"] is not None:
        noarb_rows.update({f"calendar_{k}": v for k, v in noarb["calendar"].items()})
    sections.append(_llm_csv("No-arbitrage diagnostics", pd.Series(noarb_rows, name="value")))

    term_rows = [{
        "requested_tenor": b.tenor, "actual_dte": b.dte, "expiry": b.expiry,
        "monthly": b.monthly, "forward": b.forward, "atf_iv_%": b.atf * 100.0,
        "total_atf_variance": b.atf * b.atf * b.t,
    } for b in (snap.term_bundles or [])]
    sections.extend([
        _llm_csv("Available term structure", pd.DataFrame(term_rows)),
        _llm_csv("Fitted skew nodes", data["grid"]),
    ])

    if metrics.get("position"):
        pos = metrics["position"]
        net = {k: v for k, v in pos["net"].items() if k != "rows"}
        net.update({"delta_direction": pos["dpos"], "gamma_direction": pos["gpos"],
                    "vega_direction": pos["vpos"]})
        sections.append(_llm_csv("Position net Greeks and premium", pd.Series(net, name="value")))
        sections.append(_llm_csv("Position legs", pd.DataFrame(
            snap.positions or [], columns=["strike", "type", "contracts"])))

    sections.extend([
        _llm_csv(f"Recent IV/RV and skew history (last {history_rows})",
                 data["iv_panel"], max_rows=history_rows),
        _llm_csv(f"Recent realized-vol estimator stack (last {history_rows})",
                 data["rv_estimators"], max_rows=history_rows),
        _llm_csv(f"Recent VVIX/VIX regime (last {history_rows})",
                 data["vix_vvix_ratio"], max_rows=history_rows),
        _llm_csv(f"Current option quotes nearest forward ({chain_rows} strikes)",
                 _near_forward_quotes(snap, chain_rows)),
    ])
    print("\n\n".join(sections).rstrip() + "\n")


def describe_run_data(d):
    """Pretty-print a summary of each item from `collect_run_data`."""
    def _show(name, obj):
        print("\n" + "=" * 78)
        print(name)
        print("-" * 78)
        if obj is None:
            print("  (none)")
        elif isinstance(obj, pd.DataFrame):
            print(f"  DataFrame {obj.shape[0]}x{obj.shape[1]}  cols={list(obj.columns)}")
            with pd.option_context("display.max_rows", 20, "display.width", 160):
                print(obj.head(20).to_string())
        elif isinstance(obj, pd.Series):
            with pd.option_context("display.max_rows", 40, "display.width", 160):
                print(obj.to_string())
        elif isinstance(obj, dict):
            if not obj:
                print("  (empty)")
            for k, v in obj.items():
                print(f"  [{k}]")
                if isinstance(v, pd.DataFrame):
                    print(v.to_string())
                else:
                    print(f"    {v}")
        else:
            print(f"  {obj}")

    print("\n" + "#" * 78)
    print("# RUN DATA INSPECTION  —  access any piece via DATA['<name>']")
    print("#" * 78)
    for name, obj in d.items():
        _show(name, obj)
    print("\n" + "#" * 78)
    print("# keys:", list(d.keys()))
    print("#" * 78)
