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
        "positions":      snap.positions,
    }


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
