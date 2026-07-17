"""Synthetic smoke test: build a fake Snapshot (no pipeline) and render every chart."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RunConfig
from . import model
from .data import Snapshot, CurveState, TermBundle
from . import analysis
from . import charts as charts_pkg


def make_fake_snapshot(with_prev=True, with_hist=True, with_pos=True):
    cfg = RunConfig(symbol="TEST")
    if not with_pos:
        cfg.positions, cfg.shares = [], 0
    z = cfg.z_grid
    spot, r, q, t = 600.0, 0.04, 0.013, 30 / 365.0
    atf = 0.18
    fwd = spot * np.exp((r - q) * t)
    one_sd = fwd * atf * np.sqrt(t)
    grid_strikes = fwd + z * one_sd
    # a realistic put-skewed smile
    skew_pct = {-3.: 0.42, -2.: 0.24, -1.: 0.10, 0.: 0.0, 1.: -0.05, 2.: -0.04, 3.: 0.02}
    grid_vols = np.array([atf * (1.0 + skew_pct[zz]) for zz in z])
    poly = model.fit_skew_curve(grid_strikes, grid_vols, cfg.poly_degree)
    miv = pd.Series(np.interp(grid_strikes, grid_strikes, grid_vols), index=grid_strikes, name="iv")

    prev_poly = None
    prev_label = None
    chain_prev = None
    if with_prev:
        gv_p = grid_vols * 0.95
        prev_poly = model.fit_skew_curve(grid_strikes, gv_p, cfg.poly_degree)
        prev_label = "2026-05-19 · 30DTE · same exp"
        chain_prev = pd.DataFrame({  # minimal columns _core() needs
            "midpoint": [1.0, 0.5, 2.0], "S": [590.0] * 3, "R": [r] * 3, "Q": [q] * 3,
            "dte": [42.0] * 3, "implied_vol": [atf * 1.02] * 3})

    iv_atm = iv_history = iv_rv = None
    if with_hist:
        idx = pd.bdate_range("2025-12-01", periods=120)
        rng = np.random.default_rng(0)
        atm_s = 0.18 + 0.03 * np.sin(np.linspace(0, 6, 120)) + rng.normal(0, 0.004, 120)
        iv_atm = pd.Series(atm_s, index=idx)
        iv_history = pd.DataFrame({
            "10d_put": atm_s + 0.05, "25d_put": atm_s + 0.025, "atm": atm_s,
            "25d_call": atm_s - 0.01, "10d_call": atm_s - 0.005}, index=idx)
        iv_rv = pd.Series(atm_s - 0.03 + rng.normal(0, 0.003, 120), index=idx)

    term_bundles = []
    for tn, dte in [(10, 12), (90, 91)]:
        b_atf = atf + (0.005 if tn == 10 else -0.01)
        b_fwd = spot * np.exp((r - q) * dte / 365.0)
        b_one = b_fwd * b_atf * np.sqrt(dte / 365.0)
        b_gs = b_fwd + z * b_one
        b_gv = np.array([b_atf * (1.0 + skew_pct[zz]) for zz in z])
        term_bundles.append(TermBundle(
            tenor=tn, monthly=(tn == 90), poly=model.fit_skew_curve(b_gs, b_gv, cfg.poly_degree),
            grid_strikes=b_gs, forward=b_fwd, atf=b_atf, one_sd=b_one, t=dte / 365.0, r=r, q=q,
            spot=spot, dte=dte, expiry=pd.Timestamp("2026-06-30"), hist=iv_history))

    # synthetic VIX / VVIX panels
    vidx = pd.bdate_range("2024-01-01", periods=400)
    rng2 = np.random.default_rng(1)
    vix_c = pd.Series(15 + 6 * np.abs(rng2.normal(0, 1, 400)).cumsum() / 30, index=vidx)
    vvix_c = pd.Series(90 + 25 * np.abs(rng2.normal(0, 1, 400)).cumsum() / 30, index=vidx)
    vix_dist = model.cumulative_price_distribution(vix_c, 1.0)
    vvix_dist = model.cumulative_price_distribution(vvix_c, 2.0)
    vix_dist_since = model.cumulative_price_distribution(vix_c.loc["2025-06-01":], 1.0)
    vvix_dist_since = model.cumulative_price_distribution(vvix_c.loc["2025-06-01":], 2.0)
    vix_vvix_ratio = model.vvix_vix_ratio_table(vix_c, vvix_c)

    snap = Snapshot(
        cfg=cfg, symbol="TEST", date="2026-06-18", requested_date="2026-06-18",
        spot=spot, forward=fwd, t=t, r=r, q=q, dte=30, atf=atf, one_sd=one_sd, z_grid=z,
        grid_strikes=grid_strikes, grid_vols=grid_vols, skew_pct=skew_pct, market_iv=miv, poly=poly,
        mkt_curve_x=np.array([]), mkt_curve_y=np.array([]), mkt_pdf_x=np.array([]), mkt_pdf_y=np.array([]),
        chain_prev=chain_prev, prev_poly=prev_poly, prev_label=prev_label,
        prev_obs_date=pd.Timestamp("2026-05-19"), term_bundles=term_bundles,
        iv_atm=iv_atm, iv_history=iv_history, iv_rv=iv_rv,
        vix_dist=vix_dist, vvix_dist=vvix_dist, vix_dist_since=vix_dist_since,
        vvix_dist_since=vvix_dist_since, vix_vvix_ratio=vix_vvix_ratio, since_when="2025-06-01",
        # note: Snapshot keeps `since_when` as the internal date field; config knob is vix_dist_since
        positions=(cfg.positions or []), shares=(cfg.shares or 0))
    fine = snap.fine_strikes(cfg.wings_on)
    snap.mkt_curve_x = fine
    snap.mkt_curve_y = snap.curve_vol(fine, CurveState.market(snap)) * 100.0
    calls = model.bs_call_vec(fine, snap.mkt_curve_y / 100.0, spot, t, r, q)
    pdf = model.implied_pdf(fine, calls, r, t)
    snap.mkt_pdf_x, snap.mkt_pdf_y = fine[2:-2], pdf[2:-2]
    return snap


def run():
    snap = make_fake_snapshot()
    cs_m = CurveState.market(snap)
    cs_s = CurveState.from_scenario(snap, "Risk-off / crash (vol up, steep put skew)")
    cs_g = CurveState.from_grid(snap, snap.grid_vols * 1.1, 0.06, 0.02, True)

    print("active charts:", [c.key for c in charts_pkg.active(snap)])
    for c in charts_pkg.active(snap):
        for tag, cs in [("market", cs_m), ("scenario", cs_s), ("grid", cs_g)]:
            fig = c.make(snap, cs)
            assert fig is not None, f"{c.key}/{tag} returned None"
            assert len(fig.data) > 0 or len(fig.layout.annotations) > 0, f"{c.key}/{tag} empty"
        print(f"  ok  {c.key:18s} ({len(c.make(snap, cs_m).data)} traces)")

    txt = analysis.render_text(snap, cs_m)
    assert "BESPOKE ANALYSIS" in txt and len(txt) > 200
    print(f"\nanalysis text: {len(txt)} chars, {txt.count(chr(10))+1} lines  OK")

    # variants: no prev / no history / no positions
    for kw in [dict(with_prev=False), dict(with_hist=False), dict(with_pos=False)]:
        s2 = make_fake_snapshot(**kw)
        keys = [c.key for c in charts_pkg.active(s2)]
        cs2 = CurveState.market(s2)
        for c in charts_pkg.active(s2):
            assert c.make(s2, cs2) is not None
        analysis.render_text(s2, cs2)
        print(f"  variant {kw}: active={keys}  OK")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    run()
