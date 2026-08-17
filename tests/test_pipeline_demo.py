"""The synthetic demo backend must drive the whole stack offline (no network/terminal)."""
from skewlab.config import RunConfig
from skewlab import data as D, charts as C, rv
from skewlab.pipeline.demo import get_demo_pipeline
from skewlab.inspect import rv_compare_frame, collect_run_data
from skewlab.charts import rv_term_structure


def _demo_cfg(**kw):
    base = dict(symbol="SPY", target_dte=30, use_intraday=False, monthly_only=False,
                use_iv_history=False, show_term_curves=False, show_vix_panels=False)
    base.update(kw)
    return RunConfig(**base)


def test_demo_snapshot_builds_sane_surface():
    cvt, opd = get_demo_pipeline()
    snap = D.fetch_snapshot(_demo_cfg(), cvt, opd)
    assert 0.05 < snap.atf < 0.60          # plausible ATM vol
    assert snap.forward > 0 and snap.spot > 0
    assert snap.grid_vols.shape == snap.z_grid.shape


def test_demo_every_active_chart_builds():
    cvt, opd = get_demo_pipeline()
    snap = D.fetch_snapshot(_demo_cfg(use_iv_history=True, iv_hist_start=None,
                                      show_term_curves=True, show_vix_panels=True), cvt, opd)
    cs = D.CurveState.market(snap)
    for chart in C.active(snap):
        fig = chart.make(snap, cs)
        assert fig is not None, f"chart {chart.key} returned None"


def test_demo_rv_compare_present():
    cvt, opd = get_demo_pipeline()
    snap = D.fetch_snapshot(_demo_cfg(use_iv_history=True, iv_hist_start=None), cvt, opd)
    f = rv_compare_frame(snap)
    assert "RV_fair" in list(f.index) and "now" in list(f.index)
    assert "rv_compare" in collect_run_data(snap)


def test_demo_vol_history_and_estimator_stack_build():
    cvt, opd = get_demo_pipeline()
    snap = D.fetch_snapshot(_demo_cfg(use_iv_history=True, iv_hist_start=None), cvt, opd)
    cs = D.CurveState.market(snap)
    assert C.vol_history.has_history(snap) and C.vol_history.has_estimators(snap)
    assert C.vol_history.make(snap, cs) is not None
    assert C.vol_history.make_estimators(snap, cs) is not None
    assert snap.rv_estimators is not None and "Mean" in snap.rv_estimators.columns


def test_demo_rv_term_structure_is_complete_and_deterministic():
    cvt, opd = get_demo_pipeline()
    snap = D.fetch_snapshot(_demo_cfg(show_rv_term_structure=True), cvt, opd)
    state = snap.rv_term
    assert state is not None and state.available
    assert tuple(state.estimator_table.index) == rv.ESTIMATOR_ROWS
    assert list(state.estimator_table.columns) == list(snap.cfg.rv_lookbacks)
    assert int(state.hf_daily["is_complete_session"].sum()) >= 180
    assert not bool(state.hf_daily["is_complete_session"].iloc[-1])
    assert int(state.iv_curve["atf_iv"].notna().sum()) == len(snap.cfg.rv_iv_tenors)
    assert state.summary["regime"] == "Front-end RV compressed"
    assert rv_term_structure.make(snap, D.CurveState.market(snap)) is not None
    data = collect_run_data(snap)
    assert not data["rv_term_table"].empty
    assert data["rv_term_metadata"]["source"] == "synthetic offline demo"
