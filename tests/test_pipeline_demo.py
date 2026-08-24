"""The synthetic demo backend must drive the whole stack offline (no network/terminal)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from dash import dcc, html

from skewlab.config import RunConfig
from skewlab import data as D, charts as C, rv
from skewlab.app import _opd_format_display_model, build_app
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
    fig = rv_term_structure.make(snap, D.CurveState.market(snap))
    assert fig is not None
    iv = next(
        trace for trace in fig.data
        if trace.name == "ATM implied volatility (forward)"
    )
    assert len(iv.x) == len(snap.cfg.rv_iv_tenors)
    assert len(iv.x) > 1
    data = collect_run_data(snap)
    assert not data["rv_term_table"].empty
    assert data["rv_term_metadata"]["source"] == "synthetic offline demo"


def _display_demo_snapshot():
    cvt, opd = get_demo_pipeline(today="2026-08-24")
    return D.fetch_snapshot(_demo_cfg(show_rv_term_structure=True), cvt, opd)


@pytest.fixture(scope="module")
def display_demo_snapshot():
    return _display_demo_snapshot()


def _walk(component):
    if component is None:
        return
    if isinstance(component, (list, tuple)):
        for child in component:
            yield from _walk(child)
        return
    yield component
    yield from _walk(getattr(component, "children", None))


def _iv_trace(fig):
    return next(
        trace for trace in fig.data
        if trace.name == "ATM implied volatility (forward)"
    )


def test_estimator_display_model_matches_opd_format_display_conventions():
    table = pd.DataFrame(
        {5: [0.1234, -0.01], 20: [0.1876, 0.02]},
        index=pd.Index(["Mean Volatility", "test negative"], name="RV Rolling"),
    )
    rendered = _opd_format_display_model(table, pct_cols=list(table.columns), axis=1)
    first, negative = rendered["rows"]
    assert [cell["text"] for cell in first["cells"]] == ["12.34%", "18.76%"]
    assert negative["cells"][0]["style"]["color"] == "red"
    backgrounds = [cell["style"]["backgroundColor"] for cell in first["cells"]]
    assert backgrounds == ["#ffffd9", "#081d58"]
    assert backgrounds[0] != backgrounds[1]


def test_estimator_display_model_leaves_nan_blank_and_neutral():
    table = pd.DataFrame(
        {5: [np.nan], 20: [0.10]},
        index=pd.Index(["partial"], name="RV Rolling"),
    )
    rendered = _opd_format_display_model(table, pct_cols=list(table.columns), axis=1)
    missing = rendered["rows"][0]["cells"][0]
    assert missing["text"] == ""
    assert missing["style"] == {"backgroundColor": "#fff", "color": "#374151"}


def test_complete_dash_layout_has_inline_shaded_semantic_rv_table(display_demo_snapshot):
    app = build_app(display_demo_snapshot)
    components = list(_walk(app.layout))
    estimator_cells = [component for component in components if isinstance(component, html.Td)]
    assert len(estimator_cells) == 10 * 9
    assert all("backgroundColor" in cell.style and "color" in cell.style
               for cell in estimator_cells)
    backgrounds = {cell.style["backgroundColor"] for cell in estimator_cells}
    assert len(backgrounds) > 2
    assert "#ffffd9" in backgrounds
    assert "#081d58" in backgrounds
    assert not any(isinstance(component, dcc.Markdown) for component in components)
    assert app.callback_map


def test_normal_demo_plots_complete_aligned_forward_atm_iv_curve(display_demo_snapshot):
    state = display_demo_snapshot.rv_term
    assert state.iv_curve["is_aligned"].all()
    assert state.iv_curve["observation_date"].nunique() == 1
    assert set(state.iv_curve.index) == {10, 20, 30, 45, 60, 90, 120, 180}

    fig = rv_term_structure.make(
        display_demo_snapshot, D.CurveState.market(display_demo_snapshot)
    )
    iv = _iv_trace(fig)
    assert len(iv.x) == 8
    assert list(iv.x) == sorted(iv.x)
    assert list(iv.x) == [10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0]
    assert list(iv.text)[-1] == "ATM IV"
    assert iv.line.color == "#2f6feb"
    assert iv.line.width == 4.5
    assert iv.marker.symbol == "diamond"
    assert iv.marker.line.color == "white"
    assert iv.legendrank == 1
    assert fig.layout.template is not None


def test_stale_auxiliary_tenor_is_excluded_but_diagnostics_remain(display_demo_snapshot):
    curve = display_demo_snapshot.rv_term.iv_curve.copy()
    curve.loc[10, "observation_date"] = (
        pd.Timestamp(display_demo_snapshot.date) - pd.Timedelta(days=1)
    )
    curve.loc[10, "is_aligned"] = False
    curve.loc[10, ["atf_iv", "integrated_variance"]] = np.nan
    state = replace(display_demo_snapshot.rv_term, iv_curve=curve)
    snap = replace(display_demo_snapshot, rv_term=state)

    iv = _iv_trace(rv_term_structure.make(snap))
    assert 10.0 not in list(iv.x)
    assert len(iv.x) == 7
    assert pd.notna(curve.loc[10, "actual_dte"])
    assert pd.notna(curve.loc[10, "expiry"])
    assert pd.notna(curve.loc[10, "observation_date"])


def test_main_snapshot_anchor_remains_when_all_auxiliary_tenors_are_rejected(
    display_demo_snapshot,
):
    curve = display_demo_snapshot.rv_term.iv_curve.copy()
    curve["is_aligned"] = False
    curve[["atf_iv", "integrated_variance"]] = np.nan
    state = replace(display_demo_snapshot.rv_term, iv_curve=curve)
    snap = replace(display_demo_snapshot, rv_term=state)

    iv = _iv_trace(rv_term_structure.make(snap))
    assert list(iv.x) == [float(snap.dte)]
    assert list(iv.y) == [float(snap.atf) * 100.0]
    assert list(iv.text) == ["ATM IV"]


def test_stale_chain_is_rejected_at_the_data_boundary():
    cfg = RunConfig(
        symbol="SPY", date="2026-08-24", target_dte=30, rv_iv_tenors=(10,),
        use_intraday=False, monthly_only=False,
    )
    cvt, opd = get_demo_pipeline(today="2026-08-24")

    class StaleCVT:
        def get_quick_option_chain(self, *args, **kwargs):
            chain = cvt.get_quick_option_chain(*args, **kwargs)
            stale = pd.Timestamp("2026-08-21")
            chain["observation_date"] = stale
            chain.attrs["observation_date"] = str(stale.date())
            return chain

    curve, warnings = D._fetch_rv_iv_curve(cfg, StaleCVT(), opd, [])
    row = curve.loc[10]
    assert not bool(row["is_aligned"])
    assert row["observation_date"] == pd.Timestamp("2026-08-21")
    assert pd.notna(row["actual_dte"])
    assert pd.notna(row["expiry"])
    assert pd.isna(row["atf_iv"])
    assert pd.isna(row["integrated_variance"])
    assert any("does not match main snapshot" in warning for warning in warnings)
