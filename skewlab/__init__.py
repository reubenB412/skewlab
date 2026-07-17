"""skewlab — a layered, modular rebuild of the Moontower skew/vol dashboard.

Layers (each importable on its own; only `data` and `run` touch the pipeline):

    config   RunConfig dataclass + scenario presets — every knob, no side effects.
    model    pure quant engine (Black-Scholes, skew fit, Breeden-Litzenberger, stats).
    data     I/O: fetch_snapshot(cfg, cvt, opd) -> immutable Snapshot; CurveState knobs.
    theme    one shared plotly template + legend/colour constants.
    analysis metrics(snap, cs) + text / HTML narrative renderers.
    charts   one module per chart, each a pure make(snap, cs) -> Figure, + a registry.
    app      Dash dashboard built generically from the chart registry.
    run      entry point: config -> snapshot -> serve / headless.

Quick start (from notebook/main/v8/):
    from skewlab.run import main
    snap = main(symbol="SPY", use_iv_history=True)
"""
from __future__ import annotations

from .config import RunConfig, SCENARIOS
from .data import Snapshot, CurveState, TermBundle, fetch_snapshot

__all__ = ["RunConfig", "SCENARIOS", "Snapshot", "CurveState", "TermBundle", "fetch_snapshot"]
__version__ = "0.1.0"
