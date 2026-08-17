"""Public demo entry point: build one deterministic snapshot and render it."""
from __future__ import annotations

from .config import RunConfig
from . import data as _data
from . import app as _app


def get_pipeline(demo=None):
    """Return the repository's deterministic, offline ``(cvt, opd)`` demo adapter."""
    del demo
    from .pipeline.demo import get_demo_pipeline
    return get_demo_pipeline()


def main(cfg: RunConfig | None = None, *, serve: bool = True, port: int = 8050,
         open_browser: bool | None = None, demo: bool | None = None, **overrides):
    """Build the snapshot and (optionally) serve the dashboard.

    Pass a `RunConfig`, or keyword overrides (symbol=..., date=..., use_iv_history=...).
    The public package always uses its synthetic offline backend.
    Returns the `Snapshot` so you can keep working with it in a notebook.
    """
    cfg = cfg or RunConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cvt, opd = get_pipeline(demo=demo)

    snap = _data.fetch_snapshot(cfg, cvt, opd)
    print(_app.analysis.render_text(snap, _data.CurveState.market(snap)))

    if serve:
        ob = cfg.open_in_browser if open_browser is None else open_browser
        _app.serve(snap, port=port, open_browser=ob)
    return snap


if __name__ == "__main__":
    main()
