"""skewlab.run — the entry point.

Wires the pipeline (cvt / opd) to the package: build a `RunConfig`, fetch one
`Snapshot`, then either launch the Dash dashboard or render headless static HTML.

Run from notebook/main/v8/:
    python -m skewlab.run
or in the VS Code interactive window:
    from skewlab.run import main; snap = main(symbol="SPY")
"""
from __future__ import annotations

import os
import warnings
from sys import path as _syspath

import numpy as np

# --- path bootstrap so the PRIVATE production pipeline imports cleanly when present ---
_syspath.append("..")      # notebook/main/  — where CapriciousVolTamer & FetchData live
_syspath.append("../..")
warnings.filterwarnings("ignore")
np.set_printoptions(legacy="1.25")

from .config import RunConfig
from . import data as _data
from . import app as _app


def _demo_requested(demo):
    if demo is not None:
        return bool(demo)
    return os.environ.get("SKEWLAB_DEMO", "").strip().lower() in ("1", "true", "yes", "on")


def get_pipeline(demo=None):
    """Return the ``(cvt, opd)`` data backend.

    Uses the private production pipeline (``CapriciousVolTamer``) when it's importable and
    the demo hasn't been forced; otherwise falls back to the fully synthetic, offline
    :mod:`skewlab.pipeline.demo` backend. Force the demo with ``demo=True`` or the
    ``SKEWLAB_DEMO=1`` environment variable.
    """
    if not _demo_requested(demo):
        try:
            from CapriciousVolTamer import cvt, opd
            return cvt, opd
        except Exception as e:                     # no private pipeline available -> synthetic
            print(f"[pipeline] production pipeline unavailable ({type(e).__name__}); "
                  f"using the synthetic demo backend. Set SKEWLAB_DEMO=1 to silence.")
    else:
        print("[pipeline] demo backend requested -> synthetic data (offline).")
    from .pipeline.demo import get_demo_pipeline
    return get_demo_pipeline()


def main(cfg: RunConfig | None = None, *, serve: bool = True, port: int = 8050,
         open_browser: bool | None = None, demo: bool | None = None, **overrides):
    """Build the snapshot and (optionally) serve the dashboard.

    Pass a `RunConfig`, or keyword overrides (symbol=..., date=..., use_iv_history=...).
    ``demo=True`` (or SKEWLAB_DEMO=1) forces the synthetic offline backend.
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
