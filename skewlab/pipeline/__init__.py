"""skewlab.pipeline — the data-source boundary.

skewlab's I/O layer (`skewlab.data.fetch_snapshot`) never imports a data vendor directly;
it receives two injected objects, ``cvt`` (option chains + composite realized vol) and
``opd`` (trading calendar, OHLCV, IV-history panels, VIX/VVIX, trade ledger). Anything that
implements that small surface can drive the dashboard.

Two backends live behind this boundary:

* the private production pipeline (``from CapriciousVolTamer import cvt, opd``) — not shipped
  in this repo; it talks to a ThetaData terminal, yfinance and a local trade ledger; and
* :mod:`skewlab.pipeline.demo` — a fully synthetic, offline, reproducible backend so the
  whole thing runs with no network, no terminal and no credentials (used for the public
  demo, the tests and CI).

``skewlab.run.get_pipeline()`` picks the production backend when it's importable and falls
back to the demo one otherwise (or when ``SKEWLAB_DEMO=1`` / ``demo=True``).
"""
from __future__ import annotations

from .demo import get_demo_pipeline

__all__ = ["get_demo_pipeline"]
