"""skewlab.pipeline — the data-source boundary.

skewlab's I/O layer (`skewlab.data.fetch_snapshot`) receives two injected objects: ``cvt``
(option chains, composite RV, and RV term source) and ``opd`` (calendar, OHLCV, IV-history
panels, and VIX/VVIX). The quant modules do not perform I/O.

The public launcher selects :mod:`skewlab.pipeline.demo`, a fully synthetic and reproducible
adapter used by the dashboard, tests, and CI. It requires no network or credentials.
"""
from __future__ import annotations

from .demo import get_demo_pipeline

__all__ = ["get_demo_pipeline"]
