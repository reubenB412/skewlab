"""skewlab.charts — the chart registry.

Every chart is a pure `make(snap, cs, **kw) -> go.Figure`. The registry lets the app
(and any caller) iterate over charts generically instead of hard-wiring each one.

A chart entry is described by a `Chart` record:
  key      stable id used in callbacks / div ids
  title    human label
  make     the builder (snap, cs) -> figure (or None to skip)
  needs    predicate(snap) -> bool: is this chart relevant for this snapshot?
  reacts   True if the figure changes when the sliders move (re-rendered live)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import (curve, distribution, rv_vs_iv, strike_vol_change, vix_distribution,
               vvix_vix_ratio, position, pnl, iv_history)


@dataclass(frozen=True)
class Chart:
    key: str
    title: str
    make: Callable
    needs: Callable = lambda snap: True
    reacts: bool = True


REGISTRY = [
    Chart("curve", "Skew curve", curve.make),
    Chart("distribution", "Implied distribution", distribution.make),
    Chart("rv_vs_iv", "RV vs IV (fair value)", rv_vs_iv.make,
          needs=lambda s: getattr(s, "rv_iv", None) is not None
                          and np.isfinite(float(getattr(s, "rv_iv", float("nan")))),
          reacts=False),
    Chart("strike_vol_change", "Strike vol changes", strike_vol_change.make,
          needs=lambda s: s.prev_poly is not None),
    Chart("vix_distribution", "VIX / VVIX distribution", vix_distribution.make,
          needs=lambda s: s.vix_dist is not None or s.vvix_dist is not None, reacts=False),
    Chart("vix_distribution_since", "VIX / VVIX distribution (since date)",
          vix_distribution.make_since,
          needs=lambda s: bool(s.since_when) and
                          (s.vix_dist_since is not None or s.vvix_dist_since is not None),
          reacts=False),
    Chart("vvix_vix_ratio", "VVIX / VIX ratio", vvix_vix_ratio.make,
          needs=lambda s: s.vix_vvix_ratio is not None and not getattr(s.vix_vvix_ratio, "empty", True),
          reacts=False),
    Chart("position", "Position & Greeks", position.make, needs=lambda s: s.has_positions),
    Chart("pnl", "P&L decomposition", pnl.make,
          needs=lambda s: s.has_positions and s.prev_poly is not None),
    Chart("iv_history", "IV history & regime", iv_history.make,
          needs=lambda s: s.iv_history is not None and not getattr(s.iv_history, "empty", True),
          reacts=False),
]

BY_KEY = {c.key: c for c in REGISTRY}


def active(snap):
    """Charts relevant for this snapshot, in registry order."""
    return [c for c in REGISTRY if c.needs(snap)]
