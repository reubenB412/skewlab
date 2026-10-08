"""Reproduce the numerical checks quoted in the README."""
from __future__ import annotations

import numpy as np

from . import model
from .config import RunConfig
from .data import fetch_snapshot
from .pipeline.demo import get_demo_pipeline


def demo_checks() -> dict[str, float | int]:
    cfg = RunConfig(
        symbol="SPY",
        date="2026-08-24",
        target_dte=23,
        use_intraday=False,
        monthly_only=False,
        use_iv_history=True,
        iv_hist_target_dte=30,
        iv_hist_start=None,
        show_term_curves=True,
        show_rv_term_structure=True,
        show_vix_panels=True,
        open_in_browser=False,
    )
    cvt, opd = get_demo_pipeline(today="2026-08-24")
    snap = fetch_snapshot(cfg, cvt, opd)

    x = np.asarray(snap.mkt_pdf_x, float)
    pdf = np.asarray(snap.mkt_pdf_y, float)
    density_area = float(np.sum((pdf[:-1] + pdf[1:]) / 2.0 * np.diff(x)))
    min_g, _ = model.svi_min_butterfly_g(
        snap.poly.svi_params,
        np.linspace(-0.5, 0.5, 400),
    )
    rv_term = snap.rv_term
    if rv_term is None:
        raise RuntimeError("fixed demo did not produce the RV term structure")

    return {
        "density_area": density_area,
        "minimum_density": float(np.min(pdf)),
        "minimum_durrleman_g": min_g,
        "completed_rv_sessions": int(rv_term.hf_daily["is_complete_session"].sum()),
        "aligned_atm_iv_maturities": int(rv_term.iv_curve["atf_iv"].notna().sum()),
        "rv_5_20": float(rv_term.summary["slope_5_20"]),
        "rv_10_30": float(rv_term.summary["slope_10_30"]),
    }


def main() -> None:
    checks = demo_checks()
    print("\nFixed synthetic demo: SPY, 2026-08-24")
    print(f"density area                 {checks['density_area']:.5f}")
    print(f"minimum sampled density      {checks['minimum_density']:.8f}")
    print(f"minimum Durrleman g(k)        {checks['minimum_durrleman_g']:.4f}")
    print(f"completed RV sessions        {checks['completed_rv_sessions']}")
    print(f"aligned ATM-IV maturities     {checks['aligned_atm_iv_maturities']}")
    print(f"RV(5) / RV(20)               {checks['rv_5_20']:.3f}")
    print(f"RV(10) / RV(30)              {checks['rv_10_30']:.3f}")


if __name__ == "__main__":
    main()
