"""skewlab.positions — tiny helpers for editing an option book.

A position book is a list of legs, each ``(strike, "P"/"C", contracts)`` with
positive contracts = long and negative = short. These helpers are pure: they take a
list (or ``None``) and return a NEW list, so there's no hidden global state.

    from skewlab.positions import add_position, remove_position
    book = []
    book = add_position(book, 740, "P", 100)     # long 100 of the 740 put
    book = add_position(book, 810, "C", -50)      # short 50 of the 810 call
    book = remove_position(book, 740, "P")        # drop the 740 put leg
"""
from __future__ import annotations


def add_position(positions, strike, kind, contracts):
    """Return a new book with this leg added (replacing any existing leg at the same
    strike + kind). kind is 'P' or 'C'; contracts +long / -short."""
    book = remove_position(positions, strike, kind)
    book.append((float(strike), str(kind).upper()[0], int(contracts)))
    return book


def remove_position(positions, strike, kind):
    """Return a new book with any leg matching this strike + kind removed."""
    k = str(kind).upper()[0]
    return [p for p in (positions or [])
            if not (float(p[0]) == float(strike) and str(p[1]).upper()[0] == k)]


# =====================================================================================
# Trade-ledger -> position book
# =====================================================================================
# The pipeline's `opd.fetch_trades_ledger()` reads the Excel ledger and returns a dict
# keyed by Trade_ID (e.g. "EWY_short_strangle_20260610"). Each value looks like:
#   {'symbol': 'EWY', 'dte': 37, 's0': ..., 'options': [ {leg}, ... ],
#    'delta_hedging_log': [ {hedge}, ... ], ...}
# where each option leg is:
#   {'trade_date','expiry','op_type'('p'/'c'),'strike','premium','iv_option',
#    'contracts','tr_type'('s'/'b'),'closed','close_date','close_premium'}
# `closed == 0` (0.0/None/'0') means the leg is still OPEN.


def _is_open(closed):
    """True if a ledger leg/hedge is still open (closed flag is 0 / blank / None)."""
    if closed is None:
        return True
    if isinstance(closed, str):
        return closed.strip().lower() in ("", "0", "0.0", "open", "o", "no", "n", "false")
    try:
        return float(closed) == 0.0
    except (TypeError, ValueError):
        return True


def _signed_contracts(leg):
    """Signed contracts for a leg: +long for buys ('b'), -short for sells ('s')."""
    n = int(leg.get("contracts") or 0)
    tr = str(leg.get("tr_type") or "").strip().lower()
    return -n if tr.startswith("s") else n


def _ensure_ledger(opd, refresh=False):
    """Populate opd.trades_dict from the Excel ledger if needed; return it."""
    td = getattr(opd, "trades_dict", None)
    if refresh or not td:
        opd.trades_dict = opd.fetch_trades_ledger()
    return opd.trades_dict


# --- symbol equivalence groups -------------------------------------------------------
# Pull positions on RELATED instruments onto one underlying's smile. Each group maps a
# ledger symbol -> the factor that rescales ITS strikes into a common (SPY-level) space.
# e.g. SPX trades ~10x SPY, so SPX strike 6000 * 0.1 = 600 lands on SPY's axis; ES/MES
# (S&P futures) also trade at ~index level, so 0.1 too.  Add more groups as needed.
SYMBOL_GROUPS = [
    {"SPY": 1.0, "SPX": 0.1, "SPXW": 0.1, "XSP": 1.0, "ES": 0.1, "MES": 0.1},
    {"QQQ": 1.0, "NDX": 0.025, "MNQ": 0.025, "NQ": 0.025},   # NDX ~40x QQQ
]


def resolve_symbol_scales(symbol):
    """Return ``{ledger_symbol: strike_scale}`` mapping each related symbol's strikes into
    ``symbol``'s strike space. If ``symbol`` is in a known group, every member is included
    rescaled relative to ``symbol``; otherwise just ``{symbol: 1.0}``.

    Example: resolve_symbol_scales("SPY") -> {"SPY":1.0,"SPX":0.1,"MES":0.1,...}
             resolve_symbol_scales("SPX") -> {"SPX":1.0,"SPY":10.0,"MES":1.0,...}
    """
    s = str(symbol).upper()
    for g in SYMBOL_GROUPS:
        if s in g:
            base = g[s]
            return {k.upper(): (v / base) for k, v in g.items()}
    return {s: 1.0}


def ledger_keys(opd, symbol=None, refresh=False):
    """Quiet stand-in for ``opd.print_trade_list()`` — return the Trade_ID keys
    (sorted by their trailing YYYYMMDD) WITHOUT printing them.

    Pass ``symbol`` to keep only trades whose ``symbol`` field matches (case-insensitive).
    """
    trades = _ensure_ledger(opd, refresh)
    items = sorted(trades.items(), key=lambda kv: str(kv[0]).split("_")[-1])
    if symbol is not None:
        sym = str(symbol).upper()
        items = [(k, v) for k, v in items if str(v.get("symbol", "")).upper() == sym]
    return [k for k, _ in items]


def open_legs_from_ledger(opd, symbol, expiry=None, refresh=False, verbose=True,
                          symbol_scales=None):
    """Build a skewlab position book from the OPEN option legs of ``symbol`` in the ledger.

    - Matches on the trade's ``symbol`` field (not the key prefix), case-insensitive.
    - ``symbol_scales`` (optional) = ``{ledger_symbol: strike_scale}`` to ALSO pull related
      instruments and rescale their strikes onto ``symbol``'s axis (see resolve_symbol_scales,
      e.g. SPY also pulling SPX/MES). Defaults to just ``{symbol: 1.0}``.
    - Keeps only legs with ``closed == 0`` (open).
    - If ``expiry`` (a 'YYYYMMDD' string) is given, keeps only legs on that expiry.
    - Nets contracts across legs sharing the same (scaled strike, P/C); +long / -short.

    Returns ``(book, meta)``:
      ``book`` = list of ``(strike, 'P'/'C', contracts)`` ready for ``cfg.positions``;
      ``meta`` = list of ``(trade_id, ledger_symbol, expiry, orig_strike, scaled_strike,
                 kind, signed_contracts)`` contributors.
    """
    trades = _ensure_ledger(opd, refresh)
    scales = symbol_scales or {str(symbol).upper(): 1.0}
    scales = {str(k).upper(): float(v) for k, v in scales.items()}
    exp = str(expiry) if expiry is not None else None

    agg, meta = {}, []
    for tid, tr in trades.items():
        tsym = str(tr.get("symbol", "")).upper()
        if tsym not in scales:
            continue
        fac = scales[tsym]
        for leg in tr.get("options", []) or []:
            if not _is_open(leg.get("closed")):
                continue
            if exp is not None and str(leg.get("expiry")) != exp:
                continue
            if leg.get("strike") is None or leg.get("op_type") is None:
                continue
            orig = float(leg["strike"])
            strike = round(orig * fac, 4)
            kind = str(leg["op_type"]).upper()[0]      # 'P' / 'C'
            n = _signed_contracts(leg)
            if n == 0:
                continue
            agg[(strike, kind)] = agg.get((strike, kind), 0) + n
            meta.append((tid, tsym, leg.get("expiry"), orig, strike, kind, n))

    book = [(k[0], k[1], v) for k, v in sorted(agg.items()) if v != 0]

    if verbose:
        base = str(symbol).upper()
        if not meta:
            print(f"[ledger] no OPEN {base} option legs found.")
        else:
            syms = sorted({m[1] for m in meta})
            exps = sorted({str(m[2]) for m in meta})
            related = [s for s in syms if s != base]
            note = f" (+{','.join(related)} strike-scaled onto {base})" if related else ""
            print(f"[ledger] {base}{note}: {len(meta)} open leg(s) over expiries {exps} "
                  f"-> {len(book)} netted leg(s): {book}")
            if exp is None and len(exps) > 1:
                print(f"[ledger] WARNING: legs span multiple expiries {exps}; the dashboard "
                      f"values the book at a SINGLE expiry. Pass expiry='YYYYMMDD' to isolate one.")
            if related:
                print(f"[ledger] NOTE: related-symbol legs are strike-scaled only — contract "
                      f"multipliers differ (e.g. MES vs SPY), so notional isn't normalised.")
    return book, meta


def open_shares_from_ledger(opd, symbol, refresh=False):
    """Net OPEN underlying shares from the delta-hedging log for ``symbol`` (+long / -short)."""
    trades = _ensure_ledger(opd, refresh)
    sym = str(symbol).upper()
    total = 0.0
    for tr in trades.values():
        if str(tr.get("symbol", "")).upper() != sym:
            continue
        for h in tr.get("delta_hedging_log", []) or []:
            if _is_open(h.get("closed")):
                total += float(h.get("amount") or 0.0)
    return total
