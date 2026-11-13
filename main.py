#!/usr/bin/env python3
"""Polymarket next-CEO-of-NVIDIA forecaster (twopiece design).

For every market we publish 240 rows (one per forecast minute) of five
quantiles [0.05, 0.25, 0.5, 0.75, 0.95] for the Yes price.

Shape of the forecast
----------------------
* The median is persistence: the last observed price, no drift and no
  pull back toward any mean.
* The inner band (25% / 75%) is a tight Gaussian around that median,
  with a spread that grows with the price level: tiny markets barely
  move, the big favourites get more room.
* The outer band (5% / 95%) is empirical and asymmetric: for each grid
  horizon we look at what h-minute moves actually did over the last
  18 hours of that market, take the 5th / 95th percentile of those
  moves, and interpolate between horizons in sqrt(h). Rare large
  (mostly downward) jumps are therefore priced in without widening
  the quiet-minute inner band.
* A market with little history falls back to the Gaussian form for
  all five quantiles, so a newly listed market can never crash us.

Standard library only, no network, no randomness, no wall clock:
two runs on the same input are byte-identical.
"""

import json
import math
import os
import sys

# Normal quantiles for the five levels we emit, in order.
Z_SCORES = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
STANDARD_TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Horizons (minutes) at which empirical move distributions are measured.
GRID_HORIZONS = (1, 2, 4, 8, 16, 32, 64, 120, 240)

WINDOW = 1081          # last 18h of 1-minute prices (1080 steps + endpoint)
MIN_HISTORY = 120      # below this, the empirical tails are not trusted
MIN_CHANGES = 20       # below this, a horizon falls back to scaled 1-min moves
OUTER_SCALE = 0.9      # shrink on the empirical outer quantiles
FORECAST_MINUTES = 240


def _clamp01(v):
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def level_sigma(x):
    """Gaussian spread per sqrt(minute), growing with the price level."""
    frac = (x - 0.01) / 0.19
    if frac < 0.0:
        frac = 0.0
    elif frac > 1.0:
        frac = 1.0
    return 0.0003 + 0.0007 * frac


def _empirical_quantile(sorted_values, tau):
    """Linear-interpolation quantile of an already sorted list."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] + weight * (sorted_values[hi] - sorted_values[lo])


def _changes(series, step):
    return [series[i + step] - series[i] for i in range(len(series) - step)]


def _outer_band(series):
    """Return (low, high) lists: empirical 5%/95% moves per grid horizon.

    Each list is aligned with GRID_HORIZONS. A horizon with too few
    observed moves borrows the 1-minute move distribution, scaled by
    the horizon length.
    """
    one_min = _changes(series, 1)
    one_min_sorted = sorted(one_min)
    low_at = []
    high_at = []
    for h in GRID_HORIZONS:
        moves = _changes(series, h)
        if len(moves) < MIN_CHANGES:
            lo = _empirical_quantile(one_min_sorted, 0.05) * h
            hi = _empirical_quantile(one_min_sorted, 0.95) * h
        else:
            ordered = sorted(moves)
            lo = _empirical_quantile(ordered, 0.05)
            hi = _empirical_quantile(ordered, 0.95)
        low_at.append(lo)
        high_at.append(hi)
    return low_at, high_at


def _interp_sqrt(grid_values, minute):
    """Interpolate a grid-horizon series at `minute`, linear in sqrt(h)."""
    target = math.sqrt(minute)
    roots = [math.sqrt(h) for h in GRID_HORIZONS]
    if target <= roots[0]:
        return grid_values[0]
    if target >= roots[-1]:
        return grid_values[-1]
    for k in range(len(roots) - 1):
        if roots[k] <= target <= roots[k + 1]:
            span = roots[k + 1] - roots[k]
            frac = (target - roots[k]) / span if span else 0.0
            return grid_values[k] + frac * (grid_values[k + 1] - grid_values[k])
    return grid_values[-1]


def _tidy(row):
    """Clamp to [0, 1], round to 6 decimals, force non-decreasing."""
    out = []
    prev = 0.0
    for v in row:
        v = round(_clamp01(v), 6) + 0.0
        if v < prev:
            v = prev
        out.append(v)
        prev = v
    return out


def _gaussian_rows(x, sigma):
    """All five quantiles from the normal form around x."""
    rows = []
    for minute in range(1, FORECAST_MINUTES + 1):
        spread = sigma * math.sqrt(minute)
        rows.append(_tidy([x + z * spread for z in Z_SCORES]))
    return rows


def forecast_market(prices):
    """240 x 5 quantile rows for one market's price history."""
    if not prices:
        x = 0.5
        return _gaussian_rows(x, 0.001)
    x = prices[-1]
    if len(prices) < 2:
        return _gaussian_rows(x, 0.001)
    if len(prices) < MIN_HISTORY:
        return _gaussian_rows(x, level_sigma(x))

    recent = prices[-WINDOW:]
    low_at, high_at = _outer_band(recent)
    sigma = level_sigma(x)
    rows = []
    for minute in range(1, FORECAST_MINUTES + 1):
        root = math.sqrt(minute)
        q05 = x + _interp_sqrt(low_at, minute) * OUTER_SCALE
        q95 = x + _interp_sqrt(high_at, minute) * OUTER_SCALE
        inner = 0.674490 * sigma * root
        rows.append(_tidy([q05, x - inner, x, x + inner, q95]))
    return rows


def _reorder(rows, requested_taus):
    """Emit rows in the case's quantile order (kit guarantees standard)."""
    if not requested_taus:
        return rows
    order = []
    for tau in requested_taus:
        best = min(range(len(STANDARD_TAUS)),
                   key=lambda i: abs(STANDARD_TAUS[i] - tau))
        order.append(best)
    if order == list(range(len(STANDARD_TAUS))):
        return rows
    return [[row[i] for i in order] for row in rows]


def process_case(case):
    markets = case.get("markets") or []
    requested = case.get("quantiles") or list(STANDARD_TAUS)
    forecasts = []
    for market in markets:
        prices = market.get("prices") or []
        rows = forecast_market([float(p) for p in prices])
        forecasts.append({
            "id": market.get("id"),
            "q": _reorder(rows, requested),
        })
    return {"forecasts": forecasts}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = process_case(case)
        out_name = name[:-3] + ".out"
        with open(os.path.join(out_dir, out_name), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(result, separators=(",", ":")) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
