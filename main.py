#!/usr/bin/env python3
"""Two-piece quantile forecaster for Polymarket's "Next CEO of NVIDIA?" event.

Design (validated on 13 realised cases against the house baseline):
  * the median is persistence -- the checkpoint price, no drift, no
    mean-reversion (both lose badly on this market);
  * the inner quantiles (25% / 75%) are a tight normal whose sigma grows
    with the price level (tiny markets barely move, big markets move more);
  * the outer quantiles (5% / 95%) are the market's own empirical
    h-minute change quantiles over the last 18 hours, interpolated in
    sqrt(h) and shrunk by 0.9 -- wide asymmetric tails that pay exactly
    when the rare large (mostly downward) jumps happen;
  * markets with under 120 minutes of history fall back to the normal
    form, so a newly listed market can never crash the program.

Pure standard library, deterministic (no randomness, no clock, no
network), one forecast file per input case.
"""

import json
import math
import os

# Horizons (minutes) at which empirical change quantiles are tabulated.
GRID = (1, 2, 4, 8, 16, 32, 64, 120, 240)
# Last 18 hours of 1-minute prices: 1080 minutes plus the endpoint.
WINDOW = 1081
# Outer (5% / 95%) quantiles are shrunk toward the median by this factor.
OUTER_SCALE = 0.9
# Below this many prices the empirical tails are not trusted.
SHORT_HISTORY = 120
# Standard-normal quantiles at the 5% and 25% levels.
Z95 = 1.644854
Z75 = 0.674490


def clamp01(value):
    """Clamp a price to the unit interval."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def round6(value):
    """Clamp to [0, 1] and round to 6 decimals (never -0.0)."""
    return round(clamp01(value) * 1000000) / 1000000 + 0.0


def level_sigma(x):
    """Inner normal sigma for a market priced at x.

    0.0003 for x <= 0.01, rising linearly to 0.0010 for x >= 0.20.
    """
    t = (x - 0.01) / 0.19
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return 0.0003 + 0.0007 * t


def _quantile(sorted_values, tau):
    """Linear-interpolation quantile of an already-sorted list."""
    n = len(sorted_values)
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = lo + 1 if lo + 1 < n else n - 1
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _changes(series, horizon):
    """All h-minute changes of a price series, oldest first."""
    return [series[i + horizon] - series[i]
            for i in range(len(series) - horizon)]


def _outer_band(series):
    """Per-grid-horizon (5%, 95%) empirical change quantiles.

    A horizon with fewer than 20 changes borrows the 1-minute changes
    scaled by that horizon instead of trusting a tiny sample.
    """
    one_minute = _changes(series, 1)
    band = {}
    for horizon in GRID:
        changes = _changes(series, horizon)
        if len(changes) < 20:
            changes = [c * horizon for c in one_minute] if one_minute else [0.0]
        ordered = sorted(changes)
        band[horizon] = (_quantile(ordered, 0.05) * OUTER_SCALE,
                         _quantile(ordered, 0.95) * OUTER_SCALE)
    return band


def _normal_rows(x, sigma, horizon):
    """Plain symmetric-normal quantiles around x (fallback shape)."""
    rows = []
    for minute in range(1, horizon + 1):
        spread = math.sqrt(minute)
        rows.append([round6(x - Z95 * sigma * spread),
                     round6(x - Z75 * sigma * spread),
                     round6(x),
                     round6(x + Z75 * sigma * spread),
                     round6(x + Z95 * sigma * spread)])
    return rows


def forecast_market(prices, horizon=240):
    """Forecast one market: 240 rows of [q05, q25, q50, q75, q95]."""
    if not prices:
        return [[0.5] * 5 for _ in range(horizon)]
    x = prices[-1]
    if len(prices) < 2:
        return _normal_rows(x, 0.001, horizon)
    if len(prices) < SHORT_HISTORY:
        return _normal_rows(x, level_sigma(x), horizon)

    band = _outer_band(prices[-WINDOW:])
    sigma = level_sigma(x)
    roots = {h: math.sqrt(h) for h in GRID}
    rows = []
    for minute in range(1, horizon + 1):
        spread = math.sqrt(minute)
        below = max(g for g in GRID if g <= minute)
        above = min(g for g in GRID if g >= minute)
        if below == above:
            low, high = band[below]
        else:
            weight = (spread - roots[below]) / (roots[above] - roots[below])
            low = band[below][0] * (1.0 - weight) + band[above][0] * weight
            high = band[below][1] * (1.0 - weight) + band[above][1] * weight
        inner = Z75 * sigma * spread
        row = [round6(x + low),
               round6(x - inner),
               round6(x),
               round6(x + inner),
               round6(x + high)]
        # Rows must be non-decreasing; an all-rising window can put the
        # empirical 5% quantile above the tight 25% one, so repair it.
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def main():
    in_dir = os.environ.get('MAKEFUN_INPUT_DIR') or '/input'
    out_dir = os.environ.get('MAKEFUN_OUTPUT_DIR') or '/output'
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith('.in'):
            continue
        with open(os.path.join(in_dir, name), encoding='utf-8') as handle:
            case = json.load(handle)
        horizon = int(case.get('horizon_minutes', 240))
        forecasts = [{'id': market['id'],
                      'q': forecast_market(market['prices'], horizon)}
                     for market in case['markets']]
        out_name = name[:-3] + '.out'
        with open(os.path.join(out_dir, out_name), 'w', encoding='utf-8') as handle:
            json.dump({'forecasts': forecasts}, handle, separators=(',', ':'))
            handle.write('\n')


if __name__ == '__main__':
    main()
