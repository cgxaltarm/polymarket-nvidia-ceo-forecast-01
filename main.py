#!/usr/bin/env python3
"""Polymarket next-CEO-of-NVIDIA forecaster — round-3 candidate (r3).

Design (validated in scratch-nvidia-r3, see RESEARCH_R3.md). Builds on r2
(scratch-nvidia-r2/main.py); the new parts are the REGIME scaling and the
re-tuned pooling/pull parameters:

* Median: persistence + horizon-grown sum-to-one pull, target sum 0.99
  (the realised 4h sum mean-reverts to ~0.99, not 1.00).
  med(h) = x + 1.1*(1-exp(-h/240)) * (x*0.99/S - x), S = sum of prices.
* Inner band (25%/75%): level-dependent normal, ADAPTIVE to recent
  jumpiness: sigma = clip(robust_sigma_120m, 0.75x, 4x) * level_sigma(x),
  level_sigma(x) = 0.0004 + 0.0010*clamp((x-0.01)/0.19).
  Blended 75/25 with the POOLED empirical interquartile range.
* Outer band (5%/95%): per-market empirical h-change quantiles over the
  last 12h (720 min), changes WINSORIZED at +-0.05, interpolated in
  sqrt(h) between grid horizons, blended 75/25 with the POOLED outer
  (changes of ALL markets normalised by level_sigma, pooled, quantiles
  taken over the pooled sample, then re-scaled by this market's sigma).
  Down tail widened x1.2 (realised jumps are mostly downward).
* REGIME scaling (new in r3): r = clip(robust_sigma_120m /
  robust_sigma_720m, 0.5, 1.0). When the last 2h were much calmer than
  the last 12h, spreads shrink (inner normal sigma and outer scale both
  x r); when recent vol is at/above window vol, no change (cap 1.0).
  This is what buys the quiet-regime gain: the 12h empirical tails are
  too fat once the market calms down. The pooled IQR is NOT regime
  scaled (it is already cross-market smoothed).
* Short history (<120 prices): plain level-sigma normal for all 5 quantiles.

Validated (grade_entry.py + 37-case harness incl. 9 cases never used in
selection): pooled practice-6 = 0.2655 (r2 0.2719, live 0.3135),
held-out quiet 28 cases = 0.1120 (r2 0.1162), fresh-9 = 0.1153 (r2 0.1182),
paired t on held = 4.12 (25/28 wins). Still short of the <=0.255 bar:
the remaining loss is dominated by unpredictable persistent level shifts
(oracle-median analysis, RESEARCH_R2.md section 4). See RESEARCH_R3.md.

Standard library only, no network, no randomness, no wall clock:
two runs on the same input are byte-identical.
"""

import json
import math
import os
import sys

Z_SCORES = (-1.644854, -0.674490, 0.0, 0.674490, 1.644854)
STANDARD_TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
GRID_HORIZONS = (1, 2, 4, 8, 16, 32, 64, 120, 240)

WINDOW = 720           # last 12h of 1-minute prices for empirical outer
POOL_WINDOW = 720      # same window for the pooled outer
WINSOR_CAP = 0.05      # winsorize h-changes at +-0.05 before quantiles
OUTER_SCALE = 1.0
O05 = 1.2              # widen the down tail (realised jumps mostly downward)
POOL_BLEND = 0.75      # per-market outer vs pooled outer
POOL_INNER_BLEND = 0.75  # normal inner vs pooled empirical IQR
MIN_HISTORY = 120
MIN_CHANGES = 20
FORECAST_MINUTES = 240

# adaptive inner sigma (robust 120-min vol, clipped to a multiple of level sigma)
ADAPT_WINDOW = 120
ADAPT_LO = 0.75
ADAPT_HI = 4.0

# sum-to-one pull: strength 1.1, grows with horizon, time constant 240 min,
# target sum 0.99 (realised 4h sum mean-reverts to ~0.99, not 1.00)
PULL_TAU = 240.0
PULL_A = 1.1
S_INF = 0.99

# regime scaling: shrink spreads when the last 2h were much calmer than 12h
REG_LO = 0.5
REG_HI = 1.0


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
    return 0.0004 + 0.0010 * frac


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


def _robust_sigma(prices, window):
    """Jump-robust 1-min vol: sqrt(p_nonzero) * std(nonzero changes)."""
    seg = prices[-window - 1:] if len(prices) > window + 1 else prices
    diffs = [seg[i + 1] - seg[i] for i in range(len(seg) - 1)]
    nz = [v for v in diffs if v != 0]
    if len(nz) < 10:
        return 0.0003
    p = len(nz) / len(diffs)
    m = sum(nz) / len(nz)
    var = sum((v - m) ** 2 for v in nz) / (len(nz) - 1)
    return math.sqrt(p) * math.sqrt(var)


def _quantile_grid(series, taus):
    """Per-grid-horizon empirical quantiles of h-changes (winsorized)."""
    one_min = _changes(series, 1)
    out = []
    for h in GRID_HORIZONS:
        moves = _changes(series, h)
        if len(moves) < MIN_CHANGES:
            moves = [c * h for c in one_min]
        moves = [max(-WINSOR_CAP, min(WINSOR_CAP, c)) for c in moves]
        ordered = sorted(moves)
        out.append([_empirical_quantile(ordered, t) for t in taus])
    return out


def _pooled_quantile_grid(all_prices):
    """Cross-market pooled quantiles of level-normalised h-changes.

    For each market, take its last POOL_WINDOW minutes, divide each
    h-change by that market's level_sigma, pool across markets, and take
    the five quantiles. Returns {h: [q05, q25, q50, q75, q95]} in
    normalised units (multiply by a market's level_sigma to use).
    """
    pooled = {}
    for h in GRID_HORIZONS:
        norm = []
        for prices in all_prices:
            seg = prices[-POOL_WINDOW - 1:] if len(prices) > POOL_WINDOW + 1 else prices
            sig = level_sigma(prices[-1]) if prices else 0.0003
            moves = _changes(seg, h)
            if len(moves) < MIN_CHANGES:
                moves = [c * h for c in _changes(seg, 1)]
            for c in moves:
                norm.append(max(-WINSOR_CAP, min(WINSOR_CAP, c)) / sig)
        ordered = sorted(norm)
        pooled[h] = [_empirical_quantile(ordered, t) for t in STANDARD_TAUS]
    return pooled


def forecast_market(prices, total_sum, pooled):
    """240 x 5 quantile rows for one market's price history."""
    if not prices:
        return _gaussian_rows(0.5, 0.001)
    x = prices[-1]
    if len(prices) < 2:
        return _gaussian_rows(x, 0.001)
    if len(prices) < MIN_HISTORY:
        return _gaussian_rows(x, level_sigma(x))

    seg = prices[-WINDOW - 1:] if len(prices) > WINDOW + 1 else prices
    q_grid = _quantile_grid(seg, STANDARD_TAUS)  # per grid h: 5 quantiles
    lo_at = [q[0] for q in q_grid]
    hi_at = [q[4] for q in q_grid]

    # pooled outer/inner, in normalised units -> scale by this market's sigma
    sig0 = level_sigma(x)
    pool_lo_at = [pooled[h][0] * sig0 for h in GRID_HORIZONS]
    pool_hi_at = [pooled[h][4] * sig0 for h in GRID_HORIZONS]
    pool_q25_at = [pooled[h][1] * sig0 for h in GRID_HORIZONS]
    pool_q75_at = [pooled[h][3] * sig0 for h in GRID_HORIZONS]

    # adaptive inner sigma from recent robust vol
    rsig = _robust_sigma(prices, ADAPT_WINDOW)
    sig = min(max(rsig, ADAPT_LO * sig0), ADAPT_HI * sig0)
    # regime scale: recent (120m) robust vol vs window (720m) robust vol
    rsig_w = _robust_sigma(prices, WINDOW)
    ratio = (rsig / rsig_w) if rsig_w and rsig_w > 0 else 1.0
    rscale = min(max(ratio, REG_LO), REG_HI)
    sig = sig * rscale
    oscale = OUTER_SCALE * rscale

    rows = []
    for minute in range(1, FORECAST_MINUTES + 1):
        root = math.sqrt(minute)
        # median: horizon-grown sum-to-one pull
        if total_sum:
            med = x + PULL_A * (1 - math.exp(-minute / PULL_TAU)) * (x * S_INF / total_sum - x)
        else:
            med = x
        q05 = _interp_sqrt(lo_at, minute)
        q95 = _interp_sqrt(hi_at, minute)
        if pooled is not None:
            p05 = _interp_sqrt(pool_lo_at, minute)
            p95 = _interp_sqrt(pool_hi_at, minute)
            q05 = (1 - POOL_BLEND) * q05 + POOL_BLEND * p05
            q95 = (1 - POOL_BLEND) * q95 + POOL_BLEND * p95
        q05 = med + q05 * oscale * O05
        q95 = med + q95 * oscale
        inner = 0.67449 * sig * root
        if pooled is not None:
            p25 = _interp_sqrt(pool_q25_at, minute)
            p75 = _interp_sqrt(pool_q75_at, minute)
            inner_lo = (1 - POOL_INNER_BLEND) * inner + POOL_INNER_BLEND * (-p25)
            inner_hi = (1 - POOL_INNER_BLEND) * inner + POOL_INNER_BLEND * p75
        else:
            inner_lo = inner_hi = inner
        rows.append(_tidy([q05, med - inner_lo, med, med + inner_hi, q95]))
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
    all_prices = [[float(p) for p in (m.get("prices") or [])] for m in markets]
    total_sum = sum(pr[-1] for pr in all_prices if pr)
    pooled = _pooled_quantile_grid(all_prices) if len(markets) > 1 else None
    forecasts = []
    for market, prices in zip(markets, all_prices):
        rows = forecast_market(prices, total_sum, pooled)
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
