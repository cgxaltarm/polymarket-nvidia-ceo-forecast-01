#!/usr/bin/env python3
"""NVIDIA next-CEO forecaster -- round-3 design.

Implements the r3 candidate described in RESEARCH_R3.md section 2, built on
the r2 base design in RESEARCH_R2.md section 3:

* Median: persistence pulled toward the sum-to-one target.  The pull is
  horizon-grown, tau 240, strength 1.1, target sum 0.99:
      med(h) = x + 1.1 * (1 - exp(-h/240)) * (x * 0.99 / S - x)
  where x is the checkpoint price and S is the sum of checkpoint prices
  over all markets in the case.
* Inner quantiles (25% / 75%): a tight normal centred on the median, with a
  level-dependent sigma adapted by the robust 120-min volatility
  (clipped to [0.75x, 4x] the level sigma), blended 75/25 with the pooled
  empirical inner quantiles (the pooled distribution's own q25/q75,
  normalised by level sigma and re-scaled).  The normal half-width is
  scaled by the regime factor r; the pooled part is not.
* Outer quantiles (5% / 95%): per-market empirical quantiles of h-minute
  price changes over the last 12h, winsorised to +/-0.05, blended 75/25
  with the cross-market pooled outer (changes normalised by level sigma,
  pooled over all markets, re-scaled).  The blended outer is scaled by
  the regime factor r, and the down tail (q05) is widened by 1.2.
* Regime factor: r = clip(rsig120 / rsig720, 0.5, 1.0), where rsig is the
  robust 1-min volatility sqrt(p_nonzero) * std(nonzero diffs) (floor
  0.0003 when fewer than 10 nonzero diffs).  Quiet now -> shrink; jumpy
  now -> keep (never widen above 1.0).
* Level sigma: 0.0004 + 0.0010 * clamp((x - 0.01) / 0.19, 0, 1).
* Empirical quantiles are computed on the grid horizons
  [1, 2, 4, 8, 16, 32, 64, 120, 240] and interpolated in sqrt(h).
* Short history (< 120 prices): level-sigma normal fallback for all five
  quantiles, median = last price (no sum-pull).

Stdlib only, deterministic: no randomness, no wall-clock, no network.
Reads $MAKEFUN_INPUT_DIR (else /input), writes $MAKEFUN_OUTPUT_DIR
(else /output), one .out per .in, JSON {"forecasts": [...]}.
"""

import json
import math
import os

Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
GRID_H = [1, 2, 4, 8, 16, 32, 64, 120, 240]
SQRT_H = {h: math.sqrt(h) for h in GRID_H}

# r3 parameters
LEVEL_S0 = 0.0004
LEVEL_S1 = 0.0010
RSIG_FLOOR = 0.0003
OUTER_WINDOW = 720
WINSOR = 0.05
POOL_B = 0.75        # pooled weight for the outer quantiles
POOL_INNER = 0.75    # pooled weight for the inner quantiles
SUM_TAU = 240.0
SUM_STRENGTH = 1.1
SUM_TARGET = 0.99
OUTER_Q05_SCALE = 1.2
REGIME_LO = 0.5
REGIME_HI = 1.0
ADAPT_LO = 0.75
ADAPT_HI = 4.0
SHORT_LEN = 120


def clamp01(x):
    return min(1.0, max(0.0, x))


def round6(x):
    return round(min(1, max(0, x)) * 1e6) / 1e6 + 0.0


def quantiles(changes, taus=TAUS):
    """Linear-interpolation quantiles of a list of changes."""
    ch = sorted(changes)
    n = len(ch)
    out = []
    for tau in taus:
        idx = tau * (n - 1)
        lo = int(math.floor(idx))
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        out.append(ch[lo] * (1 - frac) + ch[hi] * frac)
    return out


def h_changes(seg, h):
    return [seg[i + h] - seg[i] for i in range(len(seg) - h)]


def empirical_quantiles(seg, h):
    """Empirical quantiles of h-minute changes in seg, winsorised +/-0.05.
    Falls back to scaled 1-minute changes when fewer than 20 changes."""
    ch = h_changes(seg, h)
    if len(ch) < 20:
        ch1 = h_changes(seg, 1)
        ch = [c * h for c in ch1] if ch1 else [0.0]
    ch = [min(WINSOR, max(-WINSOR, c)) for c in ch]
    return quantiles(ch)


def robust_sigma(prices, window):
    """Robust 1-min vol: sqrt(p_nonzero) * std(nonzero diffs) over the
    last `window` minutes.  Floor 0.0003 when fewer than 10 nonzero diffs."""
    seg = prices[-window - 1:] if len(prices) > window + 1 else prices
    d = [seg[i + 1] - seg[i] for i in range(len(seg) - 1)]
    nz = [v for v in d if v != 0]
    if len(nz) < 10:
        return RSIG_FLOOR
    p = len(nz) / len(d)
    m = sum(nz) / len(nz)
    var = sum((v - m) ** 2 for v in nz) / (len(nz) - 1)
    return math.sqrt(p) * math.sqrt(var)


def level_sigma(x):
    frac = min(1.0, max(0.0, (x - 0.01) / 0.19))
    return LEVEL_S0 + LEVEL_S1 * frac


def interp_grid(eq, h):
    """Interpolate per-grid-horizon quantiles in sqrt(h) space."""
    s = math.sqrt(h)
    lo_h = max([g for g in GRID_H if g <= h], default=1)
    hi_h = min([g for g in GRID_H if g >= h], default=240)
    if lo_h == hi_h:
        return eq[lo_h]
    w = (s - SQRT_H[lo_h]) / (SQRT_H[hi_h] - SQRT_H[lo_h])
    return [eq[lo_h][j] * (1 - w) + eq[hi_h][j] * w for j in range(len(eq[lo_h]))]


def forecast_case(case):
    markets = case.get('markets', [])
    horizon = int(case.get('horizon_minutes', 240) or 240)
    if not markets:
        return {'forecasts': []}

    total = sum(m['prices'][-1] for m in markets if m.get('prices'))

    # Per-market precomputation.
    info = []
    for m in markets:
        pr = m.get('prices') or []
        if not pr:
            info.append({'m': m, 'pr': [], 'x': 0.0, 'level': LEVEL_S0,
                         'adapt': LEVEL_S0, 'r': 1.0, 'own': None})
            continue
        x = pr[-1]
        lev = level_sigma(x)
        rs120 = robust_sigma(pr, 120)
        rs720 = robust_sigma(pr, 720)
        r = rs120 / rs720 if rs720 > 0 else 1.0
        r = min(REGIME_HI, max(REGIME_LO, r))
        adapt = min(ADAPT_HI * lev, max(ADAPT_LO * lev, rs120))
        seg = pr[-(OUTER_WINDOW + 1):] if len(pr) > OUTER_WINDOW + 1 else pr
        own = {h: empirical_quantiles(seg, h) for h in GRID_H}
        info.append({'m': m, 'pr': pr, 'x': x, 'level': lev,
                     'adapt': adapt, 'r': r, 'own': own})

    # Cross-market pooled distribution per grid horizon: every market's
    # winsorised h-minute changes, normalised by its level sigma.
    pooled = {}
    for h in GRID_H:
        pool = []
        for inf in info:
            pr = inf['pr']
            if not pr:
                continue
            seg = pr[-(OUTER_WINDOW + 1):] if len(pr) > OUTER_WINDOW + 1 else pr
            ch = h_changes(seg, h)
            if len(ch) < 20:
                ch1 = h_changes(seg, 1)
                ch = [c * h for c in ch1] if ch1 else [0.0]
            ch = [min(WINSOR, max(-WINSOR, c)) for c in ch]
            pool.extend([c / inf['level'] for c in ch])
        pooled[h] = quantiles(pool) if pool else [0.0, 0.0, 0.0, 0.0, 0.0]

    forecasts = []
    for inf in info:
        pr = inf['pr']
        x = inf['x']
        lev = inf['level']
        rows = []
        if not pr:
            rows = [[0.5, 0.5, 0.5, 0.5, 0.5] for _ in range(horizon)]
            forecasts.append({'id': inf['m']['id'], 'q': rows})
            continue
        short = len(pr) < SHORT_LEN
        for h in range(1, horizon + 1):
            if short:
                # Level-sigma normal fallback, median = last price.
                row = [x + z * lev * math.sqrt(h) for z in Z]
            else:
                target = x * SUM_TARGET / total if total else x
                med = x + SUM_STRENGTH * (1 - math.exp(-h / SUM_TAU)) * (target - x)
                own_q = interp_grid(inf['own'], h)
                pool_q = interp_grid(pooled, h)
                pool_o05 = pool_q[0] * lev
                pool_o95 = pool_q[4] * lev
                o05 = (1 - POOL_B) * own_q[0] + POOL_B * pool_o05
                o95 = (1 - POOL_B) * own_q[4] + POOL_B * pool_o95
                o05 *= inf['r']
                o95 *= inf['r']
                o05 *= OUTER_Q05_SCALE
                normal_half = 0.67449 * inf['adapt'] * math.sqrt(h) * inf['r']
                inner_down = (1 - POOL_INNER) * normal_half + POOL_INNER * (-pool_q[1] * lev)
                inner_up = (1 - POOL_INNER) * normal_half + POOL_INNER * (pool_q[3] * lev)
                row = [med + o05, med - inner_down, med, med + inner_up, med + o95]
            row = [round6(v) for v in row]
            for i in range(1, 5):
                if row[i] < row[i - 1]:
                    row[i] = row[i - 1]
            rows.append(row)
        forecasts.append({'id': inf['m']['id'], 'q': rows})
    return {'forecasts': forecasts}


def main():
    in_dir = os.environ.get('MAKEFUN_INPUT_DIR', '/input')
    out_dir = os.environ.get('MAKEFUN_OUTPUT_DIR', '/output')
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith('.in'):
            continue
        with open(os.path.join(in_dir, name)) as f:
            case = json.load(f)
        result = forecast_case(case)
        out_name = name[:-3] + '.out'
        with open(os.path.join(out_dir, out_name), 'w') as f:
            json.dump(result, f)
            f.write('\n')


if __name__ == '__main__':
    main()
