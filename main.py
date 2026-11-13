#!/usr/bin/env python3
"""NVIDIA next-CEO forecaster -- round-6 design.

Implements the r6 candidate from scratch-nvidia-r6/RESEARCH_R6.md: r3 plus
TWO changes, both inside forecast_case (everything else is r3 unchanged):

* Activity gate (from r5; D4 diagnostic): the r3 pull toward the
  sum-to-one target only lives in markets that recently moved.  In
  quiet markets the pull slope is ~0 pooled, so pulling them is wrong.
      d120 = prices[-1] - prices[-121] (or prices[-1] - prices[0] if <121)
      |d120| <= 0.0035  ->  med(h) = x  (no pull, any h)
      active            ->  med(h) = x + 1.1 * (1 - exp(-h/240))
                                  * (x * 0.99 / S - x)
  Scale stays 1.1 (r3) -- the r5 scale 1.234 adds nothing on top of the
  floor.  S is the sum of last prices over all markets (NO reallocation).
* Floor-aware down-quantile shrink (DeepSeek Ide 2; D5 diagnostic): for
  small markets (x < 0.02), the down offsets are shrunk by
      fshr = x / (x + 0.01)
  Downward moves are bounded by the floor at 0 (max -x), while the model
  offsets are ~lev (roughly constant 0.0004-0.0014) regardless of price --
  for x = 0.005 the pooled inner_down ~0.021 is 4x the price itself, so
  q25/q05 clamp at 0 and the quantiles are wasted.  The shrink makes the
  down side scale with price (like a multiplicative move model).  The UP
  side is not floor-bounded, so inner_up and o95 are NOT shrunk.
  Applied to inner_down and o05 (after x1.2 and x r).  No zero-atom
  collapse (r5 Z-all adds nothing on top of the floor).

Unchanged from r3: inner quantiles (level sigma adapted by robust 120-min
vol, blended 75/25 with the pooled empirical inner), outer quantiles
(per-market empirical + pooled blend 75/25, regime factor
r = clip(rsig120/rsig720, 0.5, 1.0), down tail widened 1.2x), short
history (< 120 prices) level-sigma normal fallback (no gate, no floor).

Validated (fast harness eval_r6b.py, 37 cases; standalone grade to confirm):
r6 pooled-6 0.260404, held-28 0.109288, fresh-9 0.112361
vs r3 pooled-6 0.265358, held-28 0.111951, fresh-9 0.115096
vs r5 pooled-6 0.262740, held-28 0.111369, fresh-9 0.114466.
Paired vs r3: held 27/28 wins (t=-11.32), fresh 9/9 (t=-5.42).
Reserve candidate (floor C=0.01 + gate hard0, |d120| == 0, parameter-free):
pooled-6 0.262504, held-28 0.109341, fresh-9 0.112069.
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
GATE_D120 = 0.0035    # r6: |d120| <= this -> market quiet, no sum-pull
SUM_STRENGTH_ACTIVE = 1.1  # r6: pull strength for active markets (r3 scale)
FLOOR_C = 0.01        # r6: floor shrink fshr = x / (x + FLOOR_C)
FLOOR_X_MAX = 0.02    # r6: shrink down offsets only for x < FLOOR_X_MAX
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
                         'adapt': LEVEL_S0, 'r': 1.0, 'own': None,
                         'd120': 0.0})
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
        # r6: activity gate (same segment as the harness: last 721 prices).
        d120 = pr[-1] - pr[-121] if len(pr) > 120 else pr[-1] - pr[0]
        # r6: floor shrink for small markets (down side only).
        fshr = x / (x + FLOOR_C) if x < FLOOR_X_MAX else 1.0
        info.append({'m': m, 'pr': pr, 'x': x, 'level': lev,
                     'adapt': adapt, 'r': r, 'own': own,
                     'd120': d120, 'fshr': fshr})

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
                if abs(inf['d120']) <= GATE_D120:
                    med = x
                else:
                    med = x + SUM_STRENGTH_ACTIVE * (1 - math.exp(-h / SUM_TAU)) * (target - x)
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
                # r6: floor-aware shrink of the DOWN offsets only.
                inner_down *= inf['fshr']
                o05 *= inf['fshr']
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
