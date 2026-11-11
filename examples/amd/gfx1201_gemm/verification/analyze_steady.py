"""Summarize a steady.py result: paired gains by time window, drift, telemetry."""
import json
from pathlib import Path
import random
import statistics as st
import sys

BASE = 'current_best'


def boot(values, n=4000, seed=1):
    r = random.Random(seed)
    meds = sorted(st.median(r.choices(values, k=len(values))) for _ in range(n))
    return meds[int(.025 * n)], meds[int(.975 * n)]


def nearest(tele, t, col):
    # telemetry rows are time ordered
    lo, hi = 0, len(tele) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if tele[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid
    return tele[lo][col]


def main(path):
    d = json.loads(Path(path).read_text())
    rounds = d['rounds']
    names = [n for n in d['candidates']]
    start = d['start_monotonic']
    tele = d.get('telemetry') or []
    cols = d['telemetry_columns']
    print(f"{path}: mode={d['performance_mode']} rounds={len(rounds)} timed={d['timed_seconds']:.1f}s pauses={[(round(p['begin']-start,1), round(p['end']-p['begin'],1)) for p in d.get('pauses', [])]}")
    print('post-validation all passed:', all(v['passed'] for v in d['post_timing_validation'].values()))
    def window(rows, label):
        if len(rows) < 3:
            return
        line = f"  {label:<22} n={len(rows):3d}"
        base = st.median(r[BASE]['event_ms'] for r in rows)
        line += f" base={base:.4f}ms"
        for name in names:
            if name == BASE:
                continue
            ratios = [(r[BASE]['event_ms'] / r[name]['event_ms'] - 1) * 100 for r in rows]
            lo, hi = boot(ratios)
            wins = sum(x > 0 for x in ratios)
            line += f" | {name[:22]}: {st.median(ratios):+.2f}% [{lo:+.2f},{hi:+.2f}] {wins}/{len(rows)}"
        print(line)
    window(rounds, 'all')
    t_of = lambda r: r[r['order'][0]]['t0'] - start
    edges = [0, 10, 20, 40, 60, 90, 120, 150, 180, 1e9]
    for a, b in zip(edges, edges[1:]):
        window([r for r in rounds if a <= t_of(r) < b], f't in [{a},{b if b < 1e9 else "end"})s')
    # wall vs event
    for name in names:
        ev = st.median(r[name]['event_ms'] for r in rounds)
        wall = st.median(r[name]['wall_ms'] for r in rounds)
        print(f"  {name:<24} event {ev:.4f} ms  wall {wall:.4f} ms  wall-event {wall-ev:+.4f}")
    # time course of base and winner, with telemetry
    print('  time course (median per 10 s bucket):')
    win = 'earlyB_double_LDS_bank'
    bucket = 10
    tmax = max(t_of(r) for r in rounds)
    b0 = 0
    while b0 <= tmax:
        rows = [r for r in rounds if b0 <= t_of(r) < b0 + bucket]
        if rows:
            line = f"    {b0:4d}s n={len(rows):2d} base {st.median(r[BASE]['event_ms'] for r in rows):.4f}"
            if win in names:
                line += f" win {st.median(r[win]['event_ms'] for r in rows):.4f}"
            if 'tutorial_control' in names:
                line += f" tut {st.median(r['tutorial_control']['event_ms'] for r in rows):.4f}"
            if tele:
                seg = [x for x in tele if b0 <= x[0] - start < b0 + bucket]
                if seg:
                    f = [x[1] for x in seg if x[1] is not None]
                    p = [x[2] for x in seg if x[2] is not None]
                    tj = [x[4] for x in seg if x[4] is not None]
                    line += f" | sclk {st.median(f)/1e6:.0f} MHz (min {min(f)/1e6:.0f} max {max(f)/1e6:.0f}) power {st.median(p)/1e6:.0f} W (max {max(p)/1e6:.0f}) Tj {st.median(tj)/1000:.0f} C"
            print(line)
        b0 += bucket
    if tele:
        # per-candidate sclk/power during its own timed interval
        print('  telemetry during each candidate\'s timed samples (median):')
        for name in names:
            f, p = [], []
            for r in rounds:
                s = r[name]
                seg = [x for x in tele if s['t0'] <= x[0] <= s['t1']]
                f += [x[1] for x in seg if x[1] is not None]
                p += [x[2] for x in seg if x[2] is not None]
            if f:
                print(f"    {name:<24} sclk {st.median(f)/1e6:.0f} MHz  power {st.median(p)/1e6:.1f} W  samples {len(f)}")
        lat = [x[-1] for x in tele]
        print(f"  telemetry reads: {len(tele)}, median read {st.median(lat)*1000:.2f} ms, max {max(lat)*1000:.2f} ms")


if __name__ == '__main__':
    for p in sys.argv[1:]:
        main(p)
        print()
