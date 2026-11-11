"""Long paired timing run with per-sample timestamps and power/clock telemetry.

Each sample is identical to the study's run_experiment.py sample (5 untimed
launches, synchronize, one event pair around `batch` queued launches), and the
candidate order rotates and reverses the same way. Additions: the run lasts for
a requested duration, every sample is timestamped, host wall time is recorded
next to the event time, and a background thread logs read-only sysfs telemetry.
Run through run_gpu.py. Nothing in the study tree is modified.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
import threading
import time

STUDY = Path('/app/att/triton-gemm-asm')
sys.path.insert(0, str(STUDY))
import torch
import triton
from experiment import compile_kernel, export_baseline, load_variant, validate

HWMON = next(Path('/sys/class/drm/card0/device/hwmon').glob('hwmon*'))
SENSORS = dict(sclk_hz='freq1_input', power_uw='power1_average', edge_mc='temp1_input', junction_mc='temp2_input', mem_mc='temp3_input', fan_rpm='fan1_input', pwm='pwm1')


class Telemetry(threading.Thread):
    def __init__(self, interval):
        super().__init__(daemon=True)
        self.interval, self.rows, self.stop_flag = interval, [], threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            t = time.monotonic()
            row = [t]
            for name in SENSORS.values():
                try:
                    row.append(int((HWMON / name).read_text()))
                except (OSError, ValueError):
                    row.append(None)
            row.append(time.monotonic() - t)
            self.rows.append(row)
            self.stop_flag.wait(self.interval)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--entries', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seconds', type=float, default=150)
    ap.add_argument('--batch', type=int, default=7)
    ap.add_argument('--warm', type=int, default=5, help='untimed launches before each sample (study default 5)')
    ap.add_argument('--telemetry-interval', type=float, default=0.1, help='0 disables the telemetry thread')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--pauses', nargs='*', default=[], help='AT:SECONDS idle pauses, e.g. 60:5 (synchronized, GPU idle)')
    ap.add_argument('--shape', nargs=3, type=int, default=[8192, 8192, 8192])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    m, n, k = args.shape
    a = torch.randn((m, k), device='cuda', dtype=torch.float16)
    b = torch.randn((k, n), device='cuda', dtype=torch.float16)
    saved_a, saved_b = a.clone(), b.clone()
    backing = torch.full((m * n + 512,), -123., device='cuda', dtype=torch.float16)
    c = backing[256:-256].view(m, n)
    ref = torch.matmul(a.float(), b.float())
    standard, values, grid = compile_kernel(a, b, c, json.loads((STUDY / 'results/tune8192/config.json').read_text()))
    c.fill_(float('nan'))
    standard[grid](*values)
    torch.cuda.synchronize()
    assert validate(c, ref)['passed']
    exact = c.clone()
    entries = json.loads(args.entries.read_text())
    result = dict(shape_mnk=[m, n, k], seed=args.seed, argv=sys.argv, launches_per_sample=args.batch, warm_launches=args.warm,
                  performance_mode=Path('/sys/class/drm/card0/device/power_dpm_force_performance_level').read_text().strip(),
                  torch=torch.__version__, triton=triton.__version__, entries=entries, candidates={}, rounds=[],
                  telemetry_columns=['monotonic', *SENSORS, 'read_seconds'],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def check():
        row = validate(c, ref, exact)
        row['inputs_unchanged'] = bool(torch.equal(a.view(torch.int16), saved_a.view(torch.int16)) and torch.equal(b.view(torch.int16), saved_b.view(torch.int16)))
        row['guards_intact'] = bool((backing[:256] == -123).all() and (backing[-256:] == -123).all())
        row['passed'] &= row['inputs_unchanged'] and row['guards_intact']
        return row

    launches = {}
    for entry in entries:
        name = entry['name']
        cfg = entry['config']
        if isinstance(cfg, str):
            cfg = json.loads(Path(cfg).read_text())
        fn = compile_kernel if entry['module'] == 'tutorial' else importlib.import_module(entry['module']).compile_kernel
        base, kernel_args, kernel_grid = fn(a, b, c, cfg)
        export_baseline(base, args.out / name / 'generated')
        selected = load_variant(base, Path(entry['variant']), args.out / name / 'selected') if entry.get('variant') else base
        c.fill_(float('nan'))
        selected[kernel_grid](*kernel_args)
        torch.cuda.synchronize()
        row = dict(hsaco_sha256=hashlib.sha256(selected.asm['hsaco']).hexdigest(), shared_bytes=selected.metadata.shared, validation=check())
        result['candidates'][name] = row
        assert row['validation']['passed'], (name, row)
        launches[name] = lambda selected=selected, kernel_grid=kernel_grid, kernel_args=kernel_args: selected[kernel_grid](*kernel_args)

    def sample(fn):
        for _ in range(args.warm):
            fn()
        torch.cuda.synchronize()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.monotonic()
        begin.record()
        for _ in range(args.batch):
            fn()
        end.record()
        end.synchronize()
        t1 = time.monotonic()
        return dict(event_ms=begin.elapsed_time(end) / args.batch, wall_ms=(t1 - t0) * 1000 / args.batch, t0=t0, t1=t1)

    names = list(launches)
    random.Random(args.seed).shuffle(names)
    telemetry = Telemetry(args.telemetry_interval) if args.telemetry_interval > 0 else None
    if telemetry:
        telemetry.start()
    start = time.monotonic()
    result['start_monotonic'] = start
    repeat = 0
    pauses = sorted(tuple(float(x) for x in item.split(':')) for item in args.pauses)
    result['pauses'] = []
    while time.monotonic() - start < args.seconds:
        if pauses and time.monotonic() - start >= pauses[0][0]:
            torch.cuda.synchronize()
            t = time.monotonic()
            time.sleep(pauses[0][1])
            result['pauses'].append(dict(begin=t, end=time.monotonic(), before_round=repeat))
            pauses.pop(0)
        order = names[repeat % len(names):] + names[:repeat % len(names)]
        if (repeat // len(names)) % 2:
            order = order[::-1]
        row = dict(round=repeat, order=order)
        for name in order:
            row[name] = sample(launches[name])
        result['rounds'].append(row)
        repeat += 1
    result['timed_seconds'] = time.monotonic() - start
    if telemetry:
        telemetry.stop_flag.set()
        telemetry.join()
        result['telemetry'] = telemetry.rows
    result['post_timing_validation'] = {}
    for name, launch in launches.items():
        c.fill_(float('nan'))
        launch()
        torch.cuda.synchronize()
        result['post_timing_validation'][name] = check()
    (args.out / 'result.json').write_text(json.dumps(result, default=str) + '\n')
    assert all(v['passed'] for v in result['post_timing_validation'].values())
    print(json.dumps(dict(rounds=repeat, timed_seconds=result['timed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
