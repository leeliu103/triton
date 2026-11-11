"""Bit-compare EVERY launch of each kernel against standard Triton.

Two launch patterns per input set:
  interleaved  one output buffer; fill, launch, compare, repeat
  burst        several output buffers; launch back to back with nothing
               between kernels, then compare all of them
Comparisons stay on the GPU; only mismatch counts are copied to the host.
Run through run_gpu.py. Nothing in the study tree is modified.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

STUDY = Path('/app/att/triton-gemm-asm')
sys.path.insert(0, str(STUDY))
import torch
import triton
from experiment import compile_kernel, export_baseline, load_variant, validate

GUARD = 256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--entries', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seeds', nargs='+', type=int, required=True)
    ap.add_argument('--pattern', choices=['normal', 'uniform'], default='normal')
    ap.add_argument('--interleaved', type=int, default=200, help='launches per kernel per seed')
    ap.add_argument('--burst-rounds', type=int, default=20)
    ap.add_argument('--buffers', type=int, default=16)
    ap.add_argument('--shape', nargs=3, type=int, default=[8192, 8192, 8192])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    m, n, k = args.shape
    entries = json.loads(args.entries.read_text())
    result = dict(shape_mnk=[m, n, k], pattern=args.pattern, seeds=args.seeds, argv=sys.argv,
                  performance_mode=Path('/sys/class/drm/card0/device/power_dpm_force_performance_level').read_text().strip(),
                  torch=torch.__version__, triton=triton.__version__, entries=entries, runs=[])
    backings = [torch.full((m * n + 2 * GUARD,), -123., device='cuda', dtype=torch.float16) for _ in range(args.buffers)]
    outs = [x[GUARD:-GUARD].view(m, n) for x in backings]
    standard_config = json.loads((STUDY / 'results/tune8192/config.json').read_text())

    def guards():
        return all(bool((x[:GUARD] == -123).all() and (x[-GUARD:] == -123).all()) for x in backings)

    def save():
        (args.out / 'result.json').write_text(json.dumps(result, indent=2, default=str) + '\n')

    started = time.monotonic()
    for seed in args.seeds:
        torch.manual_seed(seed)
        if args.pattern == 'normal':
            a = torch.randn((m, k), device='cuda', dtype=torch.float16)
            b = torch.randn((k, n), device='cuda', dtype=torch.float16)
        else:
            a = torch.rand((m, k), device='cuda', dtype=torch.float16) * 2 - 1
            b = torch.rand((k, n), device='cuda', dtype=torch.float16) * 2 - 1
        saved_a, saved_b = a.clone(), b.clone()
        ref = torch.matmul(a.float(), b.float())
        standard, values, grid = compile_kernel(a, b, outs[0], standard_config)
        outs[0].fill_(float('nan'))
        standard[grid](*values)
        torch.cuda.synchronize()
        standard_check = validate(outs[0], ref)
        assert standard_check['passed'], standard_check
        exact = outs[0].clone()
        exact_bits = exact.view(torch.int16)
        # The reference itself must be repeatable, otherwise "bit-exact" is undefined.
        outs[0].fill_(float('nan'))
        standard[grid](*values)
        reference_repeat_changed = int((outs[0].view(torch.int16) != exact_bits).sum())
        del ref
        for entry in entries:
            name = entry['name']
            row = dict(seed=seed, name=name)
            result['runs'].append(row)
            cfg = entry['config']
            if isinstance(cfg, str):
                cfg = json.loads(Path(cfg).read_text())
            fn = compile_kernel if entry['module'] == 'tutorial' else importlib.import_module(entry['module']).compile_kernel
            base, kernel_args, kernel_grid = fn(a, b, outs[0], cfg)
            directory = args.out / f'seed{seed}' / name
            export_baseline(base, directory / 'generated')
            selected = load_variant(base, Path(entry['variant']), directory / 'selected') if entry.get('variant') else base
            assert kernel_args[2] is outs[0]
            per_buffer = [(*kernel_args[:2], c, *kernel_args[3:]) for c in outs]
            row.update(hsaco_sha256=hashlib.sha256(selected.asm['hsaco']).hexdigest(), shared_bytes=selected.metadata.shared,
                       reference_repeat_changed_elements=reference_repeat_changed)
            t0 = time.monotonic()
            # interleaved
            counts = torch.zeros(max(args.interleaved, 1), device='cuda', dtype=torch.int64)
            for i in range(args.interleaved):
                outs[0].fill_(float('nan'))
                selected[kernel_grid](*per_buffer[0])
                counts[i] = (outs[0].view(torch.int16) != exact_bits).sum()
            torch.cuda.synchronize()
            inter = counts[:args.interleaved].cpu().tolist()
            # burst
            burst = []
            for _ in range(args.burst_rounds):
                for c in outs:
                    c.fill_(float('nan'))
                torch.cuda.synchronize()
                for values_j in per_buffer:
                    selected[kernel_grid](*values_j)
                counts = torch.stack([(c.view(torch.int16) != exact_bits).sum() for c in outs])
                torch.cuda.synchronize()
                burst += counts.cpu().tolist()
            inputs_unchanged = bool(torch.equal(a.view(torch.int16), saved_a.view(torch.int16)) and torch.equal(b.view(torch.int16), saved_b.view(torch.int16)))
            for label, values_list in (('interleaved', inter), ('burst', burst)):
                bad = [i for i, v in enumerate(values_list) if v]
                row[label] = dict(launches=len(values_list), failing_launches=len(bad), first_failing_index=bad[0] if bad else None,
                                  max_changed_elements=max(values_list, default=0), total_changed_elements=sum(values_list),
                                  failing_counts_first_32=[values_list[i] for i in bad[:32]])
            row.update(inputs_unchanged=inputs_unchanged, guards_intact=guards(), seconds=time.monotonic() - t0)
            save()
            print(json.dumps(row), flush=True)
            del base, selected
        del a, b, saved_a, saved_b, exact, standard
        torch.cuda.empty_cache()
    result['elapsed_seconds'] = time.monotonic() - started
    summary = {}
    for row in result['runs']:
        s = summary.setdefault(row['name'], dict(hsaco_sha256=row['hsaco_sha256'], launches=0, failing_launches=0, seeds=0,
                                                 all_inputs_unchanged=True, all_guards_intact=True))
        assert s['hsaco_sha256'] == row['hsaco_sha256']
        s['seeds'] += 1
        for label in ('interleaved', 'burst'):
            s['launches'] += row[label]['launches']
            s['failing_launches'] += row[label]['failing_launches']
        s['all_inputs_unchanged'] &= row['inputs_unchanged']
        s['all_guards_intact'] &= row['guards_intact']
    result['summary'] = summary
    save()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
