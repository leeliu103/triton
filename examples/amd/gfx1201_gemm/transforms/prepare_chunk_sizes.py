"""Offline two/eight-cluster GEMM schedules with explicit B register assignment."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

from prepare_residency import ROOT, set_resources, sha, instructions, assemble
sys.path.insert(0, str(ROOT / 'pingpong/asm'))
from prepare_clustered import regs

BAR = ['s_barrier_signal -1', 's_barrier_wait -1']


def semantic_outputs(ops):
    """Compare per-accumulator expression histories independent of B register IDs."""
    state = {r: ('input', r) for r in range(256)}
    outputs = defaultdict(list)
    requests = []
    for op in ops:
        if not op.startswith(('v_', 'buffer_load', 'ds_load')):
            continue
        head, rest = op.split(' ', 1)
        dst, tail = rest.split(',', 1)
        values = tuple(state[r] for r in regs(tail))
        if head.startswith('v_wmma'):
            expression = (head, values)
            outputs[tuple(regs(dst))].append(expression)
        else:
            expression = (head, tail, values)
            if head.startswith('buffer_load'):
                requests.append((op, values))
        for index, reg in enumerate(regs(dst)):
            state[reg] = (expression, index)
    return dict(outputs), requests


def transform(source, count, phase, mp, cp):
    assert count in (2, 8)
    first = source.index('\tds_load_2addr_b64')
    matches = list(re.finditer(r'^\tv_wmma_[^\n]+\n', source, re.MULTILINE))
    assert len(matches) == 64
    last = matches[-1].end()
    original = source[first:last]
    ops = instructions(original)
    wmma = [op for op in ops if op.startswith('v_wmma')]
    lds = [op for op in ops if op.startswith('ds_load')]
    first_prefetch = original.index('\ts_cbranch_scc0 .Laonly_skip_B_prefetch\n')
    last_prefetch = original.index('.Laonly_skip_B_prefetch:\n') + len('.Laonly_skip_B_prefetch:\n')
    prefetch = original[first_prefetch:last_prefetch]
    load_groups = [[op for op in lds if ('offset0:' not in op if k == 0 else f'offset0:{k*64} ' in op)] for k in range(4)]
    assert all(len(group) == 8 for group in load_groups)
    # B ranges never overlap the original A registers 139..170 or address regs.
    existing = [129, 133, 171, 175, 198, 202, 206, 210]
    two_buffers = [existing, list(range(216, 248, 4))]
    fine_buffer = [198, 202, 206, 210]
    memory_clusters, compute_clusters, waits = [], [], []
    def rename_load(op, target):
        return re.sub(r'v\[\d+:\d+\]', f'v[{target}:{target+3}]', op, count=1)
    def rename_wmma(op, target):
        fields = op.split(', ')
        assert len(fields) == 4
        fields[1] = f'v[{target}:{target+3}]'
        return ', '.join(fields)
    for k in range(4):
        original_registers = {tuple(regs(op.split(',', 1)[0])): n for n, op in enumerate(load_groups[k])}
        indexed = []
        for op in wmma[k*16:(k+1)*16]:
            bregs = tuple(regs(op.split(', ')[1]))
            assert bregs in original_registers
            indexed.append((op, original_registers[bregs]))
        assert Counter(index for _, index in indexed) == Counter({index: 2 for index in range(8)})
        if count == 2:
            if k % 2 == 0:
                memory_clusters.append([])
                compute_clusters.append([])
                waits.append(2 if k == 0 else 0)
            targets = two_buffers[k % 2]
            memory_clusters[-1].extend(rename_load(op, targets[index]) for index, op in enumerate(load_groups[k]))
            compute_clusters[-1].extend(rename_wmma(op, targets[index]) for op, index in indexed)
        else:
            for half in range(2):
                lo = half * 4
                memory_clusters.append([rename_load(load_groups[k][lo+i], fine_buffer[i]) for i in range(4)])
                compute_clusters.append([rename_wmma(op, fine_buffer[index-lo]) for op, index in indexed if lo <= index < lo+4])
                waits.append(3-k if half == 0 else None)
    assert len(memory_clusters) == len(compute_clusters) == count
    assert all(len(group) == 64//count for group in compute_clusters)
    code = []
    def emit(op):
        code.append(('' if op.endswith(':') else '\t') + op + '\n')
    def barrier():
        for op in BAR:
            emit(op)
    def conditional(high, label):
        emit('s_bitcmp1_b32 ttmp8, 27')
        emit(f's_cbranch_scc{0 if high else 1} .Lpp2_skip_{label}')
        barrier()
        emit(f'.Lpp2_skip_{label}:')
    if phase:
        conditional(True, 'entry')
    matrices_done = 0
    for index, (memory, compute, wait) in enumerate(zip(memory_clusters, compute_clusters, waits)):
        if mp:
            emit(f's_setprio {mp}')
        for op in memory:
            emit(op)
        emit('s_wait_dscnt 0x0')
        if wait is not None:
            emit(f's_wait_loadcnt 0x{wait:x}')
        if mp:
            emit('s_setprio 0')
        if phase:
            barrier()
        for i, op in enumerate(compute):
            emit(op)
            matrices_done += 1
            if cp and i == 0:
                emit(f's_setprio {cp}')
            if matrices_done == 52:
                if phase:
                    emit('s_cmp_lg_u32 s2, 0x4000')
                code.append(prefetch)
        if cp:
            emit('s_setprio 0')
        if phase:
            barrier()
    if phase:
        conditional(False, 'exit')
    emit('s_barrier_signal -1')
    if phase:
        emit('s_cmp_lg_u32 s2, 0x4000')
    replacement = ''.join(code)
    assert semantic_outputs(ops) == semantic_outputs(instructions(replacement))
    changed = source[:first] + replacement + source[last:]
    vgprs = 248 if count == 2 else 215
    changed = set_resources(changed, vgprs=vgprs)
    return changed, {'kind': 'explicit_B_register_cluster_schedule', 'clusters': count,
        'LDS_loads_per_memory_cluster': 32//count, 'WMMA_per_compute_cluster': 64//count,
        'phase_shift': phase, 'memory_priority': mp, 'compute_priority_after_first_WMMA': cp,
        'named_VGPR': vgprs, 'expected_full_CTA_resident_waves_per_SIMD': 4 if count == 2 else 6,
        'matrix_instruction_order': 'original' if count == 2 else 'pair both M halves for each four B fragments',
        'per_accumulator_K_order_and_symbolic_operand_values_identical': True,
        'global_B_prefetch_request_values_identical': True,
        'next_B_prefetch_after_emitted_WMMA': 52,
        'B_memory_is_read_only_throughout_asymmetric_region': True,
        'no_extra_VALU_or_WMMA_instructions': True,
        'GPU_validation_required': True}


def main():
    outroot = Path(__file__).parent / 'chunk-sizes'
    outroot.mkdir(parents=True, exist_ok=False)
    source_path = ROOT / 'variants/phase2-aonly-bprefetch/bpref12_split8/kernel.amdgcn'
    source = source_path.read_text()
    parent = json.loads(source_path.with_suffix('.amdgcn.transform.json').read_text())
    baseline = Path(parent['source_baseline'])
    entries, manifest = [], []
    for count in (2, 8):
        for label, phase, mp, cp in [('control', False, 0, 0), ('phase_cp1', True, 0, 1), ('phase_mp1', True, 1, 0)]:
            changed, operation = transform(source, count, phase, mp, cp)
            for shared in ([16384] if count == 2 else [16384, 49152]):
                resource_operation = dict(operation,
                    expected_full_CTA_resident_waves_per_SIMD=4 if shared == 49152 or count == 2 else 6)
                name = f'clusters{count}_{label}_lds{shared//1024}'
                out = outroot / name
                out.mkdir()
                path = out / 'kernel.amdgcn'
                path.write_text(changed)
                binary = assemble(changed, baseline, out / 'assembled')
                proof = dict(parent, parent_variant=str(source_path), parent_variant_sha256=sha(source),
                             output_sha256=sha(changed), required_dynamic_shared_bytes=shared,
                             generator_sha256=sha(Path(__file__).read_text()), assembly=binary,
                             operations=parent['operations'] + [resource_operation], gpu_executed=False,
                             limits='Requires root numeric validation and SQTT actual residency check.')
                path.with_suffix('.amdgcn.transform.json').write_text(json.dumps(proof, indent=2) + '\n')
                entry = {'name': name, 'module': 'gluon.direct_load',
                         'config': str(ROOT / 'results/phase2-direct-load-controls/config.json'), 'variant': str(path)}
                entries.append(entry)
                manifest.append(dict(entry, hsaco_sha256=binary['hsaco_sha256'], operation=resource_operation,
                                     required_dynamic_shared_bytes=shared))
                print(json.dumps(manifest[-1]), flush=True)
    (outroot / 'entries.json').write_text(json.dumps(entries, indent=2) + '\n')
    (outroot / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
