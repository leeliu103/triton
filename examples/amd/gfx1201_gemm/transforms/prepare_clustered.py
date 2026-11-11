"""Regroup the exact final gfx1201 GEMM into CDNA-style LDS/WMMA clusters.

The phase-shifted region starts after the existing whole-block B publication and
ends before B reuse. LDS is read-only throughout that region. Its high-wave
entry and low-wave exit conditional barriers balance all barrier generations.
Only root may launch the resulting experimental kernels.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

ROOT = Path('/app/att/triton-gemm-asm')
sys.path.insert(0, str(ROOT / 'gluon'))
from prepare_direct_asm import sha, instructions, hot_loop
from prepare_aonly_aprefetch import assemble

BAR = ['s_barrier_signal -1', 's_barrier_wait -1']


def regs(s):
    result = []
    for match in re.finditer(r'\bv(?:\[(\d+):(\d+)\]|(\d+))', s):
        if match[3] is not None:
            result.append(int(match[3]))
        else:
            result.extend(range(int(match[1]), int(match[2]) + 1))
    return result


def dataflow(ops):
    """Exact source-value audit of vector operands, ignoring timing and branches."""
    state = {r: ('input', r) for r in range(215)}
    events = []
    for op in ops:
        if not op.startswith(('v_', 'buffer_load', 'ds_load')):
            continue
        head, rest = op.split(' ', 1)
        dest, tail = rest.split(',', 1)
        reads = tuple(state[r] for r in regs(tail))
        if head.startswith('v_wmma'):
            events.append((op, reads))
        elif head.startswith('buffer_load'):
            events.append((op, reads))
        for index, register in enumerate(regs(dest)):
            state[register] = (head, op, index, reads)
    return events


def transform(source, phase, priority, wavebit=2):
    begin, end, hot = hot_loop(source)
    first = hot.index('\tds_load_2addr_b64')
    last_match = list(re.finditer(r'^\tv_wmma_[^\n]+\n', hot, re.MULTILINE))[-1]
    last = last_match.end()
    segment = hot[first:last]
    ops = instructions(segment)
    matrices = [x for x in ops if x.startswith('v_wmma_')]
    lds = [x for x in ops if x.startswith('ds_load_')]
    assert len(matrices) == 64 and len(lds) == 32
    assert not any(x.startswith('ds_store') for x in ops)
    groups = []
    for group in range(4):
        wanted = [x for x in lds if ('offset0:' not in x if group == 0 else f'offset0:{64 * group} ' in x)]
        assert len(wanted) == 8
        groups.append(wanted)
    assert ops.count('s_barrier_signal -1') == 1
    prefetch_first = segment.index('\ts_cbranch_scc0 .Laonly_skip_B_prefetch\n')
    prefetch_last = segment.index('.Laonly_skip_B_prefetch:\n') + len('.Laonly_skip_B_prefetch:\n')
    prefetch = segment[prefetch_first:prefetch_last]
    # No vector arithmetic outside the unchanged WMMA/LDS/prefetch is omitted.
    assert Counter(x for x in ops if x.startswith('v_') and not x.startswith('v_wmma')) == Counter(x for x in instructions(prefetch) if x.startswith('v_'))
    code = []
    def emit(line):
        code.append(('' if line.endswith(':') else '\t') + line + '\n')
    def conditional_barrier(high, name):
        # gfx12 workgroup wave ID is ttmp8[29:25], also used by this exact
        # generated source's initial s_bfe_u32 s13, ttmp8, 0x50019.
        emit(f's_bitcmp1_b32 ttmp8, {25 + wavebit}')
        emit(f's_cbranch_scc{0 if high else 1} .Lpp_skip_{name}')
        for op in BAR:
            emit(op)
        emit(f'.Lpp_skip_{name}:')
    if phase:
        conditional_barrier(True, 'entry')
    for group in range(4):
        for op in groups[group]:
            emit(op)
        emit('s_wait_dscnt 0x0')
        # A0/A4, A1/A5, A2/A6, A3/A7 respectively. All A loads were
        # issued before this region, and next B is not loaded until group 3.
        emit(f's_wait_loadcnt 0x{3-group:x}')
        if phase:
            for op in BAR:
                emit(op)
        for index, op in enumerate(matrices[group * 16:(group + 1) * 16]):
            emit(op)
            if priority and index == 0:
                emit(f's_setprio {priority}')
            if group == 3 and index == 3:
                # Conditional phase barriers overwrite SCC. Restore original
                # loop predicate before the unchanged final-iteration guard.
                if phase:
                    emit('s_cmp_lg_u32 s2, 0x4000')
                code.append(prefetch)
        if priority:
            emit('s_setprio 0')
        if phase:
            for op in BAR:
                emit(op)
    if phase:
        conditional_barrier(False, 'exit')
    # Existing reuse arrival is delayed until all staggered readers reconverge.
    emit('s_barrier_signal -1')
    if phase:
        emit('s_cmp_lg_u32 s2, 0x4000')
    replacement = ''.join(code)
    assert dataflow(ops) == dataflow(instructions(replacement))
    changed = source[:begin] + hot[:first] + replacement + hot[last:] + source[end:]
    assert [x for x in instructions(changed) if x.startswith('v_wmma')] == [x for x in instructions(source) if x.startswith('v_wmma')]
    assert Counter(x for x in instructions(changed) if x.startswith(('v_', 'ds_', 'buffer_'))) == Counter(x for x in instructions(source) if x.startswith(('v_', 'ds_', 'buffer_')))
    return changed, {
        'kind': 'immutable_B_region_four_LDS_WMMA_clusters',
        'phase_shift': phase, 'compute_priority_after_first_wmma': priority,
        'wave_group_bit': wavebit if phase else None,
        'cluster_count': 4, 'WMMA_per_compute_cluster': 16,
        'LDS_loads_per_memory_cluster': 8,
        'B_write_publication_and_reuse_outside_phase_shift_region': True,
        'conditional_barriers_balanced_per_wave': True if phase else None,
        'memory_clusters_wait_for_own_LDS_and_A_operands': True,
        'WMMA_operand_symbolic_dataflow_identical': True,
        'WMMA_instruction_order_identical': True,
        'all_vector_memory_instructions_identical_multiset': True,
        'next_B_prefetch_after_WMMA': 52,
        'extra_VALU': 0, 'extra_VGPR': 0,
        'required_dynamic_shared_bytes': 16384,
        'gpu_executed': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'variants/phase2-aonly-bprefetch/bpref12_split8/kernel.amdgcn')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'clustered')
    args = parser.parse_args()
    source = args.source.read_text()
    parent = json.loads(args.source.with_suffix('.amdgcn.transform.json').read_text())
    assert sha(source) == parent['output_sha256']
    baseline = Path(parent['source_baseline'])
    args.out.mkdir(parents=True, exist_ok=False)
    variants = [
        ('clustered_control', False, 0, 2),
        ('phase_half', True, 0, 2),
        ('phase_half_prio1', True, 1, 2),
        ('phase_evenodd_prio1', True, 1, 0),
        ('phase_bit1_prio1', True, 1, 1),
    ]
    manifest = []
    for name, phase, priority, bit in variants:
        changed, operation = transform(source, phase, priority, bit)
        out = args.out / name
        out.mkdir()
        path = out / 'kernel.amdgcn'
        path.write_text(changed)
        binary = assemble(changed, baseline, out / 'assembled')
        proof = dict(parent, parent_variant=str(args.source.resolve()),
                     parent_variant_sha256=sha(source), output_sha256=sha(changed),
                     generator_sha256=sha(Path(__file__).read_text()),
                     operations=parent['operations'] + [operation], assembly=binary,
                     limits='Experimental asymmetric barriers: numerical/guard/runtime/SQTT validation required.')
        path.with_suffix('.amdgcn.transform.json').write_text(json.dumps(proof, indent=2) + '\n')
        row = dict(name=name, assembly=str(path), binary=str(out / 'assembled/kernel.hsaco'),
                   hsaco_sha256=binary['hsaco_sha256'], operation=operation)
        manifest.append(row)
        print(json.dumps(row), flush=True)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
