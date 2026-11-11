"""Two-version B LDS storage removes the separate reuse barrier.

Precompute alternate bank addresses once; unroll the even128-step K64 loop by
two so alternating16KiB banks require no per-iteration address arithmetic.
"""
from collections import Counter
import json
from pathlib import Path
import re

from prepare_ready_waits import ROOT, sha, instructions, assemble, vgprs
from prepare_residency import set_resources
from path_audit import operations, uniform_path, dynamic_vector_histogram

SOURCE = Path(__file__).parent/'early-b/consumer_lds_B_after0/kernel.amdgcn'


def lds_address_proof(body):
    stores = [op for op in instructions(body) if op.startswith('ds_store_b64 ')]
    reads = [op for op in instructions(body) if op.startswith('ds_load_2addr_b64 ')]
    assert len(stores) == 8 and len(reads) == 32
    store_bytes = Counter()
    read_bytes = [set() for _ in range(8)]
    store_xors = [0, 0x808, 0x1010, 0x1818, 0x2020, 0x2828, 0x3030, 0x3838]
    for wave in range(8):
        for lane in range(32):
            store_base = wave*256 | (lane & 16)*8 | (lane & 1)*64 | ((lane << 2) & 56)
            read_base = ((lane << 11) & 0x3800) | ((lane & 16) << 4) | ((lane & 15) << 3)
            for op in stores:
                address_reg = int(re.match(r'ds_store_b64 v(\d+),', op)[1])
                address = store_base ^ store_xors[address_reg-182]
                store_bytes.update(range(address, address+8))
            for op in reads:
                address_reg = int(re.search(r', v(\d+)', op)[1])
                address = read_base ^ ((address_reg-190)*8)
                offsets = dict((key, int(value)) for key, value in re.findall(r'(offset[01]):(\d+)', op))
                for key in ('offset0', 'offset1'):
                    start = address+offsets.get(key, 0)*8
                    read_bytes[wave].update(range(start, start+8))
    expected = set(range(16384))
    assert set(store_bytes) == expected and set(store_bytes.values()) == {1}
    assert all(read == expected for read in read_bytes)
    assert not expected & {address+16384 for address in expected}
    return {'CTA_store_bytes': 16384, 'CTA_store_multiplicity': 1,
            'each_wave_reads_exact_same_16384_byte_bank': True,
            'bank_ranges': [[0, 16383], [16384, 32767]],
            'banks_disjoint': True, 'all_effective_addresses_below_required_LDS_bytes': True}


def audit_schedule(prologue, bodies, epilogue):
    pending, histogram = [], Counter()
    dynamic = Counter(op for op in operations(prologue+epilogue) if op.startswith(('v_', 'ds_', 'buffer_')))
    requests = checked = 0

    def step(op):
        nonlocal requests, checked
        active = frozenset().union(*(regs for _, regs in pending))
        if op.startswith('buffer_load_b128 '):
            dst, tail = op.split(',', 1)
            assert not vgprs(tail) & active, (op, 'RAW pending global')
            assert not vgprs(dst) & active, (op, 'WAW pending global')
            pending.append((requests, vgprs(dst)))
            requests += 1
            histogram[op] += 1
        elif op.startswith('s_wait_loadcnt '):
            threshold = int(op.split()[1], 0)
            del pending[:max(0, len(pending)-threshold)]
        elif op.startswith(('v_', 'ds_')):
            assert not vgprs(op) & active, (op, sorted(vgprs(op) & active))
            checked += 1

    for op in operations(prologue):
        if op.startswith('buffer_load_b128 '):
            step(op)
    states = []
    for iteration in range(128):
        for op in uniform_path(operations(bodies[iteration % 2]), iteration < 127):
            step(op)
            if op.startswith(('v_', 'ds_', 'buffer_')):
                dynamic[op] += 1
        if iteration in (0, 1, 126, 127):
            states.append({'iteration': iteration, 'bank': iteration % 2, 'pending_at_exit': len(pending)})
    assert requests == 1536 and not pending
    assert set(histogram.values()) == {128}
    return {'passed': True, 'modeled_K64_steps': 128, 'unrolled_loop_trip_count': 64,
            'global_requests': requests, 'every_original_global_request_exactly_128_times': True,
            'checked_vector_LDS_instructions': checked, 'states': states}, dynamic


def transform(source):
    loop_start = source.index('.LBB0_1:')
    head_end = source.index('\n', loop_start)+1
    branch = '\ts_cbranch_scc1 .LBB0_1\n'
    loop_end = source.index(branch, head_end)
    before, header, body, after = source[:loop_start], source[loop_start:head_end], source[head_end:loop_end], source[loop_end+len(branch):]
    assert before.count('\ts_barrier_signal -1\n') == 1
    assert body.count('\ts_barrier_signal -1\n') == 2
    assert body.count('\ts_barrier_wait -1\n') == 2
    counters = re.findall(r'^\ts_addk_co_i32 (s\d+), 0x80\n', body, re.M)
    assert len(counters) == 1
    counter = counters[0]
    assert body.count(f'\ts_cmp_lg_u32 {counter}, 0x4000\n') == 1
    assert before.count(f'\ts_mov_b32 {counter}, 0\n') == 1
    original_body = body
    # One reuse wait precedes stores; the other wait publishes those stores.
    first_store = body.index('\tds_store_b64 ')
    reuse_wait = body.rfind('\ts_barrier_wait -1\n', 0, first_store)
    assert reuse_wait >= 0
    body = body[:reuse_wait]+body[reuse_wait:].replace('\ts_barrier_wait -1\n', '', 1)
    final_signal = body.rfind('\ts_barrier_signal -1\n')
    assert final_signal > body.rfind('\tv_wmma_')
    body = body[:final_signal]+body[final_signal:].replace('\ts_barrier_signal -1\n', '', 1)
    assert body.count('\ts_barrier_signal -1\n') == body.count('\ts_barrier_wait -1\n') == 1
    # Original addresses182..197 are invariant for the entire K loop.
    for op in instructions(original_body):
        if op.startswith(('v_', 'buffer_load', 'ds_load')):
            assert not vgprs(op.split(',', 1)[0]) & set(range(182, 198))
    address_map = {old: 215+old-182 for old in range(182, 198)}
    alternate_addresses = ''.join(f'\tv_add_nc_u32_e32 v{new}, 0x4000, v{old}\n' for old, new in address_map.items())
    before = before.replace('\ts_barrier_signal -1\n', alternate_addresses, 1)
    # Clone only executable assembly/labels. Existing debug labels stay unique.
    body1 = ''.join(line for line in body.splitlines(keepends=True)
                    if not line.strip().startswith(('.loc', '.Ltmp')))
    labels = re.findall(r'^(\.L[^:\s]+):', body1, re.M)
    for label in labels:
        body1 = body1.replace(label, label+'_bank1')

    def remap_address(match):
        op = match[0]
        if op.startswith('\tds_store'):
            return re.sub(r'(?<= v)\d+(?=,)', lambda m: str(address_map[int(m[0])]), op, count=1)
        return re.sub(r'(?<=, v)\d+', lambda m: str(address_map[int(m[0])]), op, count=1)

    body1 = re.sub(r'^\tds_(?:store_b64|load_2addr_b64)[^\n]+', remap_address, body1, flags=re.M)
    inverse = {new: old for old, new in address_map.items()}
    restored_ops = []
    for op in instructions(body1):
        for label in labels:
            op = op.replace(label+'_bank1', label)
        if op.startswith('ds_store'):
            op = re.sub(r'(?<= v)\d+(?=,)', lambda m: str(inverse[int(m[0])]), op, count=1)
        elif op.startswith('ds_load'):
            op = re.sub(r'(?<=, v)\d+', lambda m: str(inverse[int(m[0])]), op, count=1)
        restored_ops.append(op)
    assert restored_ops == instructions(body)
    result = before+header+body+'.Lsat_bank1_iteration:\n'+body1+branch+after
    result = set_resources(result, vgprs=231)
    audit, dynamic = audit_schedule(before, (body, body1), after)
    # Normalize only bank addresses and subtract the sixteen prologue adds:
    # every original vector/memory operation executes exactly as before.
    normalized = Counter()
    for op, count in dynamic.items():
        if op in instructions(alternate_addresses):
            assert count == 1
            continue
        if op.startswith('ds_store'):
            op = re.sub(r'(?<= v)\d+(?=,)', lambda m: str(inverse.get(int(m[0]), int(m[0]))), op, count=1)
        elif op.startswith('ds_load'):
            op = re.sub(r'(?<=, v)\d+', lambda m: str(inverse.get(int(m[0]), int(m[0]))), op, count=1)
        normalized[op] += count
    assert normalized == dynamic_vector_histogram(source)
    proof = {'kind': 'two_version_B_LDS_publication_only_pipeline',
             'K64_steps': 128, 'K_loop_unroll_factor': 2, 'loop_trip_count': 64,
             'loop_counter_register': counter,
             'alternate_address_registers': address_map,
             'precomputed_address_instruction_count_per_wave': 16,
             'extra_dynamic_ordinary_VALU_wave_instructions_8192cube': 262144,
             'old_modeled_combined_service_numerator': 2221293568,
             'new_modeled_combined_service_numerator': 2221555712,
             'per_iteration_address_arithmetic_added': 0,
             'each_cloned_iteration_identical_after_LDS_address_and_label_normalization': True,
             'dynamic_vector_and_memory_instruction_multiset_identical_after_bank_normalization_and_prologue_adds': True,
             'barrier_epoch_proof': {
                 'iteration_0': 'bank0 has no previous reader',
                 'iteration_1': 'bank1 has no previous reader',
                 'iteration_t_ge_2': 'store(t) follows publish(t-1); all waves reached publish(t-1) only after completing read(t-2), so bank(t%2) is free',
                 'publication': 'each wave waits dscnt0 before arrival; every read(t) follows all-wave publication(t)',
                 'no_reuse_barrier_required': True,
                 'original_dynamic_signal_wait_counts_per_CTA': [257, 256],
                 'new_dynamic_signal_wait_counts_per_CTA': [128, 128],
                 'no_unconsumed_arrival_at_kernel_exit': True},
             'LDS_address_proof': lds_address_proof(original_body),
             'global_queue_audit': audit,
             'named_VGPR': 231, 'physical_VGPR': 240,
             'dynamic_LDS_bytes': 32768, 'expected_full_CTA_resident_waves_per_SIMD': 6,
             'GPU_validation_required': True}
    return result, proof


def main():
    source = SOURCE.read_text()
    parent = json.loads(SOURCE.with_suffix('.amdgcn.transform.json').read_text())
    assert sha(source) == parent['output_sha256']
    changed, operation = transform(source)
    name = 'earlyB_double_LDS_bank'
    outroot = Path(__file__).parent/'double-lds-bank'
    out = outroot/name
    out.mkdir(parents=True, exist_ok=False)
    path = out/'kernel.amdgcn'
    path.write_text(changed)
    binary = assemble(changed, Path(parent['source_baseline']), out/'assembled')
    proof = dict(parent, parent_variant=str(SOURCE), parent_variant_sha256=sha(source),
                 output_sha256=sha(changed), operations=parent['operations']+[operation],
                 required_dynamic_shared_bytes=32768,
                 generator_sha256=sha(Path(__file__).read_text()),
                 pending_register_audit=operation['global_queue_audit'], assembly=binary, gpu_executed=False)
    path.with_suffix('.amdgcn.transform.json').write_text(json.dumps(proof, indent=2)+'\n')
    entry = {'name': name, 'module': 'gluon.direct_load',
             'config': str(ROOT/'results/phase2-direct-load-controls/config.json'), 'variant': str(path)}
    (outroot/'entries.json').write_text(json.dumps([entry], indent=2)+'\n')
    (outroot/'manifest.json').write_text(json.dumps(dict(entry, operation=operation, assembly=binary), indent=2)+'\n')
    print(json.dumps({'name': name, 'hsaco_sha256': binary['hsaco_sha256'], 'resources': [231, 240, 32768]}), flush=True)


if __name__ == '__main__':
    main()
