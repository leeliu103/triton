"""Move next B to its earliest dead-register publication/WMMA boundary.

Uses the measured consumer-LDS control. Addresses and destinations of next B
are dead for the entire current compute region in the fine register layout.
"""
import json
from pathlib import Path
import re

from prepare_ready_waits import ROOT, sha, instructions, assemble, vgprs
from prepare_chunk_sizes import semantic_outputs
from path_audit import audit_global_paths, dynamic_vector_histogram

SOURCE = Path(__file__).parent/'ready-waits/consumer_lds/kernel.amdgcn'


def move_dead_prefetch(region, marker, after_compute, compute_prefix='v_wmma_'):
    """Move a uniform guarded prefetch within a read-only compute region.

    Proves that every register written by the prefetch is dead throughout
    the intervening data instructions. Counter adjustments and final-path
    reconciliation belong to the caller, which knows the queue contracts.
    """
    begin = region.index(f'\ts_cbranch_scc0 {marker}\n')
    end = region.index(marker+':\n', begin)+len(marker+':\n')
    guarded = region[begin:end]
    payload = guarded.split('\n', 1)[1].rsplit(marker+':\n', 1)[0]
    remainder = region[:begin]+region[end:]
    computes = list(re.finditer(r'^\t'+re.escape(compute_prefix)+r'[^\n]+\n', remainder, re.M))
    insertion = computes[after_compute-1].end() if after_compute else 0
    assert insertion <= begin
    writes = frozenset().union(*(vgprs(op.split(',', 1)[0]) for op in instructions(payload)
                                if op.startswith(('v_', 'buffer_load'))))
    touched = vgprs(remainder[insertion:])
    assert not writes & touched, sorted(writes & touched)
    return remainder, payload, insertion, {
        'prefetch_written_registers': sorted(writes),
        'prefetch_written_registers_dead_throughout_remaining_compute': True,
        'after_WMMA': after_compute,
    }


def transform(source, after_wmma):
    first = source.index('\tds_load_2addr_b64')
    last = list(re.finditer(r'^\tv_wmma_[^\n]+\n', source, re.M))[-1].end()
    original = source[first:last]
    region, payload, insertion, proof = move_dead_prefetch(original, '.Laonly_skip_B_prefetch', after_wmma)
    suffix = region[insertion:]
    before_waits = re.findall(r's_wait_loadcnt (0x[0-9a-f]+)', suffix)
    # Four future-B requests follow current A. Waiting for each current A
    # therefore leaves four more requests outstanding on recurrent paths.
    suffix = re.sub(r's_wait_loadcnt (0x[0-9a-f]+)',
                    lambda m: f's_wait_loadcnt 0x{int(m[1], 0)+4:x}', suffix)
    replacement = ('\ts_cbranch_scc0 .Lsat_B_final_drain\n'+payload+
                   '\ts_branch .Lsat_B_prefetch_done\n'
                   '.Lsat_B_final_drain:\n\ts_wait_loadcnt 0x0\n'
                   '.Lsat_B_prefetch_done:\n')
    changed_region = region[:insertion]+replacement+suffix
    assert semantic_outputs(instructions(original)) == semantic_outputs(instructions(changed_region))
    result = source[:first]+changed_region+source[last:]
    assert dynamic_vector_histogram(source) == dynamic_vector_histogram(result)
    proof.update(kind='dead_register_next_B_prefetch', old_after_WMMA=52,
                 A_wait_thresholds_after_prefetch_before=[int(x, 0) for x in before_waits],
                 A_wait_thresholds_after_prefetch_new=[int(x, 0)+4 for x in before_waits],
                 final_no_prefetch_path_drains_current_A_before_join=True,
                 symbolic_accumulator_histories_identical=True,
                 exact_dynamic_vector_memory_instruction_multiset=True,
                 named_VGPR=215, physical_VGPR=216, dynamic_LDS_bytes=16384,
                 additional_vector_instructions=0, additional_VGPR=0,
                 GPU_validation_required=True)
    return result, proof


def main():
    outroot = Path(__file__).parent/'early-b'
    source = SOURCE.read_text()
    parent = json.loads(SOURCE.with_suffix('.amdgcn.transform.json').read_text())
    assert sha(source) == parent['output_sha256']
    entries, manifest = [], []
    for after_wmma in (0, 8, 16, 32):
        name = f'consumer_lds_B_after{after_wmma}'
        changed, operation = transform(source, after_wmma)
        pending = audit_global_paths(changed)
        out = outroot/name
        out.mkdir(parents=True, exist_ok=False)
        path = out/'kernel.amdgcn'
        path.write_text(changed)
        binary = assemble(changed, Path(parent['source_baseline']), out/'assembled')
        proof = dict(parent, parent_variant=str(SOURCE), parent_variant_sha256=sha(source),
                     output_sha256=sha(changed), operations=parent['operations']+[operation],
                     generator_sha256=sha(Path(__file__).read_text()),
                     pending_register_audit=pending, assembly=binary, gpu_executed=False)
        path.with_suffix('.amdgcn.transform.json').write_text(json.dumps(proof, indent=2)+'\n')
        entry = {'name': name, 'module': 'gluon.direct_load',
                 'config': str(ROOT/'results/phase2-direct-load-controls/config.json'), 'variant': str(path)}
        entries.append(entry)
        manifest.append(dict(entry, hsaco_sha256=binary['hsaco_sha256'], operation=operation, pending_register_audit=pending))
        print(json.dumps({'name': name, 'hsaco_sha256': binary['hsaco_sha256']}), flush=True)
    (outroot/'entries.json').write_text(json.dumps(entries, indent=2)+'\n')
    (outroot/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
