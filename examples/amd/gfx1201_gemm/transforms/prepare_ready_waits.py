"""CPU-only ready-wait controls of the frozen fine-eight-cluster baseline."""
import json
from pathlib import Path
import re
import sys

ROOT = Path('/app/att/triton-gemm-asm')
sys.path[:0] = [str(ROOT/'gluon'), str(ROOT/'pingpong2/asm')]
from prepare_direct_asm import sha, instructions
from prepare_aonly_aprefetch import assemble
from audit_pending_aonly import audit
from prepare_chunk_sizes import semantic_outputs
from consumer_waits import Access, QueueContract, Request, rewrite_consumer_waits

SOURCE = ROOT/'pingpong2/asm/chunk-sizes/clusters8_control_lds16/kernel.amdgcn'


def vgprs(code):
    out = set()
    for lo, hi, single in re.findall(r'\bv\[(\d+):(\d+)\]|\bv(\d+)\b', code):
        out.update(range(int(lo), int(hi)+1) if lo else [int(single)])
    return frozenset(out)


def decode(op):
    """Conservative whole-VGPR access adapter for this compute region."""
    if op.startswith(('v_', 'buffer_load', 'ds_load')):
        head, operands = op.split(' ', 1)
        dst, tail = operands.split(',', 1)
        queue = 'lds' if head.startswith('ds_load') else 'global' if head.startswith('buffer_load') else None
        return Access(vgprs(tail), vgprs(dst), queue)
    assert not op.startswith(('buffer_', 'ds_')), op
    assert not vgprs(op), op
    return Access()


def region(source):
    first = source.index('\tds_load_2addr_b64')
    last = list(re.finditer(r'^\tv_wmma_[^\n]+\n', source, re.M))[-1].end()
    raw = source[first:last]
    stream = [line.strip() for line in raw.splitlines() if line.strip()]
    assert all(not line.startswith((';', '.loc', '.Ltmp')) for line in stream)
    return first, last, stream


def initial_global(source):
    # The last B packing fence (loadcnt8) completes all four initial B
    # requests. Eight current A loads remain conservatively pending.
    prefix = source.split('.LBB0_1:', 1)[1].split('\tds_load_2addr_b64', 1)[0]
    loads = [op for op in instructions(prefix) if op.startswith('buffer_load')]
    assert len(loads) == 8
    assert [op for op in instructions(prefix) if op.startswith('s_wait_loadcnt')][-1] == 's_wait_loadcnt 0x8'
    return tuple(Request(f'A{i}', decode(op).writes) for i, op in enumerate(loads))


def transform(source, lds, global_):
    first, last, stream = region(source)
    queues = {}
    if lds:
        queues['lds'] = QueueContract('s_wait_dscnt')
    if global_:
        queues['global'] = QueueContract('s_wait_loadcnt', initial_global(source))
    # Conditional B is present on recurrent iterations and absent on the
    # final one. Drain current A first so all post-join A operands are ready.
    boundary = stream.index('s_cbranch_scc0 .Laonly_skip_B_prefetch')
    fences = {boundary: ['global']} if global_ else {}
    changed, proof = rewrite_consumer_waits(stream, queues, decode, fences)
    assert semantic_outputs(instructions('\n'.join(stream))) == semantic_outputs(instructions('\n'.join(changed)))
    # Managed waits differ; every data operation and every control-flow
    # instruction, including the original WMMA order, is identical.
    omit = tuple(q.wait_opcode for q in queues.values())
    assert [op for op in stream if not op.startswith(omit)] == [op for op in changed if not op.startswith(omit)]
    result = source[:first] + ''.join(('' if op.endswith(':') else '\t')+op+'\n' for op in changed) + source[last:]
    proof.update(kind='consumer_ready_waits', manages_LDS=lds, manages_global=global_,
                 initial_global_order=[r.name for r in initial_global(source)],
                 unchanged_data_and_control_instructions=True,
                 symbolic_accumulator_histories_identical=True,
                 conditional_B_prefetch_current_A_fence=global_,
                 queue_contract='ordered completion for same-kind ds_load_2addr_b64 and buffer_load_b128 requests',
                 named_VGPR=215, physical_VGPR=216, dynamic_LDS_bytes=16384,
                 GPU_validation_required=True)
    return result, proof


def emit(source_path, name, changed, operation, outroot, dynamic_shared=16384):
    source = source_path.read_text()
    parent = json.loads(source_path.with_suffix('.amdgcn.transform.json').read_text())
    assert sha(source) == parent['output_sha256']
    out = outroot/name
    out.mkdir(parents=True, exist_ok=False)
    path = out/'kernel.amdgcn'
    path.write_text(changed)
    pending = audit(path)
    binary = assemble(changed, Path(parent['source_baseline']), out/'assembled')
    proof = dict(parent, parent_variant=str(source_path), parent_variant_sha256=sha(source),
                 output_sha256=sha(changed), required_dynamic_shared_bytes=dynamic_shared,
                 operations=parent['operations']+[operation], pending_register_audit=pending,
                 generator_sha256=sha(Path(__file__).read_text()),
                 scheduler_sha256=sha(Path(__file__).with_name('consumer_waits.py').read_text()),
                 assembly=binary, gpu_executed=False)
    path.with_suffix('.amdgcn.transform.json').write_text(json.dumps(proof, indent=2)+'\n')
    entry = {'name': name, 'module': 'gluon.direct_load',
             'config': str(ROOT/'results/phase2-direct-load-controls/config.json'), 'variant': str(path)}
    return entry, dict(entry, hsaco_sha256=binary['hsaco_sha256'], operation=operation)


def main():
    outroot = Path(__file__).parent/'ready-waits'
    source = SOURCE.read_text()
    entries, manifest = [], []
    for name, lds, global_ in [('consumer_lds', True, False), ('consumer_A', False, True), ('consumer_both', True, True)]:
        changed, operation = transform(source, lds, global_)
        entry, record = emit(SOURCE, name, changed, operation, outroot)
        entries.append(entry)
        manifest.append(record)
        print(json.dumps({'name': name, 'hsaco_sha256': record['hsaco_sha256']}), flush=True)
    (outroot/'entries.json').write_text(json.dumps(entries, indent=2)+'\n')
    (outroot/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
