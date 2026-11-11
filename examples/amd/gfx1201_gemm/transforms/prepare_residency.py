"""Offline LDS/VGPR residency controls and persistent wrappers of exact GEMM ASM."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path('/app/att/triton-gemm-asm')
sys.path.insert(0, str(ROOT / 'gluon'))
from prepare_direct_asm import sha, instructions
from prepare_aonly_aprefetch import assemble


def set_resources(source, vgprs=None, sgprs=None):
    if vgprs is not None:
        assert 215 <= vgprs <= 256
        source, n = re.subn(r'(\.amdhsa_next_free_vgpr\s+)\d+', rf'\g<1>{vgprs}', source)
        assert n == 1
        source, n = re.subn(r'(\.vgpr_count:\s+)\d+', rf'\g<1>{vgprs}', source)
        assert n == 1
        source, n = re.subn(r'(\.set \.Lgemm.num_vgpr,\s*)\d+', rf'\g<1>{vgprs}', source)
        assert n == 1
    if sgprs is not None:
        assert sgprs == 83
        source, n = re.subn(r'(\.amdhsa_next_free_sgpr\s+)\d+', rf'\g<1>{sgprs}', source)
        assert n == 1
        source, n = re.subn(r'(\.sgpr_count:\s+)\d+', rf'\g<1>{sgprs+2}', source)
        assert n == 1
        source, n = re.subn(r'(\.set \.Lgemm.numbered_sgpr,\s*)\d+', rf'\g<1>{sgprs}', source)
        assert n == 1
    return source


def persistent(source, grid):
    assert grid in (32, 64, 96)
    original_ops = instructions(source)
    # Every original launch input that the body overwrites must be restored.
    assert not re.search(r'\bv215\b|\bs8[012]\b|\bs\[[^\]]*8[012][^\]]*\]', source)
    assert not any('exec' in op for op in original_ops)
    assert source.count('ttmp9') == 3
    body = source.replace('ttmp9', 's82')
    head = '\ts_ashr_i32 s2, s82, 31\n'
    assert body.count(head) == 1
    prologue = ('\ts_mov_b64 s[80:81], s[0:1]\n'
                '\ts_mov_b32 s82, ttmp9\n'
                '\tv_mov_b32_e32 v215, v0\n'
                '.Lpp2_outer_tile:\n'
                '\ts_mov_b64 s[0:1], s[80:81]\n'
                '\tv_mov_b32_e32 v0, v215\n')
    body = body.replace(head, prologue + head)
    end = '\ts_nop 0\n\ts_sendmsg sendmsg(MSG_DEALLOC_VGPRS)\n\ts_endpgm\n'
    assert body.count(end) == 1
    tail = ('\ts_wait_storecnt 0x0\n'
            '\ts_barrier_wait -1\n'
            f'\ts_add_u32 s82, s82, {grid}\n'
            '\ts_cmp_lt_u32 s82, 2048\n'
            '\ts_cbranch_scc1 .Lpp2_outer_tile\n')
    body = body.replace(end, tail + end)
    # Strip only the wrapper and reverse the three substituted tile-ID reads:
    # the exact original instruction stream and all arithmetic must remain.
    restored = body.replace(prologue, '').replace(tail, '').replace('s82', 'ttmp9')
    assert restored == source
    body = set_resources(body, vgprs=216, sgprs=83)
    assignments = [list(range(block, 2048, grid)) for block in range(grid)]
    assert sorted(tile for tiles in assignments for tile in tiles) == list(range(2048))
    assert max(map(len, assignments)) - min(map(len, assignments)) <= 1
    return body, {
        'kind': 'persistent_outer_tile_loop', 'physical_grid': [grid, 1, 1],
        'original_logical_tiles': 2048, 'logical_tile_schedule': 'pid = physical_block + iteration * physical_grid',
        'logical_tiles_exactly_once': True,
        'tiles_per_physical_block_minmax': [min(map(len, assignments)), max(map(len, assignments))],
        'original_instruction_stream_restored_by_stripping_wrapper': True,
        'original_TTMP9_tile_index_reads_redirected_to_s82': 3,
        'saved_launch_inputs': {'s80_s81': 'kernarg pointer', 's82': 'logical tile ID', 'v215': 'original v0'},
        'output_stores_complete_before_reusing_any_vector_registers': True,
        'original_final_pending_LDS_reuse_arrival_consumed_before_next_tile': True,
        'VGPR_deallocation_only_after_final_logical_tile': True,
        'named_vgprs': 216, 'physical_vgpr_allocation': 216,
        'named_sgprs_including_VCC': 85, 'RDNA_SGPR_allocation_is_fixed': True,
        'target_average_waves_per_SIMD': grid * 8 / 128,
        'residency_limit_is_grid_not_per_WGP_reservation': True,
        'SQTT_must_verify_distribution_and_actual_residency': True,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path(__file__).parent / 'variants')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / 'results/phase2-direct-load-controls/config.json').read_text())
    sources = {
        'current': ROOT / 'variants/phase2-aonly-bprefetch/bpref12_split8/kernel.amdgcn',
        'pingpong': ROOT / 'pingpong/asm/clustered/phase_half_prio1/kernel.amdgcn',
    }
    entries, manifest = [], []
    for family, path in sources.items():
        source = path.read_text()
        parent = json.loads(path.with_suffix('.amdgcn.transform.json').read_text())
        assert sha(source) == parent['output_sha256']
        baseline = Path(parent['source_baseline'])
        specifications = [('lds48', None, 49152), ('lds64', None, 65536), ('vgpr241', 241, 16384)]
        for grid in (32, 64, 96):
            specifications.append((f'persistent{grid}', 'persistent', grid))
        for label, choice, number in specifications:
            name = f'{family}_{label}'
            selected_config = dict(config)
            module = 'gluon.direct_load'
            if choice == 'persistent':
                result, operation = persistent(source, number)
                shared = 16384
                selected_config['persistent_grid'] = number
                module = 'pingpong2.asm.residency_adapter'
            else:
                result = set_resources(source, vgprs=choice)
                shared = number
                assert instructions(result) == instructions(source)
                operation = {'kind': 'resource_only_residency_control',
                             'VGPR_padding': choice, 'dynamic_LDS_bytes': shared,
                             'instruction_stream_identical': True,
                             'target_steady_state_waves_per_SIMD': 4,
                             'SQTT_must_verify_actual_residency': True}
            out = args.out / name
            out.mkdir()
            asm = out / 'kernel.amdgcn'
            asm.write_text(result)
            (out / 'config.json').write_text(json.dumps(selected_config, indent=2) + '\n')
            binary = assemble(result, baseline, out / 'assembled')
            proof = dict(parent, parent_variant=str(path), parent_variant_sha256=sha(source),
                         output_sha256=sha(result), required_dynamic_shared_bytes=shared,
                         generator_sha256=sha(Path(__file__).read_text()),
                         operations=parent['operations'] + [operation], assembly=binary,
                         gpu_executed=False, limits='Root must validate numeric correctness and SQTT residency before interpreting timing.')
            asm.with_suffix('.amdgcn.transform.json').write_text(json.dumps(proof, indent=2) + '\n')
            row = {'name': name, 'module': module, 'config': str(out / 'config.json'), 'variant': str(asm)}
            entries.append(row)
            manifest.append(dict(row, hsaco_sha256=binary['hsaco_sha256'], operation=operation))
            print(json.dumps(manifest[-1]), flush=True)
    (args.out / 'entries.json').write_text(json.dumps(entries, indent=2) + '\n')
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
