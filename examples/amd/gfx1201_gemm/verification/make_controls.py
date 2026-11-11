"""Build deliberately broken copies of the double-LDS-bank winner.

These are negative controls for the bit-exact check: each removes one piece of
the synchronization the winner relies on. They only change which LDS data a
wave may observe; global addresses, bounds and launch metadata are untouched.
Nothing under the original study tree is modified.
"""
import hashlib
import json
from pathlib import Path
import re

STUDY = Path('/app/att/triton-gemm-asm')
WINNER = STUDY / 'saturation/asm/double-lds-bank/earlyB_double_LDS_bank/kernel.amdgcn'
CONFIG = str(STUDY / 'results/phase2-direct-load-controls/config.json')
OUT = Path(__file__).resolve().parent / 'controls'


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def single_bank(text):
    """Both loop bodies use bank 0, with the winner's one barrier per step."""
    new, count = re.subn(r'(\tv_add_nc_u32_e32 v2(?:1[5-9]|2\d|30), )0x4000(, v1(?:8[2-9]|9[0-7])\n)', r'\g<1>0x0\2', text)
    assert count == 16, count
    return new, 'alternate LDS addresses v215..v230 set equal to bank 0 (16 adds of 0x4000 become adds of 0)'


def no_barrier(text):
    """Remove the publication barrier in both loop bodies."""
    new, count = re.subn(r'\ts_barrier_signal -1\n\ts_barrier_wait -1\n', '', text)
    assert count == 2, count
    return new, 'both s_barrier_signal/s_barrier_wait pairs deleted'


def no_store_fence(text):
    """Signal publication without first waiting for this wave's LDS stores."""
    new, count = re.subn(r'\ts_wait_dscnt 0x0\n(\ts_barrier_signal -1\n)', r'\1', text)
    assert count == 2, count
    return new, 'the s_wait_dscnt 0 before each publication signal deleted'


def main():
    text = WINNER.read_text()
    parent = json.loads(WINNER.with_suffix('.amdgcn.transform.json').read_text())
    assert parent['output_sha256'] == sha(text)
    OUT.mkdir(exist_ok=False)
    entries = [dict(name='winner', module='gluon.direct_load', config=CONFIG, variant=str(WINNER))]
    for fn in (single_bank, no_barrier, no_store_fence):
        new, edit = fn(text)
        assert new != text
        directory = OUT / fn.__name__
        directory.mkdir()
        asm = directory / 'kernel.amdgcn'
        asm.write_text(new)
        sidecar = dict(input_sha256=parent['input_sha256'], output_sha256=sha(new),
                       required_dynamic_shared_bytes=parent['required_dynamic_shared_bytes'],
                       kind='negative_control_not_a_candidate', parent_variant=str(WINNER),
                       parent_sha256=sha(text), edit=edit, purpose=fn.__doc__)
        asm.with_suffix('.amdgcn.transform.json').write_text(json.dumps(sidecar, indent=2) + '\n')
        entries.append(dict(name='control_' + fn.__name__, module='gluon.direct_load', config=CONFIG, variant=str(asm)))
        print(fn.__name__, sidecar['output_sha256'])
    (OUT / 'entries.json').write_text(json.dumps(entries, indent=2) + '\n')


if __name__ == '__main__':
    main()
