"""Move next A0 into its dead register range before next B, without copies.

This is an exact-shape schedule transformation of an audited B-only prefetch
variant. The current A0 destination v167:170 is later reused by LDS, but its
last current-iteration use is WMMA40. All current A requests are drained before
issuing next A0 there. Next A0 and B0..B3 cross the loop boundary, followed by
the next loop's A1..A7. This gives identical request order in the prologue and
subsequent iterations, so B packing uses loadcnt10/9/7.
"""
import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

from prepare_direct_asm import sha, instructions, hot_loop
from prepare_direct_prefetch import regs
from audit_pending_aonly import audit


def transform(source):
    begin,end,hot=hot_loop(source)
    address="\tv_add_nc_u32_e32 v214, s2, v181\n"
    a0="\tbuffer_load_b128 v[167:170], v214, s[4:7], null offen\n"
    assert hot.count(address)==hot.count(a0)==1
    assert hot.count("\ts_clause 0x7\n")==1
    hot=hot.replace(address,"",1).replace(a0,"",1)
    hot=hot.replace("\ts_clause 0x7\n","\ts_clause 0x6\n",1)
    # B0..B3 now follow prefetched A0. Seven A requests follow them.
    prefix_end=hot.index("\tv_wmma_")
    prefix,body=hot[:prefix_end],hot[prefix_end:]
    assert re.findall(r"^\ts_wait_loadcnt (0x[0-9a-f]+)$",prefix,re.M)==["0xb","0xa","0x8","0x7"]
    for old,new in (("0xb","0xa"),("0xa","0x9"),("0x8","0x7")):
        # Replace using a temporary marker to avoid changing the preceding edit.
        prefix=prefix.replace("\ts_wait_loadcnt "+old+"\n","\ts_wait_loadcnt NEW"+new+"\n",1)
    prefix=prefix.replace("\ts_wait_loadcnt 0x7\n","",1).replace("NEW","")
    hot=prefix+body
    # Before the new A0 request there must be no current-A request pending.
    assert hot.count("\ts_wait_loadcnt 0x0\n")==1
    hot=hot.replace("\ts_wait_loadcnt 0x0\n","",1)
    matrices=list(re.finditer(r"^\tv_wmma_[^\n]+\n",hot,re.M))
    assert len(matrices)==64
    at=matrices[39].end()
    tail=hot[at:]
    assert not regs(a0.split(",",1)[0]) & regs("\n".join(
        inst for inst in instructions(tail) if inst.startswith(("v_","ds_"))))
    assert not any(inst.startswith("s_wait_loadcnt") for inst in instructions(tail))
    # v214 only addresses A loads and does not carry numerical operands.
    references=[inst for inst in instructions(source) if 214 in regs(inst)]
    assert len(references)==9 and all(inst.startswith(("v_add_nc_u32_e32 v214,","buffer_load_b128")) for inst in references)
    early=("\ts_wait_loadcnt 0x0\n"
           "\ts_cbranch_scc0 .Laonly_skip_A0_prefetch\n"+address+a0+
           ".Laonly_skip_A0_prefetch:\n")
    hot=hot[:at]+early+hot[at:]
    prologue=source[:begin]
    first_B=prologue.index("\tv_add_nc_u32_e32 v130, 0xffff4000, v180\n")
    assert prologue.count("\ts_mov_b32 s2, 0\n")==1
    assert first_B>prologue.index("\ts_mov_b32 s2, 0\n")
    prologue=prologue[:first_B]+address+a0+prologue[first_B:]
    result=prologue+hot+source[end:]
    for family in ("v_wmma_","ds_"):
        assert [x for x in instructions(source) if x.startswith(family)]==[x for x in instructions(result) if x.startswith(family)]
    assert regs(source)==regs(result)
    # Account for guarded next-tile blocks explicitly: the final iteration
    # skips them. Apart from scalar control, exact dynamic vector work agrees.
    def dynamic_vector_hist(text):
        b,e,h=hot_loop(text)
        guarded=re.findall(r"\ts_cbranch_scc0 (\.Laonly_skip_[A-Za-z0-9_]+)\n(.*?)\1:\n",h,re.S)
        for label,block in guarded:
            h=h.replace("\ts_cbranch_scc0 "+label+"\n"+block+label+":\n","")
        hist=Counter(x for x in instructions(text[:b]+text[e:]) if x.startswith("v_"))
        hist.update({x:n*128 for x,n in Counter(x for x in instructions(h) if x.startswith("v_")).items()})
        for _,block in guarded:
            hist.update({x:n*127 for x,n in Counter(x for x in instructions(block) if x.startswith("v_")).items()})
        return hist
    assert dynamic_vector_hist(source)==dynamic_vector_hist(result)
    proof={"kind":"next_A0_prefetch_after_WMMA40_before_next_B",
        "prefetch_A_registers":list(range(167,171)),"remaining_WMMA":24,
        "prefetch_destination_dead_until_next_iteration":True,
        "current_A_requests_drained_before_A0_prefetch":True,
        "prologue_and_recurrent_load_order":"A0 B0 B1 B2 B3 A1 A2 A3 A4 A5 A6 A7",
        "B_packing_waits":[10,9,7],"A0_ready_from_B_wait":True,
        "A_address_uses_advanced_s2":True,"A_address_operation_moved_not_duplicated":True,
        "last_iteration_skips_A0_and_B_prefetch":True,
        "extra_vector_copies":0,"extra_dynamic_vector_instructions":0,
        "WMMA_and_LDS_instruction_sequences_identical":True,
        "named_VGPRs":215,"dynamic_LDS_bytes":16384}
    return result,proof


def assemble(text,baseline_path,outdir):
    import torch
    from triton.backends.amd.compiler import HIPBackend
    from triton.backends.compiler import GPUTarget
    from triton._C.libtriton import llvm
    sys.path.insert(0,"/app/att/sage-h3-asm")
    from asm_loader import _validate_assembly,_binary_metadata
    metadata=json.loads(baseline_path.with_name("kernel.json").read_text())
    parsed,_=_validate_assembly(baseline_path.read_text(),text,metadata)
    backend=HIPBackend(GPUTarget("hip","gfx1201",32));options=backend.parse_options(metadata)
    runtime=importlib.import_module("triton.runtime.driver")
    def forbidden(*args,**kwargs):raise RuntimeError("CPU assembly attempted GPU initialization")
    with patch.object(torch.cuda,"_lazy_init",forbidden),patch.object(runtime,"_create_driver",forbidden):
        llvm.init_targets();binary=backend.make_hsaco(text,metadata,options)
        outdir.mkdir();(outdir/"kernel.hsaco").write_bytes(binary)
        assert _binary_metadata(outdir/"kernel.hsaco")==parsed
        assert not torch.cuda.is_initialized() and runtime.driver._active is None
    return {"cpu_only":True,"hsaco_sha256":__import__("hashlib").sha256(binary).hexdigest(),
            "binary_metadata_matches_validated_source":True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args();source=args.source.read_text()
    parent=json.loads(args.source.with_suffix(".amdgcn.transform.json").read_text())
    assert sha(source)==parent["output_sha256"]
    result,operation=transform(source)
    args.out.mkdir(parents=True,exist_ok=False)
    path=args.out/"kernel.amdgcn";path.write_text(result)
    baseline_path=Path(parent["source_baseline"])
    assert sha(baseline_path.read_text())==parent["input_sha256"]
    pending=audit(path)
    binary=assemble(result,baseline_path,args.out/"assembled")
    proof={"input_sha256":parent["input_sha256"],"output_sha256":sha(result),
        "source_baseline":str(baseline_path),"parent_variant":str(args.source.resolve()),
        "parent_variant_sha256":sha(source),"generator_sha256":sha(Path(__file__).read_text()),
        "required_dynamic_shared_bytes":16384,"operations":parent["operations"]+[operation],
        "pending_register_audit":pending,"assembly":binary,"gpu_executed":False,
        "limits":"CPU checks do not replace GPU numerical, guard, timing and SQTT validation."}
    path.with_suffix(".amdgcn.transform.json").write_text(json.dumps(proof,indent=2)+"\n")
    print(json.dumps({"path":str(path),"operation":operation,"audit":pending,"assembly":binary}),flush=True)


if __name__=="__main__":main()
