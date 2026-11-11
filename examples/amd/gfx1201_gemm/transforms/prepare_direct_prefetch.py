"""Pipeline the next direct-GEMM tile through dead current operand registers.

At WMMA48,16current WMMA instructions remain. Their operands occupy disjoint
registers from all48next-global words and two temporary address words. Four
loads need16restoring copies after WMMA64; the other eight land directly at
their original destinations. No new VGPR or LDS allocation is required.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from prepare_direct_asm import sha, instructions, hot_loop, split_entry, regroup, shared_ab


def regs(text):
    result=set(map(int,re.findall(r"\bv(\d+)\b",text)))
    for a,b in re.findall(r"\bv\[(\d+):(\d+)\]",text):
        result.update(range(int(a),int(b)+1))
    return result


def transform(source):
    begin,end,hot=hot_loop(source)
    last_load="\tbuffer_load_b128 v[198:201], v128, s[4:7], null offen offset:3670016\n"
    prolog_end=hot.index(last_load)+len(last_load)
    prolog=hot[:prolog_end]
    body=hot[prolog_end:]
    assert len([x for x in instructions(prolog) if x.startswith("buffer_load")])==12
    label_line=prolog.splitlines(keepends=True)[0]
    assert label_line.startswith(".LBB0_1:")
    prolog=prolog[len(label_line):]
    wmma_matches=list(re.finditer(r"^\tv_wmma_[^\n]+\n",body,re.MULTILINE))
    assert len(wmma_matches)==64
    injection=wmma_matches[47].end()
    final_ops=instructions(body[injection:])
    assert sum(x.startswith("v_wmma_") for x in final_ops)==16
    assert all(x.startswith(("v_wmma_","s_wait_dscnt","s_cbranch","s_delay_alu","s_barrier_signal")) for x in final_ops),final_ops
    live=regs("\n".join(x for x in final_ops if x.startswith("v_wmma_")))
    destinations=[(129,132),(174,177),(182,185),(186,189),(154,157),(158,161),(170,173),
                  (146,149),(178,181),(133,136),(194,197),(198,201)]
    incoming=set(r for a,b in destinations for r in range(a,b+1))
    assert len(incoming)==48
    assert not incoming&live,(sorted(incoming&live))
    assert not {235,236}&(incoming|live)
    prefetch=["s_cbranch_scc0 .Lasm_skip_prefetch",
        "v_add_nc_u32_e32 v235, s3, v216",
        "buffer_load_b128 v[129:132], v217, s[8:11], null offen",
        "buffer_load_b128 v[174:177], v235, s[4:7], null offen",
        "buffer_load_b128 v[182:185], v235, s[4:7], null offen offset:524288",
        "buffer_load_b128 v[186:189], v235, s[4:7], null offen offset:1048576",
        "buffer_load_b128 v[154:157], v235, s[4:7], null offen offset:1572864",
        "buffer_load_b128 v[158:161], v235, s[4:7], null offen offset:2097152",
        "buffer_load_b128 v[170:173], v235, s[4:7], null offen offset:2621440",
        "v_add_nc_u32_e32 v236, 0xffffc000, v217",
        "buffer_load_b128 v[146:149], v236, s[8:11], null offen",
        "v_add_nc_u32_e32 v236, 0xffff4000, v217",
        "buffer_load_b128 v[178:181], v236, s[8:11], null offen",
        "v_add_nc_u32_e32 v236, 0xffff8000, v217",
        "buffer_load_b128 v[133:136], v236, s[8:11], null offen",
        "buffer_load_b128 v[194:197], v235, s[4:7], null offen offset:3145728",
        "buffer_load_b128 v[198:201], v235, s[4:7], null offen offset:3670016"]
    prefetch_text="".join("\t"+x+"\n" for x in prefetch)+".Lasm_skip_prefetch:\n"
    restore_pairs=[(137+i,174+i) for i in range(4)]+[(141+i,182+i) for i in range(4)]+[(150+i,186+i) for i in range(4)]+[(190+i,146+i) for i in range(4)]
    assert not {a for a,b in restore_pairs}&{b for a,b in restore_pairs}
    restore=["s_cbranch_scc0 .Lasm_skip_restore","s_wait_loadcnt 0x0"]
    restore += [f"v_mov_b32_e32 v{a}, v{b}" for a,b in restore_pairs]
    restore_text="".join("\t"+x+"\n" for x in restore)+".Lasm_skip_restore:\n"
    body=body[:injection]+prefetch_text+body[injection:]
    branch="\ts_cbranch_scc1 .LBB0_1\n"
    body=body.replace(branch,restore_text+branch)
    result=source[:begin]+prolog+label_line+body+source[end:]
    assert [x for x in instructions(source) if x.startswith("v_wmma_")]==[x for x in instructions(result) if x.startswith("v_wmma_")]
    assert [x for x in instructions(source) if x.startswith("ds_")]==[x for x in instructions(result) if x.startswith("ds_")]
    assert regs(result)==regs(source)
    # Each next load is into its original destination or one fresh temporary.
    source_regs=[(129,132),(137,140),(141,144),(150,153),(154,157),(158,161),(170,173),
                 (190,193),(178,181),(133,136),(194,197),(198,201)]
    state={r:("initial",r) for r in range(239)}
    expected={}
    for index,((a,b),(x,y)) in enumerate(zip(destinations,source_regs)):
        for offset in range(4):
            state[a+offset]=(index,offset)
            expected[x+offset]=(index,offset)
    for dst,src in restore_pairs:
        state[dst]=state[src]
    assert all(state[r]==value for r,value in expected.items())
    return result,{"kind":"global_prefetch_after_WMMA48_no_new_registers",
        "wmma_lead":16,"temporary_next_load_words":48,"extra_copy_words":16,
        "global_destination_ranges":destinations,"final_WMMA_live_registers":sorted(live),
        "prefetch_destinations_disjoint_from_remaining_WMMA_operands":True,
        "prefetch_addresses_disjoint_from_data_and_current_operands":True,
        "restored_original_global_result_words_symbolically_equal":True,
        "first_iteration_uses_original_global_prologue":True,
        "last_iteration_skips_all_next_tile_loads":True,
        "same_global_request_order_with_next_tile_addresses":True,
        "original_WMMA_LDS_sequences_and_resources_unchanged":True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    original=args.baseline.read_text()
    for name,batch,split in [("prefetch16",0,False),("prefetch16_splitentry",0,True),
                              ("prefetch16_scratch2_splitentry",2,True),
                              ("prefetch16_shared_ab_splitentry",6,True)]:
        changed=original; operations=[]
        if batch==6:
            changed,audit=shared_ab(changed);operations.append(audit)
        elif batch:
            changed,audit=regroup(changed,batch);operations.append(audit)
        if split:
            changed,audit=split_entry(changed);operations.append(audit)
        changed,audit=transform(changed);operations.append(audit)
        directory=args.out/name;directory.mkdir(parents=True,exist_ok=False)
        target=directory/"kernel.amdgcn";target.write_text(changed)
        provenance={"input_sha256":sha(original),"output_sha256":sha(changed),
            "required_dynamic_shared_bytes":max(16384,batch*8192),"operations":operations,
            "source_baseline":str(args.baseline.resolve()),"generator_sha256":sha(Path(__file__).read_text()),
            "gpu_executed":False,"limits":"Requires GPU numerical/guard/performance validation."}
        target.with_suffix(".amdgcn.transform.json").write_text(json.dumps(provenance,indent=2)+"\n")
        print(json.dumps({"variant":name,"amdgcn_sha256":sha(changed),"directory":str(directory)}),flush=True)


if __name__=="__main__":
    main()
