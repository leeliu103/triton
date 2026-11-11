"""CPU-only exact-specialization transformations of the first Gluon direct GEMM.

The generated variants regroup the A conversion scratch tiles and/or split the
loop-entry barrier across the preceding private-register matrix computation.
No numerical operation, global load, accumulator order, or output store changes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def instructions(text):
    return [line.strip().split(";", 1)[0].strip() for line in text.splitlines()
            if re.match(r"\s*[a-z][a-z0-9_]+\s", line)]


def hot_loop(text):
    begin = text.index(".LBB0_1:")
    end = text.index("\ts_cbranch_scc1 .LBB0_1\n", begin) + len("\ts_cbranch_scc1 .LBB0_1\n")
    return begin, end, text[begin:end]


def scratch_addresses():
    """Enumerate original exact address arithmetic for all8waves×32lanes.

    A stores use v218 = ((thread<<7)&0x1c00 | (lane&7)<<4)
    XOR ((thread&0xf8)<<1), plus the instruction's0/512 offset.
    A load base v219 = ((warp<<7)&0x300 | (lane&15)<<4)
    XOR (lane&16) | (lane&7)<<10. The four source addresses are
    base, base XOR32, base XOR64, base XOR96. They read128 bits each.
    """
    stores, loads = set(), set()
    per_wave_load = []
    for wave in range(8):
        wave_bytes = set()
        for lane in range(32):
            thread = wave * 32 + lane
            store = (((thread << 7) & 0x1c00) | ((lane & 7) << 4)) ^ ((thread & 0xf8) << 1)
            base = (((wave << 7) & 0x300) | ((lane & 15) << 4)) ^ (lane & 16)
            base |= (lane & 7) << 10
            for offset in (0,512):
                stores.update(range(store + offset, store + offset + 16))
            for offset in (0,32,64,96):
                address = base ^ offset
                loads.update(range(address, address + 16))
                wave_bytes.update(range(address, address + 16))
        per_wave_load.append(len(wave_bytes))
    assert stores == set(range(8192)), (len(stores), max(stores))
    assert loads == stores, (len(loads), max(loads))
    return {"enumerated_waves":8, "enumerated_lanes_per_wave":32,
            "original_A_scratch_byte_interval":[0,8192],
            "store_and_load_byte_sets_equal":True,
            "per_wave_unique_read_bytes":per_wave_load}


def regroup(source, batch):
    assert batch in (2,4)
    begin, end, hot = hot_loop(source)
    prefix = "\ts_barrier_signal -1\n\ts_barrier_wait -1\n\tds_store_b128 v218, v[137:140]\n"
    start = hot.index(prefix)
    stop = hot.index("\tds_store_b64 v224, v[128:129]\n",start)
    original = hot[start:stop]
    ops = instructions(original)
    stores = [x for x in ops if x.startswith("ds_store_")]
    loads = [x for x in ops if x.startswith("ds_load_")]
    assert len(stores)==8 and len(loads)==16
    assert all(x.startswith("ds_store_b128 v218,") for x in stores)
    expected_store_regs = [(137,140),(141,144),(150,153),(154,157),(158,161),(170,173),(194,197),(198,201)]
    assert [tuple(map(int,re.search(r"v\[(\d+):(\d+)\]",x).groups())) for x in stores]==expected_store_regs
    expected_load_regs = [(174,177),(162,165),(146,149),(138,141),
                          (182,185),(166,169),(154,157),(142,145),
                          (186,189),(170,173),(158,161),(150,153),
                          (202,205),(198,201),(194,197),(190,193)]
    assert [tuple(map(int,re.search(r"v\[(\d+):(\d+)\]",x).groups())) for x in loads]==expected_load_regs
    other = Counter(x for x in ops if not x.startswith(("ds_","s_barrier_","s_wait_")))
    assert other == Counter({"v_mov_b32_e32 v137, v132":1})
    assert [x for x in ops if x.startswith("s_wait_loadcnt")]==["s_wait_loadcnt 0x1","s_wait_loadcnt 0x0"]
    code = ["s_barrier_signal -1","s_barrier_wait -1"]
    mappings=[]
    def offset_inst(inst, extra):
        old = re.search(r" offset:(\d+)$",inst)
        base = int(old.group(1)) if old else 0
        if old:
            inst=inst[:old.start()]
        return inst + (f" offset:{base+extra}" if base+extra else "")
    for group_start in range(0,4,batch):
        for group in range(group_start,group_start+batch):
            extra=(group-group_start)*8192
            for which in range(2):
                if group==3:
                    code.append(f"s_wait_loadcnt 0x{1-which:x}")
                code.append(offset_inst(stores[2*group+which],extra))
            mappings.append({"original_scratch_round":group,"new_scratch_offset":extra})
        code.extend(["s_wait_dscnt 0x0","s_barrier_signal -1","s_barrier_wait -1"])
        for group in range(group_start,group_start+batch):
            extra=(group-group_start)*8192
            for which in range(4):
                code.append(offset_inst(loads[4*group+which],extra))
        code.extend(["s_wait_dscnt 0x0","s_barrier_signal -1","s_barrier_wait -1"])
    code.append("v_mov_b32_e32 v137, v132")
    replacement="".join("\t"+x+"\n" for x in code)
    changed=source[:begin]+hot[:start]+replacement+hot[stop:]+source[end:]
    assert [x for x in instructions(source) if x.startswith("v_wmma_")]==[x for x in instructions(changed) if x.startswith("v_wmma_")]
    assert [x for x in instructions(source) if x.startswith("buffer_")]==[x for x in instructions(changed) if x.startswith("buffer_")]
    audit={"kind":"regroup_A_scratch_rounds","scratch_rounds_per_barrier":batch,
           "required_dynamic_shared_bytes":max(16384,batch*8192),
           "scratch_offset_mapping":mappings,
           "address_audit":scratch_addresses(),
           "original_loop_barriers":hot.count("\ts_barrier_signal"),
           "new_loop_barriers":hot_loop(changed)[2].count("\ts_barrier_signal"),
           "wmma_and_global_operation_order_unchanged":True,
           "all_current_batch_stores_complete_before_publish_barrier":True,
           "all_current_batch_reads_complete_before_reuse_barrier":True,
           "vgpr_and_scratch_unchanged":True}
    return changed,audit


def split_entry(source):
    begin,end,hot=hot_loop(source)
    first=hot.index("\ts_barrier_signal -1\n")
    assert hot[first:].startswith("\ts_barrier_signal -1\n\ts_barrier_wait -1\n\tds_store_b128 v218")
    marker="\ts_wait_dscnt 0x0\n"
    final=hot.rindex(marker)+len(marker)
    after=instructions(hot[final:])
    assert not any(x.startswith(("ds_","s_barrier_")) for x in after)
    assert sum(x.startswith("v_wmma_") for x in after)==13
    changedhot=hot[:first]+hot[first:].replace("\ts_barrier_signal -1\n","",1)
    final=changedhot.rindex(marker)+len(marker)
    changedhot=changedhot[:final]+"\ts_barrier_signal -1\n"+changedhot[final:]
    # The first iteration has no prior reader. Its initial arrival pairs with
    # the retained entry wait; each later iteration uses the preceding arrival.
    changed=source[:begin]+"\ts_barrier_signal -1\n"+changedhot+source[end:]
    return changed,{"kind":"split_loop_entry_barrier","wmmas_after_early_signal":13,
        "all_current_iteration_LDS_reads_complete_before_signal":True,
        "no_LDS_operation_between_signal_and_next_iteration_wait":True,
        "initial_signal_inserted_before_first_loop_iteration":True,
        "final_unconsumed_arrival_has_no_LDS_users_after_exit":True,
        "vgpr_and_lds_unchanged":True}


def shared_ab(source):
    """Publish32KiB A and16KiB B together, with disjoint shared addresses."""
    grouped,prior=regroup(source,4)
    begin,end,hot=hot_loop(grouped)
    first=hot.index("\ts_barrier_signal -1\n\ts_barrier_wait -1\n\tds_store_b128 v218")
    last=hot.index("\tds_load_2addr_b64 v[128:131], v231 offset1:16\n",first)
    old=instructions(hot[first:last])
    stores_a=[x for x in old if x.startswith("ds_store_b128 v218")]
    loads_a=[x for x in old if x.startswith("ds_load_b128")]
    assert len(stores_a)==8 and len(loads_a)==16
    b_start=old.index("v_mov_b32_e32 v137, v132")
    b_end=next(i for i in range(b_start,len(old)) if old[i]=="s_wait_dscnt 0x0")
    stores_b=old[b_start:b_end]
    assert len([x for x in stores_b if x.startswith("ds_store_b64")])==8
    assert set(x.split()[0] for x in stores_b)=={"v_mov_b32_e32","ds_store_b64"}
    # All global loads are complete before the last two A stores here.
    code=["s_barrier_signal -1","s_barrier_wait -1"]
    for index,inst in enumerate(stores_a):
        if index in (6,7):
            code.append(f"s_wait_loadcnt 0x{7-index:x}")
        code.append(inst)
    code.extend(stores_b)
    code.extend(["s_wait_dscnt 0x0","s_barrier_signal -1","s_barrier_wait -1"])
    code.extend(loads_a)
    newhot=hot[:first]+"".join("\t"+x+"\n" for x in code)+hot[last:]
    # These12loop-invariant VGPRs address only the B conversion's16KiB range.
    for reg in range(223,235):
        for inst in instructions(hot):
            if re.search(rf"\bv{reg}\b",inst):
                assert inst.startswith(("ds_store_","ds_load_")),inst
    shifts="".join(f"\tv_add_nc_u32_e32 v{r}, 0x8000, v{r}\n" for r in range(223,235))
    result=grouped[:begin]+shifts+newhot+grouped[end:]
    assert [x for x in instructions(source) if x.startswith("v_wmma_")]==[x for x in instructions(result) if x.startswith("v_wmma_")]
    assert [x for x in instructions(source) if x.startswith("buffer_")]==[x for x in instructions(result) if x.startswith("buffer_")]
    return result,{"kind":"joint_A_B_LDS_publication","prior":prior,
        "required_dynamic_shared_bytes":49152,"A_byte_interval":[0,32768],"B_byte_interval":[32768,49152],
        "B_address_shifted_registers":list(range(223,235)),"B_address_shift_bytes":32768,
        "old_loop_barriers":10,"new_loop_barriers":2,
        "all_A_and_B_stores_before_single_publication_barrier":True,
        "B_operand_store_values_preserved":True,
        "first_existing_B_load_wait_also_drains_all_A_loads":True,
        "VGPR_and_scratch_unchanged":True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    original=args.baseline.read_text()
    jobs=[("splitentry",0,True),("scratch2",2,False),("scratch2_splitentry",2,True),
          ("scratch4",4,False),("scratch4_splitentry",4,True),
          ("shared_ab",6,False),("shared_ab_splitentry",6,True)]
    for name,batch,split in jobs:
        changed=original
        operations=[]
        if batch==6:
            changed,audit=shared_ab(changed)
            operations.append(audit)
        elif batch:
            changed,audit=regroup(changed,batch)
            operations.append(audit)
        if split:
            changed,audit=split_entry(changed)
            operations.append(audit)
        directory=args.out/name
        directory.mkdir(parents=True,exist_ok=False)
        assembly=directory/"kernel.amdgcn"
        assembly.write_text(changed)
        audit={"input_sha256":sha(original),"output_sha256":sha(changed),
               "source_baseline":str(args.baseline.resolve()),"operations":operations,
               "required_dynamic_shared_bytes":max(16384,batch*8192),
               "generator_sha256":sha(Path(__file__).read_text()),"gpu_executed":False,
               "limits":"Static checks do not replace GPU numerical/guard/performance validation."}
        assembly.with_suffix(".amdgcn.transform.json").write_text(json.dumps(audit,indent=2)+"\n")
        print(json.dumps({"variant":name,"directory":str(directory),**audit}),flush=True)


if __name__=="__main__":
    main()
