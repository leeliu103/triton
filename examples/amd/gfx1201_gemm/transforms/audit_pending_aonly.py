"""Conservative ordered-loadcounter audit across A-only prefetch loop boundaries."""
import argparse
import json
from pathlib import Path
import re

from prepare_direct_prefetch import regs


def audit(path):
    text=path.read_text()
    prologue,hot=text.split(".LBB0_1:",1)
    hot=hot.split("s_cbranch_scc1 .LBB0_1",1)[0]
    initial=[x.strip() for x in prologue.splitlines() if x.strip().startswith("buffer_load_b128")]
    pending=[];count=0;checked=0
    def step(inst):
        nonlocal pending,count,checked
        active=set().union(*(r for _,r in pending)) if pending else set()
        if inst.startswith("buffer_load_b128"):
            dest,rest=inst.split(",",1)
            assert not regs(rest)&active,(inst,regs(rest)&active)
            assert not regs(dest)&active,(inst,"pending overwrite",regs(dest)&active)
            pending.append((count,regs(dest)));count+=1
        elif inst.startswith("s_wait_loadcnt"):
            n=int(inst.split()[1],0)
            pending=pending[max(0,len(pending)-n):] if n else []
        elif inst.startswith(("v_","ds_")):
            assert not regs(inst)&active,(inst,regs(inst)&active)
            checked+=1
    for inst in initial:step(inst)
    states=[]
    for iteration in range(3):
        code=hot
        if iteration==2:
            code=re.sub(r"\ts_cbranch_scc0 (\.Laonly_skip_[A-Za-z0-9_]+)\n.*?\1:\n","",code,flags=re.S)
        for raw in code.splitlines():
            inst=raw.strip().split(";",1)[0].strip()
            if re.match("[a-z][a-z0-9_]+ ",inst):step(inst)
        states.append({"iteration":iteration,"pending_requests_at_exit":len(pending)})
    assert count==36 and not pending,(count,pending)
    report={"passed":True,"modeled_iterations":3,"global_requests":count,
        "checked_vector_LDS_instructions":checked,"states":states,
        "assumption":"ordered buffer-load completion count, same model as baseline compiler-generated waits"}
    (path.parent/"pending_global_register_audit.json").write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths",nargs="+",type=Path)
    args=parser.parse_args()
    for path in args.paths:print(json.dumps({"path":str(path),**audit(path)}),flush=True)
