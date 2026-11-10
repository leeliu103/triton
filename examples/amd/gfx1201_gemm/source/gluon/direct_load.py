"""Direct WMMA-layout global loads to test the cost of operand LDS conversion."""
import triton
from triton.experimental.gluon import language as gl
from . import phase2_initial as base


def candidates():
    rows=[]
    for tm,tn,tk,warps in ((256,128,64,8),(128,128,64,8),(256,128,32,8)):
        for operands in (("A","B") if tm==128 else ("A",)):
            rows.append(dict(BLOCK_SIZE_M=tm,BLOCK_SIZE_N=tn,BLOCK_SIZE_K=tk,
                GROUP_SIZE_M=4,num_warps=warps,num_stages=1,pipeline="direct",
                b_order="kn",split_n=True,direct_operands=operands))
    rows.append(dict(rows[0],split_n=False))
    return rows


def specialization(shape,strides,config):
    values,options=base.specialization(shape,strides,config)
    constants=dict(zip(base.gemm.arg_names[3:],values))
    mma=constants["MMA"]
    if "A" in config["direct_operands"]:
        constants["A_LAYOUT"]=gl.DotOperandLayout(0,mma,8)
    if "B" in config["direct_operands"]:
        constants["B_LAYOUT"]=gl.DotOperandLayout(1,mma,8)
    return tuple(constants[name] for name in base.gemm.arg_names[3:]),options


def compile_kernel(a,b,c,config):
    m,k=a.shape;n=b.shape[1]
    values,options=specialization((m,n,k),(*a.stride(),*b.stride(),*c.stride()),config)
    args=(a,b,c,*values)
    grid=(triton.cdiv(m,config["BLOCK_SIZE_M"])*triton.cdiv(n,config["BLOCK_SIZE_N"]),1,1)
    return base.gemm.warmup(*args,grid=grid,**options),args,grid


gemm=base.gemm
