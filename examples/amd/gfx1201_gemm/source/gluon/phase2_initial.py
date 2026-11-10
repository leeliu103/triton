"""FP16 GEMM with explicit operand layouts and a two-bank LDS pipeline.

The two-bank pipeline reads the current bank into private operand registers,
stores the next global tile into the other bank, and uses one barrier to both
publish that next bank and finish all reads of the current bank. The WMMA then
uses only private registers, allowing waves to progress into the following
iteration while other waves finish their matrix instructions.

No device is initialized at import. ``compile_kernel`` is the GPU-harness
adapter; ``offline_compile.py`` provides explicitly device-free compilation.
"""

import math

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDWMMALayout
from triton.experimental.gluon.language.amd.cdna3 import buffer_load, buffer_store
from triton.experimental.gluon.language.amd.rdna4 import wmma


@gluon.jit
def _load_tiles(A, B, pid_m, pid_n, k_start,
                M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
                AM: gl.constexpr, AK: gl.constexpr, BK: gl.constexpr, BN: gl.constexpr,
                TM: gl.constexpr, TN: gl.constexpr, TK: gl.constexpr,
                A_LAYOUT: gl.constexpr, B_LAYOUT: gl.constexpr):
    ar = pid_m * TM + gl.arange(0, TM, layout=gl.SliceLayout(1, A_LAYOUT))
    ac = k_start + gl.arange(0, TK, layout=gl.SliceLayout(0, A_LAYOUT))
    br = k_start + gl.arange(0, TK, layout=gl.SliceLayout(1, B_LAYOUT))
    bc = pid_n * TN + gl.arange(0, TN, layout=gl.SliceLayout(0, B_LAYOUT))
    a_mask = gl.full([TM, TK], True, gl.int1, A_LAYOUT)
    b_mask = gl.full([TK, TN], True, gl.int1, B_LAYOUT)
    if M % TM != 0:
        a_mask = a_mask & (ar[:, None] < M)
    if N % TN != 0:
        b_mask = b_mask & (bc[None, :] < N)
    if K % TK != 0:
        a_mask = a_mask & (ac[None, :] < K)
        b_mask = b_mask & (br[:, None] < K)
    a = buffer_load(A, ar[:, None] * AM + ac[None, :] * AK, a_mask, 0)
    b = buffer_load(B, br[:, None] * BK + bc[None, :] * BN, b_mask, 0)
    return a, b


@gluon.jit
def _bank_tile(A, B, a_current, b_current, a_next, b_next, acc,
               pid_m, pid_n, k_start,
               M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
               AM: gl.constexpr, AK: gl.constexpr, BK: gl.constexpr, BN: gl.constexpr,
               TM: gl.constexpr, TN: gl.constexpr, TK: gl.constexpr,
               A_LAYOUT: gl.constexpr, B_LAYOUT: gl.constexpr,
               MMA: gl.constexpr, PREFETCH: gl.constexpr):
    if PREFETCH:
        next_a, next_b = _load_tiles(A, B, pid_m, pid_n, k_start + TK,
            M, N, K, AM, AK, BK, BN, TM, TN, TK, A_LAYOUT, B_LAYOUT)
    a = a_current.load(gl.DotOperandLayout(0, MMA, 8))
    b = b_current.load(gl.DotOperandLayout(1, MMA, 8))
    if PREFETCH:
        a_next.store(next_a)
        b_next.store(next_b)
        gl.barrier()
    acc = wmma(a, b, acc)
    a_current._keep_alive()
    b_current._keep_alive()
    return acc


@gluon.jit
def gemm(A, B, C,
         M: gl.constexpr, N: gl.constexpr, K: gl.constexpr,
         AM: gl.constexpr, AK: gl.constexpr, BK: gl.constexpr, BN: gl.constexpr,
         CM: gl.constexpr, CN: gl.constexpr,
         TM: gl.constexpr, TN: gl.constexpr, TK: gl.constexpr, GROUP_M: gl.constexpr,
         MMA: gl.constexpr, A_LAYOUT: gl.constexpr, B_LAYOUT: gl.constexpr,
         A_SHARED: gl.constexpr, B_SHARED: gl.constexpr, PIPELINE: gl.constexpr):
    gl.static_assert(K > 0)
    pid = gl.program_id(0)
    num_m: gl.constexpr = gl.cdiv(M, TM)
    num_n: gl.constexpr = gl.cdiv(N, TN)
    group = pid // (GROUP_M * num_n)
    first_m = group * GROUP_M
    group_m = gl.minimum(num_m - first_m, GROUP_M)
    inner = pid % (GROUP_M * num_n)
    pid_m = first_m + inner % group_m
    pid_n = inner // group_m
    acc = gl.full([TM, TN], 0.0, gl.float32, MMA)

    if PIPELINE == "direct":
        for k in range(gl.cdiv(K, TK)):
            a, b = _load_tiles(A, B, pid_m, pid_n, k * TK,
                M, N, K, AM, AK, BK, BN, TM, TN, TK, A_LAYOUT, B_LAYOUT)
            a = gl.convert_layout(a, gl.DotOperandLayout(0, MMA, 8))
            b = gl.convert_layout(b, gl.DotOperandLayout(1, MMA, 8))
            acc = wmma(a, b, acc)
    else:
        a, b = _load_tiles(A, B, pid_m, pid_n, 0,
            M, N, K, AM, AK, BK, BN, TM, TN, TK, A_LAYOUT, B_LAYOUT)
        a0 = gl.allocate_shared_memory(gl.float16, [TM, TK], A_SHARED, a)
        b0 = gl.allocate_shared_memory(gl.float16, [TK, TN], B_SHARED, b)
        a1 = gl.allocate_shared_memory(gl.float16, [TM, TK], A_SHARED)
        b1 = gl.allocate_shared_memory(gl.float16, [TK, TN], B_SHARED)
        gl.barrier()
        tiles: gl.constexpr = gl.cdiv(K, TK)
        pairs: gl.constexpr = (tiles - 1) // 2
        for pair in range(pairs):
            acc = _bank_tile(A, B, a0, b0, a1, b1, acc, pid_m, pid_n, pair * 2 * TK,
                M, N, K, AM, AK, BK, BN, TM, TN, TK, A_LAYOUT, B_LAYOUT, MMA, True)
            acc = _bank_tile(A, B, a1, b1, a0, b0, acc, pid_m, pid_n, (pair * 2 + 1) * TK,
                M, N, K, AM, AK, BK, BN, TM, TN, TK, A_LAYOUT, B_LAYOUT, MMA, True)
        if tiles % 2 == 0:
            acc = _bank_tile(A, B, a0, b0, a1, b1, acc, pid_m, pid_n, (tiles - 2) * TK,
                M, N, K, AM, AK, BK, BN, TM, TN, TK, A_LAYOUT, B_LAYOUT, MMA, True)
            acc = _bank_tile(A, B, a1, b1, a0, b0, acc, pid_m, pid_n, (tiles - 1) * TK,
                M, N, K, AM, AK, BK, BN, TM, TN, TK, A_LAYOUT, B_LAYOUT, MMA, False)
        else:
            acc = _bank_tile(A, B, a0, b0, a1, b1, acc, pid_m, pid_n, (tiles - 1) * TK,
                M, N, K, AM, AK, BK, BN, TM, TN, TK, A_LAYOUT, B_LAYOUT, MMA, False)
        a0._keep_alive()
        b0._keep_alive()
        a1._keep_alive()
        b1._keep_alive()
    rows = pid_m * TM + gl.arange(0, TM, layout=gl.SliceLayout(1, MMA))
    cols = pid_n * TN + gl.arange(0, TN, layout=gl.SliceLayout(0, MMA))
    buffer_store(acc.to(gl.float16), C, rows[:, None] * CM + cols[None, :] * CN,
                 (rows[:, None] < M) & (cols[None, :] < N))


def _warp_bases(warps, split_n):
    bits = int(math.log2(warps))
    if split_n:
        return [[0, 1]] + [[1 << i, 0] for i in range(bits - 1)]
    return [[1 << i, 0] for i in range(bits)]


def specialization(shape, strides, config):
    """Return positional constexpr values and compiler options; CPU only."""
    m, n, k = shape
    tm, tn, tk = (config[name] for name in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"))
    warps = config["num_warps"]
    mma = AMDWMMALayout(2, True, _warp_bases(warps, config.get("split_n", True)))
    a_layout = gl.BlockedLayout([1, 8], [4, 8], [warps, 1], [1, 0])
    # Four K elements per lane enable 2x2 in-thread transpose into K-major LDS.
    b_layout = gl.BlockedLayout([4, 8], [2, 16], [warps, 1], [1, 0])
    a_shared = gl.SwizzledSharedLayout(8, 1, min(8, tk // 8), [1, 0])
    b_order = config.get("b_order", "kn")
    if b_order == "kn":
        b_shared = gl.SwizzledSharedLayout(8, 1, min(8, tk // 8), [0, 1])
    else:
        b_shared = gl.SwizzledSharedLayout(8, 1, min(8, tn // 8), [1, 0])
    values = (m, n, k, *strides, tm, tn, tk, config.get("GROUP_SIZE_M", 4),
              mma, a_layout, b_layout, a_shared, b_shared, config["pipeline"])
    options = {"num_warps": warps, "num_stages": config.get("num_stages", 1),
               "waves_per_eu": config.get("waves_per_eu", 0)}
    return values, options


def compile_kernel(a, b, c, config):
    m, k = a.shape
    n = b.shape[1]
    assert b.shape[0] == k and tuple(c.shape) == (m, n)
    values, options = specialization((m, n, k), (*a.stride(), *b.stride(), *c.stride()), config)
    args = (a, b, c, *values)
    grid = (triton.cdiv(m, config["BLOCK_SIZE_M"]) * triton.cdiv(n, config["BLOCK_SIZE_N"]), 1, 1)
    return gemm.warmup(*args, grid=grid, **options), args, grid


compile = compile_kernel


def candidates():
    """Initial tractable LDS/tile controls; all configurations are JSON-safe."""
    rows = []
    for tm, tn, tk, warps in ((128,128,32,8), (128,128,64,8),
                             (256,128,32,8), (128,256,32,8),
                             (128,64,64,4), (64,128,64,4),
                             (128,128,32,4)):
        for b_order in ("kn", "nk"):
            rows.append(dict(BLOCK_SIZE_M=tm, BLOCK_SIZE_N=tn, BLOCK_SIZE_K=tk,
                             GROUP_SIZE_M=4, num_warps=warps, num_stages=1,
                             pipeline="two_bank", b_order=b_order, split_n=True))
    for tm, tn, tk, warps in ((256,128,64,8), (128,128,64,8)):
        rows.append(dict(BLOCK_SIZE_M=tm, BLOCK_SIZE_N=tn, BLOCK_SIZE_K=tk,
                         GROUP_SIZE_M=4, num_warps=warps, num_stages=1,
                         pipeline="direct", b_order="kn", split_n=True))
    return rows
