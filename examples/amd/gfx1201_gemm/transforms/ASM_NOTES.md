# Dependency scheduling experiments for gfx1201

These are CPU-assembled, exact-source transformations of the R9700 GEMM
experiment. GPU validation and measurements are owned by the main experiment
runner. Every variant retains the original compiler assembly hash in its
`.amdgcn.transform.json`; changing runtime shape or input identity must not
circumvent that binding.

The frozen fine-eight-cluster baseline is
`../../pingpong2/asm/chunk-sizes/clusters8_control_lds16/kernel.amdgcn`,
HSACO SHA256 `7e5e7a726987ffd777be2aa70adec4c6401116dc1d3f73810cfdf0e798670a34`.
It computes FP16 GEMM with FP32 accumulation, M=N=K=8192, a 256×128×64 tile,
eight wave32 waves per CTA, 215 named / 216 allocated VGPRs, and 16 KiB LDS.

## Mechanisms

`consumer_waits.py` contains a reusable instruction-stream transformation.
The caller supplies a register-access decoder, queue wait opcodes, an explicit
list of requests pending at entry, and any required queue fences. For every
register read or overwrite, it finds the youngest conflicting pending request
and emits the least restrictive wait that guarantees completion. If four
ordered LDS requests are outstanding and the instruction needs only the
oldest result, `s_wait_dscnt 3` suffices; requesting all four would require 0.
Later independent WMMAs can execute while newer requests finish.

The engine preserves every data instruction and its order. It is restricted
to explicitly ordered, same-kind request queues. It does not infer memory
aliasing or reconcile arbitrary control-flow joins. Region boundaries retain
the original publication/store fences; callers reconcile conditional request
histories explicitly. No predicted latency or priority setting enters the
algorithm.

`lds_lookahead.py` provides a second reusable transformation. Under an explicit
immutable-memory contract, it moves each read to the earliest position allowed
by register RAW/WAR/WAW dependencies, request ordering and control fences.
The consumer-wait engine then recalculates waits for the new order. One variant
reuses dead existing B registers; another alternates two B register banks and
remains within 240 allocated VGPRs. Neither adds VALU or memory instructions.

`prepare_early_b.py` applies the same dead-register principle across iterations.
The fixed-B-register layout leaves the next-B global destinations dead after
publication of current B. The next four B requests can therefore move from
after WMMA 52 to publication or after WMMA 8/16/32. A waits after this point allow
four additional outstanding requests. The final iteration skips future B and
drains current A before joining that common code. An explicit CFG-path audit
checks all 128 iterations, 1536 global requests and every register consumer.

`prepare_double_b_bank.py` adds two versions of the B LDS tile. Iteration t
uses bank t%2. Before a wave can overwrite bank t%2 at iteration t≥2, it must
pass publication(t−1). Every wave reaches that publication only after finishing
its reads at iteration t−2. The previous publication therefore also protects
reuse; the separate reuse barrier is redundant. One publication barrier per
iteration remains, including its store-completion fence.

The double-LDS variant precomputes 16 alternate addresses once and unrolls the
even 128-step loop by two. It uses 231 named / 240 allocated VGPRs and 32 KiB LDS,
still permitting six full-CTA waves per SIMD under the known resource limits.
Each bank contains exactly 16,384 bytes, written once per CTA and read in full
by each wave. Dynamic signal/wait counts per CTA drop from 257/256 to 128/128.
There is no pending barrier arrival at exit. The previous persistent wrapper,
which expects such an arrival, must not be applied to this variant unchanged.

## Generated entries

| Entry file | Candidates | Source |
|---|---:|---|
| `ready-waits/entries.json` | 3 | Frozen baseline; LDS, A, or both consumer waits |
| `early-b/entries.json` | 4 | Consumer-LDS; next B after 0/8/16/32 WMMAs |
| `lds-lookahead/entries.json` | 2 | Consumer-LDS; dead-register or second-bank LDS lookahead |
| `combined/entries.json` | 2 | Earliest next B combined with either LDS lookahead |
| `tutorial-transfer/entries.json` | 2 including control | Standard Triton tutorial and generic LDS lookahead |
| `double-lds-bank/entries.json` | 1 | Earliest B+consumer-LDS; double LDS bank, publication only |
| `shape-transfer-m4096-v2/entries.json` | 2 | Fresh M4096 fine baseline and double LDS bank |

Each generator writes assembly, assembled HSACO, proof metadata and launch
entries without initializing a GPU. The assembler validates code metadata
against the source. CPU proofs establish the intended transformations; they
do not replace numerical, timing or SQTT validation.

## Transfer to the original Triton tutorial

The same queue engine was applied without register-number changes to the
tutorial's main compute region, which stages both A and B in 48 KiB LDS and
uses `ds_load_b128`. It reconstructs all 125 original instructions, including
every LDS wait threshold and location, exactly. LLVM already implements
consumer-ready waits in that region; the hand-reclustered fine baseline had
replaced them with group drains.

The same request-lookahead engine then moves five tutorial LDS requests across
independent WMMAs. It preserves ALU-delay, control-flow and VMEM instructions
as motion fences, keeps the register/LDS footprint and all data operations,
and checks pending global registers explicitly. This tests transfer to a
different kernel layout. Any performance conclusion belongs to its GPU result.

`prepare_shape_transfer.py` also performs a fresh compiler specialization for
M=4096,N=8192,K=8192 and reapplies the complete transformation chain. That
compiler chooses different scalar register roles for the A descriptor and
K counter. A reversible adapter lets the older B-first scheduling helper
operate on canonical role names and immediately restores the actual physical
registers. The final assembly retains the freshly compiled prologue,
epilogue, scalar values and original input hash. Its 1024 output tiles are
enumerated to prove exact coverage, with no change to the 128-step K loop.

This factory is an explicit second-shape test, not a claim of arbitrary
layout or shape support. Its first compilation attempt stopped at the old
helper's scalar-register assertion and produced no runnable variant; only
the `shape-transfer-m4096-v2` entries are launch candidates.

## Accounting and interpretation

Most variants alter only waits, control instructions and scheduling, so their
useful matrix work and ordinary VALU work match the baseline. The double-LDS
variant adds 16 ordinary VALU instructions per wave in the prologue, a total
of 262,144 wave instructions for 8192³. Its modeled combined service numerator
is therefore 2,221,555,712 rather than 2,221,293,568. This small extra setup work
must be included when calculating a modeled VALU service fraction.

For the fresh M4096 specialization, instruction accounting gives
1,110,638,592 service cycles for the fine schedule and 1,110,769,664 for
the double-bank schedule. The difference is 131,072 added setup instructions.
These are computed from the actual emitted source: the compiler's slightly
different prologue means simply halving the 8192³ numerator is inaccurate.

An increased modeled fraction or SQTT busy fraction alone is not a speedup.
Use paired elapsed timings, bitwise comparison with the matching compiler
kernel, actual SQTT unit spans/residency and valid counters together. The
instruction-based service estimate and physical busy mask remain separate
measurements; neither establishes a universal achievable 100% bound.
