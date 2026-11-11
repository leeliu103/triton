# gfx1201 GEMM: Gluon and direct assembly optimization

> **Update, 2026-11-11.** Two later kernels, `eight_chunk_asm` and
> `double_lds_bank_asm`, are now in this package, and the results were
> independently re-verified. See [VERIFICATION.md](VERIFICATION.md). Points that
> change how this summary should be read:
>
> - The runtimes below come from short runs (under about 20 seconds). On this
>   card the clock is set by a 300 W power cap and then by the 110 °C junction
>   limit. In sustained runs the two later kernels were about 2–7% slower than
>   in short runs and the tutorial 4–7% slower. The four kernels below were not
>   re-timed sustained.
> - Coverage percentages are modeled service on one SIMD per capture. Variation
>   of up to about 0.9 points was seen between two captures of one binary, and
>   the percentage is not a speed metric: a busier kernel runs at a lower clock
>   under the power cap.
> - The statement below that B double buffering did not help refers to the
>   earlier attempt on `optimized_asm`. A later two-LDS-bank design on top of
>   consumer-ready waits and earlier B prefetch (`double_lds_bank_asm`) is
>   faster by a small margin (+0.2–0.4% for the two-bank step alone).
> - The exact Triton build was `3.8.0+git4cff872c.rocm10.0.0`.
>
> The text below is otherwise unchanged from the original study of the first
> four kernels.

On a Radeon AI PRO R9700, the selected FP16 GEMM improved from **9.913 ms to
8.569 ms** for **M = N = K = 8192**. This is **15.76% higher throughput**
(13.62% less latency), reaching **128.31 TFLOP/s**. Combined WMMA and ordinary
VALU coverage increased to **92.30%** in sqttkit. **100% was not achieved.**

This directory preserves the three requested kernel versions, their exact
measured assembly and executable code objects, and the high-level sources.
The unchanged tutorial baseline is included as a numerical and timing reference.
See [README.md](README.md) for execution instructions and [manifest.json](manifest.json)
for hashes, argument layouts, launch geometry, and required dynamic LDS.

## Kernels and measured performance

| Kernel | Artifact | Median runtime | WMMA coverage | Combined VALU coverage |
|---|---|---:|---:|---:|
| Original standard Triton | [tutorial_baseline](kernels/tutorial_baseline/kernel.amdgcn) | 9.913 ms | 72.18% | 77.10% |
| Previous best ASM | [previous_best_asm](kernels/previous_best_asm/kernel.amdgcn) | 9.363 ms | 78.79% | 85.08% |
| New A-direct Gluon | [a_direct_gluon](kernels/a_direct_gluon/kernel.amdgcn) | 8.697 ms | 86.80% | 89.78% |
| Gluon + optimized ASM | [optimized_asm](kernels/optimized_asm/kernel.amdgcn) | **8.569 ms** | **89.24%** | **92.30%** |

The runtime columns come from the **same unprofiled AUTO run**, with 31
interleaved rounds and seven launches per sample. The final ASM's median paired
throughput gains are **15.7610% versus the tutorial**, **9.3165% versus the
previous ASM**, and **1.5236% versus generated A-direct Gluon**. Separate medians
need not divide to exactly the median paired ratio. Individual samples and
correctness results are in [evidence/final_benchmark.json](evidence/final_benchmark.json).
The other seed measured +15.51% versus the tutorial.

Coverage columns come from separate warmed AUTO SQTT captures of each exact
binary. They describe the selected SIMD's full trace envelope, not the same
time interval as the unprofiled benchmark. WMMA is included in combined VALU;
the two columns must not be added.

The baseline comes from the kernel body and HIP configurations in the
[standard Triton matmul tutorial at the recorded revision](https://github.com/triton-lang/triton/blob/4b61eda79b69a1f28db9e98bc0b0428e1b2d6a92/python/tutorials/03-matrix-multiplication.py).
Its selected configuration is tile 256×128×64, group M4, eight warps, two stages,
and matrix non-K dimension 16. Inputs and output are FP16, with FP32 accumulation.
All published artifacts here are restricted to contiguous row-major 8192³ GEMM
on gfx1201; they are not a general matmul implementation.

## What changed

**Previous best ASM** keeps the standard tutorial's computation. It moves three
late B loads earlier, restores their values to the original registers, hoists
late LDS reads, and signals the buffer-reuse barrier before the last 16 WMMAs.
It uses 256 named VGPRs, 49,152 bytes of dynamic LDS, and no scratch. Unlike the
later ASM, this version adds vector restoration copies.

**New A-direct Gluon** changes the operand layout at a higher level. A is loaded
directly into its WMMA register layout, all eight warps are distributed along M,
and B remains cooperatively loaded and converted through LDS. The exact source
is [source/gluon/direct_load.py](source/gluon/direct_load.py), using the frozen
[phase2_initial.py](source/gluon/phase2_initial.py) and selected
[config.json](source/gluon/config.json). Against the initial Gluon implementation,
loop barrier signals fall from 10 to 2 and modeled LDS service falls from 96 to
56 cycles per wave/K64 iteration. The kernel still issues eight A and four B
vector loads. A's LDS reads disappear, but additional B reads under the new warp
mapping keep the total LDS-read count unchanged. It uses 214 named VGPRs,
16,384 bytes of dynamic LDS, and no scratch.

**Gluon + optimized ASM** orders B loads before A, waits for A only at its
consumers, and prefetches the next B tile into dead registers with 12 WMMAs
remaining. It signals the LDS-reuse barrier with eight WMMAs remaining; the wait
stays before LDS overwrite. First-iteration B loads are peeled into the prologue,
and the final iteration skips the next-tile prefetch. It uses 215 named VGPRs,
16,384 bytes of dynamic LDS, and no scratch. This edit preserves dynamic
WMMA/ordinary-VALU work and adds **no vector restoration copies**.

## Coalescing and vector width

The direct-A layout sacrifices coalescing while preserving the vector width:

| One A-load instruction per wave | Cooperative layout | Direct WMMA layout |
|---|---:|---:|
| Per-lane vector | 8 FP16 / 16 bytes | 8 FP16 / 16 bytes |
| Instruction | `buffer_load_b128` | `buffer_load_b128` |
| Row intervals touched | 4 × 128 bytes | 16 × 32 bytes |

Both request the same 32 KiB A tile once per workgroup. Controlled counters
confirm four times the A cache-line touches: full-GEMM `TCP_REQ` increases from
109,051,904 to 310,378,496, exactly matching the address-geometry prediction.
However, successive direct loads consume adjacent quarters of the same lines.
TCP misses remain similar, and GL2-to-EA requested-byte ranges overlap; the
medians are 7.8397 and 7.9566 GiB against a cooperative-A control with matching
warp mapping. This is more cache request pressure, not four times external
traffic. The LDS and synchronization savings outweigh that cost in this workload.

## Profiling, roof, and remaining bottleneck

Tools used:

- [Fixed ROCm profiler SDK and ROCm Compute](https://github.com/leeliu103/rocm-systems/tree/0f8e0287cc4b3b3e252757da2ae4e30aa4dc7e5c).
- [sqttkit, tested base revision](https://github.com/leeliu103/sqttkit/tree/ca920c1206bc82d8c7ec29eb62d6393e5160f4e7), including its gfx12 packet overlay in the SDK.
- Exact controlled instruction counters and an AUTO shader/REALTIME endpoint
  diagnostic covering all 128 SIMDs.

For the final ASM, the counter/clock method gives **92.46% combined service** and
**89.39% matrix service** over its recorded global shader interval, agreeing
closely with SQTT's 92.30% and 89.24%. Fifteen diagnostic pairs validate 245,760
timed wave records. Exact work is 134,217,728 WMMAs and 208,044,032 inclusive VALU
instructions. `SQ_INST_CYCLES_VALU` is 2,221,309,952; removing one scalar reciprocal
cycle per wave leaves 2,221,293,568 WMMA + ordinary VALU service cycles.

One native FP16 WMMA performs 8192 FLOPs with 16 nominal service cycles. Across
128 SIMD32s, the matrix roof is **65.536 × shader GHz TFLOP/s**. The selected
diagnostic's median clock was 2.288220 GHz. A pure WMMA calibration approached
99% service capacity, but its higher clock is not substituted for the GEMM's
clock. This GEMM's fixed instruction inventory has an optimistic matrix-only
ceiling of 96.6772%, even at 100% combined service, because ordinary VALU shares
capacity with WMMA.

The final SQTT trace has 4,096 complete waves, 128 detailed waves, and zero lost
packets, unknown instruction PCs, or incomplete waves. It needed no ambiguity
resolutions from the optional local association overlay used elsewhere in the
study. Six waves are resident for 96.35% of the envelope; coverage during those
periods is 93.15%.

The first B-ready wait falls from 617 to 2 cycles median, and the first B-store
gap from 501 to 35 cycles. Remaining gaps mainly involve B publication/LDS reads
(390-cycle median gap) and the long tail of B global-load completion. These
overlapping gap associations are not additive stall percentages. Moving B's
LDS publication earlier lowered coverage to 91.53% and was 0.47–0.69% slower;
it re-exposed global-load waits. Other tile sizes, A prefetch, and B double
buffering also failed to improve the selected version.

ROCm Compute's controlled replay confirms LDS bank-conflict/indexed-active
ratio falling from 8.3333% for the previous ASM to 0% with a nonzero denominator.
Controlled replay timing and wait percentages differ substantially from AUTO.
AUTO SQ/cache counters returned impossible zeros for known nonzero work and
were rejected. Raw controlled counter JSON supplies exact instruction work;
Compute exports round large counts. SQTT and endpoint measurements describe
modeled service utilization, **not physical exclusive busy masks**. See
[evidence/profiling.json](evidence/profiling.json) for the recorded values and scopes.

## Correctness and reproducibility

The selected ASM passed two normal-input seeds, uniform [-1,1], zeros, and
constant 0.125. In each case all 67,108,864 output elements are bitwise identical
to standard Triton. No elements exceed the FP32-reference tolerance
`0.015625 + 0.001 * abs(reference)`. Inputs remain unchanged, all outputs are
finite, and output guards survive poisoned-output launches. Evidence for each
pattern is retained in [evidence/](evidence/).

The measured stack was Torch 2.12.0+rocm10.0.0 and Triton 3.8.0. The new portable
launcher uses the exact archived HSACO files; it checks their hashes and fixed
launch contract. Recompiling relocated Python sources may change debug paths
and executable hashes. Those new binaries must be validated separately rather
than silently substituted for the measured artifacts. Historical timings came
from the Triton launcher; new HIP-launcher timings are separate measurements.

The packaged launcher was validated on the R9700 against all four archived
code objects. A fresh Triton cache forced compilation of the relocated Gluon
source: its executable hash changed, and all output elements still matched the
archived Gluon kernel bit for bit. Inputs and guards passed in both paths. See
[evidence/package_validation.json](evidence/package_validation.json).

The small binaries are included to preserve executable identity. The original
source paths in assembly debug directives are retained for that same reason;
they are not runtime dependencies. Raw SQTT streams, large profiler databases,
and GPU record snapshots are not part of this compact package.
