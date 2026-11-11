# Verification of the double-LDS-bank kernel (2026-11-11)

This note adds two kernels to the package and reports an independent
re-verification of them. It also corrects how the earlier headline numbers
should be read. Everything here is for contiguous row-major FP16 GEMM with FP32
accumulation, M = N = K = 8192, on one Radeon AI PRO R9700 (gfx1201).

| Kernel | What it is | Named VGPRs | Dynamic LDS |
|---|---|---:|---:|
| [eight_chunk_asm](kernels/eight_chunk_asm/kernel.amdgcn) | A-direct Gluon with the WMMA body reordered into eight LDS-read/WMMA chunks | 215 | 16,384 bytes |
| [double_lds_bank_asm](kernels/double_lds_bank_asm/kernel.amdgcn) | The same, plus consumer-ready waits, earlier B prefetch and two alternating LDS banks | 231 | **32,768 bytes** |

`double_lds_bank_asm` declares no static LDS and addresses two 16 KiB banks. It
must be launched with at least 32,768 bytes of dynamic LDS. The HSACO hash does
not encode this; [manifest.json](manifest.json) does, and `benchmark.py` uses it.

## Result in one paragraph

The double-bank kernel is bit-exact in every one of 8,784 checked launches and
is faster than the eight-chunk kernel by about **1.5% in runs under 20 seconds**.
In sustained runs the lead was **1.1–1.3% in two runs and about 0.5% in a third
run that throttled harder**. Against the unchanged Triton tutorial it is about
**18% faster in short runs and 19–21% faster sustained**. On this card the
shader clock is set by a 300 W power cap and then by the 110 °C junction limit,
so throughput depends on when it is measured: about 130–132 TFLOP/s in runs
under 20 seconds and about 124–128 TFLOP/s sustained.

## Correctness

Every launch was compared bit for bit against the standard Triton kernel's
output for the same inputs, on the GPU, with the output buffer filled with NaN
before each launch.

| Kernel | Launches compared | Failing launches |
|---|---:|---:|
| double_lds_bank_asm | 8,784 | 0 |
| eight_chunk_asm | 8,000 | 0 |
| Intermediate (consumer waits + early B) | 8,000 | 0 |
| Control: both loop bodies use one LDS bank | 78 | 78 |
| Control: publication barrier removed | 78 | 78 |
| Control: store fence before the barrier removed | 78 | 78 |

- Inputs: eight normal-distributed seeds and two uniform seeds in AUTO mode,
  plus two normal seeds in `profile_standard` for the double-bank kernel.
- Two launch patterns per input set: fill, launch, compare, repeat; and 16
  launches back to back into 16 different output buffers, then compare.
- Inputs were unchanged and output guards intact in every run. The reference
  kernel reproduced its own output bit for bit on every input set.
- The three controls are deliberately broken copies of the double-bank kernel
  (in [verification/controls](verification/controls)). They show that the check
  detects an LDS race, including one that changes as few as about 1,600 of
  67,108,864 output elements.

Limits of this evidence:

- The controls fail on every launch. They prove the check detects such races;
  they do not show that the soak would provoke a rare, narrow timing window.
- All launches ran on an otherwise idle GPU.
- Supporting analysis: an independent CPU simulation of the full executed path
  (19,386 vector, LDS and buffer instructions per wave path; both the
  global-load and LDS counters; all 128 K steps) found no use of a register
  before its load was guaranteed complete, and confirmed the per-wave order
  `8 stores → fence → barrier → 32 reads` on alternating banks. The simulator
  is not included in this package.
- That analysis assumes in-order completion of same-kind requests and that the
  barrier releases only after all eight waves signal. The compiler-generated
  kernel for this configuration relies on the same two properties, so
  bit-exactness against it cannot test them independently.

## Speed

Unprofiled AUTO mode. Each sample is five untimed launches, a synchronize, and
one event pair around seven queued launches. Candidate order rotates and
reverses each round. Gains are medians of within-round ratios; brackets are 95%
bootstrap intervals of that median.

| Run | Double-bank vs eight-chunk | Rounds won | Two-bank step alone (vs early B) |
|---|---:|---:|---:|
| Study's original 31-round command, re-run | +1.51% [+1.18, +1.70] | 30/31 | +0.38% |
| Packaged `benchmark.py`, 31 rounds, HIP launcher | +1.60% | 31/31 | not included |
| 180 s continuous | +1.19% [+1.14, +1.25] | 332/332 | +0.31% |
| 120 s continuous, telemetry thread off | +1.13% [+1.08, +1.17] | 222/222 | +0.21% |
| 180 s with three idle pauses | +1.27% [+1.22, +1.32] | 257/258 | +0.35% |
| 75 s continuous (the telemetry-table run below) | +1.41% [+1.23, +1.52] | 115/138 | +0.32% |

- **The lead shrinks when the card throttles harder.** In the 75 s run the card
  dropped to about 276 W and 2.12 GHz after 40 s. From there the lead was
  +0.5% [+0.1, +1.4] with 42 of 63 rounds won, double-bank time was 8.88 ms
  (123.8 TFLOP/s) and the tutorial 10.62 ms. In the 180 s and 120 s runs the
  card held 292–299 W at 110 °C and the lead stayed at 1.1–1.2%.
- The two-bank step by itself is small. After the first 40 s of the 120 s run
  it was +0.10% with an interval that includes zero.
- Wall-clock time per launch in the batches is 0.04 ms above the event time, so
  the batch event timing is a fair throughput measure.
- Sustained double-bank time is 8.59–8.63 ms (127.5–128.0 TFLOP/s) in the 120 s
  and 180 s runs and 8.88 ms in the harder-throttled part of the 75 s run; the
  short re-run gave 8.43 ms (130.5 TFLOP/s) and the packaged benchmark 8.33 ms
  (132.1 TFLOP/s).
- The unchanged tutorial is 9.97 ms in the short re-run and 10.39–10.62 ms
  sustained. It slows by a further 2% or so once the junction reaches 110 °C;
  the cause was not established.
- `profile_standard` is not a usable fixed-clock control for speed: launches
  take about 43 ms there at 1.58 GHz and the tutorial is the fastest kernel.

Earlier figures in [SUMMARY.md](SUMMARY.md) (for example 8.569 ms for
`optimized_asm`) came from short runs and describe the short-run state.

## What sets the clock

Telemetry from the card's hwmon sensors during a continuous 75 s run:

| Time under load | Power | Junction | Shader clock | Fan |
|---|---:|---:|---:|---:|
| 0–5 s | 299 W (cap 300 W) | 65 °C | 2.30 GHz | 850 rpm |
| 15–20 s | 299 W | 104 °C | 2.24 GHz | 1,010 rpm |
| 40 s onward | 276–277 W | 110 °C (critical limit) | 2.12 GHz | 2,870 of 5,000 rpm |

This table is from the 75 s run. In the 180 s run the 110 °C state held
292–299 W and 2.18–2.20 GHz, so the depth of throttling varies between runs.

During a 30 s idle pause the junction fell to 38 °C; over the first 10 s back
under load it averaged 78 °C and double-bank time was 8.37 ms, against 8.56 ms
just before the pause. Speed therefore follows the power and thermal state, on
this machine with its automatic fan curve.

The two clock levels in the study's 15-pair clock run (2.41–2.48 GHz for the
first six pairs, 2.23–2.30 GHz after) are consistent with a start-of-load
transient above the cap followed by the 300 W power-capped clock; that run was
too short to reach the thermal limit. This was not re-measured.

Consequence for reading utilization: at a fixed power budget a busier kernel
runs at a lower clock. Modeled WMMA service rose from about 72% (tutorial) to
about 91% (double-bank), but speed rose 18–21%.

## Utilization, restated

Utilization was not re-measured in this verification. The study's values were
recomputed from its raw traces and records and are arithmetically correct.

| Kernel | WMMA + ordinary VALU | WMMA only | Scope |
|---|---:|---:|---|
| double_lds_bank_asm | 93.6–94.6% | 90.5–91.4% | Two warmed SQTT captures, one SIMD each |
| double_lds_bank_asm | 94.1–95.1% | 91.0–91.9% | Counters + shader clock, pairs 6–14 at 2.23–2.30 GHz |
| double_lds_bank_asm | 89.0–94.0% | lower | Same diagnostic, pairs 0–5 at 2.41–2.48 GHz |
| eight_chunk_asm | 93.5–93.7% | 90.4–90.5% | Two warmed SQTT captures |
| Unchanged tutorial | 76–77% | about 72% | Historical |

- These are modeled figures (16 cycles per FP16 WMMA, 1 per ordinary VALU), not
  a hardware busy signal. The 16-cycle cadence is supported by an in-shader
  clock calibration (16.15 cycles per WMMA) and a 16-cycle floor in the traces.
- The percentage is fixed work divided by one SIMD's envelope. Capture-to-
  capture variation reached 0.9 points for the double-bank binary (0.13 for
  eight-chunk), as large as the biggest difference between the two kernels, so
  it does not separate them. The paired timing does.
- The 94.60% counter/clock figure in the study report (not part of this
  package) is the median of a 15-pair run that spans both clock levels.

## Where the remaining idle is

A census over the four full warmed traces, recomputed independently from raw
packets:

- Consecutive WMMAs from the same wave execute 16 cycles apart (about 90% of
  pairs exactly 16). When the previous VALU instruction is another wave's WMMA,
  the spacing is 18 in about 96% of pairs and never 16 in 580,126 pairs. With
  ordinary VALU from other waves in between, the minimum is 17 plus their count.
- These short handoff holes are about 2.1% of the envelope in the double-bank
  kernel (33–39% of its modeled idle) and 1.7% in the eight-chunk kernel (about
  27%), stable to 0.04 points between captures.
- The double-bank kernel has shorter same-wave runs (5.0 WMMAs on average
  against 5.6), so it pays more for handoffs; its coverage advantage comes from
  fewer long holes.
- In the double-bank kernel, about 54% of run ends are at an 8-WMMA chunk boundary, 8% at a K-step
  boundary, and about 26% are a wave whose next instruction is a WMMA with no
  pending explicit wait while an older wave is issued instead.

Not established: whether the extra 1–2 cycles are idle time or WMMA service the
16-cycle model does not count, and what arbitration rule picks the next wave.
If it is service, that part of the gap cannot be recovered by scheduling.

## Scope of the method

- About three quarters of the total gain over the tutorial comes from the
  A-direct Gluon layout; the assembly scheduling steps after it add a few
  percent in total.
- Transfer was shown only for M = 4096 with the same K, where the compiled hot
  loop differs from the 8192³ one in two scalar registers. Different K, different
  N and other kernel families are untested, and the two-bank generator is
  specialized to K = 8192.
- All measurements are from one card in one session per run group. No RDNA3 or
  other gfx12 device was tested.

## Stack

Torch 2.12.0+rocm10.0.0; Triton wheel `3.8.0+git4cff872c.rocm10.0.0` (reported
as 3.8.0); Linux 6.8.0. Profiling stack as listed in [SUMMARY.md](SUMMARY.md).
Raw results are in [verification/results](verification/results) and
[evidence/package_validation_20261111.json](evidence/package_validation_20261111.json).
