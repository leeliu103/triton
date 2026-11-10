# How we measured 92.30% combined VALU utilization

For the optimized 8192³ FP16 GEMM on the Radeon AI PRO R9700, **92.30% is the
primary result from sqttkit's VALU-unit timeline**. A separate calculation using
hardware instruction counters and shader timestamps gives **92.46%**. We use
the second result to cross-check the first.

Both estimate how much of the available shader execution capacity is occupied
by WMMA and ordinary vector instructions. Neither reads a physical, exclusive
“VALU busy” bit for every cycle. The distinction matters when interpreting the
remaining space below 100%.

The exact measured kernel is [optimized_asm](kernels/optimized_asm/kernel.amdgcn),
whose HSACO SHA-256 is:

```text
37b7f40ab00f1fc1a7da6864dd1b14288f3a300c6ea08fdba8092326864bfcd0
```

[manifest.json](manifest.json) identifies the executable and launch contract.
[evidence/profiling.json](evidence/profiling.json) contains the retained
aggregate measurements, and [SUMMARY.md](SUMMARY.md) describes the optimization.

## The intuition: execution slots and work to fill them

Imagine one SIMD's modeled VALU service capacity as a row of slots, one slot per
shader cycle. For the instruction mix in this kernel:

- A native `v_wmma_f32_16x16x16_f16` occupies **16 modeled service cycles**.
- Each ordinary VALU instruction in this particular binary occupies **one
  modeled service cycle**. This is not a universal rule for all VALU opcodes.
- Time with no WMMA or ordinary VALU interval is uncovered time in this model.

Here is an illustrative 20-cycle window, not an excerpt from the measurement:

```text
Cycle:             01234567890123456789
WMMA service:      WWWWWWWWWWWWWWWW....   16 cycles
Ordinary VALU:     ................vv..    2 cycles
Combined coverage: ##################..   18 / 20 = 90%
```

WMMA-only coverage is 80%; combined coverage is 90%. They answer different
questions. Ordinary vector work, including address calculations and data
packing, occupies capacity even though it does not itself perform matrix FLOPs.

A **service width is not result latency**. A 16-cycle modeled WMMA span does not
mean every dependent instruction can consume its result after exactly 16 cycles.
The model describes execution-resource service; dependencies and waiting are
separate. Likewise, an encoded VOPD instruction can contain two operations while
occupying one modeled cycle. We do not count that as two cycles of coverage.

Multiple waves can be resident on a SIMD. If one wave waits for memory while
another performs WMMA, that time remains covered. Adding all waves' wait times
and subtracting them from elapsed time would therefore give the wrong answer.
Residency, active-thread fraction, and this service-utilization percentage are
different metrics.

The **VALU unit lane** in the SQTT plot is a timeline row for an execution
source, not one of the 32 data lanes inside a SIMD. Our percentage measures
coverage over time, not the fraction of threads or data lanes that are active.

## Approach 1: measure coverage on the SQTT VALU-unit timeline

The intuitive question is: **Over the complete recorded interval, how much time
is covered by a WMMA or VALU span?**

We captured the sixth GEMM dispatch in AUTO mode, after five untraced warmup
dispatches, using the supplied fixed ROCm profiler SDK. The SQTT configuration
selected shader engine 0, target CU 0, SIMD 0. The captured code object was
checked against the hash above.

The relevant sqttkit implementation is the EXEC-to-instruction matching and
unit model in `units.py`. It reconstructs
instruction and execution events from raw SQTT packets. We use known instruction
PCs from the exact code object to identify WMMA and other instruction families.
Execution packets anchor the intervals; opcode-specific model widths determine
their lengths. For the end-stamped WMMA and ordinary VALU classes here, a model
span is `[EXEC timestamp - width, EXEC timestamp)`. It is not the interval from
instruction issue until a later instruction finishes waiting.

We merge the intervals from all detailed waves on the selected SIMD. The
numerator is their **union**: a cycle covered by multiple included intervals is
counted once. The denominator is the selected SIMD's complete recorded envelope,
from its earliest recorded wave start to its latest recorded wave end. We do
not crop away startup, drain, or gaps to improve the percentage.

```text
SQTT coverage = 100 × length(union of included unit intervals)
                     / length(full selected-SIMD envelope)
```

Both lengths are in shader cycles, so this ratio needs no GHz conversion.

The final capture gives:

| Quantity | Shader cycles | Full-envelope percentage |
|---|---:|---:|
| Full selected-SIMD envelope | 18,801,031 | 100% |
| WMMA union | 16,777,216 | 89.235617% |
| Other VALU union, including the tiny scalar-TRANS contribution | 576,650 | 3.067119% |
| Combined VALU union | 17,353,866 | **92.302736%** |
| Uncovered time | 1,447,165 | 7.697264% |

There is no modeled WMMA/other-VALU intersection in this capture, so those two
union lengths add here. We still compute a union rather than assuming that
property for every kernel.

```text
17,353,866 / 18,801,031 × 100 = 92.302735951...%
```

The WMMA count also closes exactly: 128 detailed waves execute 8,192 WMMAs each,
giving 1,048,576 instructions. Multiplying by 16 service cycles gives 16,777,216.
Counting WMMA instructions as a percentage of all instruction counts would not
measure utilization: it would ignore both the different widths and idle time.

For exact accounting, the published `combined_VALU` lane includes 128 one-cycle
scalar reciprocal intervals, one `v_s_rcp_f32` per detailed wave. Of these, 118
cycles overlap ordinary VALU intervals. Thus:

```text
WMMA width sum                           16,777,216
Ordinary VALU width sum                      576,640
Scalar reciprocal width sum                     128
Inclusive width sum                      17,353,984
Subtract overlap                                118
Inclusive interval union                 17,353,866

Strict WMMA + ordinary VALU union         17,353,856
Strict main-only coverage                 92.302682762...%
```

Both inclusive and main-only values round to **92.30%**. The counter method below
uses the main-only definition, excluding that scalar reciprocal. This tiny
definition difference is recorded rather than mistaken for a measurement error.
For a kernel with substantial transcendental work, the overlap and subtraction
would need a new audit; the same shortcut would not automatically apply.

Trace validation matters as much as the division. This capture consumed
12,152,549 packets, contained 4,096 complete wave records with 128 detailed waves,
and associated 3,258,112 instruction occurrences with known PCs. It had zero
packet-loss markers, unknown instruction PCs, incomplete waves, or context
save/restore markers. The optional local association overlay used elsewhere in
the study made **zero resolutions for this kernel**.

The timeline's zoomed display is not the measurement denominator. For example,
the retained representative window has 95.00% coverage, while the full envelope
has 92.30%. We report the full-envelope value. Similarly, restricting to periods
with six resident waves gives 93.15%; that is a conditional statistic, not the
headline result.

## Approach 2: divide required service by available capacity

The intuitive question is: **How many SIMD service cycles does this instruction
stream require, and how many SIMD cycles were available while it ran?**

If 128 SIMDs each offer one modeled slot per shader cycle, their total supply is
128 times the shader-cycle duration. Dividing the work by that supply estimates
the fraction used. This method does not reconstruct a colored timeline.

### Count the required work

We collected ordinary PMC counters through the fixed SDK in controlled
`profile_standard` mode, summed the relevant hardware instances, checked for
saturation, and validated the wave and WMMA counts. Controlled mode was needed
because AUTO SQ instruction/wait/cache counters returned impossible zeros for
known nonzero work in this environment.

Only deterministic instruction work transfers from this controlled capture to
the AUTO run. The controlled timing and waiting behavior do **not** transfer.
We require the same executable hash, input shape, launch configuration, and
validated execution path.

| Counter / quantity | Exact whole-GPU count |
|---|---:|
| `SQ_WAVES` | 16,384 |
| `SQ_INSTS_VEC32_VALU_WMMA` | 134,217,728 |
| `SQ_INSTS_VALU`, inclusive encoded instructions | 208,044,032 |
| `SQ_INST_CYCLES_VALU`, weighted service demand | 2,221,309,952 |

`SQ_INST_CYCLES_VALU` is **work**, not elapsed exclusive busy time. Its sum can
count work from execution paths that overlap. The instruction mix and SQTT
opcode/width audit establish how to use it for this kernel.

The whole-GPU numerator is:

```text
WMMA service = 134,217,728 × 16
             = 2,147,483,648 SIMD cycles

Ordinary VALU count = 208,044,032 - 134,217,728 - 16,384
                    = 73,809,920
Ordinary service    = 73,809,920 × 1 SIMD cycle

Main service demand D = 2,147,483,648 + 73,809,920
                      = 2,221,293,568 SIMD cycles

Counter cross-check: 2,221,309,952 - 16,384 = 2,221,293,568
```

The subtraction removes one one-cycle scalar reciprocal per wave. There is no
substantial vector TRANS workload here. The ordinary one-cycle widths are
specific to this verified stream; other kernels may contain multi-cycle integer
or transcendental instructions. The optimized ASM and generated A-direct Gluon
have the same exact vector work, so the gain is not extra instructions merely
making the unit look busier.

### Measure the available shader-cycle capacity

In a separate AUTO run, a diagnostic version records shader-cycle and REALTIME
timestamps around the numerical kernel, together with SIMD identity. It keeps
the numerical instruction text and adds boundary instrumentation and SGPR
state, without increasing VGPR or LDS allocation. Each production launch is
paired with a diagnostic launch. There are five warmup pairs and 15 timed pairs.
Every timed diagnostic retains all 16,384 wave records: **245,760 checked records**
over 128 SIMDs.

Shader-cycle counts tell us how many local clock ticks passed. REALTIME ticks
provide a common time reference. We compare shader timestamps **within the same
SIMD**, avoiding an assumption that raw shader timestamp origins across SIMDs
are synchronized.

For SIMD `j`, combine its wave records into an envelope:

```text
C[j] = latest shader end - earliest shader start
R[j] = latest REALTIME end - earliest REALTIME start

effective shader frequency f = (sum(C[j]) / sum(R[j])) × Fref
global time T = (latest REALTIME end across all SIMDs
                - earliest REALTIME start across all SIMDs) / Fref

available capacity S = 128 × f × T
combined utilization = 100 × D / S
```

`Fref` is the previously calibrated 100,000,000 REALTIME ticks/s. The frequency
is a measured, envelope-weighted effective frequency for that dispatch, not
the advertised boost clock or the clock of a separate peak-throughput test.
Using the full global interval includes supply on SIMDs that start later or
finish earlier than the first/last SIMD.

For the dimensionless utilization ratio, the REALTIME conversion factor
actually cancels:

```text
S = 128 × sum(C[j]) × global_REALTIME_span_ticks / sum(R[j])
```

The reference frequency determines displayed GHz and milliseconds, but it does
not change this ratio when the same REALTIME scale is used consistently. That
does not remove the need for valid timestamps and a correctly reconstructed
interval.

### A worked example from the actual run

The median utilization happens to be **pair index 5** among the 15 timed pairs.
These are values from that single diagnostic launch:

| Quantity | Value |
|---|---:|
| Sum of same-SIMD shader envelopes | 2,400,943,491 cycles |
| Sum of same-SIMD REALTIME envelopes | 99,175,852 ticks |
| Global REALTIME span | 775,288 ticks |
| Effective shader frequency | 2.420895250791493 GHz |
| Global recorded time | 7.75288 ms |
| Full 128-SIMD supply | 2,402,420,527.610413 SIMD cycles |
| Required main service | 2,221,293,568 SIMD cycles |

```text
f = 2,400,943,491 / 99,175,852 × 100,000,000
  = 2,420,895,250.791493 cycles/s

T = 775,288 / 100,000,000
  = 0.00775288 s

S = 128 × f × T
  = 2,402,420,527.610413 SIMD cycles

100 × D / S = 92.460647187752...%
```

We calculate a utilization ratio for **each pair**, then report the median:
**92.46%**. We do not divide median work by a product of separately chosen median
clock and median time. For example, the run's median frequency is 2.288220 GHz;
it belongs to a different point in the sample distribution than the worked
example above. Substituting it into this example would be incorrect.

There is also a narrower, explicitly labeled metric:

```text
same-SIMD-envelope utilization = 100 × D / sum(C[j])
```

Its median is **92.5175%**, slightly higher because it excludes the outer gaps
before individual SIMDs start and after they finish. The main cross-check uses
the full global interval, **92.4606%**.

## Why the results agree, and what the agreement establishes

| Comparison | SQTT timeline | Counter + direct timing |
|---|---:|---:|
| Combined, as usually quoted | 92.302736% | 92.460647% |
| Strict WMMA + ordinary VALU | 92.302683% | 92.460647% |
| WMMA only | 89.235617% | 89.388332% |
| Timing scope | One selected SIMD, full trace envelope | Instrumented global interval across 128 SIMDs |

The quoted combined results differ by **0.1579 percentage points**, approximately
0.16 points. That is close agreement despite different runs and scopes. It is
not a statistical confidence interval or a guarantee of 0.16-point accuracy.

They use different observations: SQTT packet timing and interval reconstruction
versus hardware work counters plus timestamp records. But the interpretations
share assumptions about instruction service widths and the relevant capacity.
Agreement checks consistency and supports saying **“about 92% combined service
utilization”**; it does not independently prove physical busy time.

SQTT samples one SIMD and uses a software interval model. The counter method
instruments the kernel and uses an effective frequency over recorded envelopes.
The added boundary work, per-run AUTO clock variation, SIMD sampling, and interval
definitions can all contribute to differences. Neither percentage is an exact
uninstrumented whole-GPU busy-mask measurement.

The numerator/denominator units provide a useful check. SQTT divides shader
cycles by shader cycles. The formula divides **summed SIMD cycles** of work by
**summed SIMD cycles** of supply. Omitting the factor of 128, mixing milliseconds
with seconds, using raw instruction counts without weights, or using summed
wave lifetimes as capacity would produce the wrong quantity.

## What 100%, the matrix roof, and the speedup mean

**100% combined coverage** would mean no uncovered time in the chosen modeled
interval. It would not mean every instruction performs matrix arithmetic. In
this kernel, the matrix share of required main service is:

```text
2,147,483,648 / 2,221,293,568 = 96.677165...%
```

Consequently, even perfect combined service at the same instruction inventory
would have an optimistic WMMA-only ceiling of about **96.68%**. The measured
WMMA-only value, **89.24%**, is consistent with about 92.30% combined coverage
and the necessary ordinary vector work. That ceiling ignores other scheduling
and memory constraints; it is not a promise that 96.68% is achievable.

For native FP16 WMMA, 8192 FLOPs / 16 service cycles = 512 FLOPs per SIMD cycle.
Across 128 SIMDs, the pure matrix roof is **65.536 × shader GHz TFLOP/s**. A
separate pure-WMMA calibration approached 99% of its service capacity, but its
higher operating frequency must not replace the GEMM's measured frequency.

The **15.76% throughput improvement** is a separate measurement. It comes from
31 interleaved, unprofiled AUTO timing rounds of production kernels:

```text
speedup (%) = 100 × (median(T_tutorial / T_optimized) - 1)
            = 15.7610%
```

Those are the historical timings in
[evidence/final_benchmark.json](evidence/final_benchmark.json). A HIP-event
interval, a profiler callback interval, and a direct shader envelope can include
different boundary time. We do not replace the directly measured envelope with
a benchmark median or subtract a universal overhead to force the percentages
to match. ROCm Compute's controlled replay timings and per-wave waits are also
not used as AUTO performance or exclusive SIMD idle percentages.

The uncovered **7.70%** in the final SQTT envelope identifies remaining modeled
gaps, mainly around B-load tails and LDS publication/read synchronization. It
does not prove that eliminating those gaps would give an equal wall-time gain:
the instruction mix, operating clock, and timing boundary can change too.

## Recalculate the central numbers without a GPU

This example uses the actual SQTT totals and the actual matched diagnostic pair
printed above. Run it from this directory to compare with the committed aggregate
results as well:

```python
import json
from pathlib import Path

recorded = json.loads(Path("evidence/profiling.json").read_text())

sqtt = 100 * 17_353_866 / 18_801_031
sqtt_main = 100 * 17_353_856 / 18_801_031
work = 2_221_309_952 - 16_384
supply = 128 * 2_400_943_491 * 775_288 / 99_175_852
counter_clock = 100 * work / supply
reported = recorded["counter_and_clock_crosscheck"]["optimized_asm"]["summary"]

assert abs(sqtt - recorded["sqtt"]["optimized_asm"]["combined_valu_percent"]) < 1e-9
assert abs(counter_clock / 100 - reported[
    "main_combined_fraction_global_diagnostic_interval"]["median"]) < 1e-12
print(f"SQTT: {sqtt:.6f}% (main only: {sqtt_main:.6f}%)")
print(f"Counter + clock: {counter_clock:.6f}%")
print(f"Difference: {counter_clock - sqtt:.6f} percentage points")
```

The tooling provenance is the
[fixed SDK/ROCm Compute revision](https://github.com/leeliu103/rocm-systems/tree/0f8e0287cc4b3b3e252757da2ae4e30aa4dc7e5c)
and [tested sqttkit base](https://github.com/leeliu103/sqttkit/tree/ca920c1206bc82d8c7ec29eb62d6393e5160f4e7)
linked in the summary. The compact package retains aggregate profiling evidence,
not the large raw SQTT databases or all timestamp snapshots. The worked example
therefore makes the arithmetic reproducible; reproducing the measurements still
requires the profiling stack and the corresponding trace/counter/diagnostic
capture workflow.
