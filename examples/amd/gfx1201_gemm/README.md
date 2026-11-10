# R9700 / gfx1201 GEMM kernel study

This package contains **Previous best ASM**, **New A-direct Gluon**, and
**Gluon + optimized ASM**, plus the original tutorial reference. The measured
result is **15.76% higher throughput** and **92.30% combined VALU coverage** for
8192³ FP16 GEMM. Read [SUMMARY.md](SUMMARY.md) for the changes, correctness,
coalescing tradeoff, profiling methods, and remaining bottleneck.

- [kernels/](kernels/): exact `.amdgcn` and `.hsaco` artifacts.
- [manifest.json](manifest.json): hashes, target, dimensions, launch ABI, and LDS.
- [source/](source/): frozen tutorial and Gluon Python sources/configurations.
- [evidence/](evidence/): historical paired timing, correctness, and profiling data.
- [benchmark.py](benchmark.py): portable correctness/timing and capture runner.

## Run the exact kernels

Use a ROCm-enabled PyTorch installation and a gfx1201 GPU with enough free memory
for the 8192³ correctness check. The archived kernels require row-major FP16
inputs/output, FP32 accumulation, and the exact fixed shape. No Triton compiler
build or assembly rebuild is needed to run the archived code objects.

```bash
cd examples/amd/gfx1201_gemm
python benchmark.py --verify
python benchmark.py --out /tmp/gfx1201-gemm-check --repeats 7 --batch 7
```

Choose a fresh output directory. Run GPU jobs sequentially in normal AUTO mode
for performance comparisons. The runner checks all versions against the
archived tutorial and an FP32 reference before timing. It uses the current
PyTorch stream, preserves required dynamic LDS allocation, and records the
executable hashes with the result. New timings are independent of the retained
historical study; HIP and Triton launch paths may have different overheads.

For the supplied study environment, first activate its profiling setup and use
its ROCm PyTorch interpreter:

```bash
source /opt/rdna4-profiling-sqttkit/setup-env.sh
/opt/python/bin/python3 benchmark.py --out /tmp/gfx1201-gemm-check-2
```

The package itself has no dependency on the original `/app/att` workspace.
Additional patterns are available through `--pattern uniform`, `zeros`, and
`constant`; constant inputs use 0.125. See `python benchmark.py --help` for
capture and source-validation options.

To validate the relocated high-level source as well, add `--check-gluon-source`
with a compatible Triton installation. This separately checks the new compiled
output against the archived A-direct kernel; it does not substitute the new
binary into the archived timing comparison.

## Profile one kernel

The capture mode runs only the selected archived GEMM after input setup:

```bash
python benchmark.py --capture-only --kernel optimized_asm --launches 6 \
  --out /tmp/gfx1201-gemm-capture
```

Pass that command to the SDK, sqttkit, or ROCm Compute installation linked in
the summary. Both new kernels have the symbol `gemm`; the tutorial and previous
ASM use `matmul_kernel`. For SQTT, capture the sixth dispatch after five warmups,
keep the trace/database/code objects together, and require complete waves with
zero lost packets or unknown PCs. `--kernel` also accepts `previous_best_asm`,
`a_direct_gluon`, and `tutorial_baseline`.

Measure speed separately in AUTO. In the supplied stack, exact instruction
counters required controlled `profile_standard` collection; known-nonzero
AUTO counters returned invalid zeros. Controlled replay durations and wave-wait
ratios must not be treated as production timing or exclusive SIMD stalls.

## Compile the Gluon source

With the experimental AMD Gluon APIs available (tested with Triton 3.8.0):

```python
import json
from pathlib import Path
import torch
from source.gluon.direct_load import compile_kernel

a = torch.randn((8192, 8192), device="cuda", dtype=torch.float16)
b = torch.randn_like(a)
c = torch.empty_like(a)
config = json.loads(Path("source/gluon/config.json").read_text())
kernel, arguments, grid = compile_kernel(a, b, c, config)
kernel[grid](*arguments)
```

This compiles from the relocated source and may produce a new binary hash.
`source/gluon/phase2_initial.py` retains the original experiment's helper code;
the selected configuration activates its direct path with the A operand-layout
override in `direct_load.py`. The benchmark's archived `a_direct_gluon` entry
always refers to the exact binary measured in the summary.

All files follow the repository's [MIT license](../../../LICENSE). The tutorial
source attribution and profiling-stack revisions are recorded in the summary.
