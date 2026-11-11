# Verification harness (2026-11-11)

Scripts and raw results behind [VERIFICATION.md](../VERIFICATION.md). Like
`transforms/`, the scripts are copied unchanged from the study machine and use
its absolute paths and Triton-based loader; they are provenance, not a portable
runner. `benchmark.py` in the package root is the portable check.

| File | Purpose |
|---|---|
| `soak.py` | Bit-compares every launch against standard Triton, in an interleaved pattern and a back-to-back pattern over 16 output buffers |
| `make_controls.py`, `controls/` | Three deliberately broken copies of the double-bank kernel: single LDS bank, no barrier, no store fence. **Not candidates; they produce wrong results by design** |
| `steady.py` | Long paired timing run with per-sample timestamps, wall-clock cross-check, sysfs telemetry and optional idle pauses |
| `analyze_steady.py` | Paired gains by time window and telemetry summary |
| `results/` | Raw `result.json` of each run (the GPU device UUID was removed from `repro-original.json`) |

The packaged `steady.py` is the final revision, which adds fan telemetry
columns; only `steady-auto-fan.json` was produced by it. The four earlier steady
runs used the previous revision (script hash `37a0cc97…` in their results).

`soak.py` always exits 0; read pass or fail from `failing_launches` in its
result file.
