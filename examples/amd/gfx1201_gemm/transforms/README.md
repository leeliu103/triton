# Assembly transform scripts (study provenance)

These are the scripts that produced `kernels/eight_chunk_asm` and
`kernels/double_lds_bank_asm` from the compiled A-direct Gluon assembly. They
are copied unchanged from the study workspace and flattened into one directory.

**They are not relocated.** They contain absolute paths into the original
workspace and expect its directory layout, so they document the method and the
exact edits rather than run from this package. The archived kernels in
`kernels/` are the measured artifacts; `benchmark.py` runs those directly.

| File | Role |
|---|---|
| `consumer_waits.py` | Wait engine: derives each `s_wait_*` threshold from the registers the next instruction reads or writes, given ordered request queues and explicit fences |
| `prepare_ready_waits.py` | Applies the wait engine to the eight-chunk kernel (consumer-ready LDS waits) |
| `prepare_early_b.py` | Moves the next-B global loads to just after LDS publication, with adjusted A waits and a final-iteration drain |
| `prepare_double_b_bank.py` | Unrolls two K steps, alternates two LDS banks and removes the separate reuse barrier |
| `path_audit.py` | Replays the global-load queue along the executed path and checks every consumer |
| `prepare_chunk_sizes.py`, `prepare_residency.py`, `prepare_clustered.py` | Earlier steps that produced the eight-chunk baseline |
| `prepare_direct_asm.py`, `prepare_direct_prefetch.py`, `prepare_aonly_aprefetch.py`, `audit_pending_aonly.py` | Helpers imported by the above |
| `asm_loader.py` | Assembles an edited `.amdgcn`, checks its launch metadata against the compiled kernel and binds it through the `.transform.json` sidecar |
| `ASM_NOTES.md` | The study's notes on load motion, branch handling, register lifetimes and the two-bank argument. Verbatim: its paths are relative to the study's `saturation/asm/` directory; `lds_lookahead.py`, `prepare_shape_transfer.py` and the `*/entries.json` files it names are not included, and its baseline `clusters8_control_lds16` is `../kernels/eight_chunk_asm` here |

Known limits, found in review:

- The two-bank generator is specialized to K = 8192 (128 K64 steps, an even
  step count, literal loop bound and register numbers). Other K values stop at
  generation; they are not supported.
- `consumer_waits.py` handles one straight-line path. Joins and entry state are
  the caller's responsibility and are not checked by the engine.
- The saved path audit covers the global-load counter only. The LDS counter was
  audited separately with an independent simulator (0 hazards over the full
  executed path); see [VERIFICATION.md](../VERIFICATION.md).
- Partial wait thresholds rely on in-order completion of same-kind requests
  (the kernel descriptor sets `.amdhsa_memory_ordered 1`) and on the barrier
  releasing only after all eight waves signal. Compiler-generated code for the
  same kernel relies on both.
