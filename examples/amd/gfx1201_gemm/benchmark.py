"""Validate and compare the four archived gfx1201 GEMM code objects."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from hip_launcher import ROOT, HipKernel, HipRuntime, verify_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repeats", type=int, default=31)
    parser.add_argument("--batch", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--pattern", choices=("normal", "uniform", "zeros", "constant"), default="normal")
    parser.add_argument("--verify", action="store_true", help="Verify artifact hashes without loading HIP")
    parser.add_argument("--capture-only", action="store_true", help="Launch one selected kernel for a profiler")
    parser.add_argument("--kernel", default="optimized_asm")
    parser.add_argument("--launches", type=int, default=6)
    parser.add_argument("--check-gluon-source", action="store_true",
                        help="Also compile the relocated Gluon source and compare its output with the archived kernel")
    args = parser.parse_args()
    manifest = verify_artifacts()
    if args.verify:
        print(json.dumps({"passed": True, "verified_kernels": list(manifest["kernels"])}))
        return
    if not args.out:
        parser.error("--out is required for GPU execution")
    if min(args.repeats, args.batch, args.launches) < 1:
        parser.error("repeats, batch and launches must be positive")
    if args.kernel not in manifest["kernels"]:
        parser.error("--kernel must name an entry in manifest.json")
    if args.check_gluon_source and args.capture_only:
        parser.error("--check-gluon-source is a benchmark validation option")
    if args.out.exists():
        # Compute replays the application. Keep each replay in a separate directory.
        if args.capture_only:
            import os
            args.out = args.out / f"replay-{os.getpid()}"
        else:
            parser.error("Use a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=False)

    import torch

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    m = n = k = 8192

    def tensor(shape):
        if args.pattern == "normal":
            return torch.randn(shape, device="cuda", dtype=torch.float16)
        if args.pattern == "uniform":
            return torch.rand(shape, device="cuda", dtype=torch.float16) * 2 - 1
        return torch.full(shape, 0 if args.pattern == "zeros" else .125, device="cuda", dtype=torch.float16)

    a, b = tensor((m, k)), tensor((k, n))
    backing = torch.full((m * n + 512,), -123., device="cuda", dtype=torch.float16)
    c = backing[256:-256].view(m, n)
    metadata = dict(shape_mnk=[m, n, k], dtype="FP16 inputs/output; FP32 accumulation",
                    seed=args.seed, pattern=args.pattern, torch=torch.__version__,
                    device=str(torch.cuda.get_device_properties(0)), dense_flops=2 * m * n * k,
                    launch_method="HIP module launch of hash-checked archived code objects")
    runtime = HipRuntime()
    metadata["hip_runtime_library"] = runtime.library_path
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    names = [args.kernel] if args.capture_only else list(manifest["benchmark_order"])
    kernels = {}
    try:
        for name in names:
            kernels[name] = HipKernel(name, runtime, manifest)
        functions = {name: kernel.bind(a, b, c) for name, kernel in kernels.items()}
        metadata["hsaco_sha256"] = {name: manifest["kernels"][name]["artifacts"]["hsaco"]["sha256"]
                                      for name in names}
        (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if args.capture_only:
            for _ in range(args.launches):
                functions[args.kernel]()
            torch.cuda.synchronize()
            result = dict(metadata, mode="capture", kernel=args.kernel, launches=args.launches)
        else:
            reference = torch.matmul(a.float(), b.float())
            saved_a, saved_b = a.clone(), b.clone()
            validation = {}
            tutorial_output = None
            gluon_output = None
            for name in names:
                c.fill_(float("nan"))
                functions[name]()
                torch.cuda.synchronize()
                error = (c.float() - reference).abs()
                tolerance = .015625 + .001 * reference.abs()
                check = dict(finite=bool(torch.isfinite(c).all()), max_abs_error=float(error.max()),
                             outside_tolerance=int((error > tolerance).sum()),
                             tolerance="abs_error <= 0.015625 + 0.001 * abs(FP32 reference)",
                             guards_intact=bool((backing[:256] == -123).all() and (backing[-256:] == -123).all()),
                             inputs_unchanged=bool(torch.equal(a.view(torch.int16), saved_a.view(torch.int16))
                                                   and torch.equal(b.view(torch.int16), saved_b.view(torch.int16))))
                if name == "tutorial_baseline":
                    tutorial_output = c.clone()
                else:
                    check["changed_elements_vs_tutorial"] = int((c.view(torch.int16) != tutorial_output.view(torch.int16)).sum())
                    check["bitwise_equal_to_tutorial"] = check["changed_elements_vs_tutorial"] == 0
                check["passed"] = (check["finite"] and check["outside_tolerance"] == 0 and check["guards_intact"]
                                   and check["inputs_unchanged"] and check.get("bitwise_equal_to_tutorial", True))
                validation[name] = check
                (args.out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
                if not check["passed"]:
                    raise RuntimeError(f"{name} failed validation: {check}")
                if name == "a_direct_gluon":
                    gluon_output = c.clone()

            source_validation = None
            if args.check_gluon_source:
                from source.gluon.direct_load import compile_kernel

                config = json.loads((ROOT / "source/gluon/config.json").read_text())
                compiled, arguments, grid = compile_kernel(a, b, c, config)
                c.fill_(float("nan"))
                compiled[grid](*arguments)
                torch.cuda.synchronize()
                changed = int((c.view(torch.int16) != gluon_output.view(torch.int16)).sum())
                source_validation = dict(
                    changed_elements_vs_archived_gluon=changed,
                    inputs_unchanged=bool(torch.equal(a.view(torch.int16), saved_a.view(torch.int16))
                                          and torch.equal(b.view(torch.int16), saved_b.view(torch.int16))),
                    guards_intact=bool((backing[:256] == -123).all() and (backing[-256:] == -123).all()),
                    hsaco_sha256=hashlib.sha256(compiled.asm["hsaco"]).hexdigest(),
                    amdgcn_sha256=hashlib.sha256(compiled.asm["amdgcn"].encode()).hexdigest(),
                    note="Recompiled source is checked separately; archived code objects own the timing comparison.")
                source_validation["passed"] = (changed == 0 and source_validation["inputs_unchanged"]
                                                 and source_validation["guards_intact"])
                (args.out / "source-validation.json").write_text(json.dumps(source_validation, indent=2) + "\n")
                if not source_validation["passed"]:
                    raise RuntimeError(f"Relocated Gluon source failed validation: {source_validation}")

            def timed(fn):
                for _ in range(5):
                    fn()
                torch.cuda.synchronize()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(args.batch):
                    fn()
                end.record()
                end.synchronize()
                return begin.elapsed_time(end) / args.batch

            pairs = []
            for repeat in range(args.repeats):
                order = names[repeat % len(names):] + names[:repeat % len(names)]
                if (repeat // len(names)) % 2:
                    order = order[::-1]
                row = dict(round=repeat, order=order)
                row.update({name: timed(functions[name]) for name in order})
                pairs.append(row)
            medians = {name: statistics.median(row[name] for row in pairs) for name in names}
            result = dict(metadata, mode="benchmark", repeats=args.repeats, launches_per_sample=args.batch,
                          validation=validation, source_validation=source_validation, pairs=pairs, median_ms=medians,
                          tflops={name: 2 * m * n * k / (time * 1e9) for name, time in medians.items()},
                          paired_speedup_vs_tutorial={name: statistics.median(row["tutorial_baseline"] / row[name]
                                                                           for row in pairs) for name in names},
                          optimized_paired_speedup={name: statistics.median(row[name] / row["optimized_asm"]
                                                                           for row in pairs) for name in names})
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, indent=2))
    finally:
        torch.cuda.synchronize()
        for kernel in kernels.values():
            kernel.close()


if __name__ == "__main__":
    main()
