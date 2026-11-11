"""Load a separately assembled gfx1201 kernel with a Gluon kernel's launch ABI.

No global Triton override or compiler-cache entry is changed. Construction and
assembly are CPU-only; the returned CompiledKernel initializes HIP handles on
its first launch. Pass every original positional argument, including constexpr
arguments, to ``variant[(grid_x, 1, 1)](*args)``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import yaml


LLVM_DIR = Path("/opt/python/lib/python3.14/site-packages/_rocm_sdk_devel/lib/llvm/bin")
_RESOURCE_DESCRIPTOR_FIELDS = {
    "group_segment_fixed_size",
    "private_segment_fixed_size",
    "next_free_vgpr",
    "next_free_sgpr",
}
_RESOURCE_METADATA_FIELDS = {
    ".group_segment_fixed_size",
    ".private_segment_fixed_size",
    ".vgpr_count",
    ".sgpr_count",
    ".vgpr_spill_count",
    ".sgpr_spill_count",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, default=repr) + "\n").encode()


def _write_once(path: Path, data: bytes) -> None:
    """Allow repeat use of identical artifacts without overwriting evidence."""
    if path.exists():
        if path.read_bytes() != data:
            raise FileExistsError(f"Refusing to replace a different artifact: {path}")
        return
    path.write_bytes(data)


def _metadata_json(compiled) -> dict:
    paths = [Path(p) for k, p in compiled.metadata_group.items() if k.endswith(".json")]
    if len(paths) != 1:
        raise ValueError(f"Expected one Triton metadata JSON, found {paths}")
    return json.loads(paths[0].read_text())


def _verify_specialization(compiled, asm_path: Path, assembly: str) -> dict:
    """Bind edited ISA to the exact compiled specialization it came from.

    The launch ABI alone cannot distinguish different S/H/dtype/LSE constants:
    those values are baked into instructions. Transform producers retain both
    hashes in a sibling ``.amdgcn.transform.json`` file. Identical assembly is
    its own proof and does not need a transform sidecar.
    """
    baseline_hash = _sha256(compiled.asm["amdgcn"].encode())
    variant_hash = _sha256(assembly.encode())
    sidecar_path = asm_path.with_suffix(asm_path.suffix + ".transform.json")
    proof = {
        "baseline_amdgcn_sha256": baseline_hash,
        "variant_amdgcn_sha256": variant_hash,
    }
    if sidecar_path.exists():
        transform = json.loads(sidecar_path.read_text())
        if transform.get("input_sha256") != baseline_hash:
            raise ValueError(
                "Assembly specialization mismatch: transform input_sha256 does not "
                "match the currently compiled Gluon assembly. Regenerate the variant "
                "for this S/H/dtype/LSE specialization."
            )
        if transform.get("output_sha256") != variant_hash:
            raise ValueError("Variant assembly changed after its transform sidecar was written")
        proof.update({"method": "transform_sidecar", "sidecar": str(sidecar_path), "transform": transform})
    elif variant_hash == baseline_hash:
        proof["method"] = "identical_assembly"
    else:
        raise ValueError(
            f"Edited assembly requires {sidecar_path.name} with input_sha256 and "
            "output_sha256 to bind it to its original compiled specialization"
        )
    return proof


def _asm_metadata(assembly: str) -> dict:
    match = re.search(
        r"^\s*\.amdgpu_metadata\s*$\n(.*?)^\s*\.end_amdgpu_metadata\s*$",
        assembly,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError("Full assembly must retain its .amdgpu_metadata block")
    return yaml.safe_load(match.group(1))


def _kernel_metadata(metadata: dict, name: str) -> dict:
    kernels = metadata.get("amdhsa.kernels", [])
    if len(kernels) != 1 or kernels[0].get(".name") != name:
        raise ValueError(f"Expected exactly the baseline kernel {name!r}")
    return kernels[0]


def _descriptor(assembly: str, name: str) -> dict[str, str]:
    match = re.search(
        rf"^\s*\.amdhsa_kernel\s+{re.escape(name)}\s*$\n(.*?)^\s*\.end_amdhsa_kernel\s*$",
        assembly,
        re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"Missing .amdhsa_kernel descriptor for {name}")
    result = {}
    for key, value in re.findall(r"^\s*\.amdhsa_(\w+)\s+([^\n]+)", match.group(1), re.MULTILINE):
        if key in result:
            raise ValueError(f"Duplicate descriptor field {key}")
        result[key] = value.split(";", 1)[0].strip()
    return result


def _maximum_explicit_registers(assembly: str) -> dict[str, int]:
    """Check numbered registers in instructions, excluding metadata/comments."""
    maximum = {"v": -1, "s": -1}
    for line in assembly.splitlines():
        code = line.split(";", 1)[0].split("//", 1)[0].strip()
        if not re.match(r"[a-z][a-z0-9_]*\s+", code):
            continue
        for kind, end in re.findall(r"\b([vs])\[(?:\d+:)?(\d+)\]", code):
            maximum[kind] = max(maximum[kind], int(end))
        for kind, number in re.findall(r"\b([vs])(\d+)\b", code):
            maximum[kind] = max(maximum[kind], int(number))
    return {kind: number + 1 for kind, number in maximum.items()}


def _validate_assembly(baseline_asm: str, variant_asm: str, metadata: dict) -> tuple[dict, dict]:
    name = metadata["name"]
    original = _asm_metadata(baseline_asm)
    edited = _asm_metadata(variant_asm)
    if {k: v for k, v in original.items() if k != "amdhsa.kernels"} != {
        k: v for k, v in edited.items() if k != "amdhsa.kernels"
    }:
        raise ValueError("Assembly changed the HSA target or metadata version")
    old_kernel = _kernel_metadata(original, name)
    new_kernel = _kernel_metadata(edited, name)
    for key in old_kernel.keys() | new_kernel.keys():
        if key not in _RESOURCE_METADATA_FIELDS and old_kernel.get(key) != new_kernel.get(key):
            raise ValueError(f"Assembly changed ABI/launch metadata {key}")
    old_desc = _descriptor(baseline_asm, name)
    new_desc = _descriptor(variant_asm, name)
    for key in old_desc.keys() | new_desc.keys():
        if key not in _RESOURCE_DESCRIPTOR_FIELDS and old_desc.get(key) != new_desc.get(key):
            raise ValueError(f"Assembly changed launch or numerical descriptor {key}")
    for desc_key, meta_key in (
        ("group_segment_fixed_size", ".group_segment_fixed_size"),
        ("private_segment_fixed_size", ".private_segment_fixed_size"),
        ("kernarg_size", ".kernarg_segment_size"),
        ("next_free_vgpr", ".vgpr_count"),
    ):
        if int(new_desc[desc_key], 0) != new_kernel[meta_key]:
            raise ValueError(f"Descriptor {desc_key} disagrees with HSA {meta_key}")
    used = _maximum_explicit_registers(variant_asm)
    for kind, key in (("v", "next_free_vgpr"), ("s", "next_free_sgpr")):
        if used[kind] > int(new_desc[key], 0):
            raise ValueError(f"Instructions use {used[kind]} {kind.upper()}GPRs but descriptor {key} is {new_desc[key]}")
    if new_kernel[".wavefront_size"] != metadata["warp_size"]:
        raise ValueError("HSA wavefront size disagrees with Triton launch metadata")
    if int(new_desc["wavefront_size32"], 0) != int(metadata["warp_size"] == 32):
        raise ValueError("Descriptor wavefront size disagrees with launch metadata")
    return edited, new_desc


def _binary_metadata(path: Path) -> dict:
    tool = LLVM_DIR / "llvm-readelf"
    tool_path = str(tool) if tool.exists() else shutil.which("llvm-readelf")
    if not tool_path:
        raise FileNotFoundError("llvm-readelf is required to validate assembled HSA metadata")
    output = subprocess.check_output([tool_path, "--notes", str(path)], text=True)
    begin = output.find("---")
    if begin < 0:
        raise ValueError(f"No HSA metadata note in {path}")
    return yaml.safe_load(output[begin:])


def export_baseline(compiled, directory: str | Path) -> dict:
    """Save all generated IR, complete assembly, binary, and launch metadata."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    metadata = _metadata_json(compiled)
    if metadata["target"]["backend"] != "hip" or metadata["target"]["arch"] != "gfx1201":
        raise ValueError("This study loader supports only HIP gfx1201 kernels")
    if not {"amdgcn", "hsaco"}.issubset(compiled.asm):
        raise ValueError("Baseline must retain full AMDGCN and HSACO artifacts")
    files = {}
    for ext, value in compiled.asm.items():
        data = value if isinstance(value, bytes) else value.encode()
        path = directory / f"kernel.{ext}"
        _write_once(path, data)
        files[ext] = {"path": str(path), "sha256": _sha256(data), "bytes": len(data)}
    _write_once(directory / "kernel.json", _json(metadata))
    code_metadata = _asm_metadata(compiled.asm["amdgcn"])
    if _binary_metadata(directory / "kernel.hsaco") != code_metadata:
        raise ValueError("Baseline binary metadata differs from exported assembly metadata")
    _write_once(directory / "code_metadata.json", _json(code_metadata))
    manifest = {
        "kind": "gluon_baseline",
        "kernel_name": compiled.name,
        "triton_hash": compiled.hash,
        "triton_metadata": str(directory / "kernel.json"),
        "signature": {repr(k): v for k, v in compiled.src.signature.items()},
        "constexprs": {repr(k): repr(v) for k, v in getattr(compiled.src, "constants", {}).items()},
        "files": files,
        "resources": _kernel_metadata(code_metadata, compiled.name),
        "dynamic_shared_bytes": metadata["shared"],
    }
    _write_once(directory / "manifest.json", _json(manifest))
    return manifest


def load_variant(
    compiled,
    asm_path: str | Path,
    directory: str | Path,
    *,
    dynamic_shared_bytes: int | None = None,
):
    """Assemble a variant, validate its ABI, and return a lazy CompiledKernel.

    Resource changes require both descriptor directives and matching HSA YAML
    updates. Triton's launch-time dynamic LDS is distinct from the static LDS
    in the code object, and is preserved unless ``dynamic_shared_bytes`` is
    explicitly supplied. Numerical modes, kernel arguments, wave/workgroup
    modes, and launch geometry remain the baseline's. Runtime correctness is
    checked by the caller against the unchanged Gluon kernel. Edited assembly
    requires a sibling ``.amdgcn.transform.json`` recording the original and
    edited assembly SHA-256s as ``input_sha256`` and ``output_sha256``. A checked
    sidecar may specify ``required_dynamic_shared_bytes``; that allocation is
    used by default, and an explicit override cannot undercut its requirement.
    """
    from triton.backends.amd.compiler import HIPBackend
    from triton.compiler.compiler import CompiledKernel
    from triton._C.libtriton import llvm

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    asm_path = Path(asm_path).resolve()
    assembly = asm_path.read_text()
    baseline_asm = compiled.asm["amdgcn"]
    specialization = _verify_specialization(compiled, asm_path, assembly)
    metadata = _metadata_json(compiled)
    target = compiled.metadata.target
    if target.backend != "hip" or target.arch != "gfx1201":
        raise ValueError("This study loader supports only HIP gfx1201 kernels")
    code_metadata, descriptor = _validate_assembly(baseline_asm, assembly, metadata)
    kernel_metadata = _kernel_metadata(code_metadata, compiled.name)
    required_shared = specialization.get("transform", {}).get("required_dynamic_shared_bytes")
    if required_shared is not None and (type(required_shared) is not int or required_shared < 0):
        raise ValueError("required_dynamic_shared_bytes must be a nonnegative integer")
    if dynamic_shared_bytes is None and required_shared is not None:
        dynamic_shared_bytes = required_shared
    if dynamic_shared_bytes is not None:
        if type(dynamic_shared_bytes) is not int or dynamic_shared_bytes < 0:
            raise ValueError("dynamic_shared_bytes must be a nonnegative integer")
        if required_shared is not None and dynamic_shared_bytes < required_shared:
            raise ValueError("dynamic_shared_bytes is smaller than the transformation's required allocation")
        metadata["shared"] = dynamic_shared_bytes
    backend = HIPBackend(target)
    options = backend.parse_options(dict(metadata))
    # A baseline loaded from Triton's disk cache need not initialize LLVM.
    llvm.init_targets()
    binary = backend.make_hsaco(assembly, metadata, options)
    variant_hash = _sha256(
        compiled.hash.encode() + b"\0" + assembly.encode() + b"\0" + binary
        + b"\0" + str(metadata["shared"]).encode()
    )
    metadata["hash"] = variant_hash
    group = {}
    for ext, data in (("amdgcn", assembly.encode()), ("hsaco", binary), ("json", _json(metadata))):
        path = directory / f"kernel.{ext}"
        _write_once(path, data)
        group[f"kernel.{ext}"] = str(path)
    if _binary_metadata(directory / "kernel.hsaco") != code_metadata:
        raise ValueError("Assembled binary metadata differs from the validated assembly metadata")
    provenance = {
        "kind": "direct_assembly_variant",
        "variant_hash": variant_hash,
        "baseline_triton_hash": compiled.hash,
        "baseline_hsaco_sha256": _sha256(compiled.asm["hsaco"]),
        "hsaco_sha256": _sha256(binary),
        "amdgcn_sha256": _sha256(assembly.encode()),
        "source_assembly": str(asm_path),
        "specialization_proof": specialization,
        "kernel_name": compiled.name,
        "assembly_matches_baseline": assembly == baseline_asm,
        "binary_matches_baseline": binary == compiled.asm["hsaco"],
        "resources": kernel_metadata,
        "dynamic_shared_bytes": metadata["shared"],
        "required_dynamic_shared_bytes": required_shared,
        "descriptor": descriptor,
        "launch_metadata": metadata,
    }
    _write_once(directory / "manifest.json", _json(provenance))
    result = CompiledKernel(compiled.src, group, variant_hash)
    if result.kernel != binary or result.hash != variant_hash:
        raise AssertionError("Variant constructor did not retain the requested binary")
    result.asm_provenance = provenance
    return result
