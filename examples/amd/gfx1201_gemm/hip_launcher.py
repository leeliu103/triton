"""Launch the hash-checked, fixed-shape gfx1201 code objects in this example."""

import ctypes
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import sysconfig


ROOT = Path(__file__).resolve().parent


def read_manifest():
    return json.loads((ROOT / "manifest.json").read_text())


def verify_artifacts(manifest=None):
    """Check the archived ISA and ELF bytes without loading HIP or a GPU."""
    manifest = manifest or read_manifest()
    for name, kernel in manifest["kernels"].items():
        for kind in ("amdgcn", "hsaco"):
            artifact = kernel["artifacts"][kind]
            path = ROOT / artifact["path"]
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != artifact["sha256"]:
                raise ValueError(f"{name}: {kind} SHA256 differs from manifest: {actual}")
    return manifest


class HipRuntime:
    def __init__(self):
        candidates = []
        override = os.environ.get("HIP_LIBRARY_PATH")
        if override:
            p = Path(override)
            override = str(p / "libamdhip64.so") if p.is_dir() else str(p)
            candidates.append(override)
        # Torch has already initialized HIP when the benchmark reaches here.
        # find_library can name an older system ROCm beside Torch's bundled
        # runtime. Loading that second runtime would not share Torch's context.
        loaded = set()
        maps = Path("/proc/self/maps")
        if maps.exists():
            for line in maps.read_text().splitlines():
                parts = line.split(maxsplit=5)
                if len(parts) == 6 and Path(parts[5]).name.startswith("libamdhip64.so"):
                    loaded.add(str(Path(parts[5]).resolve()))
        if len(loaded) > 1:
            raise RuntimeError(f"Multiple HIP runtimes are already loaded: {sorted(loaded)}")
        if loaded:
            loaded_path = next(iter(loaded))
            if override and str(Path(override).resolve()) != loaded_path:
                raise RuntimeError("HIP_LIBRARY_PATH differs from Torch's already-loaded HIP runtime")
            candidates = [loaded_path]
        for variable in ("ROCM_PATH", "HIP_PATH"):
            if not loaded and os.environ.get(variable):
                candidates.extend(str(Path(os.environ[variable]) / sub / "libamdhip64.so")
                                  for sub in ("lib", "lib64"))
        if not loaded:
            purelib = Path(sysconfig.get_paths()["purelib"])
            candidates.extend(str(purelib / package / "lib/libamdhip64.so")
                              for package in ("_rocm_sdk_core", "_rocm_sdk_devel", "torch"))
            candidates.append("libamdhip64.so")
            found = ctypes.util.find_library("amdhip64")
            if found:
                candidates.append(found)
        failures = []
        for candidate in dict.fromkeys(candidates):
            try:
                self.lib = ctypes.CDLL(candidate)
                self.library_path = candidate
                break
            except OSError as error:
                failures.append(str(error))
        else:
            raise RuntimeError("Cannot load libamdhip64; set HIP_LIBRARY_PATH or activate ROCm.\n"
                               + "\n".join(failures))
        void = ctypes.c_void_p
        self.lib.hipGetErrorString.argtypes = [ctypes.c_int]
        self.lib.hipGetErrorString.restype = ctypes.c_char_p
        self.lib.hipModuleLoadData.argtypes = [ctypes.POINTER(void), void]
        self.lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(void), void, ctypes.c_char_p]
        self.lib.hipModuleUnload.argtypes = [void]
        self.lib.hipModuleLaunchKernel.argtypes = [
            void, *([ctypes.c_uint] * 7), void, ctypes.POINTER(void), ctypes.POINTER(void)
        ]
        for name in ("hipModuleLoadData", "hipModuleGetFunction", "hipModuleUnload", "hipModuleLaunchKernel"):
            getattr(self.lib, name).restype = ctypes.c_int

    def check(self, result, operation):
        if result:
            reason = self.lib.hipGetErrorString(result)
            raise RuntimeError(f"{operation}: HIP {result}: {reason.decode() if reason else 'unknown error'}")


class HipKernel:
    def __init__(self, name, runtime=None, manifest=None):
        import torch

        self.manifest = verify_artifacts(manifest)
        self.spec = self.manifest["kernels"][name]
        self.name = name
        self.device = torch.cuda.current_device()
        arch = torch.cuda.get_device_properties(self.device).gcnArchName.split(":", 1)[0]
        if arch != self.spec["gfx"]:
            raise ValueError(f"{name} requires {self.spec['gfx']}; current device is {arch}")
        self.runtime = runtime or HipRuntime()
        self.module, self.function = ctypes.c_void_p(), ctypes.c_void_p()
        binary = (ROOT / self.spec["artifacts"]["hsaco"]["path"]).read_bytes()
        self._image = ctypes.create_string_buffer(binary)
        self.runtime.check(self.runtime.lib.hipModuleLoadData(ctypes.byref(self.module), self._image),
                           "hipModuleLoadData")
        try:
            self.runtime.check(self.runtime.lib.hipModuleGetFunction(
                ctypes.byref(self.function), self.module, self.spec["symbol"].encode()), "hipModuleGetFunction")
        except Exception:
            self.close()
            raise

    def bind(self, a, b, c):
        """Validate once and bind tensors for repeated launches on the current stream."""
        import torch

        m, n, k = self.spec["shape_mnk"]
        tensors = {"a": a, "b": b, "c": c}
        for name, shape in (("a", (m, k)), ("b", (k, n)), ("c", (m, n))):
            tensor = tensors[name]
            if (tensor.dtype != torch.float16 or not tensor.is_cuda or tensor.device.index != self.device
                    or tuple(tensor.shape) != shape or not tensor.is_contiguous()
                    or list(tensor.stride()) != self.spec["strides"][name] or tensor.data_ptr() % 16):
                raise ValueError(f"{self.name}: {name} must be aligned, contiguous FP16 {shape} on device {self.device}")
        c_begin, c_end = c.data_ptr(), c.data_ptr() + c.numel() * c.element_size()
        for tensor in (a, b):
            begin, end = tensor.data_ptr(), tensor.data_ptr() + tensor.numel() * tensor.element_size()
            if c_begin < end and begin < c_end:
                raise ValueError("C must not overlap A or B")
        values = []
        for field in self.spec["kernarg_fields"]:
            if field["type"] == "pointer":
                value = tensors[field["tensor"]].data_ptr() if "tensor" in field else field["value"]
                values.append(ctypes.c_void_p(value))
            elif field["type"] == "i32":
                values.append(ctypes.c_int32(field["value"]))
            else:
                raise ValueError(f"Unknown kernarg type: {field['type']}")
        parameters = (ctypes.c_void_p * len(values))(*(ctypes.addressof(value) for value in values))
        stream = ctypes.c_void_p(torch.cuda.current_stream(self.device).cuda_stream)

        def launch():
            # Keep both the scalar storage and tensors alive for every launch.
            _ = values, tensors
            if not self.module.value:
                raise RuntimeError("Kernel module was closed")
            self.runtime.check(self.runtime.lib.hipModuleLaunchKernel(
                self.function, *self.spec["grid"], *self.spec["block"], self.spec["dynamic_shared_bytes"],
                stream, parameters, None), "hipModuleLaunchKernel")

        return launch

    def close(self):
        """Call only after all launches using this module have completed."""
        if self.module.value:
            self.runtime.check(self.runtime.lib.hipModuleUnload(self.module), "hipModuleUnload")
            self.module = ctypes.c_void_p()
