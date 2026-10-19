"""GPU discovery, reproducibility settings, memory logging and OOM-safe batching."""
from __future__ import annotations

import os
import platform
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterator

import torch


class CudaUnavailableError(RuntimeError):
    pass


CUDA_DIAGNOSTICS = """CUDA is not available to PyTorch. Diagnostics:
  1. Run `nvidia-smi`. If it fails, the NVIDIA driver is missing or broken.
  2. Check the PyTorch build: python -c "import torch; print(torch.__version__, torch.version.cuda)"
     A version ending in '+cpu' (torch.version.cuda == None) is a CPU-only wheel.
  3. Install a CUDA wheel matching your driver (nvidia-smi shows the max CUDA version), e.g.
     pip install torch --index-url https://download.pytorch.org/whl/cu126
     (only do this with your explicit approval; it replaces the installed torch).
  4. Make sure CUDA_VISIBLE_DEVICES is not set to an empty / invalid value.
To run on CPU anyway (slow), pass --allow-cpu or set ANGOURAN_ALLOW_CPU=1."""


@dataclass
class GPUInfo:
    torch_version: str
    torch_cuda_build: str | None
    cuda_available: bool
    cudnn_version: int | None
    device_name: str | None = None
    compute_capability: str | None = None
    total_vram_gb: float | None = None
    free_vram_gb: float | None = None
    driver_version: str | None = None
    driver_cuda_version: str | None = None
    python: str = field(default_factory=platform.python_version)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        if not self.cuda_available:
            return f"PyTorch {self.torch_version} (CUDA build: {self.torch_cuda_build}) — CUDA NOT available"
        return (
            f"{self.device_name} (sm_{self.compute_capability}) | VRAM {self.free_vram_gb:.1f}/"
            f"{self.total_vram_gb:.1f} GB free | PyTorch {self.torch_version} | CUDA build {self.torch_cuda_build} | "
            f"driver {self.driver_version} (CUDA {self.driver_cuda_version}) | cuDNN {self.cudnn_version}"
        )


def _nvidia_smi(query: str) -> str | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def _driver_cuda_version() -> str | None:
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            if "CUDA Version" in line:
                return line.split("CUDA Version:")[1].split("|")[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return None


def inspect_gpu() -> GPUInfo:
    info = GPUInfo(
        torch_version=torch.__version__,
        torch_cuda_build=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        cudnn_version=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
    )
    if info.cuda_available:
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        info.device_name = props.name
        info.compute_capability = f"{props.major}{props.minor}"
        info.total_vram_gb = total / 1e9
        info.free_vram_gb = free / 1e9
        info.driver_version = _nvidia_smi("driver_version")
        info.driver_cuda_version = _driver_cuda_version()
    return info


def gpu_utilization() -> dict[str, float] | None:
    line = _nvidia_smi("utilization.gpu,memory.used,memory.total")
    if not line:
        return None
    try:
        util, used, total = (float(v) for v in line.split(","))
        return {"gpu_util_pct": util, "mem_used_mb": used, "mem_total_mb": total}
    except ValueError:
        return None


def select_device(allow_cpu: bool | None = None) -> torch.device:
    """Return CUDA device; refuse to silently fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu is None:
        allow_cpu = os.environ.get("ANGOURAN_ALLOW_CPU", "0") == "1"
    if allow_cpu:
        print("WARNING: CUDA unavailable — running on CPU because CPU execution was explicitly allowed.")
        return torch.device("cpu")
    raise CudaUnavailableError(CUDA_DIAGNOSTICS)


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    """Seed all RNGs.

    deterministic=True forces deterministic kernels (slightly slower; needed for bitwise
    reproducible ensembles). cudnn.benchmark auto-tunes conv kernels; this project uses
    only dense layers, so benchmark mode brings no speed-up and is enabled only in
    non-deterministic mode.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


@dataclass
class PerfRecord:
    name: str
    seconds: float
    peak_mem_mb: float | None
    gpu_util_pct: float | None = None


class PerfLog:
    """Collects timing + peak GPU memory for each major operation."""

    def __init__(self) -> None:
        self.records: list[PerfRecord] = []

    @contextmanager
    def track(self, name: str) -> Iterator[None]:
        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e6 if cuda else None
            util = gpu_utilization() if cuda else None
            self.records.append(PerfRecord(name, time.perf_counter() - t0, peak,
                                           util["gpu_util_pct"] if util else None))

    def as_rows(self) -> list[dict]:
        return [asdict(r) for r in self.records]


def is_oom(err: BaseException) -> bool:
    return isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()


def oom_safe_batched(fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, batch_size: int,
                     min_batch: int = 256, dim: int = 0) -> torch.Tensor:
    """Apply ``fn`` over chunks of ``x`` along ``dim``; halve the chunk size on CUDA OOM."""
    n = x.shape[dim]
    bs = max(1, min(batch_size, n))
    while True:
        try:
            outs = [fn(x.narrow(dim, i, min(bs, n - i))) for i in range(0, n, bs)]
            return torch.cat(outs, dim=dim if outs[0].dim() > 0 else 0) if len(outs) > 1 else outs[0]
        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if not is_oom(err) or bs <= min_batch:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            bs = max(min_batch, bs // 2)
            print(f"[gpu] CUDA OOM — retrying with batch size {bs}")


def estimate_tensor_mb(*shape: int, bytes_per: int = 4) -> float:
    n = 1
    for s in shape:
        n *= s
    return n * bytes_per / 1e6
