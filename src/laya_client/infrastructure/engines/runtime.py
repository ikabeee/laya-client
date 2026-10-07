"""Pre-flight checks for the inference runtime.

Everything here runs before a checkpoint is loaded, so a broken install fails at start-up with a reason
and a fix instead of serving errors (or silently running a GPU deployment on the CPU) later.

The checks that matter most in practice:

* ``laya`` / ``torch`` missing: the engine extra was never installed.
* A CPU-only torch build on a machine with an NVIDIA GPU: everything "works", just 10x slower.
* A torch build without kernels for the GPU's architecture: ``torch.cuda.is_available()`` is True, but
  the first CUDA operation fails. RTX 50-series cards (Blackwell, ``sm_120``) need a CUDA 12.8+ build.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ...domain.errors import EngineUnavailableError

#: The CUDA build of torch that `make setup` installs; the oldest one with Blackwell (sm_120) kernels.
RECOMMENDED_CUDA_BUILD = "cu128"
TORCH_CUDA_INSTALL = "pip install --index-url https://download.pytorch.org/whl/%s torch" % RECOMMENDED_CUDA_BUILD
INSTALL_HINT = "run `make setup`, or install torch for your hardware and then `pip install 'laya-client[engine]'`"


@dataclass(frozen=True)
class RuntimeInfo:
    """What the engine will run on, as verified by :func:`check_runtime`."""

    device: str
    torch_version: str
    torch_cuda: str | None = None
    accelerator: str | None = None
    laya_version: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "torch": self.torch_version,
            "torch_cuda": self.torch_cuda,
            "accelerator": self.accelerator,
            "laya": self.laya_version,
        }


def nvidia_gpu_present() -> bool:
    """Whether the host exposes an NVIDIA GPU, independently of what torch can use."""
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def _import(import_module: Callable[[str], Any], name: str) -> Any:
    try:
        return import_module(name)
    except ImportError as error:
        raise EngineUnavailableError("the '%s' package is not installed: %s" % (name, INSTALL_HINT)) from error


def _torch_build(torch: Any) -> str:
    cuda = getattr(torch.version, "cuda", None)
    return "torch %s (%s)" % (torch.__version__, "CUDA %s build" % cuda if cuda else "CPU-only build")


def _check_cuda(torch: Any, device: str) -> str:
    """Verify ``device`` can actually run a CUDA kernel. Returns a description of the GPU."""
    build = _torch_build(torch)
    if not torch.cuda.is_available():
        if getattr(torch.version, "cuda", None) is None:
            raise EngineUnavailableError(
                "LAYA_DEVICE=%s but %s cannot use a GPU. Install a CUDA build: %s" % (device, build, TORCH_CUDA_INSTALL)
            )
        raise EngineUnavailableError(
            "LAYA_DEVICE=%s but %s sees no usable GPU. Check that `nvidia-smi` works, that the NVIDIA "
            "driver supports CUDA %s, and (in Docker) that the container was started with GPU access "
            "(NVIDIA Container Toolkit, `compose.gpu.yaml`)." % (device, build, torch.version.cuda)
        )

    index = int(device.split(":", 1)[1]) if ":" in device else 0
    count = torch.cuda.device_count()
    if index >= count:
        raise EngineUnavailableError(
            "LAYA_DEVICE=%s but only %d GPU(s) are visible (check CUDA_VISIBLE_DEVICES)" % (device, count)
        )

    name = torch.cuda.get_device_name(index)
    major, minor = torch.cuda.get_device_capability(index)
    arch = "sm_%d%d" % (major, minor)
    described = "%s (%s)" % (name, arch)
    try:
        # `is_available()` only says a driver answered. Running one kernel proves this torch build was
        # compiled for this GPU: an RTX 50xx with a CUDA 12.4 build fails right here.
        probe = torch.ones(8, device=device)
        float((probe * 2).sum().item())
    except Exception as error:  # noqa: BLE001 -- any failure here means the GPU is unusable
        supported = ", ".join(torch.cuda.get_arch_list()) or "none"
        raise EngineUnavailableError(
            "%s cannot run on %s: %s. This build has kernels for: %s. RTX 50-series GPUs (Blackwell, "
            "sm_120) need a CUDA 12.8+ build: %s" % (build, described, error, supported, TORCH_CUDA_INSTALL)
        ) from error
    return described


def _mps_available(torch: Any) -> bool:
    backend = getattr(getattr(torch, "backends", None), "mps", None)
    return bool(backend and backend.is_available())


def check_runtime(
    requested_device: str | None,
    *,
    require_gpu: bool = False,
    import_module: Callable[[str], Any] = importlib.import_module,
    gpu_present: Callable[[], bool] = nvidia_gpu_present,
) -> RuntimeInfo:
    """Resolve ``requested_device`` and prove the runtime can use it, or raise ``EngineUnavailableError``.

    ``None``, ``""`` and ``"auto"`` pick CUDA, then Apple MPS, then CPU. Auto mode refuses to fall back to
    the CPU when an NVIDIA GPU is present but torch cannot use it: that is a broken install, not a choice.
    Set ``LAYA_DEVICE=cpu`` to run on the CPU deliberately.
    """
    laya = _import(import_module, "laya")
    torch = _import(import_module, "torch")
    requested = (requested_device or "auto").strip().lower()
    notes: list[str] = []

    if requested == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif gpu_present():
            raise EngineUnavailableError(
                "an NVIDIA GPU is present but %s cannot use it, so the model would silently run on the CPU. "
                "Install a CUDA build (%s), or set LAYA_DEVICE=cpu to use the CPU on purpose."
                % (_torch_build(torch), TORCH_CUDA_INSTALL)
            )
        elif _mps_available(torch):
            device = "mps"
        else:
            device = "cpu"
            notes.append("no GPU found: running on the CPU")
    else:
        device = requested

    accelerator = None
    if device.startswith("cuda"):
        accelerator = _check_cuda(torch, device)
    elif device == "mps":
        if not _mps_available(torch):
            raise EngineUnavailableError("LAYA_DEVICE=mps but this torch build or machine has no MPS support")
        accelerator = "Apple MPS"
    elif device != "cpu":
        raise EngineUnavailableError("unknown LAYA_DEVICE %r: use auto, cpu, cuda, cuda:<index> or mps" % device)

    if require_gpu and accelerator is None:
        raise EngineUnavailableError(
            "LAYA_REQUIRE_GPU=true but the engine would run on the CPU (%s)" % _torch_build(torch)
        )

    return RuntimeInfo(
        device=device,
        torch_version=str(torch.__version__),
        torch_cuda=getattr(torch.version, "cuda", None),
        accelerator=accelerator,
        laya_version=getattr(laya, "__version__", None),
        notes=tuple(notes),
    )
