"""The start-up runtime checks, against stand-ins for the ``torch`` and ``laya`` modules."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from laya_client.domain.errors import EngineUnavailableError
from laya_client.infrastructure.engines import check_runtime


class FakeTensor:
    def __init__(self, fail: Exception | None) -> None:
        self.fail = fail

    def __mul__(self, other):
        if self.fail:
            raise self.fail
        return self

    def sum(self):
        return self

    def item(self):
        return 16.0


def make_torch(*, cuda_build="12.8", available=True, count=1, capability=(12, 0), kernel_error=None, mps=False):
    return SimpleNamespace(
        __version__="2.8.0+cu128" if cuda_build else "2.8.0+cpu",
        version=SimpleNamespace(cuda=cuda_build),
        cuda=SimpleNamespace(
            is_available=lambda: available,
            device_count=lambda: count,
            get_device_name=lambda index: "NVIDIA GeForce RTX 5080",
            get_device_capability=lambda index: capability,
            get_arch_list=lambda: ["sm_80", "sm_90"] if kernel_error else ["sm_80", "sm_90", "sm_120"],
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        ones=lambda n, device: FakeTensor(kernel_error),
    )


def modules(torch=None, laya=True):
    def import_module(name):
        if name == "laya":
            if not laya:
                raise ImportError("No module named 'laya'")
            return SimpleNamespace(__version__="0.3.28")
        if name == "torch":
            if torch is None:
                raise ImportError("No module named 'torch'")
            return torch
        raise AssertionError(name)

    return import_module


def check(device, torch=None, *, laya=True, gpu_present=False, require_gpu=False):
    return check_runtime(
        device,
        require_gpu=require_gpu,
        import_module=modules(torch, laya=laya),
        gpu_present=lambda: gpu_present,
    )


def test_missing_laya():
    with pytest.raises(EngineUnavailableError, match="'laya' package is not installed.*make setup"):
        check("auto", make_torch(), laya=False)


def test_missing_torch():
    with pytest.raises(EngineUnavailableError, match="'torch' package is not installed"):
        check("auto", None)


def test_auto_picks_a_working_gpu():
    info = check(None, make_torch())
    assert info.device == "cuda"
    assert info.accelerator == "NVIDIA GeForce RTX 5080 (sm_120)"
    assert info.laya_version == "0.3.28"


def test_auto_refuses_cpu_only_torch_on_a_gpu_machine():
    with pytest.raises(EngineUnavailableError, match="NVIDIA GPU is present.*CPU-only build.*cu128"):
        check("auto", make_torch(cuda_build=None, available=False), gpu_present=True)


def test_auto_uses_the_cpu_when_there_is_no_gpu():
    info = check("auto", make_torch(cuda_build=None, available=False))
    assert info.device == "cpu" and info.accelerator is None
    assert info.notes


def test_auto_prefers_mps_on_apple_silicon():
    assert check("auto", make_torch(cuda_build=None, available=False, mps=True)).device == "mps"


def test_explicit_cpu_is_allowed_on_a_gpu_machine():
    assert check("cpu", make_torch(), gpu_present=True).device == "cpu"


def test_cuda_with_cpu_only_torch():
    with pytest.raises(EngineUnavailableError, match=r"CPU-only build\) cannot use a GPU.*cu128"):
        check("cuda", make_torch(cuda_build=None, available=False))


def test_cuda_with_no_visible_gpu():
    with pytest.raises(EngineUnavailableError, match="nvidia-smi.*NVIDIA Container Toolkit"):
        check("cuda", make_torch(available=False))


def test_cuda_index_out_of_range():
    with pytest.raises(EngineUnavailableError, match="only 1 GPU"):
        check("cuda:1", make_torch())


def test_gpu_without_kernels_for_its_architecture():
    torch = make_torch(cuda_build="12.4", kernel_error=RuntimeError("CUDA error: no kernel image is available"))
    with pytest.raises(EngineUnavailableError, match=r"sm_120.*no kernel image.*sm_80, sm_90.*cu128"):
        check("cuda", torch)


def test_require_gpu_refuses_the_cpu():
    with pytest.raises(EngineUnavailableError, match="LAYA_REQUIRE_GPU"):
        check("auto", make_torch(cuda_build=None, available=False), require_gpu=True)


def test_unknown_device():
    with pytest.raises(EngineUnavailableError, match="unknown LAYA_DEVICE"):
        check("tpu", make_torch())


def test_mps_requested_but_missing():
    with pytest.raises(EngineUnavailableError, match="MPS"):
        check("mps", make_torch(mps=False))
