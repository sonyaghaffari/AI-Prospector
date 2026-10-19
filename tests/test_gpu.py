"""CUDA availability and OOM-safe batched inference."""
import pytest
import torch

from src.utils.gpu import inspect_gpu, oom_safe_batched, select_device


def test_cuda_reported_available():
    info = inspect_gpu()
    assert info.cuda_available, info.summary()
    assert torch.cuda.is_available()


def test_select_device_returns_cuda():
    dev = select_device()
    assert dev.type == "cuda"


def test_oom_safe_batched_shrinks_on_oom():
    device = torch.device("cuda")
    calls = []

    def fn(x):
        calls.append(x.shape[0])
        if x.shape[0] > 64:
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return x.sum(dim=1, keepdim=True)

    x = torch.randn(1000, 4, device=device)
    out = oom_safe_batched(fn, x, batch_size=1000, min_batch=32)
    assert out.shape == (1000, 1)
    assert torch.allclose(out, x.sum(dim=1, keepdim=True))
    # several shrinking attempts should have occurred before converging to a working chunk size
    assert len(calls) > 4
    assert calls[-1] <= 64  # the batch size that finally succeeded, repeated for later chunks


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_tensor_ops_on_gpu():
    a = torch.randn(512, 512, device="cuda")
    b = a @ a.T
    assert b.is_cuda
    torch.cuda.synchronize()
