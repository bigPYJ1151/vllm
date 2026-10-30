# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.kernels  # noqa: F401  (registers the vllm_c IR implementations)
from vllm import ir
from vllm.platforms import CpuArchEnum, current_platform
from vllm.utils.torch_utils import set_random_seed

if not (
    current_platform.is_cpu()
    and current_platform.get_cpu_architecture() == CpuArchEnum.X86
):
    pytest.skip("skipping x86 CPU-only tests", allow_module_level=True)

DTYPES = [torch.bfloat16, torch.float32]
NUM_TOKENS = [1, 7, 83]
# 40 and 100 are not multiples of 32, so the vllm_c kernel declines them and
# the native implementation runs.
HIDDEN_SIZES = [32, 40, 100, 128, 4096]
EPS = 1e-6


def _tol(dtype: torch.dtype) -> dict[str, float]:
    return (
        {"atol": 1e-2, "rtol": 1e-2}
        if dtype != torch.float32
        else {"atol": 1e-5, "rtol": 1e-5}
    )


def _make_input(num_tokens: int, hidden_size: int, dtype: torch.dtype, strided: bool):
    last_dim = 2 * hidden_size if strided else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype)[..., :hidden_size]
    assert x.is_contiguous() != strided or num_tokens == 1
    return x


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("has_weight", [False, True])
@torch.inference_mode()
def test_cpu_rms_norm_vllm_c(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    strided: bool,
    has_weight: bool,
) -> None:
    set_random_seed(0)
    x = _make_input(num_tokens, hidden_size, dtype, strided)
    weight = torch.randn(hidden_size, dtype=dtype) if has_weight else None

    ref = ir.ops.rms_norm.impls["native"].impl_fn(x, weight, EPS)
    with ir.ops.rms_norm.set_priority(["vllm_c", "native"]):
        out = ir.ops.rms_norm(x, weight, EPS)
    torch.testing.assert_close(out, ref, **_tol(dtype))


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("has_weight", [False, True])
@torch.inference_mode()
def test_cpu_fused_add_rms_norm_vllm_c(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    strided: bool,
    has_weight: bool,
) -> None:
    set_random_seed(0)
    x = _make_input(num_tokens, hidden_size, dtype, strided)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype)
    weight = torch.randn(hidden_size, dtype=dtype) if has_weight else None

    ref_out, ref_residual = ir.ops.fused_add_rms_norm.impls["native"].impl_fn(
        x.clone(), residual.clone(), weight, EPS
    )
    with ir.ops.fused_add_rms_norm.set_priority(["vllm_c", "native"]):
        out, out_residual = ir.ops.fused_add_rms_norm(
            x.clone(), residual.clone(), weight, EPS
        )
    torch.testing.assert_close(out, ref_out, **_tol(dtype))
    torch.testing.assert_close(out_residual, ref_residual, **_tol(dtype))
