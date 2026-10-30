# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CPU shared-memory all-reduce kernel (gloo for bootstrapping)."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port

pytestmark = pytest.mark.cpu_test

if not current_platform.is_cpu():
    pytest.skip("skipping CPU-only tests", allow_module_level=True)

import vllm._C  # noqa: E402, F401

THREADS_PER_RANK = 4
# Odd lengths exercise the vector tails and uneven per-rank slices; the sizes
# straddle the 1 MiB threshold between the single- and two-barrier algorithms
# and the largest one needs several iterations per thread.
NUM_ELEMS = [1, 7, 4097, 100003, 262144 + 33, (1 << 20) + 5, 16_000_003]


def _worker(rank: int, world_size: int, port: int, dtype: torch.dtype):
    torch.set_num_threads(THREADS_PER_RANK)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    name = f"shm-allreduce-test-{port}"
    handle = torch.ops._C.init_shm_manager(name, world_size, rank, THREADS_PER_RANK)
    dist.barrier()
    torch.ops._C.join_shm_manager(handle, name)
    dist.barrier()

    for numel in NUM_ELEMS:
        generator = torch.Generator().manual_seed(rank * 1000 + numel)
        for _ in range(3):
            data = torch.randn(numel, generator=generator).to(dtype)
            expected = data.float()
            dist.all_reduce(expected)
            torch.ops._C.shm_allreduce(handle, data)
            torch.testing.assert_close(
                data.float(), expected, rtol=1e-2, atol=1e-2, equal_nan=False
            )
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.skipif(
    not hasattr(torch.ops._C, "init_shm_manager"),
    reason="CPU shm all-reduce is not built",
)
@pytest.mark.parametrize("world_size", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cpu_shm_allreduce(world_size: int, dtype: torch.dtype):
    mp.spawn(
        _worker,
        args=(world_size, get_open_port(), dtype),
        nprocs=world_size,
        join=True,
    )
