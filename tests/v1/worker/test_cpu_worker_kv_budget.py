# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CPU worker's automatic KV cache sizing must report the right cause.

A worker whose own footprint exceeds its reservation and a node that is short
of free memory need opposite remedies, so they get separate errors.
"""

from types import SimpleNamespace

import pytest

from vllm.platforms import current_platform
from vllm.utils.cpu_resource_utils import MemoryNodeInfo

# Importing the CPU worker patches torch and the GPU buffer utilities for the
# whole process (vllm/v1/worker/cpu/shm.py), so the import is deferred into
# the tests and they run only on the CPU platform.
pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu(), reason="CPU worker test; needs the CPU platform"
)

GiB = 1 << 30


def _worker(monkeypatch, *, reserved, rss, total=8 * GiB, available=4 * GiB):
    import vllm.v1.worker.cpu_worker as cpu_worker_mod
    from vllm.v1.worker.cpu_worker import CPUWorker

    worker = CPUWorker.__new__(CPUWorker)
    worker.requested_cpu_memory = reserved
    worker.rank_total_memory = total
    worker.cache_config = SimpleNamespace(kv_cache_memory_bytes=None)
    monkeypatch.setattr(CPUWorker, "_should_warm_up_model", lambda self: False)
    monkeypatch.setattr(
        cpu_worker_mod,
        "get_allowed_cpu_list",
        lambda: [SimpleNamespace(numa_node=0)],
    )
    monkeypatch.setattr(
        cpu_worker_mod,
        "get_memory_node_info",
        lambda node: MemoryNodeInfo(total_memory=total, available_memory=available),
    )
    process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=rss))
    monkeypatch.setattr(cpu_worker_mod.psutil, "Process", lambda pid: process)
    return worker


def test_footprint_above_reservation_asks_for_a_larger_reservation(monkeypatch):
    # 0.76 GiB already in use against a 0.16 GiB reservation (2% of 8 GiB).
    worker = _worker(monkeypatch, reserved=int(0.16 * GiB), rss=int(0.76 * GiB))
    with pytest.raises(ValueError) as exc:
        worker.determine_available_memory()
    msg = str(exc.value)
    assert "increase `--gpu-memory-utilization` to more than 0.10" in msg
    assert "--kv-cache-memory-bytes" in msg
    # Neither the misleading remedy nor a negative KV size.
    assert "other processes" not in msg
    assert "-0." not in msg


def test_short_node_memory_still_asks_to_free_memory(monkeypatch):
    worker = _worker(monkeypatch, reserved=6 * GiB, rss=1 * GiB, available=2 * GiB)
    with pytest.raises(ValueError, match="Reduce CPU memory used by other processes"):
        worker.determine_available_memory()


def test_budget_that_fits_sizes_the_kv_cache(monkeypatch):
    worker = _worker(monkeypatch, reserved=3 * GiB, rss=1 * GiB)
    assert worker.determine_available_memory() == 2 * GiB


def test_cbb_binding_splits_node_memory_by_cbbs_on_node(monkeypatch):
    """Each rank owns one CBB's share of its node, whatever the TP size."""
    import vllm.v1.worker.cpu_worker as cpu_worker_mod
    from vllm.v1.worker.cpu_worker import CPUWorker

    membind = []
    queried = []

    def node_info(node):
        queried.append(node)
        return MemoryNodeInfo(total_memory=100 * GiB, available_memory=100 * GiB)

    monkeypatch.setenv("VLLM_CPU_RANK_MEMORY_NODE", "1")
    monkeypatch.setenv("VLLM_CPU_CBBS_ON_NODE", "4")
    monkeypatch.setattr(
        cpu_worker_mod, "get_allowed_cpu_list", lambda: [SimpleNamespace(numa_node=0)]
    )
    monkeypatch.setattr(cpu_worker_mod, "get_visible_memory_node", lambda: [0, 1])
    monkeypatch.setattr(cpu_worker_mod, "get_memory_node_info", node_info)
    fake_torch = SimpleNamespace(
        ops=SimpleNamespace(_C=SimpleNamespace(init_cpu_memory_env=membind.append))
    )
    monkeypatch.setattr(cpu_worker_mod, "torch", fake_torch)
    monkeypatch.setattr(
        cpu_worker_mod.Worker,
        "__init__",
        lambda self, *a, **k: setattr(self, "parallel_config", SimpleNamespace()),
    )

    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            gpu_memory_utilization=0.8, kv_cache_memory_bytes=None
        )
    )
    worker = CPUWorker(vllm_config, 0, 0, "")

    assert membind == [[1]]
    assert worker.requested_cpu_memory == 20 * GiB
    assert queried == [1]
