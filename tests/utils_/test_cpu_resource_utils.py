# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cgroup memory readers and SHM allocation preflight checks."""

import json
import sys
from io import StringIO
from types import SimpleNamespace
from unittest import mock

import pytest

from vllm.platforms import CpuArchEnum
from vllm.utils import cpu_resource_utils as cru
from vllm.utils import ompmultiprocessing as omp
from vllm.utils.mem_constants import GiB_bytes

_V2_LIMIT_PATH = "/sys/fs/cgroup/memory.max"
_V2_USAGE_PATH = "/sys/fs/cgroup/memory.current"
_V1_LIMIT_PATH = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
_V1_USAGE_PATH = "/sys/fs/cgroup/memory/memory.usage_in_bytes"


@pytest.fixture(autouse=True)
def _clear_cgroup_cache():
    cru.get_cgroup_memory_limit.cache_clear()
    yield
    cru.get_cgroup_memory_limit.cache_clear()


def _stub_files(monkeypatch, files: dict):
    """Stub ``open()`` for a fixed set of paths; ``None`` -> OSError."""
    real_open = open

    def fake_open(path, *args, **kwargs):
        if path in files:
            content = files[path]
            if content is None:
                raise OSError(f"no such file: {path}")
            return StringIO(content)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)


def test_cgroup_v2_limit_and_usage(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _V2_LIMIT_PATH: f"{20 * GiB_bytes}\n",
            _V2_USAGE_PATH: f"{5 * GiB_bytes}\n",
        },
    )

    assert cru.get_cgroup_memory_limit() == 20 * GiB_bytes
    assert cru.get_cgroup_memory_usage() == 5 * GiB_bytes


def test_cgroup_v2_unlimited_falls_back_to_v1(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _V2_LIMIT_PATH: "max\n",
            _V1_LIMIT_PATH: f"{8 * GiB_bytes}\n",
            _V1_USAGE_PATH: f"{1 * GiB_bytes}\n",
        },
    )

    assert cru.get_cgroup_memory_limit() == 8 * GiB_bytes
    assert cru.get_cgroup_memory_usage() == 1 * GiB_bytes


def test_cgroup_v1_limit_and_usage(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {
            _V2_LIMIT_PATH: None,
            _V1_LIMIT_PATH: f"{8 * GiB_bytes}\n",
            _V1_USAGE_PATH: f"{2 * GiB_bytes}\n",
        },
    )

    assert cru.get_cgroup_memory_limit() == 8 * GiB_bytes
    assert cru.get_cgroup_memory_usage() == 2 * GiB_bytes


def test_cgroup_v1_unlimited_sentinel_is_ignored(monkeypatch):
    """An unlimited cgroup v1 sentinel must not be treated as a real limit."""
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(
        monkeypatch,
        {_V2_LIMIT_PATH: None, _V1_LIMIT_PATH: f"{(1 << 63) - 1}\n"},
    )

    assert cru.get_cgroup_memory_limit() is None
    assert cru.get_cgroup_memory_usage() is None


def test_cgroup_no_limit_files_present(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    _stub_files(monkeypatch, {_V2_LIMIT_PATH: None, _V1_LIMIT_PATH: None})

    assert cru.get_cgroup_memory_limit() is None
    assert cru.get_cgroup_memory_usage() is None


def test_cgroup_skipped_on_non_linux(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    _stub_files(monkeypatch, {_V2_LIMIT_PATH: f"{1 * GiB_bytes}\n"})

    assert cru.get_cgroup_memory_limit() is None
    assert cru.get_cgroup_memory_usage() is None


def test_cgroup_limit_is_cached_until_explicitly_cleared(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    files = {
        _V2_LIMIT_PATH: f"{1 * GiB_bytes}\n",
        _V2_USAGE_PATH: f"{512 << 20}\n",
    }
    _stub_files(monkeypatch, files)

    assert cru.get_cgroup_memory_limit() == 1 * GiB_bytes
    files[_V2_LIMIT_PATH] = f"{2 * GiB_bytes}\n"
    assert cru.get_cgroup_memory_limit() == 1 * GiB_bytes

    cru.get_cgroup_memory_limit.cache_clear()
    assert cru.get_cgroup_memory_limit() == 2 * GiB_bytes


def test_cgroup_usage_is_read_without_cache(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    files = {
        _V2_LIMIT_PATH: f"{1 * GiB_bytes}\n",
        _V2_USAGE_PATH: f"{512 << 20}\n",
    }
    _stub_files(monkeypatch, files)

    assert cru.get_cgroup_memory_usage() == 512 << 20
    files[_V2_USAGE_PATH] = f"{768 << 20}\n"
    assert cru.get_cgroup_memory_usage() == 768 << 20


def test_check_cgroup_memory_available_warns_on_low_headroom(monkeypatch):
    monkeypatch.setattr(cru, "get_cgroup_memory_limit", lambda: 1 << 30)
    monkeypatch.setattr(cru, "get_cgroup_memory_usage", lambda: 512 << 20)

    with (
        mock.patch.object(cru.logger, "debug") as log_debug,
        mock.patch.object(cru.logger, "warning") as log_warning,
    ):
        cru.check_cgroup_memory_available(600 << 20, "mmap")

    log_debug.assert_not_called()
    log_warning.assert_called_once()
    assert "current headroom is below" in log_warning.call_args.args[-1]


def test_check_cgroup_memory_available_logs_success_at_debug(monkeypatch):
    monkeypatch.setattr(cru, "get_cgroup_memory_limit", lambda: 1 << 30)
    monkeypatch.setattr(cru, "get_cgroup_memory_usage", lambda: 512 << 20)

    with (
        mock.patch.object(cru.logger, "debug") as log_debug,
        mock.patch.object(cru.logger, "warning") as log_warning,
    ):
        cru.check_cgroup_memory_available(256 << 20, "mmap")

    log_debug.assert_called_once()
    log_warning.assert_not_called()


@pytest.mark.parametrize(
    ("limit", "usage"),
    [(None, 512 << 20), (1 << 30, None)],
)
def test_check_cgroup_memory_available_skips_without_snapshot(
    monkeypatch, limit, usage
):
    monkeypatch.setattr(cru, "get_cgroup_memory_limit", lambda: limit)
    monkeypatch.setattr(cru, "get_cgroup_memory_usage", lambda: usage)

    with (
        mock.patch.object(cru.logger, "debug") as log_debug,
        mock.patch.object(cru.logger, "warning") as log_warning,
    ):
        cru.check_cgroup_memory_available(1 << 60, "mmap")

    log_debug.assert_not_called()
    log_warning.assert_not_called()


@pytest.fixture
def six_numa_nodes(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cru, "get_memory_affinity", lambda: [0, 1, 2, 3, 4, 5])
    monkeypatch.delenv(cru.DEVICE_CONTROL_ENV_VAR, raising=False)
    monkeypatch.delenv("VLLM_CPU_SIM_MULTI_NUMA", raising=False)


def test_visible_memory_node_uses_env_var(six_numa_nodes, monkeypatch):
    monkeypatch.setenv(cru.DEVICE_CONTROL_ENV_VAR, "5,3")
    assert cru.get_visible_memory_node() == [5, 3]


def test_visible_memory_node_assigned_ids_select_dp_rank_nodes(six_numa_nodes):
    """Each DP rank sees only the nodes assigned to it."""
    assert cru.get_visible_memory_node([0, 1]) == [0, 1]
    assert cru.get_visible_memory_node([2, 3]) == [2, 3]


def test_visible_memory_node_assigned_ids_override_env_var(six_numa_nodes, monkeypatch):
    monkeypatch.setenv(cru.DEVICE_CONTROL_ENV_VAR, "5,3")
    assert cru.get_visible_memory_node([2]) == [2]


def test_visible_memory_node_assigned_ids_are_limited_to_allowed_nodes(
    six_numa_nodes, monkeypatch
):
    monkeypatch.setattr(cru, "get_memory_affinity", lambda: [0, 1])
    assert cru.get_visible_memory_node([1, 4]) == [1]


def test_visible_memory_node_sim_multi_numa_ignores_assigned_ids(
    six_numa_nodes, monkeypatch
):
    monkeypatch.setenv("VLLM_CPU_SIM_MULTI_NUMA", "1")
    assert cru.get_visible_memory_node([2]) == [0, 1, 2, 3, 4, 5]


def test_omp_manager_binds_each_dp_rank_to_its_assigned_node(monkeypatch):
    """DP ranks with different assigned NUMA nodes get disjoint cores (#47328)."""
    # 2 nodes x 4 physical cores x 2 logical CPUs
    cpus = [
        cru.LogicalCPUInfo(id=i, physical_core=i // 2, numa_node=i // 8)
        for i in range(16)
    ]
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(cru, "get_allowed_cpu_list", lambda: cpus)
    monkeypatch.setattr(cru, "get_memory_affinity", lambda: [0, 1])
    monkeypatch.setattr(
        omp,
        "current_platform",
        SimpleNamespace(
            is_cpu=lambda: True, get_cpu_architecture=lambda: CpuArchEnum.X86
        ),
    )
    monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", "auto")
    monkeypatch.delenv("VLLM_CPU_SIM_MULTI_NUMA", raising=False)
    monkeypatch.delenv(cru.DEVICE_CONTROL_ENV_VAR, raising=False)

    def cpu_ids(dp_rank: int) -> set[int]:
        parallel_config = SimpleNamespace(
            local_world_size=1,
            data_parallel_rank_local=dp_rank,
            _api_process_count=2,
            assigned_physical_gpu_ids=[dp_rank],
        )
        manager = omp.OMPProcessManager(
            SimpleNamespace(parallel_config=parallel_config, kv_transfer_config=None)
        )
        return set(manager.cpu_lists[0])

    rank0, rank1 = cpu_ids(0), cpu_ids(1)
    assert rank0 and rank1
    assert rank0 <= set(range(8))
    assert rank1 <= set(range(8, 16))


# Trimmed `lscpu --json --extended=CPU,CORE,NODE,CACHE` of a 2-CBB-per-node
# machine: CPUs 0-1 / 2-3 on node 0 (L3 0 / 1), CPUs 4-5 / 6-7 on node 1.
_LSCPU_CACHE_JSON = """{"cpus": [
  {"cpu": 0, "core": 0, "node": 0, "l1d:l1i:l2:l3": "0:0:0:0"},
  {"cpu": 1, "core": 1, "node": 0, "l1d:l1i:l2:l3": "2:2:0:0"},
  {"cpu": 2, "core": 2, "node": 0, "l1d:l1i:l2:l3": "72:72:9:1"},
  {"cpu": 3, "core": 3, "node": 0, "l1d:l1i:l2:l3": "74:74:9:1"},
  {"cpu": 4, "core": 4, "node": 1, "l1d:l1i:l2:l3": "80:80:10:2"},
  {"cpu": 5, "core": 5, "node": 1, "l1d:l1i:l2:l3": "82:82:10:2"},
  {"cpu": 6, "core": 6, "node": 1, "l1d:l1i:l2:l3": "88:88:11:3"},
  {"cpu": 7, "core": 7, "node": 1, "l1d:l1i:l2:l3": "90:90:11:3"}
]}"""


def _cpus():
    return json.loads(_LSCPU_CACHE_JSON, object_hook=cru.LogicalCPUInfo.json_decoder)[
        "cpus"
    ]


def test_lscpu_cache_column_gives_llc_id():
    assert [x.llc_id for x in _cpus()] == [0, 0, 1, 1, 2, 2, 3, 3]


@pytest.mark.parametrize("key", ["l1d:l1i:l2:l3", "cache"])
def test_lscpu_cache_column_key_variants(key):
    cpu = cru.LogicalCPUInfo.json_decoder(
        {"cpu": 0, "core": 0, "node": 0, key: "72:72:9:1"}
    )
    assert cpu.llc_id == 1


@pytest.mark.parametrize("cache", ["-", "", "::-:-"])
def test_missing_lscpu_cache_gives_unknown_llc_id(cache):
    cpu = cru.LogicalCPUInfo.json_decoder(
        {"cpu": 0, "core": 0, "node": 0, "l1d:l1i:l2:l3": cache}
    )
    assert cpu.llc_id == -1


def test_absent_lscpu_cache_column_gives_unknown_llc_id():
    cpu = cru.LogicalCPUInfo.json_decoder({"cpu": 0, "core": 0, "node": 0})
    assert cpu.llc_id == -1


def test_group_cpus_by_cbb_follows_memory_node_order():
    groups = cru.group_cpus_by_cbb(_cpus(), [1, 0])
    assert [(g.numa_node, g.llc_id) for g in groups] == [(1, 2), (1, 3), (0, 0), (0, 1)]
    assert [[c.id for c in g.cpus] for g in groups] == [[4, 5], [6, 7], [0, 1], [2, 3]]


def test_group_cpus_by_cbb_skips_masked_nodes():
    groups = cru.group_cpus_by_cbb(_cpus(), [1])
    assert [(g.numa_node, g.llc_id) for g in groups] == [(1, 2), (1, 3)]


def test_group_cpus_by_cbb_without_llc_info_is_one_group_per_node():
    cpus = [cru.LogicalCPUInfo(i, i, i // 4) for i in range(8)]
    groups = cru.group_cpus_by_cbb(cpus, [0, 1])
    assert [[c.id for c in g.cpus] for g in groups] == [[0, 1, 2, 3], [4, 5, 6, 7]]


def _omp_manager(monkeypatch, *, local_world_size, local_dp_rank, cbbs):
    from types import SimpleNamespace

    import vllm.utils.ompmultiprocessing as omp

    monkeypatch.setattr(omp.envs, "VLLM_CPU_BIND_GRANULARITY", "cbb", raising=False)
    monkeypatch.setattr(
        omp,
        "current_platform",
        SimpleNamespace(get_cpu_architecture=lambda: omp.CpuArchEnum.X86),
    )
    monkeypatch.setattr(omp.cr_utils, "get_visible_memory_node", lambda: [0, 1])
    monkeypatch.setattr(omp.cr_utils, "get_allowed_cpu_list", lambda: _cpus())
    monkeypatch.setattr(omp.cr_utils, "get_visible_cbbs", lambda: cbbs)
    manager = omp.OMPProcessManager.__new__(omp.OMPProcessManager)
    manager.local_world_size = local_world_size
    manager.local_dp_rank = local_dp_rank
    manager.simulate_multi_node = False
    manager.reserve_cpu_num = 0
    manager.rank_memory_info = []
    return manager


def test_cbb_autobind_assigns_one_cbb_per_rank(monkeypatch):
    manager = _omp_manager(
        monkeypatch,
        local_world_size=3,
        local_dp_rank=None,
        cbbs=cru.group_cpus_by_cbb(_cpus(), [0, 1]),
    )
    cpu_lists, reserved = manager._get_autobind_cpu_ids(lambda cpus: cpus)
    assert [[c.id for c in x] for x in cpu_lists] == [[0, 1], [2, 3], [4, 5]]
    assert reserved == []
    # Two CBBs per node, whatever the world size.
    assert manager.rank_memory_info == [(0, 2), (0, 2), (1, 2)]


def test_cbb_autobind_offsets_ranks_by_dp_rank(monkeypatch):
    manager = _omp_manager(
        monkeypatch,
        local_world_size=2,
        local_dp_rank=1,
        cbbs=cru.group_cpus_by_cbb(_cpus(), [0, 1]),
    )
    cpu_lists, _ = manager._get_autobind_cpu_ids(lambda cpus: cpus)
    assert [[c.id for c in x] for x in cpu_lists] == [[4, 5], [6, 7]]
    assert manager.rank_memory_info == [(1, 2), (1, 2)]


def test_cbb_autobind_rejects_more_ranks_than_cbbs(monkeypatch):
    manager = _omp_manager(
        monkeypatch,
        local_world_size=5,
        local_dp_rank=None,
        cbbs=cru.group_cpus_by_cbb(_cpus(), [0, 1]),
    )
    with pytest.raises(AssertionError, match="Not enough allowed CBBs"):
        manager._get_autobind_cpu_ids(lambda cpus: cpus)


def test_simulated_multi_numa_takes_precedence_over_cbb(monkeypatch):
    manager = _omp_manager(
        monkeypatch,
        local_world_size=4,
        local_dp_rank=0,
        cbbs=cru.group_cpus_by_cbb(_cpus(), [0, 1]),
    )
    manager.simulate_multi_node = True
    manager.internal_dp_size = 1
    cpu_lists, _ = manager._get_autobind_cpu_ids(lambda cpus: cpus)
    # 4 simulated nodes of 2 CPUs each, and no CBB memory info
    assert [[c.id for c in x] for x in cpu_lists] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert manager.rank_memory_info == []
