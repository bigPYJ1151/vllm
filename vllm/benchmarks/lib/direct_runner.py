# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Process setup for benchmarks that call the model runner directly.

One process per rank, no engine core process: the single-rank env is filled in
unless a launcher (`torchrun`) already did, and CPU ranks get the OpenMP thread
and memory-node binding that the multiprocessing executor otherwise applies.
"""

import os
import socket
import sys
from typing import Any

from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform

_BOUND_ENV = "VLLM_BENCH_CPU_THREADS_BOUND"


def init_rank_env() -> None:
    if "RANK" in os.environ:
        return  # Launched by torchrun.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    os.environ.update(
        RANK="0",
        LOCAL_RANK="0",
        WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )


def bind_cpu_threads(engine_args: EngineArgs) -> None:
    """Bind this rank's OpenMP threads and memory node like a spawned CPU worker.

    Honors `VLLM_CPU_OMP_THREADS_BIND` and `VLLM_CPU_BIND_GRANULARITY`. The
    OpenMP runtime reads its affinity env when torch is imported, so the process
    re-executes itself (same pid, same arguments) with the binding env set.
    """
    if not current_platform.is_cpu() or os.environ.get(_BOUND_ENV) == "1":
        return
    from vllm.utils.ompmultiprocessing import OMPProcessManager

    manager = OMPProcessManager(engine_args.create_engine_config())
    if manager.skip_setup:
        return
    with manager.configure_omp_envs(
        rank=int(os.environ["RANK"]), local_rank=int(os.environ["LOCAL_RANK"])
    ):
        env = {**os.environ, _BOUND_ENV: "1"}
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], env)


def get_model_runner(llm: Any) -> Any:
    return llm.llm_engine.model_executor.driver_worker.worker.model_runner
