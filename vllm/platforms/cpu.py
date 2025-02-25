# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Optional

import psutil
import torch

from vllm.logger import init_logger

from .interface import Platform, PlatformEnum, _Backend

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class CpuPlatform(Platform):
    _enum = PlatformEnum.CPU
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "cpu"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
                             dtype: torch.dtype, kv_cache_dtype: Optional[str],
                             block_size: int, use_v1: bool,
                             use_mla: bool) -> str:
        if selected_backend and selected_backend != _Backend.TORCH_SDPA:
            logger.info("Cannot use %s backend on CPU.", selected_backend)
        logger.info("Using Torch SDPA backend.")
        return "vllm.attention.backends.torch_sdpa.TorchSDPABackend"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        return psutil.virtual_memory().total

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return False

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        import vllm.envs as envs
        from vllm.utils import GiB_bytes
        model_config = vllm_config.model_config
        # Reminder: Please update docs/source/features/compatibility_matrix.md
        # If the feature combo become valid
        if not model_config.enforce_eager:
            logger.warning(
                "CUDA graph is not supported on CPU, fallback to the eager "
                "mode.")
            model_config.enforce_eager = True

        if model_config.enable_sleep_mode:
            logger.warning(
                "sleep mode is not supported on CPU, disable it.")
            model_config.enable_sleep_mode = False

        cache_config = vllm_config.cache_config

        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        kv_cache_space = envs.VLLM_CPU_KVCACHE_SPACE

        if kv_cache_space >= 0:
            if kv_cache_space == 0:
                cache_config.cpu_kvcache_space_bytes = 4 * GiB_bytes  # type: ignore
                logger.warning(
                    "Environment variable VLLM_CPU_KVCACHE_SPACE (GB) "
                    "for CPU backend is not set, using 4 by default.")
            else:
                cache_config.cpu_kvcache_space_bytes = kv_cache_space * GiB_bytes  # type: ignore # noqa
        else:
            raise RuntimeError(
                "Invalid environment variable VLLM_CPU_KVCACHE_SPACE"
                f" {kv_cache_space}, expect a positive integer value.")

        scheduler_config = vllm_config.scheduler_config
        if ((scheduler_config.chunked_prefill_enabled
             or cache_config.enable_prefix_caching)
                and model_config.dtype == torch.half):
            logger.warning("Chunked-prefill on the CPU backend only does not"
                           " support fp16 for now, cast to bf16.")
            model_config.dtype = torch.bfloat16

        parallel_config = vllm_config.parallel_config
        if (parallel_config.distributed_executor_backend is None 
                or parallel_config.distributed_executor_backend != "mp"):
            parallel_config.distributed_executor_backend = "mp"

        if parallel_config.worker_cls == "auto":
            if vllm_config.speculative_config:
                parallel_config.worker_cls = \
                    "vllm.spec_decode.spec_decode_worker.create_spec_worker"
                parallel_config.sd_worker_cls = \
                    "vllm.worker.cpu_worker.CPUWorker"
            else:
                if envs.VLLM_USE_V1:
                    parallel_config.worker_cls = \
                        "vllm.v1.worker.cpu_worker.CPUWorker"
                else:
                    parallel_config.worker_cls = \
                        "vllm.worker.cpu_worker.CPUWorker"

        # Note: workaround for v1 gpu_model_runner
        vllm_config.compilation_config.cudagraph_capture_sizes = []

        from vllm.config import CompilationLevel
        compilation_config = vllm_config.compilation_config
        if compilation_config.level == CompilationLevel.DYNAMO_ONCE:
            # O2 level
            compilation_config.backend = "eager"
        elif compilation_config.level == CompilationLevel.PIECEWISE:
            # O3 level
            compilation_config.level = CompilationLevel.DYNAMO_ONCE
            compilation_config.backend = "inductor"
            compilation_config.custom_ops += ["none"]
            compilation_config.inductor_compile_config = {
                "dce": True,
                "size_asserts": False,
                "nan_asserts": False,
                "memory_planning": True,
                "max_autotune": True,
                "epilogue_fusion": True,
            }

        assert vllm_config.device_config.device_type == "cpu"

        #
        # Environment variables for CPU executor
        #

        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # Set default threads num for OpenMP parallel
        os.environ["OMP_NUM_THREADS"] = str(torch.get_num_threads())

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # MLA attention is not supported
        os.environ["VLLM_MLA_DISABLE"] = "1"

        # Note: to avoid the error 'nthreads cannot be larger than environment
        #  variable "NUMEXPR_MAX_THREADS" (64)'.
        os.environ["NUMEXPR_MAX_THREADS"] = str(len(os.sched_getaffinity(0)))

        # Intel OpenMP setting
        ld_prealod_str = os.getenv("LD_PRELOAD", "")
        if "libiomp5.so" in ld_prealod_str:
            # The time(milliseconds) that a thread should wait after
            # completing the execution of a parallel region, before sleeping.
            os.environ['KMP_BLOCKTIME'] = "1"
            # Prevents the CPU to run into low performance state
            os.environ['KMP_TPAUSE'] = "0"
            # Provides fine granularity parallelism
            os.environ['KMP_FORKJOIN_BARRIER_PATTERN'] = "dist,dist"
            os.environ['KMP_PLAIN_BARRIER_PATTERN'] = "dist,dist"
            os.environ['KMP_REDUCTION_BARRIER_PATTERN'] = "dist,dist"

        # To hint IPEX uses shared memory based AllReduce
        os.environ["LOCAL_WORLD_SIZE"] = str(
            vllm_config.parallel_config.tensor_parallel_size)

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on CPU.")
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return "vllm.lora.punica_wrapper.punica_cpu.PunicaWrapperCPU"

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """
        Get device specific communicator class for distributed communication.
        """
        return "vllm.distributed.device_communicators.cpu_communicator.CpuCommunicator"  # noqa
