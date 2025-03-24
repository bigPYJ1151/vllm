import os
from typing import Optional, Tuple

import torch

from vllm import envs
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.utils import set_random_seed
from vllm.sequence import IntermediateTensors
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.cpu_model_runner import CPUModelRunner
from vllm.v1.worker.gpu_worker import (Worker,
                                       init_worker_distributed_environment)

logger = init_logger(__name__)


class CPUWorker(Worker):

    def __init__(self, vllm_config: VllmConfig, local_rank: int, rank: int,
                 distributed_init_method: str, is_driver_worker: bool = False):
        super().__init__(vllm_config, local_rank, rank,
                         distributed_init_method, is_driver_worker=is_driver_worker)

        self.parallel_config.disable_custom_all_reduce = True

    def init_device(self):
        assert self.device_config.device_type == "cpu"
        #
        # Environment variables for CPU worker
        #

        # For init shm files
        os.environ["VLLM_INSTANCE_ID"] = self.vllm_config.instance_id

        # Disable torch async compiling which won't work with daemonic processes
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

        # Note: unique identifier for creating allreduce shared memory
        os.environ["VLLM_DIST_IDENT"] = self.distributed_init_method.split(
            ":")[-1]

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

        # Setup OpenMP threads affinity.
        omp_cpuids = envs.VLLM_CPU_OMP_THREADS_BIND
        if omp_cpuids == "all":
            self.local_omp_cpuid = "all"
        else:
            self.local_omp_cpuid = omp_cpuids.split("|")[self.rank]
            ret = torch.ops._C_utils.init_cpu_threads_env(self.local_omp_cpuid)
            if ret:
                logger.info(ret)

        # Initialize the distributed environment.
        init_worker_distributed_environment(self.parallel_config, self.rank,
                                            self.distributed_init_method,
                                            self.local_rank, "gloo")
        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Construct the model runner
        self.model_runner: CPUModelRunner = CPUModelRunner(
            self.vllm_config, torch.device("cpu"))

    def sleep(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def wake_up(self, level: int = 1) -> None:
        logger.warning("sleep mode is not supported on CPU, ignore it.")
        pass

    def determine_available_memory(self) -> int:
        return self.cache_config.cpu_kvcache_space_bytes  # type: ignore
        
    def compile_or_warm_up_model(self) -> None:
        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        self.model_runner.warming_up_model()

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> Optional[ModelRunnerOutput]:
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = IntermediateTensors(
                get_pp_group().recv_tensor_dict(
                    all_gather_group=get_tp_group()))

        output = self.model_runner.execute_model(scheduler_output,
                                                 intermediate_tensors)

        if not get_pp_group().is_last_rank:
            assert isinstance(output, IntermediateTensors)
            get_pp_group().send_tensor_dict(output.tensors,
                                            all_gather_group=get_tp_group())
            return None

        assert isinstance(output, ModelRunnerOutput)
        return output if self.is_driver_worker else None


def _get_cache_block_size_bytes(
    cache_config: CacheConfig,
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
) -> int:
    """Return the size in bytes of a single KV cache block.
    """
    head_size = model_config.get_head_size()
    num_heads = model_config.get_num_kv_heads(parallel_config)
    num_layers = model_config.get_num_layers(parallel_config)
    block_size = cache_config.block_size
    cache_dtype = cache_config.cache_dtype

    key_cache_block = block_size * num_heads * head_size
    value_cache_block = key_cache_block
    total = num_layers * (key_cache_block + value_cache_block)
    if cache_dtype == "auto":
        dtype = model_config.dtype
    else:
        dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
    dtype_size = torch.tensor([], dtype=dtype).element_size()

    return dtype_size * total
