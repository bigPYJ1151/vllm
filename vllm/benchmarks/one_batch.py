# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark a single static batch step by step, without a server or scheduler.

Like `vllm bench forward`, the batch is built by hand and fed to the model
runner's `execute_model` and `sample_tokens`. The whole batch is prefilled in one
step, then decoded one step at a time. Each step is timed, so prefill latency
and per-step decode latency are reported separately. Warmup runs precede the
measured run.

Only the model runner is timed: input preparation, model forward and sampling.
The scheduler, output processing, API server, detokenization and streaming are
not involved.

One process per rank: `torchrun --nproc-per-node=N -m vllm.entrypoints.cli.main
bench one-batch --tensor-parallel-size N ...` for multiple ranks. On CPU, each
rank's OpenMP threads and memory node are bound like a spawned CPU worker
(`VLLM_CPU_OMP_THREADS_BIND`, `VLLM_CPU_BIND_GRANULARITY`).
"""

import argparse
import json
import os
import time
from typing import Any

import numpy as np
import torch

from vllm.benchmarks.forward.batch import BenchBatch, build_batch
from vllm.benchmarks.forward.shapes import BenchCase, BenchRequest
from vllm.benchmarks.lib.direct_runner import (
    bind_cpu_threads,
    get_model_runner,
    init_rank_env,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput


def add_cli_args(parser: FlexibleArgumentParser):
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--warmup-output-len",
        type=int,
        default=32,
        help="Output length of the warmup runs (capped by --output-len).",
    )
    parser.add_argument(
        "--num-warmup-runs",
        type=int,
        default=2,
        help="Warmup runs before the measured one. The runner's Triton input "
        "kernels specialize on argument values such as the request slot, so "
        "the second run can still compile a variant.",
    )
    parser.add_argument(
        "--log-decode-steps",
        type=int,
        default=5,
        help="Print the latency of the first N decode steps.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save the results in JSON format.",
    )

    parser = EngineArgs.add_cli_args(parser)
    parser.set_defaults(enable_prefix_caching=False)


def decode_step(batch: BenchBatch, input_len: int, step: int) -> SchedulerOutput:
    """The step after `step` decodes: one token per request."""
    req_ids = batch.req_ids
    num_reqs = len(req_ids)
    output = SchedulerOutput.make_empty()
    output.scheduled_cached_reqs = CachedRequestData(
        req_ids=req_ids,
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[None] * num_reqs,
        num_computed_tokens=[input_len + step] * num_reqs,
        num_output_tokens=[step + 1] * num_reqs,
    )
    output.num_scheduled_tokens = dict.fromkeys(req_ids, 1)
    output.total_num_scheduled_tokens = num_reqs
    return output


def timed_step(runner, scheduler_output: SchedulerOutput) -> float:
    """Time input preparation, forward and sampling of one step."""
    torch.accelerator.synchronize()
    tic = time.perf_counter()
    runner.execute_model(scheduler_output)
    runner.sample_tokens(None)
    torch.accelerator.synchronize()
    return time.perf_counter() - tic


def reduce_max(latencies: list[float]) -> list[float]:
    """Slowest rank per step: collectives make ranks wait for each other."""
    from vllm.distributed.parallel_state import get_world_group

    world = get_world_group()
    if world.world_size == 1:
        return latencies
    values = torch.tensor(latencies, dtype=torch.float64)
    torch.distributed.all_reduce(
        values, op=torch.distributed.ReduceOp.MAX, group=world.cpu_group
    )
    return values.tolist()


def run_once(
    runner,
    batch: BenchBatch,
    input_len: int,
    output_len: int,
) -> dict[str, Any]:
    """Prefill, then `output_len - 1` decode steps; finish the requests."""
    batch_size = len(batch.req_ids)
    try:
        latencies = [timed_step(runner, batch.scheduler_output)]
        for step in range(output_len - 1):
            latencies.append(timed_step(runner, decode_step(batch, input_len, step)))
    finally:
        runner.execute_model(batch.finish())
    latencies = reduce_max(latencies)
    prefill, decodes = latencies[0], latencies[1:]
    total = sum(latencies)
    result: dict[str, Any] = {
        "batch_size": batch_size,
        "input_len": input_len,
        "output_len": output_len,
        "prefill_latency": prefill,
        "total_latency": total,
        "overall_throughput": output_len * batch_size / total,
        "decode_latencies": decodes,
    }
    if decodes:
        med = float(np.median(decodes))
        result["median_decode_latency"] = med
        result["median_decode_throughput"] = batch_size / med
    return result


def print_result(result: dict[str, Any], log_decode_steps: int) -> None:
    batch_size = result["batch_size"]
    prefill = result["prefill_latency"]
    print(
        f"Prefill. latency: {prefill:6.5f} s, "
        f"throughput: {batch_size / prefill:9.2f} token/s"
    )
    for i, latency in enumerate(result["decode_latencies"][:log_decode_steps]):
        print(
            f"Decode {i}. Batch size: {batch_size}, latency: {latency:6.5f} s, "
            f"throughput: {batch_size / latency:9.2f} token/s"
        )
    if "median_decode_latency" in result:
        print(
            f"Decode.  median latency: {result['median_decode_latency']:6.5f} s, "
            f"median throughput: {result['median_decode_throughput']:9.2f} token/s"
        )
    print(
        f"Total. latency: {result['total_latency']:6.3f} s, "
        f"throughput: {result['overall_throughput']:9.2f} token/s"
    )


def main(args: argparse.Namespace):
    # Model Runner V2 is the runner CPU and GPU share; the runner is driven
    # in-process, so no engine core process sits between the bench and it.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    init_rank_env()

    # A step is one runner call, and the whole batch fits in one prefill step.
    if args.async_scheduling is None:
        args.async_scheduling = False
    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = max(8192, args.batch_size * args.input_len)
    if args.max_num_seqs is None:
        args.max_num_seqs = args.batch_size
    if args.max_model_len is None:
        args.max_model_len = args.input_len + args.output_len

    engine_args = EngineArgs.from_cli_args(args)
    engine_args.distributed_executor_backend = "external_launcher"

    bind_cpu_threads(engine_args)
    from vllm import LLM, SamplingParams
    from vllm.distributed.parallel_state import get_world_group

    llm = LLM.from_engine_args(engine_args)
    runner = get_model_runner(llm)
    assert runner.model_config.max_model_len >= args.input_len + args.output_len, (
        "max_model_len must be at least input_len + output_len"
    )
    rank = get_world_group().rank
    case = BenchCase(
        "one-batch",
        (BenchRequest(query_len=args.input_len, kv_len=args.input_len),)
        * args.batch_size,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    def new_batch() -> BenchBatch:
        kv_config = runner.kv_cache_config
        return build_batch(
            case,
            kv_config.kv_cache_groups,
            kv_config.num_blocks,
            runner.model_config.get_vocab_size(),
            np.random.default_rng(args.seed),
            runner.parallel_config.decode_context_parallel_size,
            reserve_len=args.input_len + args.output_len,
            sampling_params=sampling_params,
        )

    with torch.inference_mode():
        if rank == 0:
            print("Warmup ...")
        for _ in range(args.num_warmup_runs):
            run_once(
                runner,
                new_batch(),
                args.input_len,
                min(args.warmup_output_len, args.output_len),
            )
        if rank == 0:
            print("Benchmark ...")
        result = run_once(runner, new_batch(), args.input_len, args.output_len)

    if rank != 0:
        return
    print_result(result, args.log_decode_steps)
    if args.output_json:
        result.pop("decode_latencies")
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=4)
