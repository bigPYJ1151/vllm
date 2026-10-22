# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark a single static batch step by step, without a server.

All requests are prefilled in one engine step, then decoded one engine step at
a time until they finish. Each step is timed, so prefill latency and per-step
decode latency are reported separately. One warmup run precedes the measured
run.

The headline numbers time only the model executor: the worker-side input
preparation, model forward, and sampling. The scheduler, output processing,
API server, detokenization and streaming are excluded. The full engine step
time, which includes the scheduler and output processing, is reported
alongside as `engine_*`.
"""

import argparse
import json
import os
import time
from typing import Any

import numpy as np

from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TokensPrompt
from vllm.utils.argparse_utils import FlexibleArgumentParser


def add_cli_args(parser: FlexibleArgumentParser):
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--warmup-output-len",
        type=int,
        default=32,
        help="Output length of the warmup run (capped by --output-len).",
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


class ModelStepTimer:
    """Times the model executor part of each engine step.

    A step spans from the start of `execute_model` to the end of
    `sample_tokens`, excluding the scheduler and output processing.
    """

    def __init__(self, llm):
        core = getattr(llm.llm_engine.engine_core, "engine_core", None)
        if core is None:
            raise RuntimeError(
                "Timing the model executor needs the in-process engine core; "
                "run with VLLM_ENABLE_V1_MULTIPROCESSING=0"
            )
        self.start: float | None = None
        self.elapsed: float | None = None
        executor = core.model_executor
        execute_model, sample_tokens = executor.execute_model, executor.sample_tokens

        def timed_execute_model(*args, **kwargs):
            self.start = time.perf_counter()
            return execute_model(*args, **kwargs)

        def timed_sample_tokens(*args, **kwargs):
            output = sample_tokens(*args, **kwargs)
            assert self.start is not None
            self.elapsed = time.perf_counter() - self.start
            return output

        executor.execute_model = timed_execute_model
        executor.sample_tokens = timed_sample_tokens

    def take(self) -> float:
        assert self.elapsed is not None, "no model step was timed"
        elapsed, self.elapsed = self.elapsed, None
        return elapsed


def timed_step(engine, timer: ModelStepTimer):
    tic = time.perf_counter()
    outputs = engine.step()
    engine_latency = time.perf_counter() - tic
    return outputs, timer.take(), engine_latency


def summarize(
    prefix: str,
    prefill: float,
    decodes: list[float],
    batch_size: int,
    output_len: int,
) -> dict[str, float]:
    total = prefill + sum(decodes)
    result = {
        f"{prefix}prefill_latency": prefill,
        f"{prefix}total_latency": total,
        f"{prefix}overall_throughput": output_len * batch_size / total,
    }
    if decodes:
        med = float(np.median(decodes))
        result[f"{prefix}median_decode_latency"] = med
        result[f"{prefix}median_decode_throughput"] = batch_size / med
    return result


def run_once(
    llm,
    timer: ModelStepTimer,
    prompts: list[TokensPrompt],
    output_len: int,
    temperature: float,
    log_decode_steps: int,
    verbose: bool,
) -> dict[str, Any]:
    from vllm import SamplingParams

    engine = llm.llm_engine
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=output_len,
        detokenize=False,
    )
    batch_size = len(prompts)
    for i, prompt in enumerate(prompts):
        engine.add_request(str(i), prompt, sampling_params)

    outputs, prefill_latency, engine_prefill_latency = timed_step(engine, timer)
    # Every request must have produced its first token in this step.
    assert len(outputs) == batch_size, (
        f"prefill did not complete in one step ({len(outputs)}/{batch_size} "
        "requests emitted a token); raise --max-num-batched-tokens/--max-num-seqs "
        "or reduce the batch"
    )
    if verbose:
        print(
            f"Prefill. latency: {prefill_latency:6.5f} s, "
            f"throughput: {batch_size / prefill_latency:9.2f} token/s"
        )

    decode_latencies: list[float] = []
    engine_decode_latencies: list[float] = []
    while engine.has_unfinished_requests():
        _, latency, engine_latency = timed_step(engine, timer)
        decode_latencies.append(latency)
        engine_decode_latencies.append(engine_latency)
        i = len(decode_latencies) - 1
        if verbose and i < log_decode_steps:
            print(
                f"Decode {i}. Batch size: {batch_size}, latency: {latency:6.5f} s, "
                f"throughput: {batch_size / latency:9.2f} token/s"
            )

    result: dict[str, Any] = {"batch_size": batch_size, "output_len": output_len}
    result.update(
        summarize("", prefill_latency, decode_latencies, batch_size, output_len)
    )
    result.update(
        summarize(
            "engine_",
            engine_prefill_latency,
            engine_decode_latencies,
            batch_size,
            output_len,
        )
    )
    if verbose:
        if decode_latencies:
            print(
                f"Decode.  median latency: {result['median_decode_latency']:6.5f} s, "
                f"median throughput: {result['median_decode_throughput']:9.2f} token/s"
            )
        print(
            f"Total. latency: {result['total_latency']:6.3f} s, "
            f"throughput: {result['overall_throughput']:9.2f} token/s"
        )
        print(
            "Engine step (incl. scheduler and output processing). "
            f"prefill: {result['engine_prefill_latency']:6.5f} s, "
            f"decode median: {result.get('engine_median_decode_latency', 0):6.5f} s"
        )
    return result


def check_kv_cache_capacity(llm, batch_size: int) -> None:
    """Fail before running if the KV cache cannot hold the whole batch.

    The concurrency is computed for requests of `max_model_len` tokens, which
    defaults to `input_len + output_len`.
    """
    from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config

    engine = llm.llm_engine
    core = getattr(engine.engine_core, "engine_core", None)
    if core is None:
        raise RuntimeError(
            "KV cache check needs the in-process engine core; "
            "run with VLLM_ENABLE_V1_MULTIPROCESSING=0"
        )
    max_concurrency = get_max_concurrency_for_kv_cache_config(
        engine.vllm_config, core.scheduler.kv_cache_config
    )
    print(
        f"KV cache holds {max_concurrency:.1f} concurrent requests of "
        f"{engine.model_config.max_model_len} tokens."
    )
    if max_concurrency < batch_size:
        raise RuntimeError(
            f"KV cache holds only {max_concurrency:.1f} requests of "
            f"{engine.model_config.max_model_len} tokens, less than batch size "
            f"{batch_size}; reduce --batch-size or increase the KV cache size"
        )


def main(args: argparse.Namespace):
    # Run the engine core in-process so a step does not include IPC to it.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # A step must map to one scheduling round, and the whole batch must fit in
    # one prefill step.
    if args.async_scheduling is None:
        args.async_scheduling = False
    if args.max_num_batched_tokens is None:
        args.max_num_batched_tokens = max(8192, args.batch_size * args.input_len)
    if args.max_num_seqs is None:
        args.max_num_seqs = args.batch_size
    if args.max_model_len is None:
        args.max_model_len = args.input_len + args.output_len

    engine_args = EngineArgs.from_cli_args(args)

    from vllm import LLM

    llm = LLM.from_engine_args(engine_args)
    assert llm.llm_engine.model_config.max_model_len >= (
        args.input_len + args.output_len
    ), "max_model_len must be at least input_len + output_len"
    check_kv_cache_capacity(llm, args.batch_size)
    timer = ModelStepTimer(llm)

    rng = np.random.default_rng(0)
    token_ids = rng.integers(10000, size=(args.batch_size, args.input_len))
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in token_ids.tolist()]

    print("Warmup ...")
    run_once(
        llm,
        timer,
        prompts,
        min(args.warmup_output_len, args.output_len),
        args.temperature,
        args.log_decode_steps,
        verbose=True,
    )
    print("Benchmark ...")
    result = run_once(
        llm,
        timer,
        prompts,
        args.output_len,
        args.temperature,
        args.log_decode_steps,
        verbose=True,
    )
    result["input_len"] = args.input_len

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=4)
