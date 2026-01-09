# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time

import torch

from vllm._custom_ops import cpu_gemm_wna16

if __name__ == "__main__":
    torch.ops._C_utils.init_cpu_threads_env("99")
    torch._C._cpu._init_amx()

    M = 32
    N = 9728
    K = 2560
    OUTER_iteration = 1
    Iteration = 1024 * 32

    assert N % 32 == 0
    assert K % 32 == 0

    input_shape = (M, K)
    weight_shape = (N // 16, K * 16 // 8)
    zero_shape = (1, N // 8)
    scale_shape = (1, N)

    inputs = torch.zeros((Iteration, *input_shape), dtype=torch.bfloat16)
    weights = torch.zeros((Iteration, *weight_shape), dtype=torch.int32)
    zeros = torch.zeros((Iteration, *zero_shape), dtype=torch.int32)
    scales = torch.zeros((Iteration, *scale_shape), dtype=torch.bfloat16)

    start_ns = time.perf_counter_ns()
    for j in range(OUTER_iteration):
        for i in range(Iteration):
            cpu_gemm_wna16(
                inputs[i],
                weights[i],
                scales[i],
                zeros[i],
                None,
                None,
                8,
                "amx",
            )
    end_ns = time.perf_counter_ns()
    duration_ns = (end_ns - start_ns) / (Iteration * OUTER_iteration)

    FLOPs = 2 * M * N * K
    Mems = M * K * 2 + N * K // 2
    FLOPS_G = FLOPs / duration_ns
    Bandwidth_G = Mems / duration_ns

    print(
        f"AWQ Gemm, duration: {duration_ns}, "
        f"FLOPs: {FLOPs}, Mems: {Mems},"
        f" FLOPS: {FLOPS_G}, Bandwidth: {Bandwidth_G}"
    )
