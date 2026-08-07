"""Correctness + timing-methodology sanity checks for the oneDNN vs SGLang
AMX linear-kernel benchmark. Run before trusting any `bench_linear_kernels.py`
sweep — see plan Verification items 1-3.

Standalone script, not part of the vLLM package.
"""

import sys

import torch
import torch.utils.benchmark as torch_benchmark

from bench_linear_kernels import make_tensors, run_onednn_trial, run_sgl_trial

from vllm import _custom_ops as ops
from vllm.model_executor.layers.utils import check_cpu_sgl_kernel

# (k, n) pairs: small, a real model shape (Llama-3.1-8B down_proj K/N truncated
# to satisfy AMX's k%32==0, n%16==0), and a large AMX-friendly square shape.
REPRESENTATIVE_SHAPES = [
    (256, 128),
    (14336, 4096),
    (4096, 4096),
]
M_VALUES = [1, 64, 512]
DTYPE = torch.bfloat16


def check_correctness():
    print("=== 1. Correctness guardrail (oneDNN & SGL vs fp32 reference) ===")
    for k, n in REPRESENTATIVE_SHAPES:
        for m in M_VALUES:
            for use_bias in (True, False):
                weight, x, bias, bias_f32 = make_tensors(m, k, n, DTYPE, use_bias)

                handler = ops.create_onednn_mm(weight.t(), 32)
                onednn_out = ops.onednn_mm(handler, x, bias)

                ref = torch.nn.functional.linear(
                    x.float(),
                    weight.float(),
                    bias.float() if bias is not None else None,
                ).to(DTYPE)
                torch.testing.assert_close(onednn_out, ref, rtol=5e-2, atol=5e-2)

                sgl_ok = check_cpu_sgl_kernel(n, k, DTYPE)
                if sgl_ok:
                    packed = torch.ops._C.convert_weight_packed(weight)
                    sgl_out = torch.ops._C.weight_packed_linear(
                        x, packed, bias_f32, True
                    )
                    torch.testing.assert_close(sgl_out, ref, rtol=8e-2, atol=8e-2)

                print(
                    f"  K={k:6d} N={n:6d} M={m:5d} bias={use_bias!s:5s} "
                    f"onednn=OK sgl={'OK' if sgl_ok else 'skipped (not AMX-eligible)'}"
                )
    print("Correctness guardrail PASSED\n")


def check_timing_methodology():
    print("=== 2. Timing-methodology cross-check (perf_counter_ns vs torch.utils.benchmark) ===")
    k, n, m = 4096, 4096, 512
    weight, x, bias, bias_f32 = make_tensors(m, k, n, DTYPE, True)
    handler = ops.create_onednn_mm(weight.t(), 32)

    manual_ms = run_onednn_trial(m, k, n, DTYPE, True, warmup=5, iters=20)

    timer = torch_benchmark.Timer(
        stmt="ops.onednn_mm(handler, x, bias)",
        globals={"ops": ops, "handler": handler, "x": x, "bias": bias},
        num_threads=torch.get_num_threads(),
    )
    bench_ms = timer.timeit(20).median * 1e3

    ratio = manual_ms / bench_ms
    print(f"  perf_counter_ns median: {manual_ms:.4f} ms")
    print(f"  torch.utils.benchmark median: {bench_ms:.4f} ms")
    print(f"  ratio: {ratio:.3f}")
    if not (0.5 <= ratio <= 2.0):
        print(
            "  WARNING: timings disagree by more than 2x — check for "
            "something being timed inside the loop that shouldn't be "
            "(e.g. weight packing).",
            file=sys.stderr,
        )
    else:
        print("  agreement within noise (0.5x-2x) OK\n")


def check_plausibility():
    print("=== 3. Plausibility check (large AMX-friendly shape) ===")
    k, n, m = 4096, 4096, 4096
    if not check_cpu_sgl_kernel(n, k, DTYPE):
        print("  SKIPPED: this machine doesn't support the AMX SGL kernel path")
        return

    onednn_ms = run_onednn_trial(m, k, n, DTYPE, True, warmup=5, iters=20)
    sgl_ms = run_sgl_trial(m, k, n, DTYPE, True, warmup=5, iters=20)
    flops = 2 * m * n * k
    onednn_tflops = flops / (onednn_ms * 1e-3) / 1e12
    sgl_tflops = flops / (sgl_ms * 1e-3) / 1e12

    print(f"  M=K=N=4096: oneDNN {onednn_ms:.4f} ms ({onednn_tflops:.2f} TFLOP/s)")
    print(f"  M=K=N=4096: SGL    {sgl_ms:.4f} ms ({sgl_tflops:.2f} TFLOP/s)")
    print(
        "  Eyeball check: SGL should win here and its TFLOP/s should be in "
        "the right order of magnitude for this machine's AMX bf16 peak "
        "(low hundreds of TFLOP/s per core-group on current Xeon AMX, not "
        "single digits and not >1 PFLOP/s)."
    )


if __name__ == "__main__":
    check_correctness()
    check_timing_methodology()
    check_plausibility()
    print("\nAll sanity checks completed.")
