"""oneDNN vs SGLang AMX unquantized-linear GEMM kernel benchmark (CPU).

Standalone script, not part of the vLLM package. Compares vLLM's two
competing unquantized (bf16/fp16) CPU GEMM backends —
`ops.onednn_mm`/`ops.create_onednn_mm` and SGLang's vendored
`torch.ops._C.weight_packed_linear`/`convert_weight_packed` — at the
production call pattern used by `dispatch_cpu_unquantized_gemm`
(vllm/model_executor/layers/utils.py), across the linear shapes in
model_shapes.py.

Run under `numactl` via run_sweep.sh, not directly, so `--cpu-binding-tag`
reflects the actual pinning.
"""

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from model_shapes import MODEL_TABLE, lm_head_m_sweep, m_sweep, shard

from vllm import _custom_ops as ops
from vllm.model_executor.layers.utils import check_cpu_sgl_kernel
from vllm.utils.torch_utils import set_random_seed

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}
ALL_LINEAR_TYPES = ["qkv", "o_proj", "gate_up", "down", "moe_gate", "lm_head"]

CSV_COLUMNS = [
    "model",
    "linear_type",
    "tp_size",
    "dtype",
    "M",
    "K",
    "N",
    "onednn_geomean_ms",
    "onednn_min_ms",
    "onednn_max_ms",
    "onednn_tflops",
    "sgl_geomean_ms",
    "sgl_min_ms",
    "sgl_max_ms",
    "sgl_tflops",
    "sgl_skipped_reason",
    "speedup_onednn_over_sgl",
    "winner",
    "arithmetic_intensity_flops_per_byte",
]

MD_HEADER = (
    "Model | Linear | TP | dtype | M | K | N | oneDNN (ms) | SGL (ms) | "
    "Speedup (oneDNN/SGL) | Winner | TFLOP/s (oneDNN) | TFLOP/s (SGL) | "
    "AI (FLOPs/B)"
)


def geo_mean(values: list[float]) -> float:
    logs = [math.log(v) for v in values]
    return math.exp(sum(logs) / len(logs))


def arithmetic_intensity(m: int, k: int, n: int, dtype: torch.dtype) -> float:
    """FLOPs per byte of naive (no-reuse) memory traffic for one X@W'+b call.

    FLOPs = 2*M*N*K (multiply-add). Bytes = elem_size * (M*K + N*K + M*N),
    i.e. one full read of x and weight plus one write of the output; bias
    (O(N) elements) is negligible and excluded. bf16 and fp16 both use
    2-byte elements, so this ratio is identical for both dtypes tested here.
    This is the compulsory-traffic bound (assumes no cache reuse of the
    weight across calls) — a lower bound on intensity, not a measurement.
    """
    elem_bytes = torch.finfo(dtype).bits // 8
    flops = 2 * m * n * k
    bytes_accessed = elem_bytes * (m * k + n * k + m * n)
    return flops / bytes_accessed


def build_case_list(
    model_names: list[str] | None,
    linear_types: list[str],
    tp_sizes: list[int],
    m_list_override: list[int] | None,
):
    models = MODEL_TABLE
    if model_names:
        models = [m for m in models if m.name in model_names]
        if not models:
            raise SystemExit(f"no models matched --models {model_names}")

    cases = []
    for model in models:
        in_scope = [lt for lt in model.in_scope if lt in linear_types]
        for lt in in_scope:
            for tp in tp_sizes:
                k, n = shard(model, lt, tp)
                if m_list_override is not None:
                    ms = m_list_override
                elif lt == "lm_head":
                    ms = lm_head_m_sweep(tp)
                else:
                    ms = m_sweep(tp)
                for m in ms:
                    cases.append(
                        {
                            "model": model.name,
                            "linear_type": lt,
                            "tp_size": tp,
                            "K": k,
                            "N": n,
                            "M": m,
                        }
                    )
    return cases


def make_tensors(m: int, k: int, n: int, dtype: torch.dtype, use_bias: bool):
    weight = torch.randn((n, k), dtype=dtype) / (0.5 * k**0.5)
    x = torch.randn((m, k), dtype=dtype) / (0.5 * k**0.5)
    bias = None
    bias_f32 = None
    if use_bias:
        bias = torch.randn((n,), dtype=dtype) / (0.5 * n**0.5)
        bias_f32 = bias.float()
    return weight, x, bias, bias_f32


def time_calls(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        start = time.perf_counter_ns()
        fn()
        end = time.perf_counter_ns()
        times.append((end - start) / 1e6)
    times.sort()
    mid = len(times) // 2
    return times[mid] if len(times) % 2 else (times[mid - 1] + times[mid]) / 2


def run_onednn_trial(
    m: int, k: int, n: int, dtype: torch.dtype, use_bias: bool, warmup: int, iters: int
):
    weight, x, bias, _ = make_tensors(m, k, n, dtype, use_bias)
    handler = ops.create_onednn_mm(weight.t(), 32)
    return time_calls(lambda: ops.onednn_mm(handler, x, bias), warmup, iters)


def run_sgl_trial(
    m: int, k: int, n: int, dtype: torch.dtype, use_bias: bool, warmup: int, iters: int
):
    weight, x, _, bias_f32 = make_tensors(m, k, n, dtype, use_bias)
    packed = torch.ops._C.convert_weight_packed(weight)
    return time_calls(
        lambda: torch.ops._C.weight_packed_linear(x, packed, bias_f32, True),
        warmup,
        iters,
    )


def correctness_guardrail(k: int, n: int, dtype: torch.dtype, use_bias: bool):
    m = 64
    weight, x, bias, bias_f32 = make_tensors(m, k, n, dtype, use_bias)

    handler = ops.create_onednn_mm(weight.t(), 32)
    onednn_out = ops.onednn_mm(handler, x, bias)

    packed = torch.ops._C.convert_weight_packed(weight)
    sgl_out = torch.ops._C.weight_packed_linear(x, packed, bias_f32, True)

    ref = torch.nn.functional.linear(
        x.float(), weight.float(), bias.float() if bias is not None else None
    ).to(dtype)

    torch.testing.assert_close(onednn_out, ref, rtol=5e-2, atol=5e-2)
    torch.testing.assert_close(sgl_out, ref, rtol=8e-2, atol=8e-2)
    print(
        f"[guardrail] onednn and sgl both match fp32 reference "
        f"(K={k}, N={n}, dtype={dtype}, M={m})",
        file=sys.stderr,
    )


def run_case(
    case: dict,
    dtype_name: str,
    use_bias: bool,
    repeats: int,
    warmup: int,
    iters: int,
    seed: int,
    kernels: list[str],
) -> dict:
    dtype = DTYPE_MAP[dtype_name]
    k, n, m = case["K"], case["N"], case["M"]
    flops = 2 * m * n * k

    row = {
        "model": case["model"],
        "linear_type": case["linear_type"],
        "tp_size": case["tp_size"],
        "dtype": dtype_name,
        "M": m,
        "K": k,
        "N": n,
        "onednn_geomean_ms": None,
        "onednn_min_ms": None,
        "onednn_max_ms": None,
        "onednn_tflops": None,
        "sgl_geomean_ms": None,
        "sgl_min_ms": None,
        "sgl_max_ms": None,
        "sgl_tflops": None,
        "sgl_skipped_reason": None,
        "speedup_onednn_over_sgl": None,
        "winner": None,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity(m, k, n, dtype),
    }

    onednn_geomean = None
    if "onednn" in kernels:
        onednn_trials = []
        for r in range(repeats):
            set_random_seed(seed + r)
            onednn_trials.append(
                run_onednn_trial(m, k, n, dtype, use_bias, warmup, iters)
            )
        onednn_geomean = geo_mean(onednn_trials)
        row["onednn_geomean_ms"] = onednn_geomean
        row["onednn_min_ms"] = min(onednn_trials)
        row["onednn_max_ms"] = max(onednn_trials)
        row["onednn_tflops"] = flops / (onednn_geomean * 1e-3) / 1e12

    if "sgl" not in kernels:
        row["sgl_skipped_reason"] = "sgl excluded via --kernels"
        row["winner"] = "onednn (sgl n/a)" if onednn_geomean is not None else None
        return row

    if not check_cpu_sgl_kernel(n, k, dtype):
        row["sgl_skipped_reason"] = "check_cpu_sgl_kernel=False (k%32!=0 or n%16!=0)"
        row["winner"] = "onednn (sgl n/a)" if onednn_geomean is not None else None
        return row

    sgl_trials = []
    for r in range(repeats):
        set_random_seed(seed + r)
        sgl_trials.append(run_sgl_trial(m, k, n, dtype, use_bias, warmup, iters))
    sgl_geomean = geo_mean(sgl_trials)
    sgl_tflops = flops / (sgl_geomean * 1e-3) / 1e12

    row["sgl_geomean_ms"] = sgl_geomean
    row["sgl_min_ms"] = min(sgl_trials)
    row["sgl_max_ms"] = max(sgl_trials)
    row["sgl_tflops"] = sgl_tflops

    if onednn_geomean is not None:
        speedup = sgl_geomean / onednn_geomean  # >1 means onednn is faster
        row["speedup_onednn_over_sgl"] = speedup
        row["winner"] = (
            "onednn" if speedup > 1.02 else ("sgl" if speedup < 0.98 else "tie")
        )
    else:
        row["winner"] = "sgl (onednn n/a)"
    return row


def fmt(v, digits=3):
    return "—" if v is None else f"{v:.{digits}f}"


def write_outputs(rows: list[dict], output_dir: Path, build_tag: str, cpu_binding_tag: str):
    tag = f"{build_tag}__{cpu_binding_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_sorted = sorted(
        rows,
        key=lambda r: (r["linear_type"], r["model"], r["tp_size"], r["dtype"], r["M"]),
    )

    csv_path = output_dir / f"{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows_sorted:
            writer.writerow({k: row.get(k) for k in CSV_COLUMNS})

    md_path = output_dir / f"{tag}.md"
    with open(md_path, "w") as f:
        f.write(f"# Results: {build_tag} / {cpu_binding_tag}\n\n")
        f.write(f"{MD_HEADER}\n")
        f.write("|" + "---|" * len(MD_HEADER.split("|")) + "\n")
        for row in rows_sorted:
            f.write(
                f"{row['model']} | {row['linear_type']} | {row['tp_size']} | "
                f"{row['dtype']} | {row['M']} | {row['K']} | {row['N']} | "
                f"{fmt(row['onednn_geomean_ms'])} | {fmt(row['sgl_geomean_ms'])} | "
                f"{fmt(row['speedup_onednn_over_sgl'], 3)} | {row['winner']} | "
                f"{fmt(row['onednn_tflops'], 2)} | {fmt(row['sgl_tflops'], 2)} | "
                f"{fmt(row['arithmetic_intensity_flops_per_byte'], 2)}\n"
            )

    return csv_path, md_path


def write_manifest(output_dir: Path, build_tag: str, cpu_binding_tag: str, args_dict: dict):
    tag = f"{build_tag}__{cpu_binding_tag}"

    def _git(*cmd):
        try:
            return subprocess.check_output(
                ["git", *cmd], cwd=Path(__file__).resolve().parent, text=True
            ).strip()
        except Exception as e:  # noqa: BLE001
            return f"<unavailable: {e}>"

    cpus_allowed = "<unavailable>"
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("Cpus_allowed_list:"):
                    cpus_allowed = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    manifest = {
        "build_tag": build_tag,
        "cpu_binding_tag": cpu_binding_tag,
        "args": args_dict,
        "git_rev": _git("rev-parse", "HEAD"),
        "git_diff_stat": _git("diff", "--stat"),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "amx_tile_supported": torch.cpu._is_amx_tile_supported(),
        "cpus_allowed_list_at_runtime": cpus_allowed,
    }
    manifest_path = output_dir / f"manifest_{tag}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=str, default=None, help="comma-separated model names")
    p.add_argument(
        "--linear-types", type=str, default=",".join(ALL_LINEAR_TYPES)
    )
    p.add_argument("--tp-sizes", type=str, default="1,2,4,8")
    p.add_argument("--m-list", type=str, default=None, help="override the per-tp M sweep")
    p.add_argument("--dtype", type=str, default="bf16,fp16")
    p.add_argument("--kernels", type=str, default="onednn,sgl")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--use-bias", dest="use_bias", action="store_true", default=True)
    p.add_argument("--no-use-bias", dest="use_bias", action="store_false")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--build-tag", type=str, required=True)
    p.add_argument("--cpu-binding-tag", type=str, required=True)
    p.add_argument(
        "--output-dir", type=str, default=str(Path(__file__).resolve().parent / "results")
    )
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cpu._is_amx_tile_supported():
        print(
            "WARNING: AMX tile support not detected on this machine — every "
            "SGL case will be skipped.",
            file=sys.stderr,
        )

    model_names = args.models.split(",") if args.models else None
    linear_types = args.linear_types.split(",")
    tp_sizes = [int(x) for x in args.tp_sizes.split(",")]
    dtypes = args.dtype.split(",")
    kernels = args.kernels.split(",")
    m_list_override = [int(x) for x in args.m_list.split(",")] if args.m_list else None

    repeats = args.repeats
    if args.quick:
        tp_sizes = [t for t in tp_sizes if t in (1, 8)] or [1, 8]
        repeats = min(repeats, 2)
        m_list_override = sorted(set(m_list_override or [1, 64, 512]))

    cases = build_case_list(model_names, linear_types, tp_sizes, m_list_override)
    if not cases:
        raise SystemExit("no cases selected — check --models/--linear-types/--tp-sizes")

    print(f"{len(cases)} shape cases x {len(dtypes)} dtypes to run", file=sys.stderr)

    # Correctness guardrail: first AMX-eligible case, bf16 if available.
    guardrail_dtype = DTYPE_MAP["bf16" if "bf16" in dtypes else dtypes[0]]
    for case in cases:
        if check_cpu_sgl_kernel(case["N"], case["K"], guardrail_dtype):
            correctness_guardrail(case["K"], case["N"], guardrail_dtype, args.use_bias)
            break
    else:
        print(
            "WARNING: no case in this sweep is AMX-eligible; skipping guardrail",
            file=sys.stderr,
        )

    rows = []
    total = len(cases) * len(dtypes)
    done = 0
    for dtype_name in dtypes:
        for case in cases:
            row = run_case(
                case,
                dtype_name,
                args.use_bias,
                repeats,
                args.warmup,
                args.iters,
                args.seed,
                kernels,
            )
            rows.append(row)
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  [{done}/{total}] {row['model']} {row['linear_type']} "
                      f"tp={row['tp_size']} dtype={dtype_name} M={row['M']} "
                      f"winner={row['winner']}", file=sys.stderr)

    output_dir = Path(args.output_dir)
    csv_path, md_path = write_outputs(rows, output_dir, args.build_tag, args.cpu_binding_tag)
    manifest_path = write_manifest(
        output_dir, args.build_tag, args.cpu_binding_tag, vars(args)
    )
    print(f"wrote {csv_path}\nwrote {md_path}\nwrote {manifest_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
