"""Aggregate `bench_linear_kernels.py` sweep results into a bucketed
oneDNN-vs-SGL heuristic table, across all (build_tag, cpu_binding_tag)
combinations found in results/.

Standalone script, not part of the vLLM package. Reads every
results/<build-tag>__<cpu-binding-tag>.csv and writes
results/summary_heuristic.md.
"""

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_NAME_RE = re.compile(r"^(?P<build_tag>.+)__(?P<cpu_binding_tag>[^_]+)\.csv$")


def m_bucket(m: int, tp_size: int) -> str:
    decode_default = 256 * tp_size
    prefill_default = 4096 * tp_size
    if m <= decode_default:
        return "decode"
    if m <= prefill_default:
        return "mid"
    return "prefill"


def geo_mean(values: list[float]) -> float:
    logs = [math.log(v) for v in values]
    return math.exp(sum(logs) / len(logs))


def load_all_rows() -> list[dict]:
    rows = []
    for csv_path in sorted(RESULTS_DIR.glob("*__*.csv")):
        m = CSV_NAME_RE.match(csv_path.name)
        if not m:
            continue
        build_tag = m.group("build_tag")
        cpu_binding_tag = m.group("cpu_binding_tag")
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                row["build_tag"] = build_tag
                row["cpu_binding_tag"] = cpu_binding_tag
                rows.append(row)
    return rows


def bucket_key(row: dict):
    return (
        row["build_tag"],
        row["cpu_binding_tag"],
        row["dtype"],
        row["linear_type"],
        row["tp_size"],
        m_bucket(int(row["M"]), int(row["tp_size"])),
    )


def summarize(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    ai_groups = defaultdict(list)
    for row in rows:
        if row.get("arithmetic_intensity_flops_per_byte"):
            ai_groups[bucket_key(row)].append(
                float(row["arithmetic_intensity_flops_per_byte"])
            )
        if not row.get("speedup_onednn_over_sgl"):
            continue  # SGL was skipped for this case; excluded from win-rate/geomean
        groups[bucket_key(row)].append(float(row["speedup_onednn_over_sgl"]))

    summary = []
    for key, speedups in sorted(groups.items()):
        build_tag, cpu_binding_tag, dtype, linear_type, tp_size, bucket = key
        # speedup = sgl_ms / onednn_ms; >1 means onednn is faster.
        wins_onednn = sum(1 for s in speedups if s > 1.02)
        ai_values = sorted(ai_groups.get(key, []))
        median_ai = ai_values[len(ai_values) // 2] if ai_values else None
        summary.append(
            {
                "build_tag": build_tag,
                "cpu_binding_tag": cpu_binding_tag,
                "dtype": dtype,
                "linear_type": linear_type,
                "tp_size": tp_size,
                "m_bucket": bucket,
                "n_cases": len(speedups),
                "sgl_win_rate": 1 - wins_onednn / len(speedups),
                "geomean_speedup_onednn_over_sgl": geo_mean(speedups),
                "recommendation": "onednn" if wins_onednn / len(speedups) >= 0.5 else "sgl",
                "median_ai_flops_per_byte": median_ai,
            }
        )
    return summary


def write_summary_md(summary: list[dict], build_binding_pairs: list[tuple]):
    out_path = RESULTS_DIR / "summary_heuristic.md"
    with open(out_path, "w") as f:
        f.write("# Aggregated oneDNN vs SGL heuristic (bucketed)\n\n")
        f.write(
            "Grouped by (build_tag, cpu_binding_tag, dtype, linear_type, "
            "tp_size, M-bucket); decode = M<=256*tp, mid = 256*tp<M<=4096*tp, "
            "prefill = M>4096*tp. `geomean_speedup` is the geometric mean of "
            "per-case `sgl_ms/onednn_ms` ratios (>1 = oneDNN faster). "
            "`SGL win-rate` is the fraction of individual cases where SGL "
            "had lower latency (can diverge from the geomean's sign when a "
            "few large-margin cases skew the aggregate but most individual "
            "cases lean the other way); `recommendation` follows the "
            "win-rate, not the geomean. Cases where SGL was gated out "
            "(check_cpu_sgl_kernel=False) are excluded from these "
            "aggregates. `median AI` is the median arithmetic intensity "
            "(FLOPs per byte of naive, no-reuse memory traffic — see "
            "`bench_linear_kernels.py::arithmetic_intensity`) across cases "
            "in that bucket; bf16/fp16 share the same ratio since both are "
            "2-byte types.\n\n"
        )
        f.write(
            "build_tag | cpu_binding_tag | dtype | linear | tp | M-bucket | "
            "n | SGL win-rate | geomean speedup (oneDNN/SGL) | recommendation | "
            "median AI (FLOPs/B)\n"
        )
        f.write("|" + "---|" * 11 + "\n")
        for row in summary:
            median_ai = row["median_ai_flops_per_byte"]
            ai_cell = f"{median_ai:.2f}" if median_ai is not None else "—"
            f.write(
                f"{row['build_tag']} | {row['cpu_binding_tag']} | {row['dtype']} | "
                f"{row['linear_type']} | {row['tp_size']} | {row['m_bucket']} | "
                f"{row['n_cases']} | {row['sgl_win_rate']:.2f} | "
                f"{row['geomean_speedup_onednn_over_sgl']:.3f} | "
                f"{row['recommendation']} | "
                f"{ai_cell}\n"
            )

        f.write("\n## Build-flag sensitivity (O2 vs O3, same CPU binding)\n\n")
        f.write(
            "For matching (cpu_binding_tag, dtype, linear_type, tp_size, "
            "M-bucket) keys, compare the geomean speedup across build_tags.\n\n"
        )
        _write_axis_delta(f, summary, axis="build_tag", fixed_axis="cpu_binding_tag")

        f.write("\n## CPU-binding sensitivity (192-223 vs 192-222, same build)\n\n")
        f.write(
            "For matching (build_tag, dtype, linear_type, tp_size, M-bucket) "
            "keys, compare the geomean speedup across cpu_binding_tags. This "
            "isolates the effect of dropping one thread from the NUMA-node "
            "binding, independent of the O2/O3 build-flag axis above.\n\n"
        )
        _write_axis_delta(f, summary, axis="cpu_binding_tag", fixed_axis="build_tag")

    return out_path


def _write_axis_delta(f, summary: list[dict], axis: str, fixed_axis: str):
    by_key = defaultdict(dict)
    for row in summary:
        k = (row[fixed_axis], row["dtype"], row["linear_type"], row["tp_size"], row["m_bucket"])
        by_key[k][row[axis]] = row["geomean_speedup_onednn_over_sgl"]

    variants = sorted({row[axis] for row in summary})
    if len(variants) < 2:
        f.write(f"(only one `{axis}` present in results/ so far — nothing to compare)\n")
        return

    f.write(f"{fixed_axis} | dtype | linear | tp | M-bucket | " + " | ".join(variants) + "\n")
    f.write("|" + "---|" * (5 + len(variants)) + "\n")
    for k, variant_vals in sorted(by_key.items()):
        fixed_val, dtype, linear_type, tp_size, bucket = k
        cells = [f"{variant_vals.get(v, float('nan')):.3f}" if v in variant_vals else "—" for v in variants]
        f.write(f"{fixed_val} | {dtype} | {linear_type} | {tp_size} | {bucket} | " + " | ".join(cells) + "\n")


if __name__ == "__main__":
    rows = load_all_rows()
    if not rows:
        raise SystemExit(f"no results/*__*.csv files found under {RESULTS_DIR}")
    summary = summarize(rows)
    pairs = sorted({(r["build_tag"], r["cpu_binding_tag"]) for r in rows})
    out_path = write_summary_md(summary, pairs)
    print(f"loaded {len(rows)} rows from {len(pairs)} (build_tag, cpu_binding_tag) sweeps")
    print(f"wrote {out_path}")
