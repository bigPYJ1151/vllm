# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sum the decode throughput of data-parallel `vllm bench one-batch` instances.

usage: sum_dp.py --output-json OUT instance0.json [instance1.json ...]

Prints the summed result in the format `vllm bench one-batch` prints, so the
per-model scripts report it like a single run.
"""

import argparse
import json
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("instances", nargs="+")
    args = parser.parse_args()

    results = []
    for path in args.instances:
        with open(path) as f:
            results.append(json.load(f))

    decode_throughput = sum(r["median_decode_throughput"] for r in results)
    overall_throughput = sum(r["overall_throughput"] for r in results)
    decode_latency = statistics.mean(r["median_decode_latency"] for r in results)
    prefill = statistics.mean(r["prefill_latency"] for r in results)
    total = max(r["total_latency"] for r in results)
    batch_size = results[0]["batch_size"]

    for i, r in enumerate(results):
        print(
            f"Instance {i}. decode latency: {r['median_decode_latency']:6.5f} s, "
            f"decode throughput: {r['median_decode_throughput']:9.2f} token/s"
        )
    print(
        f"Prefill. latency: {prefill:6.5f} s, "
        f"throughput: {batch_size * len(results) / prefill:9.2f} token/s"
    )
    print(
        f"Decode.  median latency: {decode_latency:6.5f} s, "
        f"median throughput: {decode_throughput:9.2f} token/s"
    )
    print(
        f"Total. latency: {total:6.3f} s, throughput: {overall_throughput:9.2f} token/s"
    )

    summary = {
        "data_parallel_size": len(results),
        "batch_size_per_instance": batch_size,
        "input_len": results[0]["input_len"],
        "output_len": results[0]["output_len"],
        "prefill_latency": prefill,
        "median_decode_latency": decode_latency,
        "median_decode_throughput": decode_throughput,
        "total_latency": total,
        "overall_throughput": overall_throughput,
        "instances": results,
    }
    with open(args.output_json, "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()
