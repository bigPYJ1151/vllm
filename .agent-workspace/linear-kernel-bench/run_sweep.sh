#!/usr/bin/env bash
# usage: run_sweep.sh <build-tag> <cpu-range> <cpu-binding-tag> [extra bench_linear_kernels.py args...]
#
# Thin numactl wrapper around bench_linear_kernels.py. Does not compute or
# guess a CPU range — pass the exact numactl --physcpubind range for the
# binding config you're testing (see the plan's two required bindings:
# 192-223 and 192-222, both --cpunodebind=0 --membind=0).
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <build-tag> <cpu-range> <cpu-binding-tag> [extra args...]" >&2
  exit 1
fi

BUILD_TAG="$1"
CPU_RANGE="$2"
CPU_BINDING_TAG="$3"
shift 3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_CSV="${SCRIPT_DIR}/results/${BUILD_TAG}__${CPU_BINDING_TAG}.csv"

if [[ -f "$OUT_CSV" ]]; then
  TS="$(date +%Y%m%d%H%M%S)"
  echo "NOTE: ${OUT_CSV} already exists; re-run detected -- suffixing this" \
       "run's cpu-binding-tag with -${TS} so the prior results are not" \
       "overwritten." >&2
  CPU_BINDING_TAG="${CPU_BINDING_TAG}-${TS}"
fi

exec numactl --cpunodebind=0 --membind=0 --physcpubind="$CPU_RANGE" \
  python3 "${SCRIPT_DIR}/bench_linear_kernels.py" \
  --build-tag "$BUILD_TAG" \
  --cpu-binding-tag "$CPU_BINDING_TAG" \
  "$@"
