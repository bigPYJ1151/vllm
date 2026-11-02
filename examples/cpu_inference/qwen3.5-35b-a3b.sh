#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Qwen3.5-35B-A3B on CPU: latency (BS=1), throughput (BS=25) and MMLU.
# usage: bash qwen3.5-35b-a3b.sh <latency|throughput|acc> [bf16|fp8]

source "$(dirname "$0")/common.sh"

MODE=${1:?usage: bash $0 <latency|throughput|acc> [bf16|fp8]}
PRECISION=${2:-bf16}
case "$PRECISION" in
bf16) MODEL=Qwen/Qwen3.5-35B-A3B EXTRA=(--dtype bfloat16) ;;
fp8) MODEL=Qwen/Qwen3.5-35B-A3B-FP8 EXTRA=(--dtype bfloat16) ;;
*)
	echo "Unknown precision $PRECISION"
	exit 1
	;;
esac

main "$MODE" "qwen3.5-35b-a3b_${PRECISION}" "$MODEL" 25 "${EXTRA[@]}"
