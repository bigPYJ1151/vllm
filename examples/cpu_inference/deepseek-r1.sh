#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# DeepSeek-R1 on CPU: latency (BS=1), throughput (BS=25) and MMLU.
# usage: bash deepseek-r1.sh <latency|throughput|acc> [fp8]

source "$(dirname "$0")/common.sh"

MODE=${1:?usage: bash $0 <latency|throughput|acc> [fp8]}
PRECISION=${2:-fp8}
[[ -z "$DP" ]] || { echo "DP is not supported: the model needs all dies (use TP)"; exit 1; }
case "$PRECISION" in
fp8) MODEL=deepseek-ai/DeepSeek-R1 EXTRA=(--dtype bfloat16) ;;
*)
	echo "Unknown precision $PRECISION"
	exit 1
	;;
esac

KV_KIB_PER_TOKEN=${KV_KIB_PER_TOKEN-73}
main "$MODE" "deepseek-r1_${PRECISION}" "$MODEL" 25 "${EXTRA[@]}"
