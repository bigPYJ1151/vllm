#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Llama-3.1-8B-Instruct on CPU: latency (BS=1), throughput (BS=30) and MMLU.
# usage: bash llama3.1-8b.sh <latency|throughput|acc> [bf16|fp16|int8|fp8]

source "$(dirname "$0")/common.sh"

MODE=${1:?usage: bash $0 <latency|throughput|acc> [bf16|fp16|int8|fp8]}
PRECISION=${2:-bf16}
case "$PRECISION" in
bf16) MODEL=unsloth/Llama-3.1-8B-Instruct EXTRA=(--dtype bfloat16) ;;
fp16) MODEL=unsloth/Llama-3.1-8B-Instruct EXTRA=(--dtype float16) ;;
int8) MODEL=RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8 EXTRA=(--dtype bfloat16) ;;
fp8) MODEL=Intel/llama-3.1-8b-instruct-fp8 EXTRA=(--dtype bfloat16) ;;
*)
	echo "Unknown precision $PRECISION"
	exit 1
	;;
esac

main "$MODE" "llama3.1-8b_${PRECISION}" "$MODEL" 30 "${EXTRA[@]}"
