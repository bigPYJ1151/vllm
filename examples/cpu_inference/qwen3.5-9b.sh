#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Qwen3.5-9B on CPU: latency (BS=1), throughput (BS=25) and MMLU.
# usage: bash qwen3.5-9b.sh <latency|throughput|acc> [bf16]

source "$(dirname "$0")/common.sh"

MODE=${1:?usage: bash $0 <latency|throughput|acc> [bf16]}
PRECISION=${2:-bf16}
case "$PRECISION" in
bf16) MODEL=Qwen/Qwen3.5-9B EXTRA=(--dtype bfloat16) ;;
*)
	echo "Unknown precision $PRECISION"
	exit 1
	;;
esac

main "$MODE" "qwen3.5-9b_${PRECISION}" "$MODEL" 25 "${EXTRA[@]}"
