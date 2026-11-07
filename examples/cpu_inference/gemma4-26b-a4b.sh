#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# gemma-4-26B-A4B on CPU: latency (BS=1), throughput (BS=25) and MMLU.
# usage: bash gemma4-26b-a4b.sh <latency|throughput|acc> [bf16]

source "$(dirname "$0")/common.sh"

MODE=${1:?usage: bash $0 <latency|throughput|acc> [bf16]}
PRECISION=${2:-bf16}
case "$PRECISION" in
bf16) MODEL=google/gemma-4-26B-A4B EXTRA=(--dtype bfloat16) ;;
*)
	echo "Unknown precision $PRECISION"
	exit 1
	;;
esac

KV_KIB_PER_TOKEN=${KV_KIB_PER_TOKEN-233}
main "$MODE" "gemma4-26b-a4b_${PRECISION}" "$MODEL" 25 "${EXTRA[@]}"
