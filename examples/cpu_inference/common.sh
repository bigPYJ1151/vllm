#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Shared helpers for the per-model CPU benchmark scripts in this folder.
# Sourced by <model>.sh; see README.md for the environment variables.

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Test configuration (override through the environment) ------------------
INPUT_TOKEN=${INPUT_TOKEN:-1024}
OUTPUT_TOKEN=${OUTPUT_TOKEN:-1024}
OUTPUT_DIR=${OUTPUT_DIR:-$PWD/cpu_inference_results}
PORT=${PORT:-8000}
ACC_PER_SUBJECT=${ACC_PER_SUBJECT:-40}     # MMLU questions per subject, 0 = all
ACC_CONCURRENCY=${ACC_CONCURRENCY:-64}
MMLU_DATA_DIR=${MMLU_DATA_DIR:-$OUTPUT_DIR/mmlu_data}
DRY_RUN=${DRY_RUN:-0}                      # 1: only print the commands

# ---- Topology: one TP rank per CBB (die) or per NUMA node -------------------
detect_topology() {
	local numa_count die_count
	numa_count=$(lscpu | awk -F: '/^NUMA node\(s\)/ {gsub(/ /, "", $2); print $2}')
	[[ "$numa_count" =~ ^[0-9]+$ ]] || numa_count=1
	die_count=$(cat /sys/devices/system/cpu/cpu*/topology/die_id 2>/dev/null | sort -nu | wc -l)
	[[ "$die_count" =~ ^[0-9]+$ ]] || die_count=0

	if ((die_count > numa_count)); then
		TOPO_MODE=cbb
		NUM_INSTANCES=$die_count
	else
		TOPO_MODE=numa
		NUM_INSTANCES=$numa_count
	fi
	TOPO_MODE=${BIND_GRANULARITY:-$TOPO_MODE}

	# TP must divide the head counts: use the largest power of two that fits.
	local tp=1
	while ((tp * 2 <= NUM_INSTANCES)); do tp=$((tp * 2)); done
	TP=${TP:-$tp}
}

setup_cpu_env() {
	detect_topology
	export VLLM_CPU_OMP_THREADS_BIND=auto
	export VLLM_CPU_BIND_GRANULARITY=$TOPO_MODE
	export VLLM_CPU_NUM_OF_RESERVED_CPU=${VLLM_CPU_NUM_OF_RESERVED_CPU:-0}
	export VLLM_CPU_ATTN_DIRECT=${VLLM_CPU_ATTN_DIRECT:-1}
	export TORCHINDUCTOR_FILE_LOCK_TIMEOUT=${TORCHINDUCTOR_FILE_LOCK_TIMEOUT:-7200}
	# Dies share a socket's memory controllers, so the engine core sleeps between
	# requests there; with one NUMA node per rank it spins.
	if [[ "$TOPO_MODE" == cbb ]]; then
		export VLLM_MQ_BUSY_LOOP_SECONDS=${VLLM_MQ_BUSY_LOOP_SECONDS:-0}
	else
		export VLLM_MQ_BUSY_LOOP_SECONDS=${VLLM_MQ_BUSY_LOOP_SECONDS:-1.0}
	fi
	mkdir -p "$OUTPUT_DIR"
	echo "Topology: ${TOPO_MODE} granularity, ${NUM_INSTANCES} instances, TP=${TP}"
}

run() {
	echo "+ $*"
	[[ "$DRY_RUN" == 1 ]] || "$@"
}

# ---- latency / throughput: `vllm bench one-batch` ----------------------------
# one_batch <tag> <model> <batch size> <extra args...>
one_batch() {
	local tag=$1 model=$2 bs=$3
	shift 3
	local base="${OUTPUT_DIR}/${tag}_bs${bs}_in${INPUT_TOKEN}_out${OUTPUT_TOKEN}"
	rm -rf /dev/shm/* 2>/dev/null
	echo "Results: ${base}.{json,log}"
	run torchrun --standalone --nproc-per-node="$TP" -m vllm.entrypoints.cli.main \
		bench one-batch --model "$model" --tensor-parallel-size "$TP" \
		--batch-size "$bs" --input-len "$INPUT_TOKEN" --output-len "$OUTPUT_TOKEN" \
		--temperature 0 --output-json "${base}.json" "$@" 2>&1 | tee "${base}.log"
	local rc=${PIPESTATUS[0]}
	grep -aE '^(Prefill|Decode|Total)\.' "${base}.log"
	return "$rc"
}

# ---- accuracy: vllm serve + MMLU 5-shot client -------------------------------
download_mmlu() {
	[[ -d "$MMLU_DATA_DIR/test" ]] && return 0
	echo "Downloading MMLU data to $MMLU_DATA_DIR"
	mkdir -p "$MMLU_DATA_DIR"
	local tmp
	tmp=$(mktemp -d)
	wget -q -O "$tmp/data.tar" https://people.eecs.berkeley.edu/~hendrycks/data.tar || return 1
	tar -xf "$tmp/data.tar" -C "$tmp" && mv "$tmp"/data/* "$MMLU_DATA_DIR"/
	rm -rf "$tmp"
}

stop_server() {
	[[ -n "$SERVER_PID" ]] || return 0
	kill -TERM -- -"$SERVER_PID" 2>/dev/null
	sleep 10
	kill -KILL -- -"$SERVER_PID" 2>/dev/null
	SERVER_PID=
}

# accuracy <tag> <model> <extra serve args...>
accuracy() {
	local tag=$1 model=$2
	shift 2
	local base="${OUTPUT_DIR}/${tag}_mmlu"
	[[ "$DRY_RUN" == 1 ]] || download_mmlu || { echo "MMLU download failed"; return 1; }
	rm -rf /dev/shm/* 2>/dev/null

	echo "Results: ${base}.{json,log}, server log ${base}_server.log"
	echo "+ vllm serve $model --tensor-parallel-size $TP --max-model-len 4096 --max-num-seqs 64 --port $PORT $*"
	if [[ "$DRY_RUN" == 1 ]]; then
		echo "+ python ${SCRIPT_DIR}/mmlu_eval.py --data-dir $MMLU_DATA_DIR --model $model --port $PORT"
		return 0
	fi
	setsid vllm serve "$model" --tensor-parallel-size "$TP" --max-model-len 4096 \
		--max-num-seqs 64 --port "$PORT" "$@" >"${base}_server.log" 2>&1 &
	SERVER_PID=$!
	trap stop_server EXIT

	# The first start compiles for several minutes.
	until curl -s "localhost:${PORT}/v1/models" >/dev/null; do
		if ! kill -0 "$SERVER_PID" 2>/dev/null; then
			echo "Server failed, see ${base}_server.log"
			return 1
		fi
		sleep 5
	done
	python "${SCRIPT_DIR}/mmlu_eval.py" --data-dir "$MMLU_DATA_DIR" --model "$model" \
		--port "$PORT" --concurrency "$ACC_CONCURRENCY" --per-subject "$ACC_PER_SUBJECT" \
		--out "${base}.json" 2>&1 | tee "${base}.log"
	local rc=${PIPESTATUS[0]}
	grep -a 'Average accuracy' "${base}.log"
	return "$rc"
}

# ---- Entry point shared by the per-model scripts ----------------------------
# main <mode> <tag> <model> <throughput batch size> <extra args...>
# <extra args> go to `vllm bench one-batch` / `vllm serve` (e.g. --dtype float16).
main() {
	local mode=${1,,} tag=$2 model=$3 tput_bs=$4
	shift 4
	setup_cpu_env
	case "$mode" in
	latency) one_batch "$tag" "$model" "${BATCH_SIZE:-1}" "$@" ;;
	throughput) one_batch "$tag" "$model" "${BATCH_SIZE:-$tput_bs}" "$@" ;;
	acc | accuracy) accuracy "$tag" "$model" "$@" ;;
	*)
		echo "Unknown mode '$mode': use latency, throughput or acc"
		return 1
		;;
	esac
}
