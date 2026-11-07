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
# DP=<n>: throughput only; n independent TP=1 instances, one per die/NUMA node,
# each at the full batch size. Exclusive with TP.
DP=${DP:-}

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
	DIES_PER_NODE=$((die_count > numa_count ? die_count / numa_count : 1))

	# TP must divide the head counts: use the largest power of two that fits.
	local tp=1
	while ((tp * 2 <= NUM_INSTANCES)); do tp=$((tp * 2)); done
	TP=${TP:-$tp}
}

# CPUs of die <i> (same die ids as detect_topology), comma separated.
die_cpus() {
	local die_id
	die_id=$(cat /sys/devices/system/cpu/cpu*/topology/die_id | sort -nu | sed -n "$(($1 + 1))p")
	grep -lx "$die_id" /sys/devices/system/cpu/cpu*/topology/die_id |
		sed 's#.*/cpu\([0-9]*\)/.*#\1#' | sort -n | paste -sd,
}

# NUMA node of a CPU, and the <i>-th NUMA node.
cpu_node() {
	ls -d /sys/devices/system/cpu/cpu"$1"/node* | sed 's#.*node##'
}
numa_node() {
	ls -d /sys/devices/system/node/node* | sed 's#.*node##' | sort -n | sed -n "$(($1 + 1))p"
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
	if [[ -n "$DP" ]]; then
		if ((DP < 1 || DP > NUM_INSTANCES)); then
			echo "DP=$DP needs between 1 and $NUM_INSTANCES (${TOPO_MODE} instances)"
			return 1
		fi
		TP=1
		echo "Topology: ${TOPO_MODE} granularity, ${NUM_INSTANCES} instances, DP=${DP} x TP=1"
	else
		echo "Topology: ${TOPO_MODE} granularity, ${NUM_INSTANCES} instances, TP=${TP}"
	fi
}

run() {
	echo "+ $*"
	[[ "$DRY_RUN" == 1 ]] || "$@"
}

# Size the KV cache for the batch instead of letting each rank claim its whole
# memory share: <batch size> x (input + output) tokens, plus 25% for block
# rounding. KV_KIB_PER_TOKEN is per rank, set by the model scripts; unset keeps
# the default sizing from --gpu-memory-utilization.
kv_cache_args() {
	KV_ARGS=()
	[[ -n "$KV_KIB_PER_TOKEN" ]] || return 0
	local bytes=$(($1 * (INPUT_TOKEN + OUTPUT_TOKEN) * KV_KIB_PER_TOKEN * 1024 * 5 / 4))
	KV_ARGS=(--kv-cache-memory-bytes "$bytes")
}

# ---- latency / throughput: `vllm bench one-batch` ----------------------------
# one_batch <tag> <model> <batch size> <extra args...>
one_batch() {
	local tag=$1 model=$2 bs=$3
	shift 3
	local base="${OUTPUT_DIR}/${tag}_bs${bs}_in${INPUT_TOKEN}_out${OUTPUT_TOKEN}"
	rm -rf /dev/shm/* 2>/dev/null
	echo "Results: ${base}.{json,log}"
	kv_cache_args "$bs"
	run torchrun --standalone --nproc-per-node="$TP" -m vllm.entrypoints.cli.main \
		bench one-batch --model "$model" --tensor-parallel-size "$TP" \
		--batch-size "$bs" --input-len "$INPUT_TOKEN" --output-len "$OUTPUT_TOKEN" \
		--temperature 0 "${KV_ARGS[@]}" --output-json "${base}.json" "$@" 2>&1 | tee "${base}.log"
	local rc=${PIPESTATUS[0]}
	grep -aE '^(Prefill|Decode|Total)\.' "${base}.log"
	return "$rc"
}

# ---- throughput with data parallelism: independent one-batch instances -------
# one_batch_dp <tag> <model> <batch size> <extra args...>
# Instance i runs the full batch on die i (cbb) or NUMA node i; vLLM binds the
# cores from the visible memory node, and from the allowed CPUs under cbb (one
# node holds several dies there). The decode throughputs are summed.
one_batch_dp() {
	local tag=$1 model=$2 bs=$3
	shift 3
	local base="${OUTPUT_DIR}/${tag}_dp${DP}_bs${bs}_in${INPUT_TOKEN}_out${OUTPUT_TOKEN}"
	rm -rf /dev/shm/* 2>/dev/null
	echo "Results: ${base}.json, per instance ${base}_<i>.{json,log}"

	local i cpus node pin pid pids=() jsons=() rc=0
	# Under taskset a rank sees one die and would claim its whole memory node
	# for the KV cache, so scale the share by the dies on a node.
	local mem_util=0.92
	[[ "$TOPO_MODE" == cbb ]] && mem_util=$(awk -v d="$DIES_PER_NODE" 'BEGIN {printf "%.3f", 0.92 / d}')
	kv_cache_args "$bs"
	for ((i = 0; i < DP; i++)); do
		if [[ "$TOPO_MODE" == cbb ]]; then
			cpus=$(die_cpus "$i")
			node=$(cpu_node "${cpus%%,*}")
			pin="taskset -c $cpus"
		else
			node=$(numa_node "$i")
			pin=
		fi
		local cmd=(env CPU_VISIBLE_MEMORY_NODES="$node" $pin
			python -m vllm.entrypoints.cli.main bench one-batch --model "$model"
			--tensor-parallel-size 1 --batch-size "$bs" --input-len "$INPUT_TOKEN"
			--output-len "$OUTPUT_TOKEN" --temperature 0
			--gpu-memory-utilization "$mem_util" "${KV_ARGS[@]}"
			--output-json "${base}_${i}.json" "$@")
		echo "+ ${cmd[*]} > ${base}_${i}.log"
		jsons+=("${base}_${i}.json")
		[[ "$DRY_RUN" == 1 ]] && continue
		"${cmd[@]}" >"${base}_${i}.log" 2>&1 &
		pids+=($!)
	done
	[[ "$DRY_RUN" == 1 ]] && return 0

	for pid in "${pids[@]}"; do
		wait "$pid" || rc=$?
	done
	if ((rc != 0)); then
		echo "An instance failed, see ${base}_<i>.log"
		return "$rc"
	fi
	python "${SCRIPT_DIR}/sum_dp.py" --output-json "${base}.json" "${jsons[@]}"
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
	until curl -s "127.0.0.1:${PORT}/v1/models" >/dev/null; do
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
	if [[ -n "$DP" ]]; then
		[[ -z "$TP" ]] || { echo "DP and TP are exclusive"; return 1; }
		[[ "$mode" == throughput ]] || { echo "DP is only supported for throughput"; return 1; }
	fi
	setup_cpu_env || return 1
	case "$mode" in
	latency) one_batch "$tag" "$model" "${BATCH_SIZE:-1}" "$@" ;;
	throughput)
		if [[ -n "$DP" ]]; then
			one_batch_dp "$tag" "$model" "${BATCH_SIZE:-$tput_bs}" "$@"
		else
			one_batch "$tag" "$model" "${BATCH_SIZE:-$tput_bs}" "$@"
		fi
		;;
	acc | accuracy) accuracy "$tag" "$model" "$@" ;;
	*)
		echo "Unknown mode '$mode': use latency, throughput or acc"
		return 1
		;;
	esac
}
