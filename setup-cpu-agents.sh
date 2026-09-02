#!/bin/bash

set -euo pipefail

BEGIN_CRON_MARKER="# BEGIN vLLM CPU CI cache cleanup"
END_CRON_MARKER="# END vLLM CPU CI cache cleanup"

COMMAND=${1:-}
if [[ $# -gt 0 ]]; then
    shift
fi

AGENTS_PER_NUMA=${AGENTS_PER_NUMA:-2}
NUM_AGENTS=${NUM_AGENTS:-}
BUILDKITE_TOKEN_VALUE=${BUILDKITE_TOKEN:-}
HF_TOKEN_VALUE=${HF_TOKEN:-}
QUEUE=${QUEUE:-intel_cpu}
TAGS_VALUE=${TAGS:-}
HOST_ID=${HOST_ID:-}
BUILDKITE_AGENT_BIN_VALUE=${BUILDKITE_AGENT_BIN:-}
LOG_DIR=${LOG_DIR:-/var/log/buildkite-cpu-agents}
CLEANUP_SCHEDULE=${CLEANUP_SCHEDULE:-@daily}
CLEANUP_KEEP_STORAGE=${CLEANUP_KEEP_STORAGE:-100GB}
CLEANUP_LOG_DAYS=${CLEANUP_LOG_DAYS:-7}
AGGRESSIVE_SYSTEM_PRUNE=${AGGRESSIVE_SYSTEM_PRUNE:-0}

usage() {
    cat <<'EOF'
Usage:
  setup-cpu-agents.sh start [options]
  setup-cpu-agents.sh stop [options]
  setup-cpu-agents.sh status [options]
  setup-cpu-agents.sh cleanup [options]
  setup-cpu-agents.sh install-cleanup-cron [options]
  setup-cpu-agents.sh uninstall-cleanup-cron [options]

Options:
  --agents-per-numa N       Agent slots to create per NUMA node. Default: 2
  --num-agents N            Total agents to launch. Default: all available slots
  --token TOKEN             Buildkite agent token. Env: BUILDKITE_TOKEN
  --hf-token TOKEN          Hugging Face token exported to agents. Env: HF_TOKEN
  --queue QUEUE             Buildkite queue tag. Default: intel_cpu
  --tags TAGS               Extra buildkite agent tags (comma-separated
                            key=value pairs), added alongside the queue tag.
                            Repeatable; values are combined. Env: TAGS
  --host-id ID              Host identifier in agent names. Default: hostname
  --agent-bin PATH          buildkite-agent path. Env: BUILDKITE_AGENT_BIN
    --log-dir DIR             PID/log directory. Default: /var/log/buildkite-cpu-agents
    --cleanup-schedule CRON   Cleanup cron schedule. Default: @daily
                            HH:MM format is also accepted.
  --cleanup-keep-storage N  BuildKit keep-storage value. Default: 100GB
  --cleanup-log-days N      Remove managed logs older than N days. Default: 7
  --aggressive-system-prune Run docker system prune during cleanup. Default: off
  -h, --help                Show this help

Environment variables with matching names can be used instead of flags.
EOF
}

log() {
    echo "--- $*"
}

warn() {
    echo "WARNING: $*" >&2
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

is_positive_integer() {
    [[ ${1:-} =~ ^[1-9][0-9]*$ ]]
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --agents-per-numa)
                AGENTS_PER_NUMA=${2:?"--agents-per-numa requires a value"}
                shift 2
                ;;
            --num-agents)
                NUM_AGENTS=${2:?"--num-agents requires a value"}
                shift 2
                ;;
            --token)
                BUILDKITE_TOKEN_VALUE=${2:?"--token requires a value"}
                shift 2
                ;;
            --hf-token)
                HF_TOKEN_VALUE=${2:?"--hf-token requires a value"}
                shift 2
                ;;
            --queue)
                QUEUE=${2:?"--queue requires a value"}
                shift 2
                ;;
            --tags)
                if [[ -n "$TAGS_VALUE" ]]; then
                    TAGS_VALUE="$TAGS_VALUE,${2:?"--tags requires a value"}"
                else
                    TAGS_VALUE=${2:?"--tags requires a value"}
                fi
                shift 2
                ;;
            --host-id)
                HOST_ID=${2:?"--host-id requires a value"}
                shift 2
                ;;
            --agent-bin)
                BUILDKITE_AGENT_BIN_VALUE=${2:?"--agent-bin requires a value"}
                shift 2
                ;;
            --log-dir)
                LOG_DIR=${2:?"--log-dir requires a value"}
                shift 2
                ;;
            --cleanup-schedule)
                CLEANUP_SCHEDULE=${2:?"--cleanup-schedule requires a value"}
                shift 2
                ;;
            --cleanup-keep-storage)
                CLEANUP_KEEP_STORAGE=${2:?"--cleanup-keep-storage requires a value"}
                shift 2
                ;;
            --cleanup-log-days)
                CLEANUP_LOG_DAYS=${2:?"--cleanup-log-days requires a value"}
                shift 2
                ;;
            --aggressive-system-prune)
                AGGRESSIVE_SYSTEM_PRUNE=1
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "unknown option: $1"
                ;;
        esac
    done
}

host_id() {
    if [[ -n "$HOST_ID" ]]; then
        echo "$HOST_ID"
    elif hostname -s >/dev/null 2>&1; then
        hostname -s
    else
        hostname
    fi
}

script_path() {
    if command -v readlink >/dev/null 2>&1; then
        readlink -f "$0" 2>/dev/null || echo "$0"
    else
        echo "$0"
    fi
}

resolve_agent_bin() {
    if [[ -n "$BUILDKITE_AGENT_BIN_VALUE" ]]; then
        [[ -x "$BUILDKITE_AGENT_BIN_VALUE" ]] || die "agent binary is not executable: $BUILDKITE_AGENT_BIN_VALUE"
        echo "$BUILDKITE_AGENT_BIN_VALUE"
    elif [[ -x ./buildkite-agent ]]; then
        echo ./buildkite-agent
    elif command -v buildkite-agent >/dev/null 2>&1; then
        command -v buildkite-agent
    else
        die "buildkite-agent not found; use --agent-bin or BUILDKITE_AGENT_BIN"
    fi
}

expand_cpu_list() {
    local cpu_list=$1
    awk -v list="$cpu_list" '
        BEGIN {
            n = split(list, parts, ",")
            for (i = 1; i <= n; i++) {
                if (parts[i] == "") {
                    continue
                }
                if (parts[i] ~ /^[0-9]+-[0-9]+$/) {
                    split(parts[i], bounds, "-")
                    for (cpu = bounds[1]; cpu <= bounds[2]; cpu++) {
                        print cpu
                    }
                } else if (parts[i] ~ /^[0-9]+$/) {
                    print parts[i]
                }
            }
        }
    '
}

join_csv() {
    awk 'BEGIN { first = 1 } { if (!first) printf ","; printf "%s", $1; first = 0 } END { printf "\n" }'
}

tag_value() {
    local key=$1 tags=$2 pair

    IFS=',' read -ra pairs <<<"$tags"
    for pair in "${pairs[@]}"; do
        if [[ "${pair%%=*}" == "$key" ]]; then
            echo "${pair#*=}"
            return 0
        fi
    done
    return 1
}

contains_word() {
    local needle=$1
    shift
    local word

    for word in "$@"; do
        [[ "$word" == "$needle" ]] && return 0
    done
    return 1
}

add_node_cpu() {
    local node=$1
    local cpu=$2

    if ! contains_word "$node" "${NODES[@]}"; then
        NODES+=("$node")
    fi
    NODE_CPUS[$node]="${NODE_CPUS[$node]:-} $cpu"
}

load_topology_from_lscpu() {
    command -v lscpu >/dev/null 2>&1 || return 1

    local cpu node online loaded
    loaded=0
    while IFS=, read -r cpu node online _; do
        [[ -z "${cpu:-}" || "$cpu" == \#* ]] && continue
        [[ "$node" == "-" || -z "$node" || "$node" == "-1" ]] && node=0
        [[ -n "${online:-}" && "$online" == "N" ]] && continue
        [[ "$cpu" =~ ^[0-9]+$ && "$node" =~ ^[0-9]+$ ]] || continue
        add_node_cpu "$node" "$cpu"
        loaded=1
    done < <(lscpu -p=CPU,NODE,ONLINE 2>/dev/null || true)

    if [[ $loaded -eq 1 ]]; then
        return 0
    fi

    while IFS=, read -r cpu node _; do
        [[ -z "${cpu:-}" || "$cpu" == \#* ]] && continue
        [[ "$node" == "-" || -z "$node" || "$node" == "-1" ]] && node=0
        [[ "$cpu" =~ ^[0-9]+$ && "$node" =~ ^[0-9]+$ ]] || continue
        add_node_cpu "$node" "$cpu"
        loaded=1
    done < <(lscpu -p=CPU,NODE 2>/dev/null || true)

    [[ $loaded -eq 1 ]]
}

load_topology_from_sysfs() {
    local online_cpus node_dir node cpulist cpu
    online_cpus=""

    if [[ -r /sys/devices/system/cpu/online ]]; then
        online_cpus=" $(expand_cpu_list "$(cat /sys/devices/system/cpu/online)" | tr '\n' ' ')"
    fi

    shopt -s nullglob
    local node_dirs=(/sys/devices/system/node/node[0-9]*)
    shopt -u nullglob

    if [[ ${#node_dirs[@]} -gt 0 ]]; then
        for node_dir in "${node_dirs[@]}"; do
            [[ -r "$node_dir/cpulist" ]] || continue
            node=${node_dir##*node}
            cpulist=$(cat "$node_dir/cpulist")
            while read -r cpu; do
                [[ -z "$cpu" ]] && continue
                if [[ -n "$online_cpus" && " $online_cpus " != *" $cpu "* ]]; then
                    continue
                fi
                add_node_cpu "$node" "$cpu"
            done < <(expand_cpu_list "$cpulist")
        done
    elif [[ -n "$online_cpus" ]]; then
        while read -r cpu; do
            [[ -n "$cpu" ]] && add_node_cpu 0 "$cpu"
        done < <(tr ' ' '\n' <<<"$online_cpus")
    fi

    [[ ${#NODES[@]} -gt 0 ]]
}

sort_topology() {
    local node sorted

    mapfile -t NODES < <(printf '%s\n' "${NODES[@]}" | sort -n)
    for node in "${NODES[@]}"; do
        sorted=$(tr ' ' '\n' <<<"${NODE_CPUS[$node]}" | sort -n | uniq | tr '\n' ' ')
        NODE_CPUS[$node]=" $sorted"
    done
}

load_topology() {
    NODES=()
    declare -gA NODE_CPUS=()

    if ! load_topology_from_lscpu; then
        load_topology_from_sysfs || die "failed to discover CPU/NUMA topology with lscpu or sysfs"
    fi
    sort_topology
}

cpu_count_for_node() {
    local node=$1

    tr ' ' '\n' <<<"${NODE_CPUS[$node]}" | awk 'NF { count++ } END { print count + 0 }'
}

partition_for_slot() {
    local node=$1
    local slot=$2
    local total base remainder size start index end cpu

    total=$(cpu_count_for_node "$node")
    if (( total < AGENTS_PER_NUMA )); then
        die "NUMA node $node has only $total CPUs, fewer than agents-per-numa=$AGENTS_PER_NUMA"
    fi

    base=$((total / AGENTS_PER_NUMA))
    remainder=$((total % AGENTS_PER_NUMA))
    if (( slot < remainder )); then
        size=$((base + 1))
        start=$((slot * size))
    else
        size=$base
        start=$((remainder * (base + 1) + (slot - remainder) * base))
    fi
    end=$((start + size))

    index=0
    for cpu in ${NODE_CPUS[$node]}; do
        if (( index >= start && index < end )); then
            echo "$cpu"
        fi
        index=$((index + 1))
    done | join_csv
}

validate_common_numbers() {
    is_positive_integer "$AGENTS_PER_NUMA" || die "--agents-per-numa must be a positive integer"
    if [[ -n "$NUM_AGENTS" ]]; then
        is_positive_integer "$NUM_AGENTS" || die "--num-agents must be a positive integer"
    fi
    is_positive_integer "$CLEANUP_LOG_DAYS" || die "--cleanup-log-days must be a positive integer"
}

selected_slots() {
    local max_agents=$1
    local requested=${NUM_AGENTS:-$max_agents}
    local launched slot node

    (( requested <= max_agents )) || die "--num-agents=$requested exceeds max agents $max_agents (numa_nodes=${#NODES[@]}, agents_per_numa=$AGENTS_PER_NUMA)"

    launched=0
    for ((slot = 0; slot < AGENTS_PER_NUMA; slot++)); do
        for node in "${NODES[@]}"; do
            echo "$node $slot"
            launched=$((launched + 1))
            if (( launched >= requested )); then
                return 0
            fi
        done
    done
}

write_metadata() {
    local meta_file=$1
    local pid=$2
    local host=$3
    local node=$4
    local slot=$5
    local core_range=$6
    local agent_name=$7
    local log_file=$8

    cat > "$meta_file" <<EOF
PID=$pid
HOST_ID=$host
NUMA_NODE=$node
AGENT_SLOT=$slot
CORE_RANGE=$core_range
QUEUE=$QUEUE
AGENT_NAME=$agent_name
LOG_FILE=$log_file
EOF
}

metadata_value() {
    local key=$1
    local file=$2

    sed -n "s/^${key}=//p" "$file" | head -n 1
}

is_live_pid() {
    local pid=${1:-}

    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" >/dev/null 2>&1
}

pid_file_from_meta() {
    local meta_file=$1

    printf '%s\n' "${meta_file%.env}.pid"
}

cleanup_state_for_meta() {
    local meta_file=$1

    rm -f "$meta_file" "$(pid_file_from_meta "$meta_file")"
}

metadata_label() {
    local meta_file=$1
    local host node slot

    host=$(metadata_value HOST_ID "$meta_file")
    node=$(metadata_value NUMA_NODE "$meta_file")
    slot=$(metadata_value AGENT_SLOT "$meta_file")

    if [[ -n "$host" || -n "$node" || -n "$slot" ]]; then
        printf 'host=%s node=%s slot=%s\n' "${host:-?}" "${node:-?}" "${slot:-?}"
    else
        basename "$meta_file"
    fi
}

preflight_start_state() {
    local meta_files meta pid label
    local -a active_entries=()

    shopt -s nullglob
    meta_files=("$LOG_DIR"/*.env)
    shopt -u nullglob

    for meta in "${meta_files[@]}"; do
        pid=$(metadata_value PID "$meta")
        label=$(metadata_label "$meta")
        if is_live_pid "$pid"; then
            active_entries+=("$label (pid=$pid)")
        else
            warn "pruning stale managed metadata before start: $label (pid=${pid:-missing})"
            cleanup_state_for_meta "$meta"
        fi
    done

    if [[ ${#active_entries[@]} -gt 0 ]]; then
        warn "active managed agents detected; start is blocked"
        for label in "${active_entries[@]}"; do
            warn "  $label"
        done
        die "refusing to start while managed agents are active; run stop first"
    fi
}

start_agents() {
    validate_common_numbers
    [[ -n "$BUILDKITE_TOKEN_VALUE" ]] || die "--token or BUILDKITE_TOKEN is required for start"
    mkdir -p "$LOG_DIR"

    local agent_bin host max_agents requested node slot core_range pid_file log_file meta_file agent_name
    local old_pid agent_pid agent_tags label_value name_suffix

    agent_bin=$(resolve_agent_bin)
    host=$(host_id)

    agent_tags="queue=$QUEUE"
    if [[ -n "$TAGS_VALUE" ]]; then
        agent_tags="$agent_tags,$TAGS_VALUE"
    fi

    label_value=$(tag_value "label" "$TAGS_VALUE") || label_value=""
    name_suffix=${label_value:-%random}

    preflight_start_state

    load_topology
    max_agents=$((${#NODES[@]} * AGENTS_PER_NUMA))
    requested=${NUM_AGENTS:-$max_agents}
    (( requested <= max_agents )) || die "--num-agents=$requested exceeds max agents $max_agents"

    log "Starting $requested CPU Buildkite agents (max=$max_agents, agents_per_numa=$AGENTS_PER_NUMA, host=$host)"
    printf '%-8s %-8s %-12s %-12s %s\n' "NUMA" "SLOT" "PID" "QUEUE" "CORE_RANGE"

    while read -r node slot; do
        [[ -n "$node" ]] || continue
        core_range=$(partition_for_slot "$node" "$slot")
        pid_file="$LOG_DIR/agent-${host}-numa${node}-slot${slot}.pid"
        log_file="$LOG_DIR/agent-${host}-numa${node}-slot${slot}.log"
        meta_file="$LOG_DIR/agent-${host}-numa${node}-slot${slot}.env"
        agent_name="cpu-${host}-numa${node}-slot${slot}-${name_suffix}"

        if [[ -f "$pid_file" ]]; then
            old_pid=$(cat "$pid_file")
            if [[ -n "$old_pid" ]] && kill -0 "$old_pid" >/dev/null 2>&1; then
                die "managed agent already running for NUMA $node slot $slot (pid=$old_pid)"
            fi
            rm -f "$pid_file"
        fi

        (
            export CORE_RANGE="$core_range"
            export NUMA_NODE="$node"
            export AGENT_SLOT="$slot"
            if [[ -n "$HF_TOKEN_VALUE" ]]; then
                export HF_TOKEN="$HF_TOKEN_VALUE"
            fi
            exec "$agent_bin" start \
                --token "$BUILDKITE_TOKEN_VALUE" \
                --tags "$agent_tags" \
                --name "$agent_name"
        ) >"$log_file" 2>&1 &

        agent_pid=$!
        echo "$agent_pid" >"$pid_file"
        write_metadata "$meta_file" "$agent_pid" "$host" "$node" "$slot" "$core_range" "$agent_name" "$log_file"
        printf '%-8s %-8s %-12s %-12s %s\n' "$node" "$slot" "$agent_pid" "$QUEUE" "$core_range"
    done < <(selected_slots "$max_agents")
}

stop_agents() {
    mkdir -p "$LOG_DIR"
    shopt -s nullglob
    local meta_files=("$LOG_DIR"/*.env)
    shopt -u nullglob

    if [[ ${#meta_files[@]} -eq 0 ]]; then
        warn "no managed agent metadata files found in $LOG_DIR"
        return 0
    fi

    local meta pid label
    for meta in "${meta_files[@]}"; do
        pid=$(metadata_value PID "$meta")
        label=$(metadata_label "$meta")
        if is_live_pid "$pid"; then
            log "Stopping agent $label pid=$pid"
            kill "$pid" || true
            for _ in 1 2 3 4 5; do
                is_live_pid "$pid" || break
                sleep 1
            done
            if is_live_pid "$pid"; then
                warn "agent pid=$pid did not stop after SIGTERM; Will stop after current job done"
            fi
        else
            warn "stale managed metadata: $label (pid=${pid:-missing})"
        fi
        cleanup_state_for_meta "$meta"
    done
}

cron_block_installed() {
    command -v crontab >/dev/null 2>&1 || return 2
    crontab -l 2>/dev/null | grep -Fq "$BEGIN_CRON_MARKER"
}

status_agents() {
    mkdir -p "$LOG_DIR"
    load_topology || true

    local max_agents meta_files meta pid live node slot queue core_range
    max_agents=0
    if [[ ${#NODES[@]} -gt 0 ]]; then
        max_agents=$((${#NODES[@]} * AGENTS_PER_NUMA))
    fi
    log "Managed CPU Buildkite agents in $LOG_DIR"
    log "NUMA nodes=${#NODES[@]} agents_per_numa=$AGENTS_PER_NUMA max_agents=$max_agents num_agents=${NUM_AGENTS:-$max_agents}"

    shopt -s nullglob
    meta_files=("$LOG_DIR"/*.env)
    shopt -u nullglob
    if [[ ${#meta_files[@]} -eq 0 ]]; then
        warn "no managed agent metadata files found"
    else
        for meta in "${meta_files[@]}"; do
            pid=$(metadata_value PID "$meta")
            node=$(metadata_value NUMA_NODE "$meta")
            slot=$(metadata_value AGENT_SLOT "$meta")
            queue=$(metadata_value QUEUE "$meta")
            core_range=$(metadata_value CORE_RANGE "$meta")
            live=stale
            if is_live_pid "$pid"; then
                live=yes
            else
                warn "removing stale managed metadata: $(metadata_label "$meta") (pid=${pid:-missing})"
                cleanup_state_for_meta "$meta"
            fi
            printf '%-6s pid=%-8s node=%-4s slot=%-4s queue=%-12s cores=%s\n' \
                "$live" "$pid" "$node" "$slot" "$queue" "$core_range"
        done
    fi

    if ! command -v crontab >/dev/null 2>&1; then
        warn "crontab command not found; cleanup cron cannot be managed on this host"
    elif cron_block_installed; then
        log "Cleanup cron: installed"
    else
        log "Cleanup cron: not installed"
    fi
}

cleanup_host() {
    mkdir -p "$LOG_DIR"

    if command -v docker >/dev/null 2>&1; then
        log "Docker disk usage before cleanup"
        docker system df || true

        log "Pruning Docker builder cache (keep-storage=$CLEANUP_KEEP_STORAGE)"
        docker builder prune --force --keep-storage "$CLEANUP_KEEP_STORAGE" || true

        log "Pruning stopped containers"
        docker container prune --force || true

        log "Pruning dangling images"
        docker image prune --force || true

        if [[ "$AGGRESSIVE_SYSTEM_PRUNE" == "1" ]]; then
            log "Running aggressive docker system prune"
            docker system prune --force || true
        fi

        log "Docker disk usage after cleanup"
        docker system df || true
    else
        warn "docker command not found; skipping Docker cleanup"
    fi

    log "Deleting managed logs older than $CLEANUP_LOG_DAYS days under $LOG_DIR"
    find "$LOG_DIR" -type f -name '*.log' -mtime +"$CLEANUP_LOG_DAYS" -delete 2>/dev/null || true
}

normalize_schedule() {
    local schedule=$1
    local hour minute

    if [[ "$schedule" == "everyday" || "$schedule" == "daily" ]]; then
        echo "@daily"
        return 0
    fi
    if [[ "$schedule" =~ ^@(annually|yearly|monthly|weekly|daily|midnight|hourly|reboot)$ ]]; then
        echo "$schedule"
        return 0
    fi
    if [[ "$schedule" =~ ^[0-9]{1,2}:[0-9]{2}$ ]]; then
        hour=${schedule%%:*}
        minute=${schedule#*:}
        echo "$minute $hour * * *"
        return 0
    fi
    if [[ $(awk '{ print NF }' <<<"$schedule") -eq 5 ]]; then
        echo "$schedule"
        return 0
    fi
    die "invalid cleanup schedule '$schedule'; use cron format, @daily, everyday, or HH:MM"
}

cron_cleanup_command() {
    local path

    path=$(script_path)
    printf 'LOG_DIR=%q CLEANUP_KEEP_STORAGE=%q CLEANUP_LOG_DAYS=%q AGGRESSIVE_SYSTEM_PRUNE=%q %q cleanup >> %q/cleanup-cron.log 2>&1' \
        "$LOG_DIR" "$CLEANUP_KEEP_STORAGE" "$CLEANUP_LOG_DAYS" "$AGGRESSIVE_SYSTEM_PRUNE" "$path" "$LOG_DIR"
}

remove_cron_block() {
    awk -v begin="$BEGIN_CRON_MARKER" -v end="$END_CRON_MARKER" '
        $0 == begin { skip = 1; next }
        $0 == end { skip = 0; next }
        !skip { print }
    '
}

install_cleanup_cron() {
    command -v crontab >/dev/null 2>&1 || die "crontab command not found; install cron package or run cleanup manually"
    mkdir -p "$LOG_DIR"

    local current desired tmp schedule block current_block
    current=$(crontab -l 2>/dev/null || true)
    schedule=$(normalize_schedule "$CLEANUP_SCHEDULE")
    block=$(printf '%s\n%s %s\n%s\n' "$BEGIN_CRON_MARKER" "$schedule" "$(cron_cleanup_command)" "$END_CRON_MARKER")
    current_block=$(printf '%s\n' "$current" | awk -v begin="$BEGIN_CRON_MARKER" -v end="$END_CRON_MARKER" '
        $0 == begin { in_block = 1 }
        in_block { print }
        $0 == end { in_block = 0 }
    ')

    if [[ "$current_block" == "$block" ]]; then
        log "Cleanup cron already installed"
        return 0
    fi

    desired=$(mktemp)
    tmp=$(mktemp)
    trap 'rm -f "$desired" "$tmp"' RETURN

    printf '%s\n' "$current" | remove_cron_block >"$tmp"
    sed '/^[[:space:]]*$/d' "$tmp" >"$desired"
    if [[ -s "$desired" ]]; then
        printf '\n' >>"$desired"
    fi
    printf '%s' "$block" >>"$desired"
    crontab "$desired"
    log "Installed cleanup cron: $schedule"
}

uninstall_cleanup_cron() {
    command -v crontab >/dev/null 2>&1 || die "crontab command not found"

    local current desired
    current=$(crontab -l 2>/dev/null || true)
    if ! printf '%s\n' "$current" | grep -Fq "$BEGIN_CRON_MARKER"; then
        log "Cleanup cron is not installed"
        return 0
    fi

    desired=$(mktemp)
    trap 'rm -f "$desired"' RETURN
    printf '%s\n' "$current" | remove_cron_block >"$desired"
    crontab "$desired"
    log "Removed cleanup cron"
}

parse_args "$@"
validate_common_numbers

case "$COMMAND" in
    start)
        start_agents
        ;;
    stop)
        stop_agents
        ;;
    status)
        status_agents
        ;;
    cleanup)
        cleanup_host
        ;;
    install-cleanup-cron)
        install_cleanup_cron
        ;;
    uninstall-cleanup-cron)
        uninstall_cleanup_cron
        ;;
    -h|--help|help)
        usage
        ;;
    "")
        usage
        exit 1
        ;;
    *)
        usage
        die "unknown command: $COMMAND"
        ;;
esac