# CPU model benchmarks

One-line latency, throughput and accuracy tests for the models validated on the
vLLM CPU backend (Intel Xeon with AMX).

## Prerequisites

A vLLM CPU build (see the CPU installation guide, or the Docker image below),
the model weights reachable from the Hugging Face cache or hub, `torchrun`,
`curl` and `wget`. The accuracy client needs `aiohttp` and `tiktoken`. Setting
`LD_PRELOAD` to tcmalloc and Intel OpenMP, as in the CPU Docker image, is
recommended.

## Building the CPU image from a clone

`docker/Dockerfile.cpu` installs the pinned torch wheel from `prebuilt/`, which
is stored with Git LFS. Run `git lfs install` before cloning (or `git lfs
install && git lfs pull` in an existing clone). Without it `prebuilt/*.whl` is a
134-byte pointer file, and the image build fails at the first requirements
install with `torch==... has an invalid package format`.

```bash
# 1. Clone with LFS (replace the URL and branch with the ones under test)
git lfs install
git clone -b <branch> <repo url> vllm
cd vllm
ls -l prebuilt/*.whl        # must be ~190 MB, not 134 bytes

# 2. Build the dev image (compiles the CPU kernels, takes 15 to 20 minutes)
docker build -f docker/Dockerfile.cpu --target vllm-dev -t vllm-cpu-dev .

# 3. Run the scripts in the container; the examples are baked into the image
docker run --rm -it --privileged --net host --shm-size 64g --entrypoint bash \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -e OUTPUT_DIR=/results -v $PWD/results:/results \
    vllm-cpu-dev -lc "cd /vllm-workspace/examples/cpu_inference && bash llama3.1-8b.sh latency"
```

`--privileged` and `--net host` are what the reference runs used; add
`-e HF_TOKEN=<token>` for gated models (for example `Intel/llama-3.1-8b-instruct-fp8`).
Models are downloaded on first use, so keep only the checkpoint under test on
disk when space is tight (DeepSeek-R1 alone needs about 650 GB).

The image also carries an `as` wrapper (`docker/cpu_as_shim.sh`, found through
`COMPILER_PATH`) so that Triton-CPU kernels, such as the sampler's, assemble on
AVX10.2/APX hosts (Diamond Rapids), where the pinned binutils rejects some LLVM
output. No extra mount or setting is needed.

## Models and scripts

Each model has its own script:

| Script | Model | Precisions |
|---|---|---|
| `llama3.1-8b.sh` | Llama-3.1-8B-Instruct | `bf16` (default), `fp16`, `int8` (W8A8), `fp8` (Intel) |
| `qwen3.5-9b.sh` | Qwen3.5-9B | `bf16` |
| `qwen3.5-35b-a3b.sh` | Qwen3.5-35B-A3B | `bf16` (default), `fp8` |
| `gemma4-26b-a4b.sh` | gemma-4-26B-A4B | `bf16` |
| `deepseek-r1.sh` | DeepSeek-R1 (671B, native FP8 blocks) | `fp8` |

```bash
bash <model>.sh <latency|throughput|acc> [precision]

bash qwen3.5-9b.sh latency
bash llama3.1-8b.sh throughput fp8
bash qwen3.5-35b-a3b.sh acc fp8
```

## What each mode runs

| Mode | Command | Reports |
|---|---|---|
| `latency` | `vllm bench one-batch`, batch size 1 | prefill latency, median decode step latency, total latency |
| `throughput` | `vllm bench one-batch`, batch size 30 (Llama) or 25 (others); with `DP=<n>`, `n` instances at that batch size | the same, plus decode and overall tokens/s (summed over the instances with `DP`) |
| `acc` | `vllm serve` + MMLU 5-shot client | average accuracy |

`vllm bench one-batch` builds one static batch (1024 input and 1024 output
tokens by default), prefills it in one step and decodes it step by step. It
drives the model runner directly, so only input preparation, the forward pass
and sampling are timed. One process runs per tensor-parallel rank (`torchrun`).
Two warmup runs precede the measured one.

`acc` starts a server, waits until it is ready, runs the MMLU client on 40
questions per subject (2280 questions, same prompts as the usual 5-shot
benchmark) and stops the server. The first start of a model compiles for
several minutes; later starts reuse the compile cache. The MMLU data is
downloaded on first use.

## Topology

The scripts bind one tensor-parallel rank to each CBB (die) when the host has
more dies than NUMA nodes (e.g. Diamond Rapids), and to each NUMA node
otherwise (e.g. Granite Rapids). TP is the largest power of two that fits the
instance count, so that it divides the head counts. The engine core sleeps
between requests with die binding and spins with NUMA binding. Everything below
can be overridden through the environment.

Per host, the scripts pick the binding that the reference runs used:

| Host | Detected binding | Equivalent explicit setting |
|---|---|---|
| Diamond Rapids (256 cores, 4 dies) | one rank per die (`cbb`), engine core sleeps | `TP=4 BIND_GRANULARITY=cbb VLLM_MQ_BUSY_LOOP_SECONDS=0` |
| Granite Rapids (384 cores, 6 NUMA nodes) | one rank per NUMA node (`numa`), engine core spins, 4 of the 6 nodes used | `TP=4 BIND_GRANULARITY=numa VLLM_MQ_BUSY_LOOP_SECONDS=1.0` |

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `OUTPUT_DIR` | `./cpu_inference_results` | Logs and JSON results |
| `INPUT_TOKEN` / `OUTPUT_TOKEN` | 1024 / 1024 | Prompt and generation length (latency, throughput) |
| `BATCH_SIZE` | 1 (latency), 30 or 25 (throughput) | Batch size |
| `TP` | detected | Tensor-parallel size (exclusive with `DP`) |
| `DP` | unset | Data-parallel instances for `throughput` (see below; exclusive with `TP`) |
| `KV_KIB_PER_TOKEN` | per model | KV cache bytes per token and rank, in KiB; sizes the KV cache (see below) |
| `BIND_GRANULARITY` | detected (`cbb` or `numa`) | Thread binding granularity |
| `ACC_PER_SUBJECT` | 40 | MMLU questions per subject, 0 for all |
| `ACC_CONCURRENCY` | 64 | Concurrent MMLU requests |
| `MMLU_DATA_DIR` | `$OUTPUT_DIR/mmlu_data` | MMLU data (downloaded if missing) |
| `PORT` | 8000 | Server port for `acc` |
| `DRY_RUN` | 0 | `1` prints the commands without running them |

Results are written to `$OUTPUT_DIR/<model>_<precision>_bs<N>_in<I>_out<O>.{json,log}`
for `latency`/`throughput` and `$OUTPUT_DIR/<model>_<precision>_mmlu.{json,log}`
(plus `_server.log`) for `acc`.

## KV cache size

`latency` and `throughput` pass `--kv-cache-memory-bytes` to `vllm bench
one-batch`: batch size x (input + output tokens) x `KV_KIB_PER_TOKEN` x 1.25
(block rounding). Without it every rank would claim its whole memory share for
the KV cache, which can exhaust the host memory with several ranks or DP
instances on one node (for example DP=4 at batch size 320). The per-model
defaults come from the auto-sized caches (KiB per token and rank: Llama 128,
Qwen3.5-9B 70, Qwen3.5-35B-A3B 57, gemma-4-26B-A4B 233, DeepSeek-R1 73); set
`KV_KIB_PER_TOKEN=` (empty) to size the cache from `--gpu-memory-utilization`
instead.

## Data-parallel throughput

`DP=<n>` replaces tensor parallelism in the `throughput` mode: it starts `n`
independent `vllm bench one-batch` instances with TP=1, one per die (or NUMA
node), waits for all of them and reports the sum of their decode throughputs.
Every instance runs the full batch size (`BATCH_SIZE` is not divided by `n`),
so the total batch is `n` times larger. `DP` and `TP` cannot be set together, and
`DP` is rejected for `latency` and `acc` and for DeepSeek-R1 (it needs all dies).

```bash
DP=4 bash llama3.1-8b.sh throughput bf16
```

Each instance sees only its memory node (`CPU_VISIBLE_MEMORY_NODES`) and, with
die binding, only the cores of its die (`taskset`); vLLM then binds the
threads automatically. Results are written to
`$OUTPUT_DIR/<model>_<precision>_dp<n>_bs<N>_in<I>_out<O>.json` (the sum and the
per-instance results) and `..._<i>.{json,log}` per instance. The instances are
only launched together, not synchronized, so the sum is accurate when their
decode phases overlap, which holds for instances of equal speed.

## Reproducing the reference runs

All reference runs use 1024 input and 1024 output tokens, greedy sampling and
0 reserved cores. TP runs use TP=4; pass it explicitly to make that independent
of the topology detection. DP runs use four TP=1 instances with a larger batch
per instance. Run from this folder; results go to `$OUTPUT_DIR`.

```bash
export OUTPUT_DIR=$PWD/results

# TP=4: latency (BS=1)
export TP=4
bash llama3.1-8b.sh latency bf16
bash llama3.1-8b.sh latency fp16
bash llama3.1-8b.sh latency int8
bash llama3.1-8b.sh latency fp8
bash qwen3.5-9b.sh latency
bash qwen3.5-35b-a3b.sh latency bf16
bash qwen3.5-35b-a3b.sh latency fp8
bash gemma4-26b-a4b.sh latency
bash deepseek-r1.sh latency

# TP=4: throughput (BS=30 for Llama, BS=25 for the others)
bash llama3.1-8b.sh throughput bf16
bash llama3.1-8b.sh throughput fp16
bash llama3.1-8b.sh throughput int8
bash llama3.1-8b.sh throughput fp8
bash qwen3.5-9b.sh throughput
bash qwen3.5-35b-a3b.sh throughput bf16
bash qwen3.5-35b-a3b.sh throughput fp8
bash gemma4-26b-a4b.sh throughput
bash deepseek-r1.sh throughput

# TP=4: MMLU 5-shot (40 questions per subject, 2280 questions)
bash llama3.1-8b.sh acc bf16
bash llama3.1-8b.sh acc fp16
bash llama3.1-8b.sh acc int8
bash llama3.1-8b.sh acc fp8
bash qwen3.5-9b.sh acc
bash qwen3.5-35b-a3b.sh acc bf16
bash qwen3.5-35b-a3b.sh acc fp8
bash gemma4-26b-a4b.sh acc
bash deepseek-r1.sh acc

# DP=4: throughput with a larger batch per instance (no TP; not for DeepSeek-R1)
unset TP
export DP=4
BATCH_SIZE=348 bash llama3.1-8b.sh throughput bf16
BATCH_SIZE=348 bash llama3.1-8b.sh throughput fp16
BATCH_SIZE=296 bash llama3.1-8b.sh throughput int8
BATCH_SIZE=296 bash llama3.1-8b.sh throughput fp8
BATCH_SIZE=108 bash qwen3.5-9b.sh throughput
BATCH_SIZE=320 bash qwen3.5-35b-a3b.sh throughput bf16
BATCH_SIZE=320 bash qwen3.5-35b-a3b.sh throughput fp8
BATCH_SIZE=96 bash gemma4-26b-a4b.sh throughput
```

Expected MMLU (2280-question subset): about 0.68–0.69 for every Llama variant,
0.59–0.60 for Qwen3.5-9B, 0.81–0.85 for Qwen3.5-35B-A3B and 0.78 for
gemma-4-26B-A4B and 0.87 for DeepSeek-R1.
