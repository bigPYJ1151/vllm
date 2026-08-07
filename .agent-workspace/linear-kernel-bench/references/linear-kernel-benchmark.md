# oneDNN vs SGLang AMX unquantized-linear GEMM benchmark (CPU)

## Background

PR [#50801](https://github.com/vllm-project/vllm/pull/50801) ("[CPU] Refine
CPU kernel dispatch") removed the SGLang AMX `weight_packed_linear` path
from vLLM's unquantized (bf16/fp16) linear dispatch
(`dispatch_cpu_unquantized_gemm` in
`vllm/model_executor/layers/utils.py`), making oneDNN (`onednn_mm`) the sole
path. The SGLang kernel is still vendored and built
(`csrc/cpu/sgl-kernels/`) and still used for the quantized int8 path — only
the unquantized bf16/fp16 dispatch lost the SGL option.

This benchmark re-evaluates that decision at the kernel level (not
end-to-end serving): does oneDNN dominate SGL across the shapes vLLM
actually produces, or is there a shape/dtype region where re-adding a
dispatch branch would help?

A separate confound was found while scoping this: SGLang's own upstream
build compiles the same vendored kernel sources with `-O3
-march=x86-64-v4 ...`, while vLLM's `cmake/cpu_extension.cmake` compiles the
identical sources under the default `RelWithDebInfo` build type (stock
`-O2 -g -DNDEBUG`). To separate "kernel design difference" from "compiler
flag gap," every sweep below was run against **both** build configs.

## Scope

- **Kernels**: `ops.onednn_mm` (oneDNN) vs `torch.ops._C.weight_packed_linear`
  (SGLang AMX), the exact production call patterns used by
  `dispatch_cpu_unquantized_gemm`.
- **Dtypes**: bf16 and fp16 only (`check_cpu_sgl_kernel`'s supported
  unquantized dtypes; SGLang has no fp32 path).
- **TP sizes**: 1, 2, 4, 8.
- **Linear types**: `qkv`, `o_proj`, `gate_up`, `down`, `moe_gate` (MoE
  router/gate only — see exclusions), `lm_head`.
- **M sweep**: log-spaced, bracketing vLLM's CPU scheduler defaults per TP
  degree (`max_num_seqs=256*tp` decode, `max_num_batched_tokens=4096*tp`
  prefill) — see `model_shapes.py::m_sweep`/`lm_head_m_sweep`.
- **Build configs**: default `-O2` (`O2-default`) and
  `CMAKE_BUILD_TYPE=Release` → `-O3` (`O3-release`).
- **CPU/NUMA bindings**: `numactl --cpunodebind=0 --membind=0` with
  `--physcpubind=192-223` (32 threads, full NUMA node 0 HT-sibling range)
  and `--physcpubind=192-222` (31 threads, one fewer).
- **Repeats**: 5 independent trials per case, geometric mean of per-trial
  medians reported (noise-robust for latency ratios).

Total: 2 builds × 2 bindings = 4 full sweeps, 5480 cases each (348 shapes ×
2 dtypes × ~4 repeats-worth of derived M values — see per-sweep manifests),
21920 rows overall.

### Exclusions

- **MoE per-expert FC1/FC2** (inside routed experts) — `cpu_moe.py`
  unconditionally uses `convert_weight_packed`/`fused_experts_cpu`; it never
  had oneDNN as a competing path, so there's no dispatch question there.
  Only the **router/gate** linear (hidden → num_experts) is in scope.
- **GLM router excluded**: GLM4-MoE's gate is a raw `torch.nn.Linear`, not
  a vLLM `LinearBase` subclass — it never reaches
  `dispatch_cpu_unquantized_gemm`. GLM-5.2-FP8 has no `moe_gate` row.
- **lm_head excluded for embedding/reranking-only models**
  (Qwen3-Embedding-8B, bge-reranker-base) — their serving paths never
  invoke the vocab-projection GEMM.
- Token embedding lookup (`input_ids → hidden`) is a pure gather, not a
  GEMM — out of scope regardless of model.

## Model shape table (TP=1 baseline; K→N per linear type)

| Model | vocab_size | qkv (K→N) | o_proj (K→N) | gate_up (K→N) | down (K→N) | moe_gate (K→N) | lm_head (K→N, padded-to-64 vocab) | in-scope types |
|---|---|---|---|---|---|---|---|---|
| Qwen3.6-35B-A3B (≈Qwen3-30B-A3B) | 151936 | 2048→5120 | 4096→2048 | — | — | 2048→128 | 2048→151936 | qkv, o_proj, moe_gate, lm_head |
| gemma-4-26B-A4B-it (local attn) | 262144 | 2816→8192 | 4096→2816 | 2816→4224 (shared FFN, gated) | 2112→2816 | 2816→128 | 2816→262144 | qkv_local, qkv_global, o_proj_local, o_proj_global, gate_up, down, moe_gate, lm_head |
| gemma-4-26B-A4B-it (global attn) | 262144 | 2816→10240 | 8192→2816 | *(same)* | *(same)* | *(same)* | *(same)* | *(see above)* |
| gpt-oss-20b | 201088 | 2880→5120 | 4096→2880 | — | — | 2880→32 | 2880→201088 | qkv, o_proj, moe_gate, lm_head |
| granite-guardian-4.1-8b (assumed dense) | 49155 | 4096→6144 | 4096→4096 | 4096→25600 (gated, assumed) | 12800→4096 | n/a | 4096→49216 | qkv, o_proj, gate_up, down, lm_head |
| bge-reranker-base (≈XLM-R-base) | 250002 | 768→2304 | 768→768 | 768→3072 (non-gated) | 3072→768 | n/a | n/a (excluded) | qkv, o_proj, gate_up, down |
| whisper-large-v3 | 51866 | 1280→3840 | 1280→1280 | 1280→5120 (non-gated) | 5120→1280 | n/a | 1280→51904 | qkv, o_proj, gate_up, down, lm_head |
| Mistral-7B-Instruct-v0.3 | 32768 | 4096→6144 | 4096→4096 | 4096→28672 (gated) | 14336→4096 | n/a | 4096→32768 | qkv, o_proj, gate_up, down, lm_head |
| Qwen3-Embedding-8B | 151936 | 4096→6144 | 4096→4096 | 4096→24576 (gated) | 12288→4096 | n/a | n/a (excluded) | qkv, o_proj, gate_up, down |
| Llama-3.1-8B-Instruct | 128256 | 4096→6144 | 4096→4096 | 4096→28672 (gated) | 14336→4096 | n/a | 4096→128256 | qkv, o_proj, gate_up, down, lm_head |
| GLM-5.2-FP8 (≈GLM-4.5/4.6 analog, bf16 shapes) | 151552 | 4096→14336 | 12288→4096 | 4096→21888 (gated, 1 dense layer) | 10944→4096 | — (excluded) | 4096→151552 | qkv, o_proj, gate_up, down, lm_head |
| phi-4 (fused qkv_proj) | 100352 | 5120→7680 | 5120→5120 | 5120→35840 (gated, assumed) | 17920→5120 | n/a | 5120→100352 | qkv, o_proj, gate_up, down, lm_head |

`N` values shown are TP=1 (unsharded); `model_shapes.py::shard()` applies
per-linear-type TP rules (column-parallel divides N, row-parallel divides
K, `moe_gate` is `ReplicatedLinear` and never sharded, `lm_head`'s vocab
dim is padded to a multiple of 64 before dividing by `tp_size`).

## Methodology

Each `(model, linear_type, tp_size, dtype, M)` case:

1. `weight = torch.randn((N,K), dtype) / (0.5*K**0.5)`, same for `x`.
2. If `check_cpu_sgl_kernel(N, K, dtype)` is false (AMX unsupported, or
   `K % 32 != 0` / `N % 16 != 0`), only oneDNN is timed;
   `sgl_skipped_reason` records why.
3. oneDNN: `handler = ops.create_onednn_mm(weight.t(), 32)` once, then
   `ops.onednn_mm(handler, x, bias)` timed (bias same dtype as `x`).
4. SGL: `packed = torch.ops._C.convert_weight_packed(weight)` once, then
   `torch.ops._C.weight_packed_linear(x, packed, bias_f32, True)` timed
   (bias **must** be float32 — `gemm.cpp` reads it as `float*` with no
   dtype check).
5. Repeated 5× with fresh tensors (incrementing seed) per case; the
   reported latency is the **geometric mean of the 5 trial medians**
   (each trial itself: 5 warmup + 20 timed iterations).
6. `speedup_onednn_over_sgl = sgl_geomean_ms / onednn_geomean_ms` — **>1
   means oneDNN is faster** (oneDNN's time is the denominator; a larger
   ratio means oneDNN completed in less time). `winner` is `onednn` when
   this ratio is `>1.02`, `sgl` when `<0.98`, else `tie`.

### Why two CPU bindings

`--physcpubind=192-223` vs `192-222` isolates whether the relative
oneDnn/SGL comparison is sensitive to exact thread count (32 vs 31 on the
same 32-physical-core NUMA node) — a sanity check that neither kernel's
relative advantage is an artifact of one specific thread count.

### Methodology caveat: thread-pinning (`OMP_PROC_BIND`/`KMP_AFFINITY`)

Mid-benchmark, we tested whether adding explicit per-core thread pinning
(`OMP_PROC_BIND=true`, on top of the existing `numactl --physcpubind`
binding) would materially change results. It should not be used as-is in
this container:

- The container preloads Intel's `libiomp5.so`, not GNU libgomp. Bare
  `OMP_PROC_BIND=true` (without `OMP_NUM_THREADS`) silently collapsed
  `torch.get_num_threads()` from 32 down to 1. Setting
  `OMP_NUM_THREADS=32` explicitly alongside it caused all 33 native
  threads to bind to a **single CPU** instead of spreading across the
  `numactl` range (verified via each thread's `Cpus_allowed_list` in
  `/proc/self/task/*/status`) — `OMP_PLACES=cores` did not fix this either.
  Both misconfigurations caused a 13-21x geomean slowdown.
- The correct Intel-native equivalent is
  `KMP_AFFINITY=granularity=fine,compact,1,0` with `OMP_NUM_THREADS=32`,
  which correctly spread one thread per core across the full `numactl`
  range.
- Compared properly (`KMP_AFFINITY` config vs the `numactl`-only baseline
  used for all sweeps below), the difference was negligible: geomean
  ratio 1.016 (oneDNN) / 1.008 (SGL), winner-label flips on only ~7% of
  cases (consistent with noise near near-tie cases, no systematic bias).

**Conclusion**: none of the sweeps below use explicit `OMP_PROC_BIND`/
`KMP_AFFINITY` pinning beyond the `numactl --physcpubind` binding already
in place; the quick A/B test above confirms this doesn't change the
results materially, and bare `OMP_PROC_BIND=true` is actively harmful in
this environment rather than a legitimate additional tuning knob.

### Correctness guardrail

Every sweep run opens with a check that `onednn_mm`, `weight_packed_linear`,
and an fp32 `torch.nn.functional.linear` reference all agree (loose
tolerance for the AMX/bf16 legs) for one representative shape, before
proceeding — this would catch a broken/stale build before burning
CPU-hours on a bad sweep. All 4 sweeps passed this guardrail.

## Arithmetic intensity (compute/memory-access ratio)

Every case's `arithmetic_intensity_flops_per_byte` column (added to all 4
result CSVs and to `summary_heuristic.md`'s bucket table) is the
naive/no-reuse roofline ratio for that one GEMM call:

```
FLOPs        = 2 * M * N * K                      (multiply-add)
bytes        = elem_size * (M*K + N*K + M*N)      (one read of x and weight, one write of output; bias is O(N), negligible)
AI (FLOPs/B) = FLOPs / bytes
             = M*N*K / (M*K + N*K + M*N)           (bf16 and fp16 are both 2-byte elements, so the ratio is identical for both dtypes tested here)
```

This is a **lower bound**, not a measurement — it assumes the weight is
re-read from memory on every call with no cache residency. Both kernels
actually pack/cache the weight once per case (`create_onednn_mm`,
`convert_weight_packed`) and reuse it across the 20 timed iterations per
trial, so real cache behavior is more favorable than this formula assumes;
it's included as a shape-only "how compute-bound is this GEMM, all else
equal" signal, not an absolute FLOPs/byte the CPU actually sustains.

Representative median AI per bucket (`O2-default`/`cpu192-223`/bf16 — AI is
build/binding/dtype-invariant, only shape-dependent):

| linear_type | tp | decode median AI | mid median AI |
|---|---|---|---|
| `qkv` | 1 | 31.6 | 722.8 |
| `qkv` | 8 | 69.8 | 622.2 |
| `down` | 1 | 31.7 | 768.0 |
| `down` | 8 | 83.6 | 955.7 |
| `moe_gate` | 1 | 25.3 | 109.4 |
| `moe_gate` | 8 | 30.7 | 119.6 |
| `lm_head` | 1 | 31.7 | *(no mid bucket — see below)* |
| `lm_head` | 8 | 114.3 | *(no mid bucket)* |

This lines up with the qualitative pattern in the Summary heuristic below:

- `qkv`/`o_proj`/`gate_up`/`down`'s `mid` bucket sits at AI≈600-1150 —
  clearly compute-bound — and that's exactly where oneDNN wins decisively.
  Their `decode` bucket spans AI≈30-85, i.e. genuinely memory-bound, which
  is where SGL is competitive-to-ahead.
- `moe_gate` never gets much above AI≈110-120 even in its `mid` bucket,
  because `N` (num_experts, 32-128) is tiny relative to `K` — the
  weight-read and output-write terms never get diluted by a large `N` the
  way the other linear types' `mid` bucket does. `moe_gate` stays in a
  memory-bound-ish regime across the entire M range sampled, consistent
  with SGL winning it everywhere rather than losing ground at higher M.
- `lm_head`'s AI (≈30-115) overlaps the same intermediate range as
  `qkv`/`down`'s ambiguous `decode` bucket, consistent with `lm_head`'s own
  near-toss-up result. `lm_head` never has a `mid`-bucket row at all —
  `lm_head_m_sweep()` caps at `256*tp` because `lm_head` only ever sees one
  row per sequence (or a handful, for logprobs), never a full prefill
  chunk — so there's no larger-M/higher-AI regime to sample for this
  linear type in the first place (see Caveats).

## How to reproduce

```bash
# Inside the vllm-dev-env dev container, from the worktree root:
cd .agent-workspace/linear-kernel-bench   # or scripts/linear-kernel-bench/ once copied to the skill dir

# Default (-O2) build:
uv pip uninstall vllm || true
unset CMAKE_BUILD_TYPE
VLLM_TARGET_DEVICE=cpu python3 setup.py develop
python3 sanity_check.py
./run_sweep.sh O2-default 192-223 cpu192-223
./run_sweep.sh O2-default 192-222 cpu192-222

# Release (-O3) build:
uv pip uninstall vllm || true
CMAKE_BUILD_TYPE=Release VLLM_TARGET_DEVICE=cpu python3 setup.py develop
python3 sanity_check.py
./run_sweep.sh O3-release 192-223 cpu192-223
./run_sweep.sh O3-release 192-222 cpu192-222

# Aggregate:
python3 analyze_results.py   # writes results/summary_heuristic.md
```

## Results

Full per-case tables (5480 rows each; `Model | Linear | TP | dtype | M | K
| N | oneDNN (ms) | SGL (ms) | Speedup (oneDNN/SGL) | Winner | TFLOP/s
(oneDNN) | TFLOP/s (SGL) | AI (FLOPs/B)`, sorted `linear_type -> model ->
tp_size -> dtype -> M`):

- [`results/O2-default__cpu192-223.md`](../results/O2-default__cpu192-223.md)
  / [`.csv`](../results/O2-default__cpu192-223.csv)
- [`results/O2-default__cpu192-222.md`](../results/O2-default__cpu192-222.md)
  / [`.csv`](../results/O2-default__cpu192-222.csv)
- [`results/O3-release__cpu192-223.md`](../results/O3-release__cpu192-223.md)
  / [`.csv`](../results/O3-release__cpu192-223.csv)
- [`results/O3-release__cpu192-222.md`](../results/O3-release__cpu192-222.md)
  / [`.csv`](../results/O3-release__cpu192-222.csv)

Each has an accompanying `manifest_<tag>.json` recording git revision (all
4: `7e724dca897dda9e3fd05abcfe5b0f6d77c564b1`, clean tree), the actual
`CMAKE_BUILD_TYPE`, and the runtime-confirmed `Cpus_allowed_list`.

## Summary heuristic

Full bucketed table (all 4 build×binding combos, by `linear_type`/`tp`/
`dtype`/M-bucket):
[`results/summary_heuristic.md`](../results/summary_heuristic.md).

Bucket boundaries: `decode = M <= 256*tp`, `mid = 256*tp < M <= 4096*tp`,
`prefill = M > 4096*tp` — the M-sweep never generates a value strictly
greater than `4096*tp`, so the `prefill` bucket never appears; everything
the sweep covers falls in `decode` or `mid`. `lm_head` never has a `mid`
row either — its own M-sweep (`lm_head_m_sweep()`) is deliberately capped
at `256*tp` (see Caveats), so all `lm_head` rows fall in `decode`.

By linear type (pattern consistent across bf16/fp16 and O2/O3 — dtype and
build flags don't change which kernel wins, see sensitivity sections
below):

- **`moe_gate`**: SGL wins decisively and consistently — win-rate
  62-100%, geomean speedup 0.65-1.05 (i.e. SGL usually *faster* by
  magnitude too, not just by case count), across every tp size and both
  `decode`/`mid` buckets. This is the one linear type where SGL is a clear,
  unambiguous win everywhere sampled.
- **`qkv` / `o_proj` / `gate_up` / `down`**: `mid` bucket is a decisive
  oneDNN win (win-rate 69-100% for oneDNN, geomean speedup 1.05-1.64).
  `decode` bucket is murkier: SGL wins the *majority of individual cases*
  by count (60-87% SGL win-rate) but the bucket's geomean is often near
  parity or even leans oneDNN (0.85-1.06) — a look at the raw per-M-value
  numbers shows why: at very small M (roughly ≤32), SGL has a slight edge
  (a few percent); from M≈64 up through the top of the `decode` range
  (256×tp), oneDNN pulls decisively ahead (up to ~2x at M=256, tp=1). The
  `decode` bucket's win-rate is being pulled toward "SGL" by the sheer
  count of very-small-M samples, while the larger-magnitude wins within
  the same bucket belong to oneDNN.
- **`lm_head`**: closest to a toss-up. Win-rate hovers around 44-56% and
  geomean speedup 1.06-1.12 (mild oneDNN lean) across tp sizes; no
  decisive winner.

## Build-flag sensitivity (O2 vs O3)

Comparing the same `(cpu_binding_tag, dtype, linear_type, tp_size,
M-bucket)` geomean-speedup value across `O2-default` and `O3-release`
(176 matched buckets): median delta **0.57%**, p90 **1.72%**, max **3.45%**.
The `-O2`/`-O3` compiler-flag gap between vLLM's default build and
SGLang's upstream build does **not** meaningfully change which kernel wins
or by how much — the earlier build-flag confound this benchmark set out to
separate turns out not to matter for this comparison. Full delta table:
[`results/summary_heuristic.md`](../results/summary_heuristic.md#build-flag-sensitivity-o2-vs-o3-same-cpu-binding).

## CPU-binding sensitivity (192-223 vs 192-222)

Comparing the same `(build_tag, dtype, linear_type, tp_size, M-bucket)`
geomean-speedup value across the two `numactl` bindings (176 matched
buckets): median delta **1.50%**, p90 **3.84%**, max **11.82%**. Mostly
noise-level; a handful of buckets (mostly `down`, higher tp, `mid` bucket)
show up to ~12% delta, which is plausibly run-to-run variance rather than
a systematic 31-vs-32-thread effect, consistent with the OMP-pinning A/B
test above showing similar-magnitude noise between otherwise-identical
runs. Full delta table:
[`results/summary_heuristic.md`](../results/summary_heuristic.md#cpu-binding-sensitivity-192-223-vs-192-222-same-build).

## Dispatch recommendation heuristic (not implemented in vLLM yet)

This is a candidate recommendation only — no vLLM code has been touched.

1. **MoE router/gate (`moe_gate`)**: re-add SGL as the dispatch target.
   This is the one linear type with a clean, consistent, magnitude-backed
   win for SGL across every tp size, dtype, and M sampled. Scope note:
   only applies to models whose gate is a vLLM `ReplicatedLinear`
   (confirmed for Qwen3-MoE-style, gpt-oss-style gates) — GLM-style raw
   `nn.Linear` gates never reach this dispatch function at all.
2. **`qkv` / `o_proj` / `gate_up` / `down`**: the data doesn't support a
   clean shape-based dispatch win. oneDNN decisively wins the bulk of the
   practical M range (`mid` bucket, and the upper half of `decode`); SGL's
   edge is confined to a thin low-M slice (M≲32) at a small (~2-5%)
   margin. Given the margin size, adding dispatch complexity for this
   slice is unlikely to be worth it — **no change recommended**; oneDNN
   staying the sole path for these types is reasonable.
3. **`lm_head`**: too close to call either way — **no change
   recommended**.

Net: of the two dispatch questions this benchmark set out to answer
("was removing SGL from unquantized-linear dispatch the right call?"),
the answer is "yes, for the non-MoE linear types the removal was correct
or at worst neutral" and "no, for the MoE router/gate specifically, SGL
looks like it should have stayed (or should be re-added)."

## Caveats

- **`lm_head` never reaches the `mid`/`prefill` M-bucket**: unlike the
  other linear types, `lm_head_m_sweep()` intentionally caps M at
  `256*tp` — `lm_head` only ever computes logits for one row per sequence
  (or a handful, for `logprobs`/multi-token proposals), never a full
  prefill-length batch, so there's no realistic larger-M shape to sample.
  This isn't a gap in sweep coverage, just a reflection of how `lm_head`
  is actually called in practice.
- **Winner-label correction**: an earlier draft of this benchmark's
  scripts had the `winner`/`recommendation` labeling logic inverted
  (comparing `speedup_onednn_over_sgl > 1.02` and calling it an SGL win,
  when a ratio `>1` — `sgl_ms/onednn_ms` — means oneDNN took *less* time).
  This was caught by cross-checking raw `onednn_geomean_ms`/
  `sgl_geomean_ms`/TFLOP/s values directly against the `winner` column
  before writing this doc. The underlying timing data was never affected
  (only the derived label); all 4 result CSVs and `summary_heuristic.md`
  were relabeled from the already-correct ratio column, no kernels were
  re-run. All conclusions above reflect the corrected labeling.
- **Assumption-based shapes**: `granite-guardian-4.1-8b` (assumed dense,
  gated FFN) and `GLM-5.2-FP8` (shapes approximated from the closest
  GLM-4.5/4.6 analog) have no public config to derive exact shapes from —
  flagged, not re-derived.
- **Gated-vs-non-gated FFN width**: assumed SwiGLU-gated (×2 N) for
  Llama/Mistral/Qwen3/Granite/Phi/gemma-shared/GLM-dense families;
  non-gated (×1 N) for BERT/XLM-R-style (bge-reranker) and Whisper.
- **GLM router excluded** (see Scope) — not a gap in the benchmark, GLM's
  gate genuinely never reaches this dispatch path.
- **MoE per-expert FC1/FC2 excluded** (see Scope) — no oneDNN path exists
  there to compare against.
- Random weight/activation tensors (not real trained weights) — this is a
  pure kernel microbenchmark; numerically-realistic (vs. random) data
  distributions were not tested and could plausibly move the AMX-vs-oneDNN
  ratio slightly, though the kernels themselves are data-independent in
  their control flow (no sparsity/branching on values), so this is a
  low-risk gap.
