"""Model linear-layer shapes and TP-sharding helpers for the oneDNN vs
SGLang AMX unquantized-GEMM benchmark. Standalone data/helper module, not
part of the vLLM package.

Shapes are TP=1 baselines derived from vLLM's modeling code (or the
closest real analog where a target checkpoint name has no public config,
noted per entry). `shard()` applies vLLM's actual TP-sharding rules
(vllm/model_executor/layers/linear.py, vocab_parallel_embedding.py) to
produce the (K, N) GEMM shape at any tp_size in {1, 2, 4, 8}.
"""

from dataclasses import dataclass

DEFAULT_VOCAB_PADDING_SIZE = 64


def pad_vocab_size(vocab_size: int, pad_to: int = DEFAULT_VOCAB_PADDING_SIZE) -> int:
    return ((vocab_size + pad_to - 1) // pad_to) * pad_to


def _divide(x: int, tp_size: int) -> int:
    assert x % tp_size == 0, f"{x} not divisible by tp_size={tp_size}"
    return x // tp_size


@dataclass
class ModelShape:
    name: str
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    vocab_size: int
    in_scope: list[str]
    gate_up: tuple[int, int] | None = None  # (K, N) at TP=1
    down: tuple[int, int] | None = None  # (K, N) at TP=1
    moe_gate: tuple[int, int] | None = None  # (K, N) at TP=1, unsharded
    is_moe: bool = False
    note: str = ""


MODEL_TABLE = [
    ModelShape(
        name="Qwen3.6-35B-A3B",
        hidden_size=2048,
        num_heads=32,
        num_kv_heads=4,
        head_dim=128,
        vocab_size=151936,
        moe_gate=(2048, 128),
        is_moe=True,
        in_scope=["qkv", "o_proj", "moe_gate", "lm_head"],
        note="shapes from Qwen3-30B-A3B analog; no public config for the "
        "exact checkpoint name",
    ),
    ModelShape(
        name="gemma-4-26B-A4B-it (local attn)",
        hidden_size=2816,
        num_heads=16,
        num_kv_heads=8,
        head_dim=256,
        vocab_size=262144,
        gate_up=(2816, 4224),
        down=(2112, 2816),
        moe_gate=(2816, 128),
        is_moe=True,
        in_scope=["qkv", "o_proj", "gate_up", "down", "moe_gate", "lm_head"],
        note="assumed shapes; sliding-window (local) attention layer variant",
    ),
    ModelShape(
        name="gemma-4-26B-A4B-it (global attn)",
        hidden_size=2816,
        num_heads=32,
        num_kv_heads=4,
        head_dim=256,
        vocab_size=262144,
        gate_up=(2816, 4224),
        down=(2112, 2816),
        moe_gate=(2816, 128),
        is_moe=True,
        in_scope=["qkv", "o_proj", "gate_up", "down", "moe_gate", "lm_head"],
        note="assumed shapes; global-attention layer variant",
    ),
    ModelShape(
        name="gpt-oss-20b",
        hidden_size=2880,
        num_heads=64,
        num_kv_heads=8,
        head_dim=64,
        vocab_size=201088,
        moe_gate=(2880, 32),
        is_moe=True,
        in_scope=["qkv", "o_proj", "moe_gate", "lm_head"],
    ),
    ModelShape(
        name="granite-guardian-4.1-8b",
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        vocab_size=49155,
        gate_up=(4096, 25600),
        down=(12800, 4096),
        in_scope=["qkv", "o_proj", "gate_up", "down", "lm_head"],
        note="assumed dense (Llama-style) architecture; no public config",
    ),
    ModelShape(
        name="bge-reranker-base",
        hidden_size=768,
        num_heads=12,
        num_kv_heads=12,
        head_dim=64,
        vocab_size=250002,
        gate_up=(768, 3072),
        down=(3072, 768),
        in_scope=["qkv", "o_proj", "gate_up", "down"],
        note="XLM-R-base analog; non-gated FFN; classifier head excluded "
        "(tiny, not comparable across models)",
    ),
    ModelShape(
        name="whisper-large-v3",
        hidden_size=1280,
        num_heads=20,
        num_kv_heads=20,
        head_dim=64,
        vocab_size=51866,
        gate_up=(1280, 5120),
        down=(5120, 1280),
        in_scope=["qkv", "o_proj", "gate_up", "down", "lm_head"],
        note="Conv1d feature-extraction stem excluded (never dispatched "
        "through dispatch_cpu_unquantized_gemm)",
    ),
    ModelShape(
        name="Mistral-7B-Instruct-v0.3",
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        vocab_size=32768,
        gate_up=(4096, 28672),
        down=(14336, 4096),
        in_scope=["qkv", "o_proj", "gate_up", "down", "lm_head"],
    ),
    ModelShape(
        name="Qwen3-Embedding-8B",
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        vocab_size=151936,
        gate_up=(4096, 24576),
        down=(12288, 4096),
        in_scope=["qkv", "o_proj", "gate_up", "down"],
        note="embedding-serving path never invokes lm_head; excluded",
    ),
    ModelShape(
        name="Llama-3.1-8B-Instruct",
        hidden_size=4096,
        num_heads=32,
        num_kv_heads=8,
        head_dim=128,
        vocab_size=128256,
        gate_up=(4096, 28672),
        down=(14336, 4096),
        in_scope=["qkv", "o_proj", "gate_up", "down", "lm_head"],
    ),
    ModelShape(
        name="GLM-5.2-FP8",
        hidden_size=4096,
        num_heads=96,
        num_kv_heads=8,
        head_dim=128,
        vocab_size=151552,
        gate_up=(4096, 21888),
        down=(10944, 4096),
        in_scope=["qkv", "o_proj", "gate_up", "down", "lm_head"],
        note="bf16 shapes from GLM-4.5/4.6 analog (dense layer only); MoE "
        "router excluded — raw nn.Linear, not dispatched through vLLM's "
        "CPU GEMM path",
    ),
    ModelShape(
        name="phi-4",
        hidden_size=5120,
        num_heads=40,
        num_kv_heads=10,
        head_dim=128,
        vocab_size=100352,
        gate_up=(5120, 35840),
        down=(17920, 5120),
        in_scope=["qkv", "o_proj", "gate_up", "down", "lm_head"],
        note="fused qkv_proj; gate_up width assumed gated (SwiGLU)",
    ),
]


def shard(model: ModelShape, linear_type: str, tp_size: int) -> tuple[int, int]:
    """Return (K, N) for one linear type at the given tp_size.

    Mirrors vllm/model_executor/layers/linear.py's sharding rules:
    QKVParallelLinear divides num_heads/num_kv_heads by tp_size (KV
    replicated once tp_size exceeds total kv heads); MergedColumnParallelLinear
    (gate_up) divides output by tp_size; RowParallelLinear (o_proj, down)
    divides input by tp_size; the MoE router gate is a ReplicatedLinear,
    unsharded; lm_head pads vocab_size to a multiple of 64 before dividing
    by tp_size (vocab_parallel_embedding.py).
    """
    if linear_type == "qkv":
        num_heads = max(model.num_heads // tp_size, 1)
        num_kv_heads = max(model.num_kv_heads // tp_size, 1)
        q_size = num_heads * model.head_dim
        kv_size = num_kv_heads * model.head_dim
        return model.hidden_size, q_size + 2 * kv_size
    if linear_type == "o_proj":
        num_heads = max(model.num_heads // tp_size, 1)
        return num_heads * model.head_dim, model.hidden_size
    if linear_type == "gate_up":
        k, n = model.gate_up
        return k, _divide(n, tp_size)
    if linear_type == "down":
        k, n = model.down
        return _divide(k, tp_size), n
    if linear_type == "moe_gate":
        return model.moe_gate
    if linear_type == "lm_head":
        n_padded = pad_vocab_size(model.vocab_size)
        return model.hidden_size, _divide(n_padded, tp_size)
    raise ValueError(f"unknown linear_type: {linear_type}")


BASE_M = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def m_sweep(tp_size: int) -> list[int]:
    """M-sweep for qkv/o_proj/gate_up/down/moe_gate: log-spaced, bracketing
    both the decode default (256*tp) and prefill default (4096*tp)."""
    prefill_default, decode_default = 4096 * tp_size, 256 * tp_size
    m = set(BASE_M) | {decode_default, prefill_default}
    v = 4096
    while v < prefill_default:
        v *= 2
        m.add(v)
    return sorted(m)


def lm_head_m_sweep(tp_size: int) -> list[int]:
    """M-sweep for lm_head: capped at the decode default (256*tp), since
    lm_head only ever sees one row per sequence (or a handful, for
    logprobs) regardless of prefill token count."""
    decode_default = 256 * tp_size
    m = {v for v in BASE_M if v <= decode_default} | {decode_default}
    return sorted(m)


if __name__ == "__main__":
    for tp in (1, 2, 4, 8):
        for model in MODEL_TABLE:
            for lt in model.in_scope:
                k, n = shard(model, lt, tp)
                print(f"{model.name:35s} {lt:10s} tp={tp} K={k:6d} N={n:6d}")
