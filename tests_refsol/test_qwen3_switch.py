from types import SimpleNamespace

import mlx.core as mx

from .tiny_llm_base import Qwen3ModelWeek1, Qwen3ModelWeek2, models
from .utils import assert_allclose


def _quantized_layer(
    out_dim: int, in_dim: int, group_size: int = 64
) -> SimpleNamespace:
    weight = mx.random.normal(shape=(out_dim, in_dim)).astype(mx.bfloat16)
    quantized_weight, scales, biases = mx.quantize(
        weight, group_size=group_size, bits=4
    )
    return SimpleNamespace(
        weight=quantized_weight,
        scales=scales,
        biases=biases,
        group_size=group_size,
        bits=4,
    )


def _fake_qwen3_mlx_model() -> SimpleNamespace:
    mx.random.seed(0)
    args = SimpleNamespace(
        num_hidden_layers=2,
        hidden_size=64,
        vocab_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        rms_norm_eps=1e-5,
        max_position_embeddings=128,
        rope_theta=10000,
        tie_word_embeddings=True,
    )
    embed_tokens = _quantized_layer(args.vocab_size, args.hidden_size)
    kv_hidden_size = args.num_key_value_heads * args.head_dim
    attn_hidden_size = args.num_attention_heads * args.head_dim
    layers = []
    for _ in range(args.num_hidden_layers):
        layers.append(
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    q_proj=_quantized_layer(attn_hidden_size, args.hidden_size),
                    k_proj=_quantized_layer(kv_hidden_size, args.hidden_size),
                    v_proj=_quantized_layer(kv_hidden_size, args.hidden_size),
                    o_proj=_quantized_layer(args.hidden_size, attn_hidden_size),
                    q_norm=SimpleNamespace(
                        weight=mx.ones((args.head_dim,), dtype=mx.bfloat16)
                    ),
                    k_norm=SimpleNamespace(
                        weight=mx.ones((args.head_dim,), dtype=mx.bfloat16)
                    ),
                ),
                mlp=SimpleNamespace(
                    gate_proj=_quantized_layer(
                        args.intermediate_size, args.hidden_size
                    ),
                    up_proj=_quantized_layer(args.intermediate_size, args.hidden_size),
                    down_proj=_quantized_layer(
                        args.hidden_size, args.intermediate_size
                    ),
                ),
                input_layernorm=SimpleNamespace(
                    weight=mx.ones((args.hidden_size,), dtype=mx.bfloat16)
                ),
                post_attention_layernorm=SimpleNamespace(
                    weight=mx.ones((args.hidden_size,), dtype=mx.bfloat16)
                ),
            )
        )
    return SimpleNamespace(
        args=args,
        model=SimpleNamespace(
            embed_tokens=embed_tokens,
            layers=layers,
            norm=SimpleNamespace(
                weight=mx.ones((args.hidden_size,), dtype=mx.bfloat16)
            ),
        ),
    )


def test_qwen3_dispatches_week1_loader():
    mlx_model = _fake_qwen3_mlx_model()

    week1 = models.dispatch_model("qwen3-0.6b", mlx_model, week=1)

    assert isinstance(week1, Qwen3ModelWeek1)


def _normalize_logits(logits: mx.array) -> mx.array:
    return logits - mx.logsumexp(logits, keepdims=True)


def test_qwen3_week1_prefill_matches_week2_cache_loader():
    mlx_model = _fake_qwen3_mlx_model()
    inputs = mx.array([[1, 5, 7, 3]], dtype=mx.int32)
    week1 = Qwen3ModelWeek1(mlx_model)
    week2 = Qwen3ModelWeek2(mlx_model)

    week1_out = _normalize_logits(week1(inputs))
    week2_out = _normalize_logits(week2(inputs, 0, week2.create_kv_cache()))

    assert_allclose(week1_out, week2_out, precision=mx.bfloat16, rtol=0.1, atol=0.5)
