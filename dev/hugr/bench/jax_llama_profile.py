# ---------------------------------------------------------------------------
# Copyright (c) 2026 The Raven authors. All rights reserved.
# SPDX-License-Identifier: ISC
# ---------------------------------------------------------------------------

"""Matched JAX benchmark for Hugr's synthetic Llama-3.2-1B-shaped workload."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


Array = jax.Array
PyTree = Any


@dataclass(frozen=True)
class Dimensions:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


LLAMA3_1B = Dimensions(
    vocab_size=128_256,
    hidden_size=2_048,
    intermediate_size=8_192,
    num_hidden_layers=16,
    num_attention_heads=32,
    num_key_value_heads=8,
    max_position_embeddings=131_072,
    rope_theta=500_000.0,
)

SMOKE = Dimensions(
    vocab_size=4_096,
    hidden_size=256,
    intermediate_size=768,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=2,
    max_position_embeddings=2_048,
    rope_theta=10_000.0,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("prefill", "decode"), default="decode")
    parser.add_argument("--preset", choices=("llama3-1b", "smoke"), default="llama3-1b")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--cache-length", type=int, default=2_048)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--intermediate-size", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--q-heads", type=int)
    parser.add_argument("--kv-heads", type=int)
    parser.add_argument("--max-position-embeddings", type=int)
    parser.add_argument("--rope-theta", type=float)
    arguments = parser.parse_args()
    if arguments.prompt_length <= 0:
        parser.error("--prompt-length must be positive")
    if arguments.cache_length <= 0:
        parser.error("--cache-length must be positive")
    if arguments.warmups < 0:
        parser.error("--warmups may not be negative")
    if arguments.iterations <= 0:
        parser.error("--iterations must be positive")
    if arguments.device < 0:
        parser.error("--device may not be negative")
    return arguments


def resolve_dimensions(arguments: argparse.Namespace) -> Dimensions:
    dimensions = LLAMA3_1B if arguments.preset == "llama3-1b" else SMOKE
    overrides = {
        "vocab_size": arguments.vocab_size,
        "hidden_size": arguments.hidden_size,
        "intermediate_size": arguments.intermediate_size,
        "num_hidden_layers": arguments.layers,
        "num_attention_heads": arguments.q_heads,
        "num_key_value_heads": arguments.kv_heads,
        "max_position_embeddings": arguments.max_position_embeddings,
        "rope_theta": arguments.rope_theta,
    }
    dimensions = replace(
        dimensions,
        **{name: value for name, value in overrides.items() if value is not None},
    )
    positive = (
        dimensions.vocab_size,
        dimensions.hidden_size,
        dimensions.intermediate_size,
        dimensions.num_hidden_layers,
        dimensions.num_attention_heads,
        dimensions.num_key_value_heads,
        dimensions.max_position_embeddings,
    )
    if any(value <= 0 for value in positive) or dimensions.rope_theta <= 0:
        raise ValueError("all model dimensions and RoPE theta must be positive")
    if dimensions.hidden_size % dimensions.num_attention_heads:
        raise ValueError("hidden size must be divisible by query heads")
    if dimensions.num_attention_heads % dimensions.num_key_value_heads:
        raise ValueError("query heads must be divisible by key/value heads")
    if dimensions.head_dim % 2:
        raise ValueError("attention head dimension must be even")
    if arguments.cache_length > dimensions.max_position_embeddings:
        raise ValueError("cache length exceeds maximum positions")
    if arguments.case == "prefill" and arguments.prompt_length > arguments.cache_length:
        raise ValueError("prompt length exceeds cache length")
    return dimensions


def block_tree(tree: PyTree) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        leaf.block_until_ready()


def make_parameters(dimensions: Dimensions, dtype: jnp.dtype) -> PyTree:
    def matrix(rows: int, columns: int) -> Array:
        return jnp.zeros((rows, columns), dtype=dtype)

    def layer() -> dict[str, PyTree]:
        kv_size = dimensions.num_key_value_heads * dimensions.head_dim
        return {
            "input_layernorm": jnp.ones((dimensions.hidden_size,), dtype=dtype),
            "q_proj": matrix(dimensions.hidden_size, dimensions.hidden_size),
            "k_proj": matrix(dimensions.hidden_size, kv_size),
            "v_proj": matrix(dimensions.hidden_size, kv_size),
            "o_proj": matrix(dimensions.hidden_size, dimensions.hidden_size),
            "post_attention_layernorm": jnp.ones(
                (dimensions.hidden_size,), dtype=dtype
            ),
            "gate_proj": matrix(dimensions.hidden_size, dimensions.intermediate_size),
            "up_proj": matrix(dimensions.hidden_size, dimensions.intermediate_size),
            "down_proj": matrix(dimensions.intermediate_size, dimensions.hidden_size),
        }

    return {
        "embed_tokens": matrix(dimensions.vocab_size, dimensions.hidden_size),
        "layers": tuple(layer() for _ in range(dimensions.num_hidden_layers)),
        "norm": jnp.ones((dimensions.hidden_size,), dtype=dtype),
    }


def make_cache(dimensions: Dimensions, cache_length: int, dtype: jnp.dtype) -> PyTree:
    shape = (
        1,
        dimensions.num_key_value_heads,
        cache_length,
        dimensions.head_dim,
    )
    return (
        tuple(
            jnp.zeros(shape, dtype=dtype) for _ in range(dimensions.num_hidden_layers)
        ),
        tuple(
            jnp.zeros(shape, dtype=dtype) for _ in range(dimensions.num_hidden_layers)
        ),
        jnp.zeros((1, cache_length), dtype=jnp.bool_),
        jnp.asarray(0, dtype=jnp.int32),
    )


def rms_norm(input: Array, scale: Array) -> Array:
    mean_square = jnp.mean(input * input, axis=-1, keepdims=True)
    return input * jax.lax.rsqrt(mean_square + input.dtype.type(1e-5)) * scale


def apply_rope(input: Array, positions: Array, dimensions: Dimensions) -> Array:
    half = dimensions.head_dim // 2
    exponents = jnp.arange(0, dimensions.head_dim, 2, dtype=jnp.float32) / np.float32(
        dimensions.head_dim
    )
    inv_freq = jnp.exp(-exponents * jnp.log(np.float32(dimensions.rope_theta)))
    angles = positions.astype(jnp.float32)[:, :, None] * inv_freq[None, None, :]
    angles = angles[:, None, :, :]
    cosine = jnp.cos(angles).astype(input.dtype)
    sine = jnp.sin(angles).astype(input.dtype)
    first, second = jnp.split(input, (half,), axis=-1)
    return jnp.concatenate(
        (first * cosine - second * sine, first * sine + second * cosine),
        axis=-1,
    )


def repeat_kv(input: Array, dimensions: Dimensions) -> Array:
    repetitions = dimensions.num_attention_heads // dimensions.num_key_value_heads
    if repetitions == 1:
        return input
    batch, kv_heads, sequence, head_dim = input.shape
    return jnp.broadcast_to(
        input[:, :, None, :, :],
        (batch, kv_heads, repetitions, sequence, head_dim),
    ).reshape(batch, dimensions.num_attention_heads, sequence, head_dim)


def append_valid(
    valid: Array, position: Array, token_valid: Array, cache_length: int
) -> Array:
    sequence = token_valid.shape[1]
    positions = position + jnp.arange(sequence, dtype=jnp.int32)
    slots = jnp.arange(cache_length, dtype=jnp.int32)
    writes = positions[None, :, None] == slots[None, None, :]
    valid_writes = jnp.sum(
        jnp.where(token_valid[:, :, None], writes, False).astype(jnp.int32),
        axis=1,
    ) > jnp.int32(0)
    return jnp.where(valid, True, valid_writes)


def update_cache(cache: Array, values: Array, writes: Array) -> Array:
    inserted = jnp.matmul(jnp.swapaxes(values, -1, -2), writes.astype(values.dtype))
    inserted = jnp.swapaxes(inserted, -1, -2)
    occupied = (jnp.sum(writes.astype(jnp.int32), axis=0) > jnp.int32(0))[
        None, None, :, None
    ]
    return jnp.where(occupied, inserted, cache)


def make_step(dimensions: Dimensions, cache_length: int):
    head_dim = dimensions.head_dim
    score_scale = 1.0 / np.sqrt(head_dim)

    def step(
        params: PyTree, cache: PyTree, input_ids: Array, token_valid: Array
    ) -> tuple[Array, PyTree]:
        keys, values, old_valid, position = cache
        sequence = input_ids.shape[1]
        valid = append_valid(old_valid, position, token_valid, cache_length)
        hidden = params["embed_tokens"][input_ids.astype(jnp.int32)]
        query_positions = (position + jnp.arange(sequence, dtype=jnp.int32))[None, :]
        slots = jnp.arange(cache_length, dtype=jnp.int32)
        writes = (position + jnp.arange(sequence, dtype=jnp.int32))[:, None] == slots[
            None, :
        ]
        output_keys: list[Array] = []
        output_values: list[Array] = []
        for layer, key_cache, value_cache in zip(
            params["layers"], keys, values, strict=True
        ):
            normalized = rms_norm(hidden, layer["input_layernorm"])
            query = normalized @ layer["q_proj"]
            key = normalized @ layer["k_proj"]
            value = normalized @ layer["v_proj"]
            query = query.reshape(
                1, sequence, dimensions.num_attention_heads, head_dim
            ).transpose(0, 2, 1, 3)
            key = key.reshape(
                1, sequence, dimensions.num_key_value_heads, head_dim
            ).transpose(0, 2, 1, 3)
            value = value.reshape(
                1, sequence, dimensions.num_key_value_heads, head_dim
            ).transpose(0, 2, 1, 3)
            query = apply_rope(query, query_positions, dimensions)
            key = apply_rope(key, query_positions, dimensions)
            key_cache = update_cache(key_cache, key, writes)
            value_cache = update_cache(value_cache, value, writes)
            key_heads = repeat_kv(key_cache, dimensions)
            value_heads = repeat_kv(value_cache, dimensions)
            scores = (query @ jnp.swapaxes(key_heads, -1, -2)) * (
                query.dtype.type(score_scale)
            )
            mask = slots[None, None, None, :] <= query_positions[:, None, :, None]
            mask = mask & valid[:, None, None, :]
            scores = jnp.where(mask, scores, query.dtype.type(-np.inf))
            probabilities = jax.nn.softmax(scores, axis=-1)
            attended = probabilities @ value_heads
            attended = attended.transpose(0, 2, 1, 3).reshape(
                1, sequence, dimensions.hidden_size
            )
            hidden = hidden + attended @ layer["o_proj"]
            normalized = rms_norm(hidden, layer["post_attention_layernorm"])
            gate = normalized @ layer["gate_proj"]
            up = normalized @ layer["up_proj"]
            hidden = hidden + (jax.nn.silu(gate) * up) @ layer["down_proj"]
            output_keys.append(key_cache)
            output_values.append(value_cache)
        hidden = rms_norm(hidden, params["norm"])
        logits = hidden @ params["embed_tokens"].T
        return logits, (
            tuple(output_keys),
            tuple(output_values),
            valid,
            position + jnp.int32(sequence),
        )

    return jax.jit(step)


def elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def percentile(sorted_samples: list[float], fraction: float) -> float:
    last = len(sorted_samples) - 1
    return sorted_samples[min(last, int(fraction * last))]


def main() -> None:
    arguments = parse_arguments()
    dimensions = resolve_dimensions(arguments)
    devices = jax.devices("gpu")
    if arguments.device >= len(devices):
        raise ValueError(
            f"CUDA device {arguments.device} unavailable; found {len(devices)}"
        )
    device = devices[arguments.device]
    dtype = jnp.float16
    sequence = arguments.prompt_length if arguments.case == "prefill" else 1
    print(
        "workload=synthetic-llama-3.2-1b-shaped implementation=jax "
        f"case={arguments.case} dtype=float16 batch=1 "
        f"sequence={sequence} cache={arguments.cache_length} "
        f"warmups={arguments.warmups} iterations={arguments.iterations}"
    )
    print(
        f"vocab={dimensions.vocab_size} hidden={dimensions.hidden_size} "
        f"intermediate={dimensions.intermediate_size} "
        f"layers={dimensions.num_hidden_layers} "
        f"q_heads={dimensions.num_attention_heads} "
        f"kv_heads={dimensions.num_key_value_heads} "
        f"head_dim={dimensions.head_dim} "
        f"max_positions={dimensions.max_position_embeddings} "
        f"rope_theta={dimensions.rope_theta:.0f} tied_embeddings=true"
    )
    print(
        "note=synthetic_zero_weights; rope=standard_not_llama3_scaled; "
        "initial_cache_position=0_with_fixed_capacity_cache"
    )
    print(f"jax={jax.__version__} backend={jax.default_backend()} device={device}")
    with jax.default_device(device):
        started = time.perf_counter_ns()
        params = make_parameters(dimensions, dtype)
        cache = make_cache(dimensions, arguments.cache_length, dtype)
        input_ids = np.arange(sequence, dtype=np.int32)[None, :]
        token_valid = jnp.ones((1, sequence), dtype=jnp.bool_)
        block_tree((params, cache, token_valid))
        print(f"parameter_and_cache_initialization_ms={elapsed_ms(started):.6f}")
        parameter_count = sum(
            int(leaf.size) for leaf in jax.tree_util.tree_leaves(params)
        )
        print(f"parameters={parameter_count} parameter_bytes={2 * parameter_count}")
        execute = make_step(dimensions, arguments.cache_length)

        def run_once() -> tuple[np.ndarray, PyTree]:
            logits, output_cache = execute(params, cache, input_ids, token_valid)
            host_logits = np.asarray(logits)
            return host_logits, output_cache

        started = time.perf_counter_ns()
        last_logits, first_cache = run_once()
        print(f"first_compile_and_execute_ms={elapsed_ms(started):.6f}")
        print(f"output_cache_length={sequence}")
        del first_cache
        for _ in range(arguments.warmups):
            last_logits, _ = run_once()
        samples = [0.0] * arguments.iterations
        for index in range(arguments.iterations):
            started = time.perf_counter_ns()
            last_logits, _ = run_once()
            samples[index] = elapsed_ms(started)
        sorted_samples = sorted(samples)
        mean = sum(samples) / arguments.iterations
        print(
            "steady_e2e_ms "
            f"mean={mean:.6f} "
            f"p10={percentile(sorted_samples, 0.10):.6f} "
            f"median={percentile(sorted_samples, 0.50):.6f} "
            f"p90={percentile(sorted_samples, 0.90):.6f} "
            f"min={sorted_samples[0]:.6f} max={sorted_samples[-1]:.6f}"
        )
        print(f"tokens_per_second={sequence * 1_000.0 / mean:.6f}")
        host_logits = np.asarray(last_logits)
        print(
            f"logits_shape={list(host_logits.shape)} "
            f"first_logit={host_logits[0, 0, 0]:.9g}"
        )


if __name__ == "__main__":
    main()
