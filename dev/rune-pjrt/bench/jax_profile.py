# ---------------------------------------------------------------------------
# Copyright (c) 2026 The Raven authors. All rights reserved.
# SPDX-License-Identifier: ISC
# ---------------------------------------------------------------------------

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


Array = np.ndarray | jax.Array


@dataclass(frozen=True)
class Workload:
    name: str
    shape: str
    inputs: tuple[np.ndarray, ...]
    body: Callable[..., jax.Array]


def pointwise(input: jax.Array) -> jax.Array:
    return jax.nn.sigmoid((input * input) + (input * 0.5))


def make_pointwise(size: int) -> Workload:
    indices = np.arange(size, dtype=np.int64)
    input = (((indices % 257) - 128) / 128.0).astype(np.float32)
    return Workload("pointwise", f"[{size}]", (input,), pointwise)


def softmax(input: jax.Array) -> jax.Array:
    return jax.nn.softmax(input, axis=1)


def rows_for_width(width: int) -> int:
    return max(1, 4 * 1024 * 1024 // width)


def make_softmax(width: int) -> Workload:
    rows = rows_for_width(width)
    row = np.arange(rows, dtype=np.int64)[:, None]
    column = np.arange(width, dtype=np.int64)[None, :]
    input = ((row * 17 - column * 13) / 256.0).astype(np.float32)
    return Workload("softmax", f"[{rows},{width}]", (input,), softmax)


def layer_norm(
    input: jax.Array, scale: jax.Array, bias: jax.Array
) -> jax.Array:
    mean = jnp.mean(input, axis=1, keepdims=True)
    centered = input - mean
    variance = jnp.mean(centered * centered, axis=1, keepdims=True)
    normalized = centered * jax.lax.rsqrt(variance + 1e-5)
    return (normalized * scale) + bias


def make_layer_norm(width: int) -> Workload:
    rows = rows_for_width(width)
    row = np.arange(rows, dtype=np.int64)[:, None]
    column = np.arange(width, dtype=np.int64)[None, :]
    input = ((row * 7 - column * 11) / 512.0).astype(np.float32)
    indices = np.arange(width, dtype=np.int64)
    scale = (0.75 + ((indices % 31) / 64.0)).astype(np.float32)
    bias = (((indices % 17) - 8) / 32.0).astype(np.float32)
    return Workload(
        "layer_norm", f"[{rows},{width}]", (input, scale, bias), layer_norm
    )


def gemm(lhs: jax.Array, rhs: jax.Array) -> jax.Array:
    return lhs @ rhs


def make_gemm(size: int) -> Workload:
    row = np.arange(size, dtype=np.int64)[:, None]
    column = np.arange(size, dtype=np.int64)[None, :]
    lhs = ((row * 3 - column) / float(size)).astype(np.float32)
    rhs = ((row + column * 2) / float(size)).astype(np.float32)
    return Workload(
        "gemm", f"[{size},{size}]x[{size},{size}]", (lhs, rhs), gemm
    )


def make_workload(case: str, size: int) -> Workload:
    if case == "pointwise":
        return make_pointwise(size)
    if case == "softmax":
        return make_softmax(size)
    if case == "layer_norm":
        return make_layer_norm(size)
    if case == "gemm":
        return make_gemm(size)
    raise ValueError(f"unknown benchmark {case}")


def elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def percentile(sorted_samples: list[float], fraction: float) -> float:
    last = len(sorted_samples) - 1
    return sorted_samples[min(last, int(fraction * last))]


def block_until_ready(values: tuple[jax.Array, ...]) -> None:
    for value in values:
        value.block_until_ready()


def run(
    workload: Workload, mode: str, warmups: int, iterations: int
) -> None:
    compiled = jax.jit(workload.body)
    if mode == "resident":
        started = time.perf_counter_ns()
        inputs = tuple(jax.device_put(value) for value in workload.inputs)
        block_until_ready(inputs)
        print(f"initial_device_put_ms={elapsed_ms(started):.6f}")
    elif mode == "host":
        inputs = workload.inputs
    else:
        raise ValueError(f"unknown mode {mode}")

    print(
        f"case={workload.name} implementation=jax mode={mode} "
        f"shape={workload.shape} warmups={warmups} iterations={iterations}"
    )
    print(
        f"jax={jax.__version__} backend={jax.default_backend()} "
        f"device={jax.devices()[0]}"
    )

    def execute() -> Array:
        output = compiled(*inputs)
        if mode == "resident":
            output.block_until_ready()
            return output
        return np.asarray(output)

    started = time.perf_counter_ns()
    last = execute()
    print(f"first_compile_and_execute_ms={elapsed_ms(started):.6f}")

    for _ in range(warmups):
        last = execute()

    samples = [0.0] * iterations
    for index in range(iterations):
        started = time.perf_counter_ns()
        last = execute()
        samples[index] = elapsed_ms(started)

    sorted_samples = sorted(samples)
    mean = sum(samples) / iterations
    print(
        "steady_e2e_ms "
        f"mean={mean:.6f} "
        f"p10={percentile(sorted_samples, 0.10):.6f} "
        f"median={percentile(sorted_samples, 0.50):.6f} "
        f"p90={percentile(sorted_samples, 0.90):.6f} "
        f"min={sorted_samples[0]:.6f} max={sorted_samples[-1]:.6f}"
    )
    host_output = np.asarray(last)
    checksum = host_output.sum(dtype=np.float64)
    print(f"checksum={checksum:.17g}")


def parse_arguments() -> tuple[str, str, int, int, int]:
    if len(sys.argv) != 6:
        raise ValueError(
            "usage: jax_profile.py "
            "(pointwise|softmax|layer_norm|gemm|suite) "
            "(resident|host) SIZE WARMUPS ITERATIONS"
        )
    case = sys.argv[1]
    mode = sys.argv[2]
    size = int(sys.argv[3])
    warmups = int(sys.argv[4])
    iterations = int(sys.argv[5])
    if size <= 0 or warmups < 0 or iterations <= 0:
        raise ValueError(
            "size and iterations must be positive; warmups may be zero"
        )
    return case, mode, size, warmups, iterations


def main() -> None:
    case, mode, size, warmups, iterations = parse_arguments()
    if jax.default_backend() != "gpu":
        raise RuntimeError("the JAX CUDA backend is unavailable")
    if case == "suite":
        workloads = (
            make_pointwise(1_048_576),
            make_softmax(768),
            make_layer_norm(768),
            make_gemm(1024),
        )
        for workload in workloads:
            run(workload, mode, warmups, iterations)
    else:
        run(make_workload(case, size), mode, warmups, iterations)


if __name__ == "__main__":
    main()
