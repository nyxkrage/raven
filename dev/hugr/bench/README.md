# Llama 3.2 1B-shaped CUDA benchmark

This benchmark measures Hugr's cached Llama inference through PJRT CUDA and a
matched JAX implementation. The default shape is Llama 3.2 1B: 128,256 tokens,
2,048 hidden units, 8,192 SwiGLU units, 16 layers, 32 query heads, 8 key/value
heads, 64 channels per head, a 131,072-token maximum context, and tied token
embeddings. It contains 1,235,814,400 synthetic parameters.

The weights are zeros, so this is a compiler/runtime performance workload rather
than a quality evaluation. Parameters remain dynamic program inputs and are not
compile-time constants. Hugr does not yet implement Llama 3's scaled RoPE, so
both programs deliberately use its current standard-RoPE graph. As in Hugr,
frequency and trigonometric calculations use float32 before their results are
cast to the model's float16 dtype, allowing the exact 500,000 theta. The model
dtype is float16 because PJRT's bfloat16 reduction and reciprocal lowering is
not complete yet.

Steady-state timing keeps weights, the all-true attention mask, and the KV cache
on the GPU. Each call includes the token-ID upload and the logits download. The
JAX runner uses the same boundary so its end-to-end samples are comparable.

The prefill and decode cases are separate processes so each retains only one
compiled copy of the 2.30-GiB parameter set. A process that uses both sequence
signatures currently retains a separate device copy for each executable.
Decode runs at position zero against the configured fixed-capacity cache.
Hugr's cached graph touches the full cache capacity regardless of the current
logical position, so this preserves the decode graph's relevant work without
compiling a second prefill executable.

Build and run a small CUDA smoke test:

```sh
dune exec dev/hugr/bench/llama_profile.exe -- \
  --preset smoke --case decode --cache-length 128 --warmups 2 --iterations 10
python dev/hugr/bench/jax_llama_profile.py \
  --preset smoke --case decode --cache-length 128 --warmups 2 --iterations 10
```

Run the default Llama 3.2 1B-shaped decode workload:

```sh
dune exec dev/hugr/bench/llama_profile.exe
python dev/hugr/bench/jax_llama_profile.py
```

Use `--case prefill --prompt-length 128` for prefill. Both programs accept the
same dimension, cache, warmup, iteration, and CUDA-device overrides; use
`--help` for the complete list.

## RTX 3090 results

These measurements use the full 1,235,814,400-parameter preset, a 2,048-token
cache, five warmups, and 20 timed calls. Token IDs cross from host to device and
logits return to the host in every sample; weights, the attention mask, and the
KV cache stay resident. Decode starts from an empty cache, while prefill appends
128 tokens.

| Case | Implementation | First compile + execute | Mean | Median | P90 | Throughput |
|---|---|---:|---:|---:|---:|---:|
| Decode, 1 token | Hugr/PJRT | — | 7.302 ms | 6.785 ms | 8.644 ms | 136.95 token/s |
| Decode, 1 token | JAX | 12.392 s | 6.464 ms | 6.379 ms | 6.938 ms | 154.69 token/s |
| Prefill, 128 tokens | Hugr/PJRT | 76.472 s | 46.210 ms | 23.137 ms | 76.127 ms | 2,769.96 token/s |
| Prefill, 128 tokens | JAX | 22.898 s | 23.068 ms | 22.806 ms | 23.555 ms | 5,548.74 token/s |

For decode, Hugr is 13.0% slower by mean and 6.4% slower by median. The
steady-state kernels are therefore already close to JAX on this model shape;
first-call tracing and latency tails remain separate performance targets.

Hugr's prefill samples are bimodal: ordinary calls match JAX closely (23.137 ms
versus 22.806 ms by median), while periodic device-buffer finalizer/GC stalls
raise the mean and P90 to roughly twice JAX's. Shape-only tracing reduced Hugr's
prefill first call from 671.649 seconds to 76.472 seconds; the remaining cold
gap and the cache-output lifetime tails are the next optimization targets.
