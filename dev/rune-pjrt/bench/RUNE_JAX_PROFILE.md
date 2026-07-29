# Rune PJRT versus JAX CUDA

This comparison separates generated GPU-kernel performance from PJRT dispatch
and host-transfer costs. It also records the runtime changes made after the
initial profile.

## Summary

- Once equivalent StableHLO reaches XLA, Rune and JAX generate effectively the
  same softmax, layer-normalization, and pointwise kernels. The selected GEMM
  kernels are within 2.6%.
- Direct transfers between `Nx_buffer` storage and PJRT, together with
  concurrent staging of multiple inputs, reduced Rune host-to-host calls from
  1.6–1.9× slower than JAX to within 4%.
- `Rune_pjrt.Device_buffer` and `jit_device` now support asynchronous,
  device-resident execution and chaining between separately compiled calls.
  Resident softmax and layer normalization are within 0.04–0.06 ms of JAX,
  GEMM is tied, and the pointwise case remains 0.11 ms behind.
- The earlier large DSL pointwise win exposed an `Nx.sigmoid` lowering problem.
  `Nx.sigmoid` now keeps its constant scalar, making ordinary Rune/XLA tied
  with JAX and within 1.5% of the DSL kernel.

The regular XLA path is consequently at roughly JAX level for meaningful
host-synchronized work, and has a usable resident path for long-running GPU
programs. The kernel DSL should be judged on irregular or structurally
specialized kernels, not the previous sigmoid result.

## Setup

- GPU: NVIDIA GeForce RTX 3090, compute capability 8.6
- Driver: 610.43.02
- CUDA runtime/toolkit reported by Rune: 13.3
- Rune PJRT plugin: local XLA build from source commit `2d878836d484`
  (2026-05-28)
- JAX and jaxlib: 0.10.0 with the CUDA 13 PJRT plugin
- Nsight Systems: 2026.1.3
- Dtype: `float32`

The Rune and JAX PJRT plugins are close but not identical XLA revisions. The
pointwise, softmax, and layer-normalization comparisons nevertheless produced
the same launch geometry, and their equivalent output checksums agree to
float32 precision.

GPU durations below are Nsight Systems averages. Host-mode profiles were used
for the final kernel comparison because their regular transfers keep GPU clocks
stable. Very short device-resident profiles otherwise measure transient lower
clocks. GEMM timing is the selected steady-state kernel after XLA autotuning,
excluding its candidate and red-zone launches.

## Generated GPU kernels

| Workload | Shape | Rune/XLA | JAX/XLA | Rune relative to JAX |
|---|---:|---:|---:|---:|
| Pointwise using old tensor sigmoid lowering | `[1,048,576]` | 15.005 µs | 9.003 µs | 1.67× slower |
| Pointwise using current scalar sigmoid lowering | `[1,048,576]` | 8.971 µs | 9.003 µs | 0.4% faster |
| Softmax | `[5,461, 768]` | 39.689 µs | 39.700 µs | tied |
| Layer normalization | `[5,461, 768]` | 41.892 µs | 41.898 µs | tied |
| GEMM | `[1,024, 1,024]²` | 87.249 µs | 85.071 µs | 2.6% slower |

The pointwise expression is:

```text
sigmoid(x * x + 0.5 * x)
```

The current scalar implementation is equivalent to:

```ocaml
let one = Nx.scalar_like value 1.0 in
Nx.div one (Nx.add one (Nx.exp (Nx.neg value)))
```

At 16,777,216 elements, the original Rune/XLA kernel took 237.976 µs, the
scalar form took 160.185 µs, and the best DSL schedule took 157.722 µs. The DSL
is therefore only 1.5% faster than correctly lowered XLA here, rather than the
previously reported 34%.

## Host boundary and resident execution

Both host-to-host columns receive ordinary host arrays, synchronously complete
one compiled graph, download the result, and materialize the host output. Input
staging and device copies may proceed asynchronously inside that call. The
first three results are means of 100 synchronized iterations after 20 warmups
in one process. GEMM is the mean of an isolated 200-iteration run after 20
warmups, which avoids suite-order tail noise from dominating the comparison.

| Workload | Rune host-to-host | JAX host-to-host | Rune/JAX |
|---|---:|---:|---:|
| Pointwise, 1,048,576 elements | 2.579 ms | 2.556 ms | 1.01× |
| Softmax, width 768 | 9.864 ms | 10.263 ms | 0.96× |
| Layer normalization, width 768 | 10.154 ms | 10.094 ms | 1.01× |
| GEMM, size 1,024 | 3.260 ms | 3.254 ms | 1.00× |

The original GEMM gap was not H2D bandwidth. Each framework spent about
0.82 ms copying its two 4 MiB inputs. Rune used
`kImmutableOnlyDuringCall`, however, which forced XLA to copy each input into
pinned staging memory synchronously. Submitting the inputs one at a time left
a 237 µs GPU bubble between their transfers. Rune now keeps the input
Bigarrays alive under `kImmutableUntilTransferCompletes`, submits all inputs,
and drains their release events before returning.

| GEMM timeline component | Rune before | Rune current | JAX |
|---|---:|---:|---:|
| Two 4 MiB H2D copies | 819.4 µs | 822.8 µs | 817.1 µs |
| Gap between H2D copies | 236.8 µs | 2.9 µs | 3.1 µs |
| Selected GEMM kernel | 88.0 µs | 86.9 µs | 85.2 µs |
| 4 MiB D2H copy | 394.9 µs | 410.3 µs | 372.6 µs |
| First H2D start to D2H end | 1,830.3 µs | 1,645.0 µs | 1,550.8 µs |

The resident comparison uploads inputs once and then includes framework
dispatch plus an explicit readiness wait on every iteration. To reduce clock
and startup bias, these are medians from 100 iterations after ten warmups, with
all four workloads run in one process.

| Workload | Rune resident | JAX resident | Rune/JAX | Absolute gap |
|---|---:|---:|---:|---:|
| Pointwise, 1,048,576 elements | 0.279 ms | 0.171 ms | 1.63× | 0.108 ms |
| Softmax, width 768 | 0.284 ms | 0.227 ms | 1.25× | 0.057 ms |
| Layer normalization, width 768 | 0.277 ms | 0.236 ms | 1.17× | 0.041 ms |
| GEMM, size 1,024 | 0.342 ms | 0.346 ms | 0.99× | 0.004 ms faster |

The ratio is largest for the memory-bound pointwise case, but its absolute gap
is 0.11 ms. A one-element pointwise call is tied within run-to-run noise at
about 0.20 ms in Rune and 0.22 ms in JAX. Splitting the Rune suite shows
roughly 0.16–0.21 ms in dispatch and output-handle creation, followed by
0.08–0.18 ms in the PJRT readiness wait. Requesting a device-completion event
directly was measured and removed because it increased total latency.

The original Nsight trace showed that Rune and JAX already spent the same time
in GPU copies; Rune's old bridge instead copied inputs and outputs through
temporary OCaml strings. The current bridge gives PJRT the contiguous
`Nx_buffer` Bigarray directly, overlaps staging across inputs, and downloads
into the final output buffer. A shared PJRT client also lets a `Device_buffer`
returned by one compiled function be consumed by another.

## The `Nx.sigmoid` finding

The previous frontend implementation built `ones_like x` and implemented
`exp2` through `exp`. During Rune tracing, the ones tensor became another
function argument:

```mlir
func.func @main(
  %arg0: tensor<1048576xf32>,
  %arg1: tensor<1048576xf32>
) -> tensor<1048576xf32>
```

The generated kernel must read the full `%arg1` tensor. The scalar formulation
instead produces one input and scalar StableHLO constants, matching JAX's
lowering. `Nx.sigmoid` now uses this formulation in the regular frontend; no
backend operation was added.

## Numerical check

The final outputs were summed in float64 on the host to avoid comparing
different float32 reduction orders:

- pointwise: identical, `606628.7620869875`;
- softmax: difference `2.02e-8` over 4,194,048 values;
- layer normalization: identical, `-14907.628868277185`;
- GEMM: identical, `1518196410.6257038`.

These checks cover the profiled deterministic inputs. New kernel work should
still use element-wise error validation, especially when tensor-core
accumulation or approximate math is involved.

## Changes and remaining work

Completed from the initial profile:

1. PJRT clients are shared by plugin and device, so compiled programs and
   resident buffers use one runtime.
2. `Device_buffer`, `jit_device`, `jits_device`, and the heterogeneous packed
   form keep typed values on the device and execute asynchronously.
3. Host transfers borrow contiguous `Nx_buffer` storage, stage multiple inputs
   concurrently, and write directly into final output storage.
4. `Nx.sigmoid` keeps its one constant scalar.
5. The benchmark now separates host, resident, compilation, dispatch, and wait
   timing.

The remaining resident difference is small in absolute terms and lies below
the StableHLO/compiler layer. Closing the pointwise gap further would mean
reducing output-buffer, PJRT dispatch, and readiness-event overhead. Kernel DSL
evaluation should continue on grouped or ragged GEMM, sparse MoE routing, and
fusions that remove unavoidable intermediates; standard XLA kernels already
match JAX.

## Reproduction

Build the Rune benchmark:

```sh
env RUNE_PJRT_CUDA_KERNELS=enabled \
  opam exec --switch=snakeml -- \
  dune build dev/rune-pjrt/bench/triton_profile.exe
```

Create the isolated JAX environment:

```sh
uv venv --python 3.12 /tmp/raven-jax-bench
uv pip install --python /tmp/raven-jax-bench/bin/python \
  'jax[cuda13-local]==0.10.0'
```

Run the JAX timing suite:

```sh
env CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=/home/carsten/packages/cudnn/9.23.1.3_cuda13/lib:/usr/local/cuda/targets/x86_64-linux/lib \
  XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/carsten/packages/xla/external/cuda_nvcc \
  /tmp/raven-jax-bench/bin/python \
  dev/rune-pjrt/bench/jax_profile.py suite host 1 20 100
```

For the resident suite reported above, use:

```sh
env CUDA_VISIBLE_DEVICES=0 \
  LD_LIBRARY_PATH=/home/carsten/packages/cudnn/9.23.1.3_cuda13/lib:/usr/local/cuda/targets/x86_64-linux/lib \
  XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/carsten/packages/xla/external/cuda_nvcc \
  /tmp/raven-jax-bench/bin/python \
  dev/rune-pjrt/bench/jax_profile.py suite resident 1 10 100
```

Run the equivalent Rune resident suite:

```sh
env CUDA_VISIBLE_DEVICES=0 \
  RUNE_PJRT_PLUGIN_PATH=/home/carsten/packages/xla/xla/pjrt/c \
  XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/carsten/packages/xla/external/cuda_nvcc \
  _build/default/dev/rune-pjrt/bench/triton_profile.exe \
  suite baseline_resident 1 1 10 100
```

Use `baseline` in place of `baseline_resident` for the host-to-host suite. The
size and configuration arguments are placeholders when `suite` is selected.
The isolated GEMM comparison uses `gemm host 1024 20 200` for JAX and
`gemm baseline 1024 1 20 200` for Rune.

Collect a JAX CUDA timeline with:

```sh
nsys profile \
  --trace=cuda --sample=none --cpuctxsw=none \
  --output=/tmp/raven-vs-jax-layernorm-w768-host \
  env CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/home/carsten/packages/cudnn/9.23.1.3_cuda13/lib:/usr/local/cuda/targets/x86_64-linux/lib \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/carsten/packages/xla/external/cuda_nvcc \
    /tmp/raven-jax-bench/bin/python \
    dev/rune-pjrt/bench/jax_profile.py layer_norm host 768 3 10
```
