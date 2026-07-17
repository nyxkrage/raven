# Causal scaled-softmax CUDA profile

Measured on 2026-07-16. The regular StableHLO baseline was captured from Raven
commit `2b4e737ba4be3ba37f20bd83db6b29ca84f3606d` before the custom-kernel
implementation. The custom result uses the experimental FFI and PPX described
in `../KERNELS.md` and the kernel in
`../kernels/causal_scaled_softmax.cu`.

## Workload and environment

- GPU: NVIDIA GeForce RTX 3090 (24 GiB), compute capability 8.6
- Driver: 610.43.02
- CUDA runtime and toolkit reported by XLA: 13.3
- cuDNN reported by XLA: 9.22
- OCaml: 5.4.1
- PJRT plugin:
  `/home/carsten/packages/xla/xla/pjrt/c/pjrt_c_api_gpu_plugin.so`
- Plugin SHA-256:
  `7f19682b8d941e76786acd25c1de1f046d69c5df7c84d72131bf3cd8a5ff1990`
- Input and output: f32 `[1, 12, S, S]`
- Operation: multiply scores by 0.125, apply `key <= query` causal masking,
  and softmax over the final dimension

The baseline is the ordinary Raven body traced through `Rune_pjrt.jit`. The
custom case calls the same annotated function through `Rune_pjrt.jit`; the PPX
and FFI path replace its body with one StableHLO custom call. Each Nsight run
contains the first execution, two warmups, and ten measured executions, for 13
identical kernel launches.

## GPU kernel result

XLA already fuses the complete regular body into one kernel named
`triton_softmax_2`. The custom path also launches exactly one kernel. It is
therefore a comparison against an existing fused implementation, not a
multi-launch baseline.

| S | XLA Triton median (us) | Custom median (us) | Custom change | Custom grid | Block |
|---:|---:|---:|---:|---:|---:|
| 128 | 5.184 | 4.576 | 11.73% faster | 1,536 | 128 |
| 256 | 10.496 | 7.807 | 25.62% faster | 3,072 | 128 |
| 512 | 39.231 | 32.159 | 18.03% faster | 6,144 | 256 |
| 1024 | 134.494 | 92.446 | 31.26% faster | 12,288 | 256 |

The custom kernel reads only the causal half of the score matrix, retains each
active score in registers across max and sum reduction, writes masked outputs
using the fallback's finite `-1e9` sentinel, and uses warp-shuffle block
reductions. Masked outputs underflow to exact zero for the measured workload.
The ordinary Raven graph also supplies two cached `S x S` mask-construction
buffers and two i32 vectors of length `S` to the Triton fusion. Avoiding those
inputs is the main structural advantage of the custom call.

Counting the custom kernel's causal score reads and full output writes gives
the following lower-bound effective bandwidth:

| S | Estimated traffic | Effective bandwidth |
|---:|---:|---:|
| 128 | 1,182,720 bytes | 258 GB/s |
| 256 | 4,724,736 bytes | 605 GB/s |
| 512 | 18,886,656 bytes | 587 GB/s |
| 1024 | 75,522,048 bytes | 817 GB/s |

At S=1024 this is about 87% of the RTX 3090's nominal 936 GB/s bandwidth. The
result is consistent with a bandwidth-bound operation whose improvement comes
from moving less data. This is an inference from lower-bound traffic, not a
hardware-counter result; Nsight Compute counters were unavailable.

## Transfers and Raven wall time

The transfer medians from the custom Nsight runs were:

| S | H2D median (us) | D2H median (us) |
|---:|---:|---:|
| 128 | 79.102 | 68.799 |
| 256 | 308.700 | 303.292 |
| 512 | 1,265.645 | 1,208.366 |
| 1024 | 4,912.345 | 4,728.764 |

Unprofiled `Runtime.execute` timings used ten warmups and 50 samples. They
include OCaml serialization, host-to-device and device-to-host copies, device
execution, allocation, and rebuilding the output Nx tensor.

| S | Baseline median (ms) | Custom median (ms) | Custom p10/p90 (ms) |
|---:|---:|---:|---:|
| 128 | 1.357 | 1.515 | 1.470 / 1.576 |
| 256 | 3.665 | 5.591 | 5.284 / 11.789 |
| 512 | 13.513 | 13.924 | 12.408 / 42.203 |
| 1024 | 328.021 | 319.343 | 313.184 / 341.204 |

These wall times do not isolate kernel quality. The custom median is slower at
S=128 through S=512 and 2.6% lower at S=1024, but the host-dominated samples
have broad tails. At S=1024 the device-kernel improvement is only 42
microseconds while one Raven call takes hundreds of milliseconds. Nsight
kernel duration is the meaningful implementation comparison until the PJRT
runtime avoids its current host round trips and serialization costs.

## Numerical validation

The PPX-generated forward and VJP calls were compiled together through Raven,
executed by PJRT CUDA, and compared with the unannotated Raven body on the CPU.
The acceptance tolerance was `3e-6`.

| Shape | Forward max abs | Row-sum max abs | Masked max abs | VJP max abs |
|---|---:|---:|---:|---:|
| `[1,2,4,4]` | 5.96e-8 | 8.94e-8 | 0 | 2.45e-9 |
| `[1,12,128,128]` | 5.96e-8 | 1.49e-7 | 0 | 7.16e-8 |
| `[1,2,129,129]` | 2.98e-8 | 1.12e-7 | 0 | 5.09e-8 |
| `[1,12,256,256]` | 2.98e-8 | 1.57e-7 | 0 | 1.05e-7 |
| `[1,12,512,512]` | 5.96e-8 | 1.62e-7 | 0 | 9.31e-8 |
| `[1,2,513,513]` | 5.96e-8 | 1.27e-7 | 0 | 8.36e-8 |
| `[1,12,1024,1024]` | 5.96e-8 | 1.74e-7 | 0 | 1.11e-7 |

An additional extreme-logit case, where the finite mask can dominate active
scores, matched exactly in both the forward and VJP. This guards the semantic
difference that an infinite mask would otherwise introduce.

A separate forward test places a transpose immediately before the custom call
and another immediately after it. It matched the fallback within `2.98e-8`,
confirming that the StableHLO operand and result layout constraints preserve the
raw kernel's packed row-major buffer contract.

## Reproduction

Build the PPX, runtime, benchmark, and CUDA library:

```sh
env RUNE_PJRT_CUDA_KERNELS=enabled \
  opam exec -- dune build dev/rune-pjrt/bench/softmax_profile.exe
```

The opt-in keeps `nvcc` and the local XLA FFI header tree out of normal
CPU-only builds and test runs.

Run an unprofiled custom-kernel sample from the executable's build directory:

```sh
cd _build/default/dev/rune-pjrt/bench
env CUDA_VISIBLE_DEVICES=0 \
  RUNE_PJRT_PLUGIN_PATH=/home/carsten/packages/xla/xla/pjrt/c \
  XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/carsten/packages/xla/external/cuda_nvcc \
  ./softmax_profile.exe kernel 1024 10 50
```

Use `baseline` in place of `kernel` for the regular StableHLO implementation.
Collect the GPU timeline with a unique output name:

```sh
nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --output=/tmp/raven-kernel-softmax-s1024 \
  env CUDA_VISIBLE_DEVICES=0 \
    RUNE_PJRT_PLUGIN_PATH=/home/carsten/packages/xla/xla/pjrt/c \
    XLA_FLAGS=--xla_gpu_cuda_data_dir=/home/carsten/packages/xla/external/cuda_nvcc \
    ./softmax_profile.exe kernel 1024 2 10

nsys stats \
  --report cuda_gpu_kern_gb_sum,cuda_gpu_mem_time_sum \
  --format csv /tmp/raven-kernel-softmax-s1024.nsys-rep
```

Nsight Compute hardware counters were unavailable on this machine with
`ERR_NVGPUCTRPERM`; all device durations above come from Nsight Systems.
