# rune-pjrt

Experimental PJRT/XLA integration for Rune.

This incubator lives under `dev/` on purpose. It is where the tracing, IR,
runtime bridge, and build story can evolve before anything is promoted into the
published `rune` package.

## Scope

- trace `Nx_effect` programs into a compact JIT IR
- run dense transformer inference with device-resident KV caches
- build toward PJRT/XLA execution on CUDA
- use a provided PJRT plugin, an official JAX PJRT wheel, or a source build
- keep generated build artifacts in `_build/`

## Current Status

The OCaml side traces `Nx_effect` programs into a compact IR, lowers a small
subset to StableHLO text, and calls PJRT in process through the C API. Host
buffers can be used with `jit`, while `jit_device` keeps typed buffers resident
on one PJRT device and lets separately compiled calls feed each other without
host transfers. Python is used only by the plugin locator/downloader; JAX is not
imported and does not participate in tracing, compilation, or execution.

For CUDA, `rune-pjrt` first looks for an explicitly provided or locally built
plugin. If none exists, it locates an installed `jax-cuda13-pjrt` or
`jax-cuda12-pjrt` wheel. As a final fallback it downloads the matching official
wheel from PyPI, verifies its SHA-256 digest, and extracts only
`xla_cuda_plugin.so` into the user cache. CUDA 13 is selected for NVIDIA drivers
580 and newer; CUDA 12 is selected for drivers 525 through 579.

Set `RUNE_PJRT_AUTO_FETCH=0` to disable network fallback. The other controls
are:

- `RUNE_PJRT_CUDA_VERSION=12|13` to override CUDA detection
- `RUNE_PJRT_JAX_VERSION` to pin a JAX PJRT wheel version
- `RUNE_PJRT_PLUGIN_CACHE` to change the extraction cache
- `RUNE_PJRT_FETCHER` to provide another downloader executable or script
- `RUNE_PJRT_PYTHON` to select the Python interpreter

To build plugins from a vendored XLA checkout instead, run:

```bash
bash dev/rune-pjrt/scripts/build_plugin.sh cpu
```

or:

```bash
bash dev/rune-pjrt/scripts/build_plugin.sh cuda
```

The helper records the real Bazel-built plugin path under
`_build/default/dev/rune-pjrt/plugins/*.path` and `rune-pjrt` loads that
artifact in place. This matters for the CUDA plugin, which relies on the
original Bazel output location for its loader `RUNPATH`.

You can also point the runtime at plugins outside `_build/` with
`RUNE_PJRT_PLUGIN_PATH`. The value is a colon-separated search path; entries may
be plugin directories, direct plugin files, `.path` files, or package roots with
plugins below them. Explicit plugins always take precedence over the wheel
fallback.

See [VENDORING.md](VENDORING.md) for the intended source layout.

## Device-resident execution

Transfer long-lived inputs once, compose compiled calls on the device, and
only copy a result back when host code needs it:

```ocaml
let normalize =
  Rune_pjrt.jit_device (fun x ->
      let mean = Nx.mean ~axes:[ 1 ] ~keepdims:true x in
      Nx.sub x mean)
in
let x_device = Rune_pjrt.Device_buffer.of_host x in
let y_device = normalize x_device in
Rune_pjrt.Device_buffer.await y_device;
let y = Rune_pjrt.Device_buffer.to_host y_device
```

`jit_device` returns before CUDA finishes. `Device_buffer.await` synchronizes
without downloading, and `Device_buffer.to_host` synchronizes as part of the
transfer. Use `jits_device` for homogeneous multi-input functions and
`jits_device_packed` when input or output dtypes differ.

## Examples

User-facing examples that select PJRT through the Rune device API live with the
`rune` and `hugr` packages.
