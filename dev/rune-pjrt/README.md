# rune-pjrt

Experimental PJRT/XLA integration for Rune.

This incubator lives under `dev/` on purpose. It is where the tracing, IR,
runtime bridge, and build story can evolve before anything is promoted into the
published `rune` package.

## Scope

- trace `Nx_effect` programs into a compact JIT IR
- validate the GPT-2 forward-pass subset needed by `packages/kaun/examples/04-gpt2`
- build toward PJRT/XLA execution on CUDA
- keep vendored third-party source trees in `vendor/`
- keep generated build artifacts in `_build/`

## Current Status

The OCaml side traces `Nx_effect` programs into a compact IR, lowers a small
subset to StableHLO text, and calls PJRT in process through the C API. Host
buffers can be used with `jit`, while `jit_device` keeps typed buffers resident
on one PJRT device and lets separately compiled calls feed each other without
host transfers. There is no Python/JAX execution path in this incubator.

To execute anything, build a vendored PJRT plugin into `_build/` first:

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
plugins below them.

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
`rune` package. The larger end-to-end GPT-2 example still lives in
`packages/kaun/examples/04-gpt2/pjrt/`.
