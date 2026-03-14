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
subset to StableHLO text, and calls PJRT in process through the C API. There is
no Python/JAX execution path in this incubator.

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

See [VENDORING.md](VENDORING.md) for the intended source layout.

## Examples

Small user-facing examples live under [`examples/`](examples):

- `01-inference`: greedy decoding through `Rune_pjrt.Causal_lm.greedy_decode`
- `02-training`: a compiled training step using `Rune.value_and_grads`
- `03-lora`: frozen-base adaptation with a low-rank update

The larger end-to-end GPT-2 example still lives in
`packages/kaun/examples/04-gpt2/pjrt/`.
