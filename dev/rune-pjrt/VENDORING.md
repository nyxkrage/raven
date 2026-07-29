# Vendoring

The package ships the PJRT C API headers needed to compile its native bridge
and the XLA FFI C header needed to compile its example CUDA kernels under
`dev/rune-pjrt/vendor/`. Their upstream Apache 2.0 license is included beside
them. Full third-party source trees belong in the repository root `vendor/`
directory.

The bundled headers make release archives self-contained; they are not an XLA
source checkout and cannot build a PJRT plugin. When no plugin is supplied,
`rune-pjrt` can extract the prebuilt CUDA plugin from the official
`jax-cuda13-pjrt` or `jax-cuda12-pjrt` wheel at runtime. Downloaded wheels and
plugins remain in the user cache and are never vendored into the source tree.

## Expected Layout

- `vendor/xla`
- additional supporting repositories under `vendor/` when the chosen upstream
  build recipe requires them

The root [`vendor/dune`](/home/carsten/raven/vendor/dune) already marks
vendored directories for Dune.

## Clone Commands

Clone upstream sources into `vendor/`:

```bash
git clone https://github.com/openxla/xla.git vendor/xla
```

Or use the helper script:

```bash
bash dev/rune-pjrt/scripts/clone_vendor.sh
```

If the selected build recipe needs extra repositories, clone those into sibling
directories under `vendor/` as well.

## Build Outputs

All generated files, native objects, and external build outputs must live under
`_build/`.

- Never commit build outputs into `vendor/`
- Never write compiled libraries back into upstream source trees
- Prefer Dune rules that redirect external build directories into
  `_build/default/dev/rune-pjrt/...`

## Plugin Builds

Build vendored PJRT plugins with:

```bash
bash dev/rune-pjrt/scripts/build_plugin.sh cpu
```

or:

```bash
bash dev/rune-pjrt/scripts/build_plugin.sh cuda
```

The helper runs Bazel with its output root under
`_build/default/dev/rune-pjrt/bazel` and records the real built plugin path in
`_build/default/dev/rune-pjrt/plugins/*.path`.

`rune-pjrt` then loads the Bazel-built plugin in place. This is required for
the CUDA plugin, whose loader `RUNPATH` is tied to the original Bazel output
layout.
