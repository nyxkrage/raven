# Vendoring

Third-party source trees for the PJRT/XLA path belong in the repository root
`vendor/` directory, not under `dev/rune-pjrt/`.

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
