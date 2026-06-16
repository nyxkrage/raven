# JIT Inference Backends

Run a tiny autoregressive inference loop through `Rune.jit` with a selectable
backend device. Backend parsing and device construction are handled by
`Rune.Backend`, so the decode code only needs a `Rune.Device.t`.

```bash
dune exec packages/rune/examples/02-jit-inference/main.exe -- tolk-cpu
dune exec packages/rune/examples/02-jit-inference/main.exe -- pjrt-cpu
dune exec packages/rune/examples/02-jit-inference/main.exe -- pjrt-cuda
```

The same backend can also be selected with `RUNE_JIT_BACKEND`.
The default backend is `pjrt-cuda`. PJRT modes use `RUNE_PJRT_DEVICE_ID` when
set.

`tolk-cpu` is included to show the same `Rune.Device` API, but the current
Rune/Tolk replay path is non-functional on this branch, so use PJRT to run the
full decode.
