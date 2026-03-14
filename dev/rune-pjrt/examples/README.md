# rune-pjrt examples

These examples are intentionally small and deterministic. They show the public
`Rune_pjrt` API that user code is expected to call, while still exercising the
real PJRT/XLA path on CPU or CUDA.

## Examples

- `01-inference`: greedy decoding with `Rune_pjrt.Causal_lm.greedy_decode`
- `02-training`: a compiled linear-regression training step using
  `Rune.value_and_grads` inside `Rune_pjrt.jits`
- `03-lora`: a frozen linear projection with a trainable low-rank adapter

## Running

Build the plugins first:

```bash
bash dev/rune-pjrt/scripts/build_plugin.sh cpu
bash dev/rune-pjrt/scripts/build_plugin.sh cuda
```

Then run any example on CPU:

```bash
RUNE_PJRT_BACKEND=cpu dune exec dev/rune-pjrt/examples/01-inference/main.exe
RUNE_PJRT_BACKEND=cpu dune exec dev/rune-pjrt/examples/02-training/main.exe
RUNE_PJRT_BACKEND=cpu dune exec dev/rune-pjrt/examples/03-lora/main.exe
```

Or on CUDA:

```bash
RUNE_PJRT_BACKEND=cuda dune exec dev/rune-pjrt/examples/01-inference/main.exe
RUNE_PJRT_BACKEND=cuda dune exec dev/rune-pjrt/examples/02-training/main.exe
RUNE_PJRT_BACKEND=cuda dune exec dev/rune-pjrt/examples/03-lora/main.exe
```

Useful environment variables:

- `RUNE_PJRT_BACKEND=cpu|cuda`
- `RUNE_PJRT_DEVICE_ID=<int>`
- `RUNE_PJRT_MAX_TOKENS=<int>` for `01-inference`
- `RUNE_PJRT_TRAIN_STEPS=<int>` and `RUNE_PJRT_LEARNING_RATE=<float>` for
  `02-training`
- `RUNE_PJRT_LORA_STEPS=<int>` and `RUNE_PJRT_LORA_LR=<float>` for `03-lora`
