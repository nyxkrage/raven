# GPT-2 Benchmarks

This benchmark compares the PJRT/XLA GPT-2 path against HuggingFace PyTorch on
the same synthetic prompt and greedy-generation workload.

## What It Measures

- `prefill`: prompt processing through the model, returning only the next-token
  argmax instead of materializing the full logits tensor to host
- `decode`: greedy decode for `max_new_tokens`
- `prefill tok/s`: prompt tokens processed per second after warmup
- `decode tok/s`: generated tokens per second after warmup
- `mean_ms`: steady-state latency after warmup

The OCaml side uses the real GPT-2 PJRT example code through
`Rune_pjrt.jit` and `Rune_pjrt.Causal_lm.greedy_decode`. The Python side uses
`transformers.GPT2LMHeadModel` with `torch.compile`.

The benchmark intentionally does not time "full prompt logits copied back to
host" as the primary forward metric, because that mostly measures output
materialization of a very large tensor rather than inference.

The benchmark uses synthetic token ids and wrapped position ids
(`position_id % n_positions`) so it can exercise `2048 + 512` tokens even
though standard GPT-2 ships with `n_positions = 1024`.

## Setup

Create a virtual environment with `uv`:

```bash
uv venv dev/rune-pjrt/bench/.venv
uv pip install --python dev/rune-pjrt/bench/.venv/bin/python torch transformers
```

Build the PJRT plugin you want to benchmark:

```bash
bash dev/rune-pjrt/scripts/build_plugin.sh cpu
bash dev/rune-pjrt/scripts/build_plugin.sh cuda
```

## Running

CUDA:

```bash
uv run --python dev/rune-pjrt/bench/.venv/bin/python \
  python dev/rune-pjrt/bench/bench_gpt2_vs_torch.py \
  --backend cuda \
  --prompt-tokens 2048 \
  --max-tokens 512 \
  --warmup 1 \
  --iterations 1
```

CPU:

```bash
uv run --python dev/rune-pjrt/bench/.venv/bin/python \
  python dev/rune-pjrt/bench/bench_gpt2_vs_torch.py \
  --backend cpu \
  --prompt-tokens 2048 \
  --max-tokens 512 \
  --warmup 1 \
  --iterations 1
```

The CPU run is supported by the harness, but at `2048 + 512` tokens it is a
very slow workload and is mainly useful for completeness checks rather than
regular iteration.

To save the combined results as JSON:

```bash
uv run --python dev/rune-pjrt/bench/.venv/bin/python \
  python dev/rune-pjrt/bench/bench_gpt2_vs_torch.py \
  --backend cuda \
  --json-output dev/rune-pjrt/bench/results-gpt2-cuda.json
```
