from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import GPT2LMHeadModel

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OCAML_EXE = ROOT / "_build" / "default" / "packages" / "kaun" / "examples" / "04-gpt2" / "pjrt" / "bench_main.exe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark rune-pjrt GPT-2 against PyTorch")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--model-id", default="gpt2")
    parser.add_argument("--ocaml-exe", type=Path, default=DEFAULT_OCAML_EXE)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def summarize(name: str, first_s: float, samples_s: list[float]) -> dict[str, float]:
    samples_ms = [sample * 1000.0 for sample in samples_s]
    return {
        "name": name,
        "first_ms": first_s * 1000.0,
        "mean_ms": statistics.mean(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def bench_timed(fn: Callable[[], Any], *, warmup: int, iterations: int, device: str) -> tuple[dict[str, float], Any]:
    sync(device)
    start = time.perf_counter()
    first_result = fn()
    sync(device)
    first_s = time.perf_counter() - start

    for _ in range(warmup):
        fn()
        sync(device)

    samples_s: list[float] = []
    last_result = first_result
    for _ in range(iterations):
        sync(device)
        start = time.perf_counter()
        last_result = fn()
        sync(device)
        samples_s.append(time.perf_counter() - start)

    return summarize(fn.__name__, first_s, samples_s), last_result


def build_ocaml_bench(exe: Path) -> None:
    rel = exe.relative_to(ROOT)
    subprocess.run(["dune", "build", str(rel)], cwd=ROOT, check=True)


def run_rune_bench(args: argparse.Namespace) -> dict[str, Any]:
    if not args.skip_build:
        build_ocaml_bench(args.ocaml_exe)

    env = os.environ.copy()
    env["RUNE_PJRT_BACKEND"] = args.backend
    env["RUNE_PJRT_BENCH_PROMPT_TOKENS"] = str(args.prompt_tokens)
    env["RUNE_PJRT_BENCH_MAX_TOKENS"] = str(args.max_tokens)
    env["RUNE_PJRT_BENCH_WARMUP"] = str(args.warmup)
    env["RUNE_PJRT_BENCH_ITERS"] = str(args.iterations)
    proc = subprocess.run(
        [str(args.ocaml_exe)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("rune-pjrt benchmark produced no stdout")
    return json.loads(lines[-1])


def run_torch_bench(args: argparse.Namespace) -> dict[str, Any]:
    if args.backend == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA backend requested but torch.cuda.is_available() is false")

    device = args.backend
    torch.set_float32_matmul_precision("high")
    model = GPT2LMHeadModel.from_pretrained(args.model_id)
    model.eval()
    model.to(device)
    prompt_len = args.prompt_tokens
    max_seq = prompt_len + args.max_tokens
    vocab_size = int(model.config.vocab_size)
    n_positions = int(model.config.n_positions)
    prompt_ids = [(((i * 17) + 11) % (vocab_size - 1)) + 1 for i in range(prompt_len)]
    input_ids = torch.tensor([prompt_ids], dtype=torch.int32, device=device).to(torch.long)
    prompt_position_ids = (
        torch.arange(prompt_len, device=device, dtype=torch.long).remainder(n_positions).unsqueeze(0)
    )
    full_position_ids = (
        torch.arange(max_seq, device=device, dtype=torch.long).remainder(n_positions).unsqueeze(0)
    )
    full_tokens = torch.zeros((1, max_seq), dtype=torch.long, device=device)
    full_tokens[:, :prompt_len] = input_ids

    def forward_fn(input_ids_t: torch.Tensor, position_ids_t: torch.Tensor) -> torch.Tensor:
        return model(input_ids=input_ids_t, position_ids=position_ids_t).logits

    def generate_fn(compiled_forward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], tokens_t: torch.Tensor) -> torch.Tensor:
        tokens = tokens_t.clone()
        pos = prompt_len - 1
        for _ in range(args.max_tokens):
            logits = compiled_forward_fn(tokens, full_position_ids)
            next_token = torch.argmax(logits[:, pos, :], dim=-1)
            pos += 1
            tokens[:, pos] = next_token
        return tokens

    compiled_forward = torch.compile(forward_fn, mode="max-autotune")

    with torch.inference_mode():
        prefill_stats, prefill_token = bench_timed(
            lambda: torch.argmax(
                compiled_forward(input_ids, prompt_position_ids)[:, prompt_len - 1, :],
                dim=-1,
            ).item(),
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        decode_stats, generated = bench_timed(
            lambda: generate_fn(compiled_forward, full_tokens),
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )

    tail_sum = int(generated[0, prompt_len:].sum().item())
    prefill_tokens_per_s = args.prompt_tokens / (prefill_stats["mean_ms"] / 1000.0)
    decode_tokens_per_s = args.max_tokens / (decode_stats["mean_ms"] / 1000.0)
    return {
        "backend": args.backend,
        "model_id": args.model_id,
        "prompt_tokens": prompt_len,
        "max_tokens": args.max_tokens,
        "max_seq": max_seq,
        "position_mode": "wrapped_mod_n_positions",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "prefill": prefill_stats,
        "prefill_token": int(prefill_token),
        "decode": decode_stats,
        "prefill_tokens_per_s": prefill_tokens_per_s,
        "decode_tokens_per_s": decode_tokens_per_s,
        "generated_tail_sum": tail_sum,
    }


def speedup(reference_ms: float, candidate_ms: float) -> float:
    return reference_ms / candidate_ms


def print_report(rune_result: dict[str, Any], torch_result: dict[str, Any]) -> None:
    rows = [
        ("prefill mean ms", rune_result["prefill"]["mean_ms"], torch_result["prefill"]["mean_ms"]),
        ("prefill tok/s", rune_result["prefill_tokens_per_s"], torch_result["prefill_tokens_per_s"]),
        ("decode mean ms", rune_result["decode"]["mean_ms"], torch_result["decode"]["mean_ms"]),
        ("decode tok/s", rune_result["decode_tokens_per_s"], torch_result["decode_tokens_per_s"]),
    ]
    print(f"backend: {rune_result['backend']}")
    print(f"model:   {rune_result['model_id']}")
    print(f"prompt_tokens: {rune_result['prompt_tokens']}")
    print(f"max_tokens:    {rune_result['max_tokens']}")
    print(f"position_mode: {rune_result['position_mode']}")
    print(f"prefill_token: {rune_result['prefill_token']}")
    print()
    print(f"{'metric':<20} {'rune-pjrt':>14} {'pytorch':>14} {'torch/rune':>12}")
    for name, rune_value, torch_value in rows:
        ratio = speedup(torch_value, rune_value)
        print(f"{name:<20} {rune_value:14.6f} {torch_value:14.6f} {ratio:12.3f}x")
    print()
    print(f"rune tail checksum:    {rune_result['generated_tail_sum']}")
    print(f"pytorch tail checksum: {torch_result['generated_tail_sum']}")


def main() -> None:
    args = parse_args()
    rune_result = run_rune_bench(args)
    torch_result = run_torch_bench(args)
    report = {
        "rune_pjrt": rune_result,
        "pytorch": torch_result,
    }
    print_report(rune_result, torch_result)
    if args.json_output is not None:
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
