# ---------------------------------------------------------------------------
# Copyright (c) 2026 The Raven authors. All rights reserved.
# SPDX-License-Identifier: ISC
# ---------------------------------------------------------------------------

"""Run Raven's CUDA unit and end-to-end tests on Modal."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal


REMOTE_REPOSITORY = Path("/workspace/raven")
REPOSITORY = (
    Path(__file__).resolve().parents[2]
    if modal.is_local()
    else REMOTE_REPOSITORY
)
OPAM_ROOT = "/opt/opam"
OPAM_SWITCH = "raven-ci"

app = modal.App("raven-cuda-ci")
plugin_cache = modal.Volume.from_name(
    "raven-ci-pjrt-cache", create_if_missing=True
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install(
        "bubblewrap",
        "ca-certificates",
        "curl",
        "git",
        "libcairo2-dev",
        "libffi-dev",
        "libgmp-dev",
        "liblapacke-dev",
        "libopenblas-dev",
        "libsdl2-dev",
        "m4",
        "opam",
        "pkg-config",
        "unzip",
        "zlib1g-dev",
    )
    .env(
        {
            "OPAMROOT": OPAM_ROOT,
            "OPAMSWITCH": OPAM_SWITCH,
            "OPAMYES": "true",
        }
    )
    .run_commands(
        "opam init --bare --disable-sandboxing -y",
        f"opam switch create {OPAM_SWITCH} ocaml-base-compiler.5.4.0 -y",
    )
    .add_local_dir(REPOSITORY / "opam", "/tmp/raven-opam/opam", copy=True)
    .add_local_file(
        REPOSITORY / "dev/rune-pjrt/rune-pjrt.opam",
        "/tmp/raven-opam/dev/rune-pjrt/rune-pjrt.opam",
        copy=True,
    )
    .add_local_file(
        REPOSITORY / "dev/rune-pjrt/ppx_rune_kernel.opam",
        "/tmp/raven-opam/dev/rune-pjrt/ppx_rune_kernel.opam",
        copy=True,
    )
    .run_commands(
        "opam pin add thumper "
        "git+https://github.com/invariant-hq/thumper.git --no-action",
        "opam install /tmp/raven-opam/opam/*.opam "
        "/tmp/raven-opam/dev/rune-pjrt/*.opam --deps-only --with-test -y"
    )
    .add_local_dir(
        REPOSITORY,
        str(REMOTE_REPOSITORY),
        ignore=[
            ".git/**",
            "_build/**",
            "_opam/**",
            "vendor/xla/**",
            "vendor/xet-core/**",
            "**/__pycache__/**",
        ],
    )
)


def run(arguments: list[str], environment: dict[str, str]) -> None:
    print(f"+ {shlex.join(arguments)}", flush=True)
    subprocess.run(
        arguments,
        cwd=REMOTE_REPOSITORY,
        env=environment,
        check=True,
    )


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={"/cache": plugin_cache},
)
def cuda_tests() -> None:
    environment = dict(os.environ)
    environment.pop("RUNE_PJRT_TEST_SKIP_CUDA", None)
    environment.update(
        {
            "RUNE_PJRT_AUTO_FETCH": "enabled",
            "RUNE_PJRT_CUDA_KERNELS": "enabled",
            "RUNE_PJRT_CUDA_VERSION": "12",
            "RUNE_PJRT_PLUGIN_CACHE": "/cache/rune-pjrt",
            "RUNE_PJRT_TEST_REQUIRE_CUDA": "1",
            "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/usr/local/cuda",
        }
    )

    run(["nvidia-smi"], environment)
    run(["nvcc", "--version"], environment)

    kernels = [
        "dev/rune-pjrt/kernels/causal_scaled_softmax.so",
        "dev/rune-pjrt/kernels/grouped_gemm.so",
    ]
    executables = [
        "dev/rune-pjrt/test/test_runtime.exe",
        "dev/rune-pjrt/test/test_cuda_ffi.exe",
        "dev/rune-pjrt/test/test_cuda_grouped_gemm.exe",
        "dev/rune-pjrt/test/test_cuda_triton.exe",
        "dev/hugr/test/test_llama.exe",
        "dev/hugr/bench/llama_profile.exe",
    ]
    run(
        [
            "opam",
            "exec",
            "--",
            "dune",
            "build",
            "--profile",
            "release",
            *kernels,
            *executables,
        ],
        environment,
    )

    for executable in executables[:-1]:
        run(
            [
                "opam",
                "exec",
                "--",
                "dune",
                "exec",
                "--profile",
                "release",
                executable,
            ],
            environment,
        )

    benchmark = executables[-1]
    common = [
        "--preset",
        "smoke",
        "--cache-length",
        "32",
        "--warmups",
        "1",
        "--iterations",
        "2",
    ]
    run(
        [
            "opam",
            "exec",
            "--",
            "dune",
            "exec",
            "--profile",
            "release",
            benchmark,
            "--",
            "--case",
            "decode",
            *common,
        ],
        environment,
    )
    run(
        [
            "opam",
            "exec",
            "--",
            "dune",
            "exec",
            "--profile",
            "release",
            benchmark,
            "--",
            "--case",
            "prefill",
            "--prompt-length",
            "8",
            *common,
        ],
        environment,
    )
    plugin_cache.commit()


@app.local_entrypoint()
def main() -> None:
    cuda_tests.remote()
