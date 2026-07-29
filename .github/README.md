# Continuous integration

GitHub Actions separates checks by their cost and platform requirements:

- `quality` checks OCaml formatting, opam manifests, and Python helpers.
- `unit` builds every ordinary Raven project and runs its test aliases on the
  minimum full-workspace OCaml compiler, the current compiler, and macOS.
- `end-to-end` builds from an exact clean-source copy and runs representative
  tokenizer, JIT, training, reinforcement-learning, and model-inference entry
  points.
- `oxcaml` builds and tests Nx with the OxCaml compiler in its isolated project.
- `cuda` sends the current source tree to an L4 GPU on Modal. It requires a
  working PJRT CUDA plugin, compiles and runs the raw-CUDA and Triton tests, runs
  Hugr's cached Llama tests, and performs smoke prefill and decode executions.

## Enabling Modal CUDA CI

Create a Modal service token and add its ID and secret to the repository as
GitHub Actions secrets named `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. Then set
the repository Actions variable `MODAL_CI_ENABLED` to `true`.

The CUDA job runs for pushes, manual workflow runs, and pull requests whose
source branch belongs to this repository. It does not receive credentials or
run for pull requests from forks.

Modal uses the official NVIDIA CUDA development image and persists downloaded
JAX PJRT wheels in the `raven-ci-pjrt-cache` Modal volume. OCaml dependencies
are cached as an image layer and are rebuilt only when an opam manifest changes.

Run the same remote job manually after configuring a local Modal token:

```sh
python -m pip install "modal>=1.4,<2"
modal run .github/modal/cuda_ci.py
```
