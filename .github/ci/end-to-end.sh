#!/usr/bin/env bash
set -eu

source_root="$(pwd)"
release_root="$(mktemp -d)"
trap 'rm -rf "$release_root"' EXIT
opam_switch="$(opam switch show)"
export OPAMSWITCH="$opam_switch"
eval "$(opam env --switch="$opam_switch" --set-switch)"

git ls-files --cached --others --exclude-standard -z \
  | tar --null --files-from=- --create \
  | tar -x -C "$release_root"
cd "$release_root"

opam exec -- dune build --profile release @install

opam exec -- dune exec --profile release \
  packages/brot/examples/10-bert-pipeline/main.exe
opam exec -- dune exec --profile release \
  packages/rune/examples/02-jit-inference/main.exe
opam exec -- dune exec --profile release \
  packages/kaun/examples/01-xor/main.exe
opam exec -- dune exec --profile release \
  packages/fehu/examples/01-random-agent/main.exe
opam exec -- dune exec --profile release dev/hugr/test/test_llama.exe

printf 'End-to-end tests passed for the archive produced from %s\n' "$source_root"
