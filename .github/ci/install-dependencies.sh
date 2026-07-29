#!/usr/bin/env bash
set -eu

opam pin add thumper git+https://github.com/invariant-hq/thumper.git --no-action
opam install opam/*.opam dev/rune-pjrt/*.opam --deps-only --with-test --yes
