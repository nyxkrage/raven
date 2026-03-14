#!/usr/bin/env bash
set -eu

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
vendor_root="$repo_root/vendor"
xla_dir="$vendor_root/xla"

mkdir -p "$vendor_root"

if [ -d "$xla_dir/.git" ]; then
  echo "vendor/xla already exists"
  exit 0
fi

git clone https://github.com/openxla/xla.git "$xla_dir"
