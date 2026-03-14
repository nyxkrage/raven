#!/usr/bin/env bash
set -eu

backend="${1:-cpu}"
repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)"
xla_root="$repo_root/vendor/xla"
build_root="$repo_root/_build/default/dev/rune-pjrt"
plugin_dir="$build_root/plugins"
bazel_args=()

detect_cuda_caps() {
  if [ -n "${HERMETIC_CUDA_COMPUTE_CAPABILITIES:-}" ]; then
    printf '%s\n' "$HERMETIC_CUDA_COMPUTE_CAPABILITIES"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local caps
    caps="$(
      nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | awk -F. 'NF == 2 { gsub(/ /, "", $1); gsub(/ /, "", $2); print "sm_" $1 $2 }' \
        | sort -u \
        | paste -sd, -
    )"
    if [ -n "$caps" ]; then
      printf '%s\n' "$caps"
      return
    fi
  fi
  printf '%s\n' "sm_80,sm_86,sm_89,compute_90"
}

case "$backend" in
  cpu)
    target="//xla/pjrt/c:pjrt_c_api_cpu_plugin.so"
    plugin_name="pjrt_c_api_cpu_plugin.so"
    user_root="$build_root/bazel/cpu"
    ;;
  cuda|gpu)
    target="//xla/pjrt/c:pjrt_c_api_gpu_plugin.so"
    plugin_name="pjrt_c_api_gpu_plugin.so"
    user_root="$build_root/bazel/cuda"
    bazel_args+=(--config=cuda)
    cuda_caps="$(detect_cuda_caps)"
    bazel_args+=("--repo_env=HERMETIC_CUDA_COMPUTE_CAPABILITIES=$cuda_caps")
    echo "using CUDA compute capabilities: $cuda_caps"
    ;;
  *)
    echo "usage: $0 [cpu|cuda]" >&2
    exit 2
    ;;
esac

if [ ! -d "$xla_root/.git" ]; then
  echo "vendor/xla is missing; clone it first" >&2
  exit 1
fi

mkdir -p "$plugin_dir"
bazel_args=(--output_user_root="$user_root" build "${bazel_args[@]}")

(
  cd "$xla_root"
  bazel "${bazel_args[@]}" "$target"
)

source_path="$xla_root/bazel-bin/xla/pjrt/c/$plugin_name"
if [ ! -f "$source_path" ]; then
  echo "built plugin not found at $source_path" >&2
  exit 1
fi

resolved_path="$(readlink -f "$source_path")"
printf '%s\n' "$resolved_path" > "$plugin_dir/$plugin_name.path"
rm -f "$plugin_dir/$plugin_name"
echo "recorded $plugin_name at $plugin_dir/$plugin_name.path -> $resolved_path"
