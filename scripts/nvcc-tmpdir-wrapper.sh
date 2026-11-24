#!/usr/bin/env bash
# Route nvcc temporary files to RAM-backed /dev/shm so CUDA kernel builds do
# not exhaust a disk-backed /tmp (ENOSPC) on CI workers.
set -euo pipefail

tmp_dir="${CUDA_BUILD_TMPDIR:-/dev/shm}"
if [[ -d "${tmp_dir}" && -w "${tmp_dir}" ]]; then
  export TMPDIR="${tmp_dir}"
fi

real_nvcc="${NVCC_REAL:-}"
if [[ -z "${real_nvcc}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    real_nvcc="$(command -v nvcc)"
  elif [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
    real_nvcc="${CUDA_HOME}/bin/nvcc"
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    real_nvcc=/usr/local/cuda/bin/nvcc
  elif [[ -x /opt/cuda/bin/nvcc ]]; then
    real_nvcc=/opt/cuda/bin/nvcc
  else
    echo "nvcc-tmpdir-wrapper: nvcc not found" >&2
    exit 127
  fi
fi

exec "${real_nvcc}" "$@"
