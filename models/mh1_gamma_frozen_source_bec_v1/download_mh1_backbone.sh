#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$(cd -- "$(dirname -- "$0")" && pwd)/external"
MODEL_PATH="${MODEL_DIR}/mace-mh-1.model"
MODEL_URL="https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model"
EXPECTED_SHA256="ec00a2705854622fbbd898ccfb7701072fcd674709102d009fb919c1b8cc5dde"

mkdir -p "${MODEL_DIR}"
if [[ ! -f "${MODEL_PATH}" ]]; then
  curl --fail --location --retry 3 "${MODEL_URL}" --output "${MODEL_PATH}"
fi

printf '%s  %s\n' "${EXPECTED_SHA256}" "${MODEL_PATH}" | sha256sum --check --status || {
  rm -f "${MODEL_PATH}"
  printf 'MH-1 backbone checksum mismatch; downloaded file was removed.\n' >&2
  exit 1
}

printf 'Verified MH-1 backbone: %s\n' "${MODEL_PATH}"
