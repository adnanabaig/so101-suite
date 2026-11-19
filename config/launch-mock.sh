#!/usr/bin/env bash
# Isolated software preview; never selects physical ports or cameras.
set -euo pipefail
suite_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$suite_root"
unset PYTHONPATH LEROBOT_HOME
for setting in ${!SO101_@}; do
    unset "$setting"
done
export SO101_HOME="$suite_root/data/mock-home"
export SO101_BACKEND=mock SO101_MOCK=1 SO101_CAMERAS=
export SO101_DATA_DIR="$suite_root/data/mock"
export SO101_DATASETS_DIR="$suite_root/datasets/mock"
export SO101_OUTPUTS_DIR="$suite_root/outputs/mock"
export SO101_UI_DIR="$suite_root/frontend/dist"
export SO101_HOST=127.0.0.1 SO101_PORT=8000 SO101_CORS_ORIGINS=
if (( $# == 0 )); then
    set -- serve
fi
exec "$suite_root/.venv/bin/so101" "$@"
