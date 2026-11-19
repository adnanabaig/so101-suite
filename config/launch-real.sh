#!/usr/bin/env bash
# Non-reloading real backend; configure and verify .env before use.
set -euo pipefail
suite_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$suite_root"
if [[ ! -f .env ]]; then
    echo 'Missing .env. Review config/rigs/so101.env.example and verify USB roles first.' >&2
    exit 1
fi
unset PYTHONPATH LEROBOT_HOME
for setting in ${!SO101_@}; do
    unset "$setting"
done
export SO101_HOME="$suite_root/data/real-home"
export SO101_BACKEND=real SO101_MOCK=0
export SO101_DATA_DIR="$suite_root/data/real"
export SO101_DATASETS_DIR="$suite_root/datasets/real"
export SO101_OUTPUTS_DIR="$suite_root/outputs/real"
export SO101_UI_DIR="$suite_root/frontend/dist"
export SO101_HOST=127.0.0.1 SO101_PORT=8000 SO101_CORS_ORIGINS=
if (( $# == 0 )); then
    set -- serve
fi
exec "$suite_root/.venv/bin/so101" "$@"
