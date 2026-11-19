#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s IMAGE@sha256:<64 lowercase hex> [OUTPUT_TAG]\n' "$0" >&2
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    usage
    exit 2
fi
base_image="$1"
output_tag="${2:-so101-thor-worker:local}"

case "$base_image" in
    nvcr.io/nvidia/?*@sha256:[0-9a-f][0-9a-f]*) ;;
    *)
        printf '%s\n' \
            'error: base must be an nvcr.io/nvidia image pinned by a lowercase sha256 digest' >&2
        exit 2
        ;;
esac
digest="${base_image##*@sha256:}"
zeros="0000000000000000000000000000000000000000000000000000000000000000"
if [ "${#digest}" -ne 64 ] || [ "$digest" = "$zeros" ]; then
    printf 'error: replace the 64-character all-zero digest placeholder\n' >&2
    exit 2
fi
case "$digest" in *[!0-9a-f]*) printf 'error: invalid sha256 digest\n' >&2; exit 2 ;; esac

script_dir="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
worker_dir="$(dirname -- "$script_dir")"
if ! source_commit="$(git -C "$worker_dir" rev-parse --verify HEAD 2>/dev/null)"; then
    printf 'error: build from a Git checkout with an exact source revision\n' >&2
    exit 2
fi
if [ -n "$(git -C "$worker_dir" status --porcelain -- .)" ]; then
    printf 'error: commit all thor-worker changes before building the qualification image\n' >&2
    exit 2
fi
exec docker build \
    --build-arg "BASE_IMAGE=$base_image" \
    --build-arg "SOURCE_COMMIT=$source_commit" \
    --tag "$output_tag" \
    --file "$worker_dir/Dockerfile" \
    "$worker_dir"
