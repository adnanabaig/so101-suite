#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s %s\n' "$0" \
        'IMAGE SOURCE_COMMIT IMAGE_IDENTITY@sha256:<digest> OUTPUT_DIR [WHEELHOUSE_DIGEST]' >&2
}

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    usage
    exit 2
fi
image="$1"
source_commit="$2"
image_identity="$3"
output_dir="$4"
wheelhouse_digest="${5:-}"
script_dir="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
# Resolved relative to this script at runtime.
# shellcheck disable=SC1091
. "$script_dir/container-image.sh"

case "$image" in '' | -* | *[[:space:]]*) printf 'error: invalid image name\n' >&2; exit 2 ;; esac
case "$source_commit" in
    *[!0-9a-f]* | '')
        printf 'error: source commit must be 40 lowercase hex characters\n' >&2
        exit 2
        ;;
esac
if [ "${#source_commit}" -ne 40 ]; then
    printf 'error: source commit must be 40 lowercase hex characters\n' >&2
    exit 2
fi
if [ "$source_commit" = "0000000000000000000000000000000000000000" ]; then
    printf 'error: source commit may not be all zeroes\n' >&2
    exit 2
fi
case "$image_identity" in
    ?*@sha256:*) ;;
    *) printf 'error: image identity must be digest-pinned\n' >&2; exit 2 ;;
esac
digest="${image_identity##*@sha256:}"
[ "${#digest}" -eq 64 ] || { printf 'error: image identity has an invalid digest\n' >&2; exit 2; }
case "$digest" in
    *[!0-9a-f]*) printf 'error: image identity has an invalid digest\n' >&2; exit 2 ;;
esac
zeros="0000000000000000000000000000000000000000000000000000000000000000"
if [ "$digest" = "$zeros" ]; then
    printf 'error: image identity may not use the placeholder digest\n' >&2
    exit 2
fi
if [ -n "$wheelhouse_digest" ]; then
    if [ "${#wheelhouse_digest}" -ne 64 ] || [ "$wheelhouse_digest" = "$zeros" ]; then
        printf 'error: wheelhouse manifest must be a non-placeholder SHA-256 digest\n' >&2
        exit 2
    fi
    case "$wheelhouse_digest" in
        *[!0-9a-f]*)
            printf 'error: wheelhouse manifest must be a lowercase SHA-256 digest\n' >&2
            exit 2
            ;;
    esac
fi
[ -d "$output_dir" ] || { printf 'error: output directory does not exist\n' >&2; exit 2; }
output_dir="$(CDPATH='' cd -- "$output_dir" && pwd -P)"
case "$output_dir" in
    *','*) printf 'error: mount path may not contain commas\n' >&2; exit 2 ;;
    / | /dev | /dev/* | /proc | /proc/* | /sys | /sys/* | /run | /run/* | /etc | /etc/*)
        printf 'error: output must be a dedicated evidence directory\n' >&2
        exit 2
        ;;
esac
shopt -s nullglob dotglob
output_entries=("$output_dir"/*)
shopt -u nullglob dotglob
if [ "${#output_entries[@]}" -ne 0 ]; then
    printf 'error: output must be a new, empty evidence directory\n' >&2
    exit 2
fi
[ "$(id -u)" -ne 0 ] || { printf 'error: run the probe as a non-root user\n' >&2; exit 2; }
host_uid="$(id -u)"
host_gid="$(id -g)"
immutable_image="$(resolve_pinned_image "$image" "$image_identity")"
runtime_args=(
    /opt/so101-thor-worker/scripts/runtime_probe.py
    --source-commit "$source_commit"
    --container-image "$image_identity"
    --target thor
    --output /output/compatibility.json
)
if [ -n "$wheelhouse_digest" ]; then
    runtime_args+=(--wheelhouse-manifest-sha256 "$wheelhouse_digest")
fi

exec docker run --rm \
    --runtime=nvidia \
    --network=none \
    --read-only \
    --cap-drop=ALL \
    --security-opt=no-new-privileges:true \
    --user="$host_uid:$host_gid" \
    --tmpfs=/tmp:rw,noexec,nosuid,nodev,size=64m \
    --mount="type=bind,src=$output_dir,dst=/output" \
    --entrypoint=python3 \
    -- "$immutable_image" \
    "${runtime_args[@]}"
