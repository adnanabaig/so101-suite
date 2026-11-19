#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s IMAGE INPUT_DIRECTORY OUTPUT_DIRECTORY\n' "$0" >&2
}

# Fixed arity is intentional: callers cannot append arbitrary Docker flags.
[ "$#" -eq 3 ] || { usage; exit 2; }
image="$1"
input_dir="$2"
output_dir="$3"
script_dir="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
# Resolved relative to this script at runtime.
# shellcheck disable=SC1091
. "$script_dir/container-image.sh"

case "$image" in '' | -* | *[[:space:]]*) printf 'error: invalid image name\n' >&2; exit 2 ;; esac
[ -d "$input_dir" ] || { printf 'error: input directory does not exist\n' >&2; exit 2; }
[ -d "$output_dir" ] || { printf 'error: output directory does not exist\n' >&2; exit 2; }
input_dir="$(CDPATH='' cd -- "$input_dir" && pwd -P)"
output_dir="$(CDPATH='' cd -- "$output_dir" && pwd -P)"
if [[ "$input_dir" == "$output_dir" \
    || "$input_dir" == "$output_dir/"* \
    || "$output_dir" == "$input_dir/"* ]]; then
    printf 'error: input and output mount paths must not overlap or contain one another\n' >&2
    exit 2
fi
case "$input_dir" in
    *,*) printf 'error: mount paths may not contain commas\n' >&2; exit 2 ;;
esac
case "$output_dir" in
    *,*) printf 'error: mount paths may not contain commas\n' >&2; exit 2 ;;
esac
case "$input_dir" in
    / | /dev | /dev/* | /proc | /proc/* | /sys | /sys/* | /run | /run/* | /etc | /etc/*)
        printf 'error: input must be a dedicated artifact directory\n' >&2
        exit 2
        ;;
esac
case "$output_dir" in
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
[ "$(id -u)" -ne 0 ] || { printf 'error: run the worker as a non-root user\n' >&2; exit 2; }
if [ ! -f "$input_dir/benchmark-config.json" ]; then
    printf 'error: input lacks benchmark-config.json\n' >&2
    exit 2
fi
if [ ! -f "$input_dir/compatibility.json" ]; then
    printf 'error: input lacks compatibility.json\n' >&2
    exit 2
fi
compatibility_reader='import sys; from pathlib import Path; '
compatibility_reader+='from benchmark_contract import read_json, validate_execution_compatibility; '
compatibility_reader+='raw=read_json(Path(sys.argv[1])); '
compatibility_reader+='value=validate_execution_compatibility(raw, '
compatibility_reader+='expected_source_commit=raw["source_commit"], '
compatibility_reader+='expected_container_image=raw["container_image"]); '
compatibility_reader+='print(value["container_image"])'
if ! image_identity="$(
    PYTHONPATH="$script_dir" python3 -c "$compatibility_reader" "$input_dir/compatibility.json"
)"; then
    printf 'error: compatibility.json is invalid\n' >&2
    exit 2
fi
[ "$image_identity" != "HOST-NATIVE" ] || {
    printf 'error: a container benchmark requires a digest-pinned compatibility probe\n' >&2
    exit 2
}
host_uid="$(id -u)"
host_gid="$(id -g)"
immutable_image="$(resolve_pinned_image "$image" "$image_identity")"

# No --device, --privileged, host namespace, Docker socket, USB, or TTY mount.
# The Jetson NVIDIA runtime supplies only its configured compute facilities.
exec docker run --rm \
    --runtime=nvidia \
    --network=none \
    --read-only \
    --cap-drop=ALL \
    --security-opt=no-new-privileges:true \
    --user="$host_uid:$host_gid" \
    --tmpfs=/tmp:rw,noexec,nosuid,nodev,size=256m \
    --mount="type=bind,src=$input_dir,dst=/input,readonly" \
    --mount="type=bind,src=$output_dir,dst=/output" \
    -- "$immutable_image" \
    --config /input/benchmark-config.json \
    --compatibility /input/compatibility.json \
    --expected-container-image "$image_identity" \
    --output /output/benchmark-result.json
