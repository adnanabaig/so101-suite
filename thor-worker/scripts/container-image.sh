#!/usr/bin/env bash

# Resolve a user-friendly local image name to the immutable image id that Docker
# will actually run, and prove it matches the identity recorded as evidence.
resolve_pinned_image() {
    local image="$1"
    local identity="$2"
    local digest inspection inspection_format repo_output
    local actual_id actual_digest repo_digest found
    inspection_format='{{println .Id}}{{range .RepoDigests}}{{println .}}{{end}}'

    case "$identity" in
        ?*@sha256:*) ;;
        *) printf 'error: container identity must be digest-pinned\n' >&2; return 2 ;;
    esac
    digest="${identity##*@sha256:}"
    if [ "${#digest}" -ne 64 ]; then
        printf 'error: container identity has an invalid digest\n' >&2
        return 2
    fi
    case "$digest" in
        *[!0-9a-f]*) printf 'error: container identity has an invalid digest\n' >&2; return 2 ;;
    esac
    if [ "$digest" = "0000000000000000000000000000000000000000000000000000000000000000" ]; then
        printf 'error: container identity may not use the placeholder digest\n' >&2
        return 2
    fi

    if ! inspection="$(docker image inspect --format "$inspection_format" -- "$image")"; then
        printf 'error: cannot inspect container image %s\n' "$image" >&2
        return 2
    fi
    actual_id="${inspection%%$'\n'*}"
    if [[ "$inspection" == *$'\n'* ]]; then
        repo_output="${inspection#*$'\n'}"
    else
        repo_output=""
    fi
    actual_digest="${actual_id#sha256:}"
    if [ "$digest" != "$actual_digest" ]; then
        found=0
        while IFS= read -r repo_digest; do
            [ -n "$repo_digest" ] || continue
            case "$repo_digest" in
                ?*@sha256:[0-9a-f]*) ;;
                *) continue ;;
            esac
            repo_digest="${repo_digest##*@sha256:}"
            if [ "${#repo_digest}" -eq 64 ] && [ "$repo_digest" = "$digest" ]; then
                found=1
                break
            fi
        done <<< "$repo_output"
        if [ "$found" -ne 1 ]; then
            printf 'error: local image does not match recorded container identity\n' >&2
            return 2
        fi
    fi
    case "$actual_id" in
        sha256:[0-9a-f]*) ;;
        *) printf 'error: Docker returned an invalid immutable image id\n' >&2; return 2 ;;
    esac
    actual_digest="${actual_id#sha256:}"
    if [ "${#actual_digest}" -ne 64 ]; then
        printf 'error: Docker returned an invalid immutable image id\n' >&2
        return 2
    fi
    case "$actual_digest" in
        *[!0-9a-f]*)
            printf 'error: Docker returned an invalid immutable image id\n' >&2
            return 2
            ;;
    esac
    printf '%s\n' "$actual_id"
}
