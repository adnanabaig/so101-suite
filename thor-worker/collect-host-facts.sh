#!/usr/bin/env bash
set -euo pipefail

# Inventory only non-sensitive host/runtime facts. In particular, do not print
# the environment, network configuration, mounted filesystems, USB tree, or
# anything under /dev.

command_value() {
    local label="$1"
    local output
    shift
    printf '%s=' "$label"
    if command -v "$1" >/dev/null 2>&1 && output="$("$@" 2>/dev/null)" && [ -n "$output" ]; then
        printf '%s\n' "${output%%$'\n'*}"
    else
        printf 'unavailable\n'
    fi
}

command_last_value() {
    local label="$1"
    local output
    shift
    printf '%s=' "$label"
    if command -v "$1" >/dev/null 2>&1 && output="$("$@" 2>/dev/null)" && [ -n "$output" ]; then
        printf '%s\n' "${output##*$'\n'}"
    else
        printf 'unavailable\n'
    fi
}

printf 'schema_version=1\n'
printf 'captured_utc='
date -u '+%Y-%m-%dT%H:%M:%SZ'
printf 'architecture='; uname -m
printf 'kernel='; uname -sr

if [ -r /etc/os-release ]; then
    # Only the public OS identity fields; do not source arbitrary shell text.
    awk -F= '/^(ID|VERSION_ID|PRETTY_NAME)=/{print "os_" tolower($1) "=" $2}' /etc/os-release
else
    printf 'os=unavailable\n'
fi

if [ -r /etc/nv_tegra_release ]; then
    printf 'l4t_release='; head -n 1 /etc/nv_tegra_release
else
    printf 'l4t_release=unavailable\n'
fi

command_value python python3 --version
command_value jetpack dpkg-query -W "-f=\${Version}\n" nvidia-jetpack
command_last_value cuda nvcc --version
command_value tensorrt dpkg-query -W "-f=\${Version}\n" tensorrt
command_value cudnn dpkg-query -W "-f=\${Version}\n" libcudnn9

printf 'memory_kib='
awk '/^MemTotal:/{print $2; found=1} END{if(!found) print "unavailable"}' /proc/meminfo
if [ -r /sys/devices/virtual/thermal/thermal_zone0/temp ]; then
    printf 'thermal_zone0_millic='; head -n 1 /sys/devices/virtual/thermal/thermal_zone0/temp
else
    printf 'thermal_zone0_millic=unavailable\n'
fi
