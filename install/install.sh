#!/usr/bin/env bash
# Install or upgrade the SO-101 suite on Linux (first target: Ubuntu 24.04 on
# an NVIDIA laptop) or macOS. Windows: install.ps1.
#
#   install/install.sh                  detect the machine and install
#   install/install.sh --dry-run        print what it would do, change nothing
#   install/install.sh --profile cpu    force the CPU build of PyTorch
#   install/install.sh --realsense      also support Intel RealSense cameras
#
# It creates a virtual environment under --prefix, installs the PyTorch build
# for this machine, then the suite with its extras, adds an app-menu launcher,
# and runs `so101 doctor` last. Doctor's verdict is advice, not a failure: right
# after a first install the checks that need a fresh login (the dialout group,
# ~/.local/bin on PATH) pass only after you log out and back in. A suite that
# doesn't import or run does fail the install. Re-run it to upgrade; a re-run
# replaces a PyTorch build that isn't this profile's (the CPU build after a
# driver update, say). It never deletes the data folder
# (datasets, checkpoints, .env). It adds settings to the data folder's .env
# (~/.local/share/so101/.env on Linux): the absolute data, datasets and outputs
# folders, so `so101` finds the same settings and data from any folder, and with
# the serial numbers below the arm ports. It shows each edit, and replaces a
# value already there only if you answer y. On the first run doctor creates the
# empty subfolders and audit database, as the first launch would.
#
# Every step that needs root (packages, groups, udev rules and the shared
# motor-device state namespace) is printed first and run with sudo only if you
# answer y in a terminal.
#
# PyTorch: torch 2.11.0 and torchvision 0.26.0 from one index, then torchcodec
# 0.11.1 from PyPI. --profile picks the build:
#   auto   Linux: from the NVIDIA driver version (nvidia-smi):
#            580 or newer   cu130, the PyPI build (CUDA 13.0)
#            570 to 579     cu128 from download.pytorch.org
#            525 to 569     cu126 from download.pytorch.org
#            no driver      cpu from download.pytorch.org
#          macOS: mps.
#   cuda   the NVIDIA build for this driver, as above; fails without a driver
#   cu130, cu128, cu126, cpu   that build, whatever the driver
#   mps    the PyPI build for Apple Silicon Macs
#
# Python: 3.12 or 3.13. On Linux the system python3.12 (Ubuntu 24.04's), else
# python3.13; it never installs another Python. On macOS uv, if installed,
# fetches 3.12.
# Free-threaded builds (3.13t) are refused: torchcodec, PyAV, MuJoCo and
# OpenCV have no wheels for them.
#
# --realsense adds the suite's realsense extra (pyrealsense2; on macOS
# pyrealsense2-macosx). Its Linux wheels are for x86-64 only. On Linux it also
# offers Intel's udev rules, which the camera needs to open without root.
#
# --follower-serial and --leader-serial (the arm adapters' USB serial numbers,
# which `python -m so101.platform_info` lists) add udev rules on Linux that name
# the adapters /dev/so101-follower and /dev/so101-leader and keep ModemManager
# off them, and set SO101_FOLLOWER_PORT and SO101_LEADER_PORT to those names
# in the data folder's .env (also the suite's defaults on Linux).
#
# Needs: bash 3.2+, internet access. Behind a TLS-inspecting proxy, export
# PIP_CERT=<CA bundle> (pip) or UV_SYSTEM_CERTS=1 (uv) first.

set -euo pipefail

# Versions LeRobot 0.6.1 accepts (torch>=2.7,<2.12; torchvision>=0.22,<0.27;
# torchcodec>=0.3,<0.12). torchcodec declares no torch requirement, so it is
# pinned here to the release built for this torch.
TORCH_VERSION="2.11.0"
TORCHVISION_VERSION="0.26.0"
TORCHCODEC_VERSION="0.11.1"
TORCH_INDEX="https://download.pytorch.org/whl"
PYPI_INDEX="https://pypi.org/simple"
# NVIDIA drivers (Linux numbers). The PyPI torch 2.11.0 wheel for Linux pins
# CUDA 13.0.2 (cuda-toolkit==13.0.2 in its metadata); the download.pytorch.org
# wheels pin CUDA 12.8.1 and 12.6.3. NVIDIA's CUDA release notes list these
# drivers for those updates. An older driver of the same major CUDA family
# (R580 for CUDA 13, R525+ for CUDA 12) runs them in minor-version
# compatibility mode, which only rules out PTX JIT; the wheels ship native code
# for every GPU the suite targets.
MIN_DRIVER_CU130_PICK="580"
MIN_DRIVER_CU130="580.95.05"
MIN_DRIVER_CU128_PICK="570"
MIN_DRIVER_CU128="570.124.06"
MIN_DRIVER_CU126="560.35.05"
MIN_DRIVER_CUDA12="525"
# System packages the Linux stack uses (Debian/Ubuntu names): FFmpeg for
# torchcodec's video decoding, libusb for RealSense, git for LeRobot's Hub
# tools, and EGL/GL for MuJoCo's offscreen rendering. libusb-1.0-0 kept its name
# through the 64-bit time_t (t64) renames: Ubuntu 24.04 ships libusb-1.0-0
# 2:1.0.27-1 and has no libusb-1.0-0t64, nor do Debian 12 and 13 (the
# packages.ubuntu.com and packages.debian.org indexes, checked 2026-10).
APT_PACKAGES="ffmpeg libusb-1.0-0 git libegl1 libgl1"
# udev rules: the arm adapters' stable names, and where the installer puts
# Intel's RealSense rules or Intel's librealsense2-udev-rules package does
# (same paths as so101.platform_info).
ARM_RULES="/etc/udev/rules.d/99-so101-arms.rules"
REALSENSE_RULES="/etc/udev/rules.d/99-realsense-libusb.rules"
REALSENSE_RULES_SYSTEM="/lib/udev/rules.d/60-librealsense2-udev-rules.rules /usr/lib/udev/rules.d/60-librealsense2-udev-rules.rules"
# The arm adapters' udev names (so101.platform_info.ARM_LINKS), and the exit
# status of `so101.platform_info --set-env` when the .env kept another value.
FOLLOWER_LINK="/dev/so101-follower"
LEADER_LINK="/dev/so101-leader"
LINUX_DEVICE_STATE_ROOT="/var/lib/so101/device-ownership-v2"
MACOS_DEVICE_STATE_ROOT="/Library/Application Support/SO101/device-ownership-v2"
SET_ENV_KEPT=3

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="auto"
PREFIX=""
PYTHON=""
EXTRAS="robot,sim,train"
WHEEL=""
FRESH=0
LAUNCHER=1
ALLOW_NO_UI=0
DRY_RUN=0
REALSENSE=0
FOLLOWER_SERIAL=""
LEADER_SERIAL=""

usage() {
    # The header comment, up to the first empty line.
    sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'EOF'
Options:
  --profile P            auto (default), cuda, cu130, cu128, cu126, cpu or mps
  --prefix DIR           where the app goes (default: ~/.local/opt/so101-suite)
  --python PY            the Python 3.12 or 3.13 to use (default: python3.12,
                         else python3.13; on macOS uv first)
  --extras LIST          suite extras (default: robot,sim,train)
  --realsense            add Intel RealSense camera support (same as adding
                         realsense to --extras)
  --follower-serial S    Linux: the follower arm adapter's USB serial number
  --leader-serial S      Linux: the leader arm adapter's USB serial number
  --wheel FILE           install this wheel (default: the newest
                         dist/so101-<version>-*.whl of the source folder's
                         version, refused if the source changed after it was
                         built; else the source folder). Every workstation
                         input must match this source and carry a verified UI
  --fresh                delete and recreate the app's virtual environment (not the data)
  --no-launcher          don't add the app-menu launcher
  --allow-no-ui          allow an API/CLI-only package without the operator UI;
                         requires --no-launcher
  --dry-run              print the commands instead of running them
  -h, --help             show this help
EOF
}

say() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

# Print a command; run it unless this is a dry run.
run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [ "$DRY_RUN" -eq 0 ]; then
        "$@"
    fi
}

# confirm QUESTION: true only if this is not a dry run, stdin and stdout are a
# terminal, and the user answers y. The installer never runs sudo otherwise.
confirm() {
    local answer=""
    if [ "$DRY_RUN" -eq 1 ] || ! [ -t 0 ] || ! [ -t 1 ]; then
        say "the installer doesn't run sudo on its own: run the commands above in a terminal when you're ready"
        return 1
    fi
    printf '%s [y/N] ' "$1"
    read -r answer || answer=""
    case "$answer" in
        y | Y | yes | Yes | YES) return 0 ;;
        *)
            say "skipped: run the commands above when you're ready"
            return 1
            ;;
    esac
}

# ask QUESTION: like confirm, for changes that need no sudo (the settings file).
ask() {
    local answer=""
    if [ "$DRY_RUN" -eq 1 ] || ! [ -t 0 ] || ! [ -t 1 ]; then
        say "not asked (no terminal): nothing was replaced"
        return 1
    fi
    printf '%s [y/N] ' "$1"
    read -r answer || answer=""
    case "$answer" in
        y | Y | yes | Yes | YES) return 0 ;;
        *) return 1 ;;
    esac
}

# version_ge A B: true if dotted version A >= B (numeric fields).
version_ge() {
    local IFS=.
    local -a a b
    read -r -a a <<<"$1"
    read -r -a b <<<"$2"
    local i x y
    for i in 0 1 2 3; do
        x="${a[$i]:-0}"
        y="${b[$i]:-0}"
        x="${x%%[!0-9]*}"
        y="${y%%[!0-9]*}"
        x=$((10#${x:-0}))
        y=$((10#${y:-0}))
        if [ "$x" -gt "$y" ]; then return 0; fi
        if [ "$x" -lt "$y" ]; then return 1; fi
    done
    return 0
}

while [ $# -gt 0 ]; do
    case "$1" in
        --profile) PROFILE="${2:?--profile needs a value}"; shift 2 ;;
        --profile=*) PROFILE="${1#*=}"; shift ;;
        --prefix) PREFIX="${2:?--prefix needs a value}"; shift 2 ;;
        --prefix=*) PREFIX="${1#*=}"; shift ;;
        --python) PYTHON="${2:?--python needs a value}"; shift 2 ;;
        --python=*) PYTHON="${1#*=}"; shift ;;
        --extras) EXTRAS="${2-}"; [ $# -ge 2 ] || die "--extras needs a value"; shift 2 ;;
        --extras=*) EXTRAS="${1#*=}"; shift ;;
        --wheel) WHEEL="${2:?--wheel needs a value}"; shift 2 ;;
        --wheel=*) WHEEL="${1#*=}"; shift ;;
        --follower-serial) FOLLOWER_SERIAL="${2:?--follower-serial needs a value}"; shift 2 ;;
        --follower-serial=*) FOLLOWER_SERIAL="${1#*=}"; shift ;;
        --leader-serial) LEADER_SERIAL="${2:?--leader-serial needs a value}"; shift 2 ;;
        --leader-serial=*) LEADER_SERIAL="${1#*=}"; shift ;;
        --fresh) FRESH=1; shift ;;
        --realsense) REALSENSE=1; shift ;;
        --no-launcher) LAUNCHER=0; shift ;;
        --allow-no-ui) ALLOW_NO_UI=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h | --help) usage; exit 0 ;;
        *) die "unknown option: $1 (see --help)" ;;
    esac
done

if [ "$ALLOW_NO_UI" -eq 1 ] && [ "$LAUNCHER" -ne 0 ]; then
    die "--allow-no-ui requires --no-launcher: a workstation launcher needs the operator web UI"
fi

# The app, launcher and data belong to the operator. A live run of the whole
# script as root would put them in root's home (or leave root-owned files in
# the invoking user's home under sudo). Individual system changes ask for
# sudo at the point where they are made. A root --dry-run is safe for audits.
if [ "$DRY_RUN" -eq 0 ] && [ "$(id -u)" = "0" ]; then
    die "run install.sh as your normal user, without sudo; it asks for sudo only for the individual system changes (packages, groups and udev rules)"
fi

# --realsense and a realsense entry in --extras mean the same thing.
EXTRAS="${EXTRAS// /}"
case ",$EXTRAS," in
    *,realsense,*) REALSENSE=1 ;;
    *) if [ "$REALSENSE" -eq 1 ]; then EXTRAS="${EXTRAS:+$EXTRAS,}realsense"; fi ;;
esac

# Serial numbers go into a udev rules file: allow only plain characters.
for serial in "$FOLLOWER_SERIAL" "$LEADER_SERIAL"; do
    case "$serial" in
        *[!A-Za-z0-9._-]*) die "a USB serial number has only letters, digits, '.', '_' and '-': got '$serial'" ;;
    esac
done
if [ -n "$FOLLOWER_SERIAL$LEADER_SERIAL" ]; then
    if [ -z "$FOLLOWER_SERIAL" ] || [ -z "$LEADER_SERIAL" ]; then
        die "give both --follower-serial and --leader-serial"
    fi
    [ "$FOLLOWER_SERIAL" != "$LEADER_SERIAL" ] || die "--follower-serial and --leader-serial are the same adapter"
fi

# --- the machine ---------------------------------------------------------------

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
    Linux | Darwin) ;;
    *) die "this script is for Linux and macOS; on Windows run install\\install.ps1 in PowerShell" ;;
esac

if [ "$OS" = "Darwin" ]; then
    if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = "1" ]; then
        die "this terminal runs under Rosetta (x86_64). Open a native arm64 terminal and run the installer again."
    fi
    if [ "$ARCH" != "arm64" ]; then
        die "Intel Macs are not supported: PyTorch 2.7+ (LeRobot's minimum) and MuJoCo 3.14 have no macOS x86_64 wheels."
    fi
    [ -z "$FOLLOWER_SERIAL" ] || die "--follower-serial and --leader-serial are for Linux (udev)"
fi

# The Python versions this install can use, standard builds only: 3.12 or 3.13,
# the suite's requires-python (>=3.12,<3.14; every pinned package has wheels for
# both). With --realsense before macOS 15 only pyrealsense2-macosx 2.54.2
# installs, which stops at 3.12. pyrealsense2 2.56's Linux wheels are x86-64 only.
PY_MAX_MINOR=13
PY_WHY=""
if [ "$OS" = "Linux" ]; then
    if [ "$REALSENSE" -eq 1 ]; then
        case "$ARCH" in
            x86_64 | amd64) ;;
            *) die "--realsense: pyrealsense2 2.56 has no Linux $ARCH wheel for Python 3.12 or 3.13 (its Linux wheels are for x86-64). Build librealsense with its Python bindings from source, then install without --realsense." ;;
        esac
    fi
elif [ "$REALSENSE" -eq 1 ]; then
    macos_major="$(sw_vers -productVersion 2>/dev/null | cut -d. -f1)" || macos_major=""
    case "$macos_major" in
        '' | *[!0-9]*) ;; # unknown: leave it to pip
        *) if [ "$macos_major" -lt 15 ]; then
            PY_MAX_MINOR=12
            PY_WHY=" (--realsense before macOS 15: pyrealsense2-macosx 2.54.2 stops at Python 3.12)"
        fi ;;
    esac
fi
PY_GIL="bool(sysconfig.get_config_var('Py_GIL_DISABLED'))"
PY_CHECK="import sys, sysconfig; raise SystemExit(not (3, 12) <= sys.version_info[:2] <= (3, $PY_MAX_MINOR) or $PY_GIL)"
if [ "$PY_MAX_MINOR" -eq 12 ]; then PY_WANT="Python 3.12"; else PY_WANT="Python 3.12 or 3.$PY_MAX_MINOR"; fi
ENV_BIN="/usr/bin/env"
[ -x "$ENV_BIN" ] || ENV_BIN="$(command -v env)"
PY_PROBE_ENV=("$ENV_BIN" -i "PATH=$PATH" "HOME=/dev/null" "PYTHONNOUSERSITE=1" "PYTHONSAFEPATH=1")
for name in LANG LANGUAGE LC_ALL LC_CTYPE; do
    value="${!name-}"
    if [ -n "$value" ]; then PY_PROBE_ENV+=("$name=$value"); fi
done
probe_python() {
    local python="$1"
    shift
    "${PY_PROBE_ENV[@]}" "$python" -I -S "$@"
}
# py_ok PYTHON: true if PYTHON runs and is a version this install can use.
py_ok() { probe_python "$1" -c "$PY_CHECK" 2>/dev/null; }
# py_version PYTHON: like 3.12.3, with a t for a free-threaded build.
py_version() {
    probe_python "$1" -c "import platform, sysconfig; print(platform.python_version() + ('t' if sysconfig.get_config_var('Py_GIL_DISABLED') else ''))" 2>/dev/null || echo "unknown"
}

# nvidia_driver: the driver version if nvidia-smi answers with one, else empty.
# Before the kernel module loads (Secure Boot's MOK enrollment, a laptop in
# integrated-only mode) nvidia-smi exits non-zero and prints its error on
# stdout, so only a successful answer that looks like a version counts.
nvidia_driver() {
    local out
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    out="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)" || return 0
    out="${out%%$'\n'*}"
    out="${out//[[:space:]]/}"
    case "$out" in
        [0-9]*) ;;
        *) return 0 ;;
    esac
    case "$out" in
        *[!0-9.]*) return 0 ;;
    esac
    printf '%s\n' "$out"
}

# cuda_profile_for_driver DRIVER: cu130, cu128, cu126, or empty if no CUDA build runs.
cuda_profile_for_driver() {
    local driver="$1"
    if [ -z "$driver" ]; then
        echo ""
    elif version_ge "$driver" "$MIN_DRIVER_CU130_PICK"; then
        echo "cu130"
    elif version_ge "$driver" "$MIN_DRIVER_CU128_PICK"; then
        echo "cu128"
    elif version_ge "$driver" "$MIN_DRIVER_CUDA12"; then
        echo "cu126"
    else
        echo ""
    fi
}

# profile_source: where $PROFILE's torch comes from, for the log.
profile_source() {
    case "$PROFILE" in
        cu130) echo "torch from PyPI, built for CUDA 13.0 (driver 580 or newer)" ;;
        cu128) echo "torch from $TORCH_INDEX/cu128, built for CUDA 12.8 (driver 570 to 579)" ;;
        cu126) echo "torch from $TORCH_INDEX/cu126, built for CUDA 12.6 (driver 525 to 569)" ;;
        cpu) echo "torch from $TORCH_INDEX/cpu, CPU only" ;;
        mps) echo "torch from PyPI, uses the Apple GPU (MPS)" ;;
    esac
}

# warn_old_driver: warn if the NVIDIA driver is older than NVIDIA lists for $PROFILE.
warn_old_driver() {
    local need family
    case "$PROFILE" in
        cu130) need="$MIN_DRIVER_CU130" family="$MIN_DRIVER_CU130_PICK" ;;
        cu128) need="$MIN_DRIVER_CU128" family="$MIN_DRIVER_CUDA12" ;;
        cu126) need="$MIN_DRIVER_CU126" family="$MIN_DRIVER_CUDA12" ;;
        *) return 0 ;;
    esac
    [ -n "$DRIVER" ] || return 0
    if version_ge "$DRIVER" "$need"; then
        return 0
    elif version_ge "$DRIVER" "$family"; then
        warn "NVIDIA driver $DRIVER is older than $need, which NVIDIA lists for the $PROFILE build. It runs in CUDA minor-version compatibility mode; if training reports CUDA errors, update the driver and re-run."
    else
        warn "NVIDIA driver $DRIVER is too old for the $PROFILE build (it needs $family or newer); torch will use the CPU until you update the driver or pick another --profile."
    fi
}

# no_driver_note: why there is no driver version, if nvidia-smi is there at all.
no_driver_note() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        warn "nvidia-smi is installed but not answering, so the NVIDIA kernel module isn't loaded: with Secure Boot on, finish the MOK enrollment at the next boot (or run sudo ubuntu-drivers install again); on a laptop in integrated-only GPU mode, switch to hybrid. Then re-run with --profile cuda."
    fi
}

DRIVER=""
case "$PROFILE" in
    auto)
        if [ "$OS" = "Darwin" ]; then
            PROFILE="mps"
        else
            DRIVER="$(nvidia_driver)"
            PROFILE="$(cuda_profile_for_driver "$DRIVER")"
            if [ -n "$PROFILE" ]; then
                say "NVIDIA driver $DRIVER: the $PROFILE build ($(profile_source))"
                warn_old_driver
            elif [ -n "$DRIVER" ]; then
                warn "NVIDIA driver $DRIVER is too old for any CUDA build (they need $MIN_DRIVER_CUDA12 or newer). Installing the CPU build; update the driver (sudo ubuntu-drivers install) and re-run with --profile cuda to use the GPU."
                PROFILE="cpu"
            else
                no_driver_note
                PROFILE="cpu"
                say "no NVIDIA driver answered: the cpu build ($(profile_source))"
            fi
        fi
        ;;
    cuda)
        [ "$OS" = "Linux" ] || die "--profile cuda needs Linux or Windows with an NVIDIA GPU"
        DRIVER="$(nvidia_driver)"
        if [ -z "$DRIVER" ]; then
            no_driver_note
            die "--profile cuda: nvidia-smi doesn't answer; install or load the NVIDIA driver first (or pass --profile cu130, cu128 or cu126 to install anyway)"
        fi
        PROFILE="$(cuda_profile_for_driver "$DRIVER")"
        [ -n "$PROFILE" ] || die "NVIDIA driver $DRIVER is too old: CUDA builds need $MIN_DRIVER_CUDA12 or newer"
        say "NVIDIA driver $DRIVER: the $PROFILE build ($(profile_source))"
        warn_old_driver
        ;;
    cu130 | cu128 | cu126 | cpu)
        [ "$OS" = "Linux" ] || die "--profile $PROFILE is for Linux and Windows; on a Mac use mps"
        DRIVER="$(nvidia_driver)"
        say "--profile $PROFILE: $(profile_source)"
        warn_old_driver
        ;;
    mps)
        [ "$OS" = "Darwin" ] || die "--profile mps is for Apple Silicon Macs"
        ;;
    *) die "unknown profile: $PROFILE (auto, cuda, cu130, cu128, cu126, cpu or mps)" ;;
esac

# One index per resolve (--index-url, never --extra-index-url: the same file
# name can differ between indexes). torchcodec always comes from PyPI.
# TORCH_LOCAL is the local version label the profile's builds carry: none on
# PyPI's, cu128, cu126 or cpu (as in 2.11.0+cpu) on download.pytorch.org's.
case "$PROFILE" in
    mps | cu130)
        TORCH_SPECS=("torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION")
        TORCH_ARGS=(--index-url "$PYPI_INDEX")
        TORCH_LOCAL=""
        ;;
    *)
        TORCH_SPECS=("torch==$TORCH_VERSION+$PROFILE" "torchvision==$TORCHVISION_VERSION+$PROFILE")
        TORCH_ARGS=(--index-url "$TORCH_INDEX/$PROFILE")
        TORCH_LOCAL="$PROFILE"
        ;;
esac
CODEC_SPEC="torchcodec==$TORCHCODEC_VERSION"

# --- where things go -------------------------------------------------------------

PREFIX="${PREFIX:-$HOME/.local/opt/so101-suite}"
case "$PREFIX" in
    /*) ;;
    *) PREFIX="$PWD/$PREFIX" ;;
esac
VENV="$PREFIX/venv"
VPY="$VENV/bin/python"

# source_version: the suite version this source folder builds (pyproject.toml's
# [project] version); empty without a pyproject.toml.
source_version() {
    local root="${1:-$ROOT}"
    [ -f "$root/pyproject.toml" ] || return 0
    awk '
        /^\[/ { project = ($0 ~ /^\[project\][ \t]*$/); next }
        project && /^version[ \t]*=/ {
            if (match($0, /"[^"]*"/)) print substr($0, RSTART + 1, RLENGTH - 2)
            exit
        }' "$root/pyproject.toml"
}

source_git_head() {
    [ -e "$ROOT/.git" ] || return 0
    command -v git >/dev/null 2>&1 || return 0
    git -C "$ROOT" rev-parse HEAD 2>/dev/null || true
}

# wheel_version FILE: the version in a suite wheel's file name
# (so101-<version>-<tags>.whl); empty if FILE isn't named like one.
wheel_version() {
    local name="${1##*/}" version
    case "$name" in
        so101-*-*.whl) ;;
        *) return 0 ;;
    esac
    version="${name#so101-}"
    printf '%s\n' "${version%%-*}"
}

# newer_source FILE: a file under src/so101 or frontend/src changed after FILE
# (bytecode caches and Finder's .DS_Store don't count); empty if none.
newer_source() {
    local dir found
    for dir in "$ROOT/src/so101" "$ROOT/frontend/src"; do
        [ -d "$dir" ] || continue
        found="$(find "$dir" -name __pycache__ -prune -o -type f ! -name '*.pyc' ! -name .DS_Store \
            -newer "$1" -print 2>/dev/null | head -n 1)" || true
        if [ -n "$found" ]; then
            printf '%s\n' "$found"
            return 0
        fi
    done
}

SOURCE_VERSION="$(source_version)"
SOURCE_GIT_HEAD="$(source_git_head)"
if [ -z "$WHEEL" ]; then
    # The newest wheel in dist/ (by modification time) of the version this
    # source folder builds, unless the source changed after it was built: an
    # old wheel left in dist/ would otherwise install old code. No wheel: the
    # source folder itself. An explicit --wheel chooses the file but still gets
    # the exact source/dependency/UI verification below.
    for candidate in "$ROOT"/dist/so101-*.whl; do
        [ -e "$candidate" ] || continue
        if [ -n "$SOURCE_VERSION" ] && [ "$(wheel_version "$candidate")" != "$SOURCE_VERSION" ]; then
            warn "ignoring $candidate: this source folder is version $SOURCE_VERSION"
            continue
        fi
        if [ -z "$WHEEL" ] || [ "$candidate" -nt "$WHEEL" ]; then
            WHEEL="$candidate"
        fi
    done
    if [ -n "$WHEEL" ]; then
        changed="$(newer_source "$WHEEL")"
        if [ -n "$changed" ]; then
            die "$WHEEL is older than the source folder ($changed changed after it was built), so it may lack the latest changes. Rebuild it with the web UI: make frontend-deps build-ui wheel (needs Node.js). Or pass the wheel you mean with --wheel FILE."
        fi
    fi
fi
WHEEL_VERSION=""
if [ -n "$WHEEL" ]; then
    [ -f "$WHEEL" ] || die "no such wheel: $WHEEL"
    WHEEL_VERSION="$(wheel_version "$WHEEL")"
    [ -n "$WHEEL_VERSION" ] || die "$WHEEL isn't named like a suite wheel (so101-<version>-py3-none-any.whl)"
    [ "${WHEEL##*/}" = "so101-$WHEEL_VERSION-py3-none-any.whl" ] \
        || die "$WHEEL isn't the supported pure-Python suite wheel name (so101-<version>-py3-none-any.whl)"
    TARGET="$WHEEL"
    SOURCE_DESC="wheel $WHEEL"
else
    [ -f "$ROOT/pyproject.toml" ] || die "no dist/so101-*.whl and no source folder at $ROOT"
    TARGET="$ROOT"
    SOURCE_DESC="source folder $ROOT"
fi

# Appending ``[extra,...]`` makes TARGET a pip requirement string. Characters
# with requirement/URL meaning would make a legal filesystem path parse as a
# marker, fragment or another requirement. Refuse before creating a venv or
# changing the machine instead of letting pip install something else.
case "$TARGET" in
    *';'* | *'['* | *']'* | *'#'* | *'?'* | *$'\n'* | *$'\r'*)
        die "the install source path contains a character pip can misparse (; [ ] # ? or a line break); move it to a plain path and re-run"
        ;;
esac

# Run the stdlib-only verifier through -c so the fake/bootstrap Python contract
# is the same on Linux, macOS and the hermetic installer tests. It checks that a
# wheel is byte-for-byte this source package and that its UI is current and
# intact; source installs get the same UI-stamp check. Only the explicit
# --no-launcher --allow-no-ui development mode may omit the UI.
INPUT_VERIFIER="$ROOT/install/verify_input.py"
VERIFY_ROOT="$ROOT"
INPUT_CHECK='import runpy, sys; sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")'
WHEEL_COPY='import os, shutil, sys
source, destination = sys.argv[1:]
descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
with open(source, "rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
    shutil.copyfileobj(incoming, outgoing, 1 << 20)
    outgoing.flush()
    os.fsync(outgoing.fileno())'
WHEEL_SHA256='import hashlib, sys
digest = hashlib.sha256()
with open(sys.argv[1], "rb") as stream:
    for block in iter(lambda: stream.read(1 << 20), b""):
        digest.update(block)
print(digest.hexdigest())'
SOURCE_TREE_SHA256='# SO101_SOURCE_TREE_SHA256
import hashlib, sys
from pathlib import Path
root = Path(sys.argv[1])
tree = hashlib.sha256()
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    relative = path.relative_to(root).as_posix()
    content = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            content.update(block)
    tree.update(f"{relative}\0{content.hexdigest()}\n".encode())
print(tree.hexdigest())'
FINAL_WHEEL_SHA256="$WHEEL_SHA256
# SO101_INPUT_FINAL_HASH"
FINAL_SOURCE_TREE_SHA256="$SOURCE_TREE_SHA256
# SO101_INPUT_FINAL_HASH"
SOURCE_COPY='import os, shutil, sys
from pathlib import Path
source, destination = map(Path, sys.argv[1:])
destination.mkdir(mode=0o700)
paths = (
    "pyproject.toml", "hatch_build.py", "README.md", "LICENSE",
    "install/verify_input.py", "src/so101", "frontend/src", "frontend/public",
    "frontend/dist", "frontend/index.html", "frontend/package.json",
    "frontend/package-lock.json", "frontend/vite.config.ts", "frontend/build-stamp.ts",
    "frontend/tsconfig.json", "frontend/tsconfig.node.json",
)
for relative in paths:
    incoming = source / relative
    outgoing = destination / relative
    if incoming.is_dir():
        outgoing.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(incoming, outgoing, symlinks=False)
    elif incoming.is_file():
        outgoing.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(incoming, outgoing)
for root, directories, files in os.walk(destination, topdown=False):
    for name in files:
        os.chmod(Path(root) / name, 0o400)
    for name in directories:
        os.chmod(Path(root) / name, 0o500)
os.chmod(destination, 0o500)'
INPUT_CHECKED=0
PRIVATE_INSTALL_ROOT=""
STAGED_WHEEL=""
STAGED_SOURCE=""
VERIFIED_WHEEL_SHA256=""
VERIFIED_SOURCE_SHA256=""

cleanup_private_install() {
    if [ -n "$PRIVATE_INSTALL_ROOT" ] && [ -d "$PRIVATE_INSTALL_ROOT" ]; then
        chmod -R u+w "$PRIVATE_INSTALL_ROOT" 2>/dev/null || true
        rm -rf "$PRIVATE_INSTALL_ROOT" || true
    fi
}

ensure_private_install_root() {
    if [ -n "$PRIVATE_INSTALL_ROOT" ]; then return 0; fi
    PRIVATE_INSTALL_ROOT="$(mktemp -d "/tmp/so101-install-private.XXXXXX")" \
        || die "couldn't create a private installation directory"
    chmod 700 "$PRIVATE_INSTALL_ROOT" || die "couldn't make the installation directory private"
    trap cleanup_private_install EXIT
}

INSTALL_ENV=()
prepare_install_environment() {
    local name value private_home private_config private_cache private_data private_state private_tmp
    if [ "$DRY_RUN" -eq 0 ]; then
        ensure_private_install_root
        private_home="$PRIVATE_INSTALL_ROOT/package-home"
        private_config="$PRIVATE_INSTALL_ROOT/package-config"
        private_cache="$PRIVATE_INSTALL_ROOT/package-cache"
        private_data="$PRIVATE_INSTALL_ROOT/package-data"
        private_state="$PRIVATE_INSTALL_ROOT/package-state"
        private_tmp="$PRIVATE_INSTALL_ROOT/package-tmp"
        mkdir -p "$private_home" "$private_config" "$private_cache" \
            "$private_data" "$private_state" "$private_tmp"
        chmod 700 "$private_home" "$private_config" "$private_cache" \
            "$private_data" "$private_state" "$private_tmp"
    else
        # A dry run must not create isolation state. The verifier needs none of
        # these locations; /dev/null prevents it from consulting user files.
        private_home="/dev/null"
        private_config="/dev/null"
        private_cache="/dev/null"
        private_data="/dev/null"
        private_state="/dev/null"
        private_tmp="/tmp"
    fi
    INSTALL_ENV=(
        "$ENV_BIN" -i
        "PATH=$PATH"
        "SHELL=/bin/sh"
        "HOME=$private_home"
        "XDG_CONFIG_HOME=$private_config"
        "XDG_CACHE_HOME=$private_cache"
        "XDG_DATA_HOME=$private_data"
        "XDG_STATE_HOME=$private_state"
        "TMPDIR=$private_tmp"
        "TEMP=$private_tmp"
        "TMP=$private_tmp"
        "NETRC=/dev/null"
        "PIP_CONFIG_FILE=/dev/null"
        "PIP_DISABLE_PIP_VERSION_CHECK=1"
        "PIP_KEYRING_PROVIDER=disabled"
        "PIP_NO_INPUT=1"
        "UV_KEYRING_PROVIDER=disabled"
        "UV_NO_CONFIG=1"
        "UV_PYTHON_INSTALL_DIR=$PREFIX/uv-python"
        "PYTHONNOUSERSITE=1"
        "PYTHONSAFEPATH=1"
    )
    for name in LANG LANGUAGE LC_ALL LC_CTYPE PIP_CERT PIP_CLIENT_CERT PIP_PROXY \
        PIP_TRUSTED_HOST \
        UV_SYSTEM_CERTS CURL_CA_BUNDLE REQUESTS_CA_BUNDLE SSL_CERT_FILE SSL_CERT_DIR \
        HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY http_proxy https_proxy all_proxy no_proxy; do
        value="${!name-}"
        if [ -n "$value" ]; then INSTALL_ENV+=("$name=$value"); fi
    done
}

isolated_python() {
    local python="$1"
    shift
    "${INSTALL_ENV[@]}" "$python" -I "$@"
}

isolated_stdlib_python() {
    local python="$1"
    shift
    "${INSTALL_ENV[@]}" "$python" -I -S "$@"
}

run_install_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    if [ "$DRY_RUN" -eq 0 ]; then
        "${INSTALL_ENV[@]}" "$@"
    fi
}

set_suite_spec() {
    if [ -n "$EXTRAS" ]; then
        SUITE_SPEC="${TARGET}[${EXTRAS}]"
    else
        SUITE_SPEC="$TARGET"
    fi
}

stage_install_input() {
    local python="$1"
    if [ "$DRY_RUN" -eq 1 ] || [ -n "$STAGED_WHEEL" ] || [ -n "$STAGED_SOURCE" ]; then
        return 0
    fi
    ensure_private_install_root
    if [ -n "$WHEEL" ]; then
        mkdir "$PRIVATE_INSTALL_ROOT/wheel" || die "couldn't create the private wheel directory"
        chmod 700 "$PRIVATE_INSTALL_ROOT/wheel" || die "couldn't make the wheel directory private"
        STAGED_WHEEL="$PRIVATE_INSTALL_ROOT/wheel/${WHEEL##*/}"
        isolated_stdlib_python "$python" -c "$WHEEL_COPY" "$WHEEL" "$STAGED_WHEEL" \
            || die "couldn't copy $WHEEL into the private verified-wheel directory"
        chmod 400 "$STAGED_WHEEL" || die "couldn't make the private wheel read-only"
        TARGET="$STAGED_WHEEL"
    else
        STAGED_SOURCE="$PRIVATE_INSTALL_ROOT/source"
        isolated_stdlib_python "$python" -c "$SOURCE_COPY" "$ROOT" "$STAGED_SOURCE" \
            || die "couldn't make a private snapshot of source folder $ROOT"
        TARGET="$STAGED_SOURCE"
        VERIFY_ROOT="$STAGED_SOURCE"
        INPUT_VERIFIER="$STAGED_SOURCE/install/verify_input.py"
        SOURCE_VERSION="$(source_version "$STAGED_SOURCE")"
    fi
    set_suite_spec
}

check_input() {
    local checked initial_sha256 final_sha256
    local -a args
    if [ "$INPUT_CHECKED" -eq 1 ]; then
        return 0
    fi
    [ -f "$INPUT_VERIFIER" ] || die "installation verifier is missing: $INPUT_VERIFIER"
    stage_install_input "$1"
    if [ -n "$STAGED_WHEEL" ]; then
        initial_sha256="$(isolated_stdlib_python "$1" -c "$WHEEL_SHA256" "$STAGED_WHEEL")" \
            || die "couldn't hash the private wheel before validation"
    elif [ -n "$STAGED_SOURCE" ]; then
        initial_sha256="$(isolated_stdlib_python "$1" -c "$SOURCE_TREE_SHA256" "$STAGED_SOURCE")" \
            || die "couldn't hash the private source snapshot before validation"
    fi
    args=(--root "$VERIFY_ROOT" --target "$TARGET" --wheel-version "$WHEEL_VERSION" --source-version "$SOURCE_VERSION" --extras "$EXTRAS")
    if [ "$ALLOW_NO_UI" -eq 1 ]; then args+=(--allow-no-ui); fi
    if [ -n "$SOURCE_GIT_HEAD" ]; then args+=(--expected-git-head "$SOURCE_GIT_HEAD"); fi
    if ! checked="$(isolated_stdlib_python "$1" -c "$INPUT_CHECK" "$INPUT_VERIFIER" "${args[@]}" 2>&1)"; then
        die "${checked##*$'\n'}"
    fi
    if [ -n "$STAGED_WHEEL" ]; then
        final_sha256="$(isolated_stdlib_python "$1" -c "$FINAL_WHEEL_SHA256" "$STAGED_WHEEL")" \
            || die "couldn't hash the private verified wheel"
        [ "$final_sha256" = "$initial_sha256" ] \
            || die "the private wheel changed while it was being validated; refusing it"
        VERIFIED_WHEEL_SHA256="$final_sha256"
    elif [ -n "$STAGED_SOURCE" ]; then
        final_sha256="$(isolated_stdlib_python "$1" -c "$FINAL_SOURCE_TREE_SHA256" "$STAGED_SOURCE")" \
            || die "couldn't hash the private verified source snapshot"
        [ "$final_sha256" = "$initial_sha256" ] \
            || die "the private source snapshot changed while it was being validated; refusing it"
        VERIFIED_SOURCE_SHA256="$final_sha256"
    fi
    INPUT_CHECKED=1
    say "input:    $checked"
}

# --extras "" installs the suite alone (mock mode and the web UI only).
set_suite_spec

# uv (macOS only): its own Python and faster installs. On Linux the venv comes
# from the system Python and pip.
USE_UV=0
UV_BIN=""
if [ "$OS" = "Darwin" ] && [ -z "$PYTHON" ] && command -v uv >/dev/null 2>&1; then
    USE_UV=1
    UV_BIN="$(command -v uv)"
fi
prepare_install_environment

say "machine:  $OS $ARCH"
say "profile:  $PROFILE (${TORCH_SPECS[*]}, then $CODEC_SPEC from PyPI)"
say "app:      $PREFIX"
say "suite:    $SOURCE_DESC [$EXTRAS]"
if [ "$USE_UV" -eq 1 ]; then say "using:    uv ($UV_BIN)"; fi
say "python:   $PY_WANT$PY_WHY"
if [ "$REALSENSE" -eq 1 ]; then say "realsense: yes"; fi
if [ "$DRY_RUN" -eq 1 ]; then say "dry run: nothing will be changed"; fi

# Rough free-space check: the CUDA build of PyTorch alone is several GB.
need_gb=3
case "$PROFILE" in cu*) need_gb=8 ;; esac
parent="$PREFIX"
while [ ! -d "$parent" ]; do parent="$(dirname "$parent")"; done
free_kb="$(df -Pk "$parent" 2>/dev/null | awk 'NR == 2 { print $4 }')" || free_kb=""
case "$free_kb" in '' | *[!0-9]*) free_kb="" ;; esac # df failed: skip the check
if [ -n "$free_kb" ] && [ "$free_kb" -lt $((need_gb * 1024 * 1024)) ]; then
    warn "only $((free_kb / 1024 / 1024)) GB free at $parent; the $PROFILE install may need about $need_gb GB"
fi

# --- virtual environment: which Python -----------------------------------------------

# venv_works: the venv's Python runs and is 3.12+. venv_ok: and this install can use it.
venv_works() {
    [ -x "$VPY" ] \
        && probe_python "$VPY" -c 'import sys; raise SystemExit(sys.version_info[:2] < (3, 12))' 2>/dev/null
}
venv_ok() { [ -x "$VPY" ] && py_ok "$VPY"; }

if [ -e "$VENV" ] && [ ! -f "$VENV/pyvenv.cfg" ]; then
    die "$VENV exists but is not a virtual environment; move it away or pick another --prefix"
fi
if [ "$FRESH" -eq 0 ] && [ -e "$VENV" ] && venv_works && ! venv_ok; then
    die "$VENV has Python $(py_version "$VPY"), but this install needs $PY_WANT (a standard build)$PY_WHY. Re-run with --fresh to recreate the app's virtual environment; the data folder is kept."
fi
NEW_VENV=1
if [ "$FRESH" -eq 0 ] && [ -e "$VENV" ] && venv_ok; then
    NEW_VENV=0
fi

VENV_PACKAGE="" # Debian/Ubuntu package with the chosen Python's venv module
if [ "$NEW_VENV" -eq 1 ] && [ "$USE_UV" -eq 0 ]; then
    if [ -z "$PYTHON" ]; then
        for candidate in python3.12 python3.13 python3; do
            if command -v "$candidate" >/dev/null 2>&1 && py_ok "$candidate"; then
                PYTHON="$candidate"
                break
            fi
        done
        if [ -z "$PYTHON" ]; then
            if [ "$OS" = "Linux" ]; then
                die "$PY_WANT not found. On Ubuntu 24.04: sudo apt install python3.12 python3.12-venv, then re-run."
            fi
            die "$PY_WANT not found$PY_WHY. Install Python 3.12 (python.org or Homebrew) or install uv, then re-run."
        fi
    fi
    py_ok "$PYTHON" || die "$PYTHON is Python $(py_version "$PYTHON"), but this install needs $PY_WANT (a standard build, not free-threaded)$PY_WHY"
    if [ "$OS" = "Linux" ]; then
        # A distribution Python (base prefix /usr) keeps venv in a separate package.
        VENV_PACKAGE="$(probe_python "$PYTHON" -c 'import sys; print("python%d.%d-venv" % sys.version_info[:2] if sys.base_prefix == "/usr" else "")')"
    fi
fi

# The wheel, before anything changes, with a Python that's already there: the
# reused venv's, else the one that creates it (uv's: once the venv exists).
if [ "$NEW_VENV" -eq 0 ]; then
    check_input "$VPY"
elif [ -n "$PYTHON" ]; then
    check_input "$PYTHON"
fi

# --- system packages (Linux) ----------------------------------------------------------

# apt_missing PACKAGE...: the packages dpkg doesn't list as installed. dpkg-query
# prints a status for each architecture a package is installed for (libgl1:amd64
# and libgl1:i386 once Steam is installed), so any one that says installed counts.
apt_missing() {
    local pkg status out="" nl='
'
    for pkg in "$@"; do
        # shellcheck disable=SC2016 # ${Status} is dpkg-query's field, not a shell variable
        status="$(dpkg-query -W -f='${Status}\n' "$pkg" 2>/dev/null)" || status=""
        case "$nl$status$nl" in
            *"${nl}install ok installed${nl}"*) ;;
            *) out="${out:+$out }$pkg" ;;
        esac
    done
    printf '%s' "$out"
}

linux_packages() {
    local wanted missing
    # shellcheck disable=SC2086 # word splitting is intended: a package list
    wanted="$(printf '%s ' $VENV_PACKAGE $APT_PACKAGES)"
    wanted="${wanted% }"
    if ! command -v dpkg-query >/dev/null 2>&1; then
        say "system packages: not a Debian or Ubuntu system; install these with your package manager: $wanted"
        return 0
    fi
    # shellcheck disable=SC2086
    missing="$(apt_missing $wanted)"
    if [ -z "$missing" ]; then
        say "system packages: all installed ($wanted)"
        return 0
    fi
    say "system packages: missing $missing. The command:"
    printf '    sudo apt-get update && sudo apt-get install %s\n' "$missing"
    if confirm "Run it now with sudo?"; then
        # shellcheck disable=SC2086
        if run sudo apt-get update && run sudo apt-get install $missing; then
            say "system packages installed"
        else
            warn "installing the system packages failed; run the command above by hand"
        fi
    fi
}

if [ "$OS" = "Linux" ]; then
    linux_packages
fi

# --- virtual environment: create or reuse ---------------------------------------------

if [ "$FRESH" -eq 1 ] && [ -e "$VENV" ]; then
    say "removing the old virtual environment (--fresh)"
    run rm -rf "$VENV"
elif [ -e "$VENV" ] && [ "$NEW_VENV" -eq 1 ]; then
    say "the virtual environment at $VENV is broken or too old; recreating it"
    run rm -rf "$VENV"
fi

if [ "$NEW_VENV" -eq 0 ]; then
    say "reusing $VENV (Python $(py_version "$VPY"))"
elif [ "$USE_UV" -eq 1 ]; then
    # uv's own CPython build (downloaded once).
    run mkdir -p "$PREFIX"
    run_install_command "$UV_BIN" venv --seed --managed-python --python 3.12 "$VENV"
else
    run mkdir -p "$PREFIX"
    if ! run_install_command "$PYTHON" -I -S -m venv "$VENV"; then
        rm -rf "$VENV"
        if [ -n "$VENV_PACKAGE" ]; then
            die "$PYTHON couldn't create the virtual environment: run sudo apt install $VENV_PACKAGE, then re-run."
        fi
        die "$PYTHON couldn't create the virtual environment (see the error above)"
    fi
fi
if [ "$DRY_RUN" -eq 0 ]; then
    check_input "$VPY" # uv's Python on macOS: not checked above
elif [ "$INPUT_CHECKED" -eq 0 ]; then
    say "dry run: the installation input is checked once uv has created the virtual environment"
fi

pip_install() {
    if [ "$USE_UV" -eq 1 ]; then
        run_install_command "$UV_BIN" pip install --python "$VPY" "$@"
    else
        run_install_command "$VPY" -I -m pip install --disable-pip-version-check "$@"
    fi
}

pip_uninstall() {
    if [ "$USE_UV" -eq 1 ]; then
        run_install_command "$UV_BIN" pip uninstall --python "$VPY" "$@"
    else
        run_install_command "$VPY" -I -m pip uninstall --disable-pip-version-check -y "$@"
    fi
}

check_verified_input() {
    local current
    if [ -n "$STAGED_WHEEL" ]; then
        current="$(isolated_stdlib_python "$VPY" -c "$WHEEL_SHA256" "$STAGED_WHEEL")" \
            || die "couldn't re-read the private verified wheel"
        [ "$current" = "$VERIFIED_WHEEL_SHA256" ] \
            || die "the private verified wheel changed after validation; refusing to install it"
    elif [ -n "$STAGED_SOURCE" ]; then
        current="$(isolated_stdlib_python "$VPY" -c "$SOURCE_TREE_SHA256" "$STAGED_SOURCE")" \
            || die "couldn't re-read the private verified source snapshot"
        [ "$current" = "$VERIFIED_SOURCE_SHA256" ] \
            || die "the private verified source snapshot changed after validation; refusing to install it"
    fi
}

# Run by the venv's Python with the wanted local label (TORCH_LOCAL) as argv[1]:
# prints the installed torch and torchvision with another label, their names on
# one line and "name version, ..." on the next. It reads package metadata;
# importing torch would take seconds.
TORCH_BUILD_CHECK='import importlib.metadata as metadata, sys
found = []
for name in ("torch", "torchvision"):
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        continue
    if version.partition("+")[2].lower() != sys.argv[1]:
        found.append((name, version))
if found:
    print(" ".join(name for name, _ in found))
    print(", ".join(name + " " + version for name, version in found))'
SELECTED_PACKAGE_CHECK='# SO101_SELECTED_PACKAGE_CHECK
import importlib.metadata as metadata, sys
expected = dict(item.split("==", 1) for item in sys.argv[1:])
found = {}
problems = []
for name, wanted in expected.items():
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError:
        problems.append(name + " is missing")
        continue
    found[name] = actual
    if actual != wanted:
        problems.append(name + " is " + actual + ", expected " + wanted)
if problems:
    raise SystemExit("; ".join(problems))
print(", ".join(name + " " + found[name] for name in expected))'

verify_selected_packages() {
    local found
    if [ "$DRY_RUN" -eq 1 ]; then
        say "verify after install: ${TORCH_SPECS[*]}, $CODEC_SPEC"
        return 0
    fi
    if ! found="$(isolated_python "$VPY" -c "$SELECTED_PACKAGE_CHECK" \
        "${TORCH_SPECS[@]}" "$CODEC_SPEC" 2>&1)"; then
        die "installed PyTorch packages do not match the selected $PROFILE profile: ${found##*$'\n'}. Re-run with --fresh; the data folder is kept."
    fi
    say "verified packages: $found"
}

# replace_other_build: uninstall the torch and torchvision a reused venv has if
# they're another build than $PROFILE's. pip and uv keep an installed
# 2.11.0+cpu for torch==2.11.0 (PEP 440: a version without a local label
# matches any local label), so a re-run with --profile cu130 after a driver
# update would keep the CPU build. The pinned +cu128, +cu126 and +cpu specs
# replace another build by themselves; uninstalling first changes nothing there.
replace_other_build() {
    local found why=""
    local -a stale
    [ "$NEW_VENV" -eq 0 ] || return 0 # a new venv has no torch yet
    if ! found="$(isolated_python "$VPY" -c "$TORCH_BUILD_CHECK" "$TORCH_LOCAL" 2>/dev/null)"; then
        warn "couldn't read which PyTorch build $VENV has; if torch isn't the $PROFILE build afterwards, re-run with --fresh"
        return 0
    fi
    [ -n "$found" ] || return 0
    read -r -a stale <<<"${found%%$'\n'*}"
    [ -n "$TORCH_LOCAL" ] || why=" (pip and uv would keep it: a pin without a +label accepts any build)"
    say "replacing ${found#*$'\n'}: not the $PROFILE build$why; uninstalling it first"
    pip_uninstall "${stale[@]}" \
        || die "couldn't uninstall ${stale[*]} from $VENV; re-run with --fresh to recreate the app's virtual environment (the data folder is kept)"
}

# --- packages --------------------------------------------------------------------

if [ "$USE_UV" -eq 0 ]; then
    run_install_command "$VPY" -I -m pip install --disable-pip-version-check \
        --index-url "$PYPI_INDEX" --upgrade pip
fi
say "step 1/3: PyTorch ($PROFILE)"
replace_other_build
pip_install ${TORCH_ARGS[@]+"${TORCH_ARGS[@]}"} "${TORCH_SPECS[@]}"
# A second resolve, from PyPI: the PyTorch CUDA indexes have no torchcodec for
# every OS, and one resolve never mixes two indexes.
pip_install --index-url "$PYPI_INDEX" "$CODEC_SPEC"
say "step 2/3: the suite and its dependencies"
# torch, torchvision and torchcodec are already satisfied, so this keeps the
# builds from step 1. feetech-servo-sdk has no wheel and builds locally.
check_verified_input
pip_install --index-url "$PYPI_INDEX" "$SUITE_SPEC"
check_verified_input
say "step 3/3: refresh the suite itself (same version numbers between builds)"
check_verified_input
if [ "$USE_UV" -eq 1 ]; then
    pip_install --index-url "$PYPI_INDEX" --no-deps --reinstall-package so101 "$TARGET"
else
    pip_install --index-url "$PYPI_INDEX" --no-deps --force-reinstall "$TARGET"
fi
check_verified_input
verify_selected_packages

# The suite must import and its so101 command must run: every step below uses
# them, so this fails the install. (so101 has no --version; --help loads the
# whole command line without reading settings.)
FRESH_HINT="Re-run with --fresh to recreate the app's virtual environment; the data folder is kept."
if [ "$DRY_RUN" -eq 1 ]; then
    run "$VENV/bin/so101" --help
else
    if ! SUITE_VERSION="$(isolated_python "$VPY" -c 'import so101; print(so101.__version__)' 2>&1)"; then
        die "the suite doesn't import in $VENV (${SUITE_VERSION##*$'\n'}). $FRESH_HINT"
    fi
    [ -x "$VENV/bin/so101" ] || die "$VENV/bin/so101 is missing: the suite's command wasn't installed. $FRESH_HINT"
    "$VENV/bin/so101" --help >/dev/null 2>&1 \
        || die "$VENV/bin/so101 doesn't run (run $VENV/bin/so101 --help to see why). $FRESH_HINT"
    say "installed: so101 $SUITE_VERSION ($VENV/bin/so101)"
fi

# --- Linux: serial access and udev rules ------------------------------------------------

# rules_installed FILE CONTENT: true if FILE holds exactly CONTENT.
rules_installed() {
    [ -f "$1" ] && [ "$(cat "$1")" = "$2" ]
}

# in_groups GROUP [USER]: true if GROUP is one of this login session's groups,
# or with USER, one of the groups USER is configured in (id -nG). No pipeline:
# under pipefail, grep -q quitting early could fail the test it answers.
in_groups() {
    local group="$1" groups
    shift
    groups="$(id -nG "$@" 2>/dev/null)" || return 1
    case " $groups " in
        *" $group "*) return 0 ;;
        *) return 1 ;;
    esac
}

# The dialout group owns the arm adapters' /dev/ttyACM* and /dev/ttyUSB*.
dialout_group() {
    local user="${USER:-$(id -un)}"
    if in_groups dialout; then
        say "serial ports: $user is in the dialout group"
    elif in_groups dialout "$user"; then
        say "serial ports: $user was added to the dialout group; log out and back in for it to take effect"
    else
        say "serial ports: $user isn't in the dialout group, so it can't open the arm adapters. The command:"
        printf '    sudo usermod -aG dialout %s\n' "$user"
        if confirm "Run it now with sudo?"; then
            if run sudo usermod -aG dialout "$user"; then
                say "added $user to dialout: log out and back in (or reboot) for it to take effect"
            else
                warn "adding $user to dialout failed; run the command above by hand"
            fi
        fi
    fi
}

# Durable device-global locks and append-only energized intent must not live in
# /tmp: age cleanup could unlink an active lock/marker. Root-owned setgid+sticky
# directories let every hardware-group member share the namespace without
# deleting another user's immutable records or permitting world access.
device_ownership_state() {
    local root group
    if [ "$OS" = "Linux" ]; then
        root="$LINUX_DEVICE_STATE_ROOT"
        group="dialout"
    else
        root="$MACOS_DEVICE_STATE_ROOT"
        group="staff"
    fi
    say "motor-device ownership state: $root (root:$group, mode 3770). The commands:"
    printf '    sudo install -d -o root -g %s -m 3770 %q %q %q\n' \
        "$group" "$root" "$root/locks" "$root/intent-events"
    printf '    sudo install -d -o root -g %s -m 3770 %q\n' \
        "$group" "$root/intent-events/aliases"
    if confirm "Create or repair the shared motor-device state directories with sudo?"; then
        if run sudo install -d -o root -g "$group" -m 3770 \
            "$root" "$root/locks" "$root/intent-events" \
            && run sudo install -d -o root -g "$group" -m 3770 \
                "$root/intent-events/aliases"; then
            say "motor-device ownership state ready"
        else
            die "couldn't create the shared motor-device state directories; real-arm commands fail closed without them"
        fi
    fi
}

# arm_rules_text FOLLOWER LEADER: the rules so101.platform_info.arm_udev_rules writes.
arm_rules_text() {
    printf '%s\n' \
        "# SO-101 arm adapters: stable names, and ModemManager leaves them alone." \
        "SUBSYSTEM==\"tty\", ATTRS{serial}==\"$1\", SYMLINK+=\"so101-follower\", ENV{ID_MM_DEVICE_IGNORE}=\"1\"" \
        "SUBSYSTEM==\"tty\", ATTRS{serial}==\"$2\", SYMLINK+=\"so101-leader\", ENV{ID_MM_DEVICE_IGNORE}=\"1\""
}

arm_udev() {
    local rules staged
    if [ -z "$FOLLOWER_SERIAL" ]; then
        say "arm names: to name the adapters /dev/so101-follower and /dev/so101-leader, plug both arms in, run $VPY -m so101.platform_info to see each adapter's serial number, then re-run with --follower-serial and --leader-serial"
        return 0
    fi
    rules="$(arm_rules_text "$FOLLOWER_SERIAL" "$LEADER_SERIAL")"
    if rules_installed "$ARM_RULES" "$rules"; then
        say "arm udev rules: already installed ($ARM_RULES)"
        return 0
    fi
    say "arm udev rules: /dev/so101-follower and /dev/so101-leader for adapters $FOLLOWER_SERIAL and $LEADER_SERIAL. The rules and commands:"
    printf '%s\n' "$rules" | sed 's/^/    | /'
    printf '    sudo install -m 0644 <a file with the rules above> %s\n' "$ARM_RULES"
    printf '    sudo udevadm control --reload-rules && sudo udevadm trigger\n'
    printf '    then unplug and replug both arms\n'
    if confirm "Run them now with sudo?"; then
        if ! staged="$(mktemp)" || ! printf '%s\n' "$rules" >"$staged"; then
            warn "couldn't stage the rules in a temporary file; run the commands above by hand"
            [ -z "${staged:-}" ] || rm -f "$staged"
            return 0
        fi
        if run sudo install -m 0644 "$staged" "$ARM_RULES" \
            && run sudo udevadm control --reload-rules \
            && run sudo udevadm trigger; then
            say "arm udev rules installed; unplug and replug both arms"
        else
            warn "installing the arm udev rules failed; run the commands above by hand"
        fi
        rm -f "$staged"
    fi
}

# show_settings OUTPUT: what so101.platform_info --set-env/--init-env printed.
show_settings() {
    [ -n "$1" ] || return 0
    printf '%s\n' "$1" | sed '1s/^/==> /; 2,$s/^/  /'
}

# The data folder's .env (in your home folder, so no sudo): the absolute storage
# folders, so every so101 command finds the same data from any folder, and with
# --follower-serial/--leader-serial the arms' udev names. Each edit is shown; a
# value the file already has is replaced only if you answer y.
settings_file() {
    local out status
    local -a ports=(--set-env "SO101_FOLLOWER_PORT=$FOLLOWER_LINK" --set-env "SO101_LEADER_PORT=$LEADER_LINK")
    if [ "$DRY_RUN" -eq 1 ]; then
        run "$VPY" -P -m so101.platform_info --init-env
        if [ -n "$FOLLOWER_SERIAL" ]; then
            run "$VPY" -P -m so101.platform_info "${ports[@]}"
        fi
        return 0
    fi
    status=0
    out="$("$VPY" -P -m so101.platform_info --init-env 2>&1)" || status=$?
    show_settings "$out"
    if [ "$status" -eq "$SET_ENV_KEPT" ]; then
        say "kept the folders the settings file already names"
    elif [ "$status" -ne 0 ]; then
        die "the data folder $DATA_DIR cannot be used for settings ($DATA_DIR/.env could not be initialized)"
    fi
    [ -n "$FOLLOWER_SERIAL" ] || return 0
    status=0
    out="$("$VPY" -P -m so101.platform_info "${ports[@]}" 2>&1)" || status=$?
    show_settings "$out"
    if [ "$status" -eq "$SET_ENV_KEPT" ]; then
        if ask "Replace them with $FOLLOWER_LINK and $LEADER_LINK?"; then
            status=0
            out="$("$VPY" -P -m so101.platform_info "${ports[@]}" --replace 2>&1)" || status=$?
            show_settings "$out"
        else
            say "kept the ports already set; the udev names are $FOLLOWER_LINK and $LEADER_LINK"
            status=0
        fi
    fi
    if [ "$status" -ne 0 ]; then
        warn "couldn't set the arm ports; put SO101_FOLLOWER_PORT=$FOLLOWER_LINK and SO101_LEADER_PORT=$LEADER_LINK in $DATA_DIR/.env"
    fi
}

# realsense_rules_valid PATH: a rules file counts only when valid USB rules
# grant access to every RealSense model this suite targets: D405, D415, D435,
# D435i and D455 (product ids 0b5b, 0ad3, 0b07, 0b3a and 0b5c). The suite's
# upstream file uses SUBSYSTEMS/ATTRS and MODE/GROUP; distro/vendor packages
# may use SUBSYSTEM/ATTR and TAG+=\"uaccess\" instead. Udev keys are
# case-sensitive, so malformed prefixes and lower-case imitations do not pass.
realsense_rules_valid() {
    local path="$1"
    [ -r "$path" ] || return 1
    awk '
        {
            line = $0
            sub(/#.*/, "", line)
            subsystem = vendor = access = product_count = 0
            count = split(line, clauses, ",")
            for (i = 1; i <= count; i++) {
                token = clauses[i]
                gsub(/[[:space:]]/, "", token)
                if (token ~ /^SUBSYSTEM(S)?=="usb"$/) subsystem = 1
                if (token ~ /^ATTR(S)?[{]idVendor[}]=="8086"$/) vendor = 1
                if (token ~ /^MODE:?="0666"$/ || token ~ /^GROUP:?="plugdev"$/ \
                        || token ~ /^TAG[+]="uaccess"$/) access = 1
                if (token ~ /^ATTR(S)?[{]idProduct[}]=="[0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f]([|][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f])*"$/) {
                    products = token
                    sub(/^ATTR(S)?[{]idProduct[}]=="/, "", products)
                    sub(/"$/, "", products)
                    product_count = split(products, product_ids, "[|]")
                }
            }
            if (subsystem && vendor && access && product_count) {
                for (j = 1; j <= product_count; j++) {
                    covered[tolower(product_ids[j])] = 1
                }
            }
        }
        END {
            complete = covered["0ad3"] && covered["0b07"] && covered["0b3a"] \
                && covered["0b5b"] && covered["0b5c"]
            exit(complete ? 0 : 1)
        }
    ' "$path"
}

# Intel's RealSense rules: print the commands; run them only if the user says yes.
realsense_udev() {
    local found="" source="" shown=""
    for found in $REALSENSE_RULES $REALSENSE_RULES_SYSTEM; do
        if realsense_rules_valid "$found"; then
            say "RealSense udev rules: already installed ($found)"
            return 0
        elif [ -e "$found" ]; then
            warn "ignoring $found: it has no usable Intel RealSense USB access rule"
        fi
    done
    if [ "$DRY_RUN" -eq 1 ]; then
        # The venv may not exist yet, so show where the path comes from.
        shown="\"\$($(printf '%q' "$VPY") -m so101.platform_info --realsense-udev-rules)\""
    else
        source="$("$VPY" -P -m so101.platform_info --realsense-udev-rules 2>/dev/null)" || source=""
        if [ -z "$source" ]; then
            warn "this build of the suite has no RealSense udev rules file. Copy config/99-realsense-libusb.rules from librealsense to $REALSENSE_RULES with sudo, run: sudo udevadm control --reload-rules && sudo udevadm trigger, then unplug and replug the camera."
            return 0
        fi
        shown="$(printf '%q' "$source")"
    fi
    say "RealSense udev rules: librealsense needs them to open the camera without root. The commands:"
    printf '    sudo cp %s %s\n' "$shown" "$REALSENSE_RULES"
    printf '    sudo udevadm control --reload-rules && sudo udevadm trigger\n'
    printf '    then unplug and replug the camera\n'
    if confirm "Run them now with sudo?"; then
        if run sudo cp "$source" "$REALSENSE_RULES" \
            && run sudo udevadm control --reload-rules \
            && run sudo udevadm trigger; then
            say "RealSense udev rules installed; unplug and replug the camera"
        else
            warn "installing the RealSense udev rules failed; run the commands above by hand"
        fi
    fi
}

# so101 and so101-suite on PATH through ~/.local/bin, which Ubuntu's ~/.profile
# puts on PATH at login.
path_links() {
    local bin="$HOME/.local/bin" name link
    if ! run mkdir -p "$bin"; then
        warn "couldn't create $bin, so so101 and so101-suite aren't on PATH; run them from $VENV/bin"
        return 0
    fi
    for name in so101 so101-suite; do
        link="$bin/$name"
        if [ -e "$link" ] && ! [ -L "$link" ]; then
            warn "$link exists and isn't a link, so it was left alone; run $VENV/bin/$name instead"
            continue
        fi
        run ln -sfn "$VENV/bin/$name" "$link" || warn "couldn't link $link; run $VENV/bin/$name instead"
    done
    case ":$PATH:" in
        *":$bin:"*) say "commands: so101 and so101-suite are on PATH ($bin)" ;;
        *) say "commands: so101 and so101-suite are in $bin, which is on PATH after you log out and back in" ;;
    esac
}

# --- the data folder ---------------------------------------------------------------

ensure_data_folder() {
    local probe=""
    if [ "$DRY_RUN" -eq 1 ]; then
        say "data folder check: create it and verify that it is writable"
        return 0
    fi
    if ! mkdir -p "$DATA_DIR"; then
        die "the data folder $DATA_DIR cannot be created or used"
    fi
    if ! probe="$(mktemp "$DATA_DIR/.so101-install-write-test.XXXXXX")"; then
        die "the data folder $DATA_DIR cannot be created or used (it is not writable)"
    fi
    if ! rm -f "$probe"; then
        die "the data folder $DATA_DIR cannot be created or used (a write test could not be removed: $probe)"
    fi
}

if [ "$DRY_RUN" -eq 1 ]; then
    DATA_DIR="(the data folder: so101.platform_info --data-dir)"
else
    DATA_DIR="$("$VPY" -P -m so101.platform_info --data-dir)" \
        || die "the suite couldn't name its data folder (python -m so101.platform_info --data-dir failed; see above). $FRESH_HINT"
fi
if [ "$LAUNCHER" -eq 1 ]; then
    case "$VPY$DATA_DIR" in
        *$'\n'* | *$'\r'*)
            die "the app or data path contains a line break, which cannot be encoded safely in a launcher; choose another --prefix/SO101_HOME or use --no-launcher"
            ;;
    esac
fi
say "data folder: $DATA_DIR"
ensure_data_folder

if [ "$OS" = "Linux" ]; then
    path_links
    dialout_group
    arm_udev
fi
device_ownership_state
settings_file
if [ "$REALSENSE" -eq 1 ]; then
    if [ "$DRY_RUN" -eq 0 ] \
        && ! isolated_python "$VPY" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("pyrealsense2") is None)'; then
        warn "--realsense: pyrealsense2 didn't get installed. This build of the suite may have no realsense extra (pip and uv only warn about that); see the install output above."
    elif [ "$OS" = "Linux" ]; then
        realsense_udev
    fi
fi

# --- launcher ------------------------------------------------------------------------

# Characters the .desktop Exec key needs escaped inside double quotes.
desktop_quote() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//\`/\\\`}"
    s="${s//\$/\\\$}"
    s="${s//%/%%}"
    printf '"%s"' "$s"
}

# write_file PATH CONTENT: false if it couldn't be written (the caller warns).
write_file() {
    local path="$1" content="$2"
    if [ "$DRY_RUN" -eq 1 ]; then
        printf '+ write %s:\n%s\n' "$path" "$content"
        return 0
    fi
    mkdir -p "$(dirname "$path")" && printf '%s\n' "$content" >"$path"
}

if [ "$LAUNCHER" -eq 1 ]; then
    if [ "$OS" = "Linux" ]; then
        APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
        DESKTOP="$APPS/so101-suite.desktop"
        if write_file "$DESKTOP" "[Desktop Entry]
Type=Application
Version=1.0
Name=SO-101 Suite
Comment=Operate, record, simulate and train the SO-101 arm
Exec=$(desktop_quote "$VPY") -P -X utf8 -m so101.launcher --home $(desktop_quote "$DATA_DIR") --pause-on-error
Terminal=true
Categories=Development;Science;Robotics;"; then
            if command -v update-desktop-database >/dev/null 2>&1; then
                run update-desktop-database "$APPS" || true
            fi
            say "launcher: SO-101 Suite in the app menu ($DESKTOP); it starts the suite in a terminal and opens the browser"
        else
            warn "couldn't write the launcher $DESKTOP; start the suite with $VENV/bin/so101-suite"
        fi
    else
        COMMAND_FILE="$HOME/Applications/SO-101 Suite.command"
        if write_file "$COMMAND_FILE" "#!/bin/bash
# SO-101 Suite: opens a Terminal window that runs the server. Closing it stops the suite.
exec $(printf '%q' "$VPY") -P -X utf8 -m so101.launcher --home $(printf '%q' "$DATA_DIR") --pause-on-error" \
            && run chmod +x "$COMMAND_FILE"; then
            say "launcher: ~/Applications/SO-101 Suite.command (double-click it, or drag it to the Dock)"
        else
            warn "couldn't write the launcher $COMMAND_FILE; start the suite with $VENV/bin/so101-suite"
        fi
    fi
fi

# --- check ---------------------------------------------------------------------------

# Last, once everything the install leaves behind is in place, and from the data
# folder, so a .env in the folder the installer was started from (a source
# checkout's, say) doesn't change what doctor checks. Its verdict is advice.
DOCTOR_OK=1
if [ "$DRY_RUN" -eq 0 ]; then
    (cd "$DATA_DIR") || die "the data folder $DATA_DIR cannot be created or used"
    say "platform report"
    (cd "$DATA_DIR" && "$VPY" -m so101.platform_info) || warn "the platform report failed"
    say "so101 doctor"
    (cd "$DATA_DIR" && "$VENV/bin/so101" doctor) || DOCTOR_OK=0
else
    run "$VPY" -m so101.platform_info
    run "$VENV/bin/so101" doctor
fi

# doctor_advice: what to do when so101 doctor exits non-zero (a FAIL row, or an
# error it printed), which a first install expects on Linux.
doctor_advice() {
    warn "so101 doctor reported a FAIL (see its table above); the install itself is complete."
    if [ "$OS" = "Linux" ]; then
        say "right after a first install, checks that need a fresh login can't pass yet: the dialout group (opening the arm adapters), ~/.local/bin on PATH (the so101 command), and the /dev/so101-follower and /dev/so101-leader names once the arms are plugged in. So log out and back in, then run so101 doctor again, and fix any FAIL that remains with the remediation it shows."
    else
        say "fix it with the remediation it shows, then run $VENV/bin/so101 doctor again"
    fi
}

say "done. Start the suite from the launcher, or run: $VENV/bin/so101-suite"
if [ "$OS" = "Linux" ]; then
    say "MuJoCo: the launcher sets MUJOCO_GL=egl when the NVIDIA driver is loaded, so the simulator renders offscreen on the GPU; export MUJOCO_GL=egl yourself when you run so101 by hand"
fi
if [ "$REALSENSE" -eq 1 ]; then
    say "RealSense: plug the camera into a USB 3 port; USB 2 limits resolution and frame rate"
    if [ "$OS" = "Darwin" ]; then
        say "RealSense on macOS: librealsense usually needs sudo to open the camera; color-only capture also works without it through the camera's UVC index (a plain camera number in SO101_CAMERAS)"
    fi
fi
if [ "$DOCTOR_OK" -eq 0 ]; then
    doctor_advice
fi
