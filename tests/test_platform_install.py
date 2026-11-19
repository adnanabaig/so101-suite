"""install/install.sh, run as dry runs on a faked machine.

``uname``, ``sw_vers``, ``sysctl``, ``id``, ``dpkg-query`` and ``nvidia-smi`` are
replaced by shims on PATH, and the Python the installer is given is a wrapper
that reports whatever version (and base prefix) the test asks for, so Ubuntu on
an NVIDIA laptop or a Mac on macOS 14 can be checked from any POSIX machine.
Dry runs install nothing and never ask questions or run sudo.

A few tests run the whole script for real (:func:`full_run`) against a faked,
already-created venv whose ``python`` answers pip and so101.platform_info
itself and whose ``so101`` command is a shim, to check what a dry run can't:
what the script does when doctor or a late step fails. Nothing is installed.

The realsense extra itself belongs to the camera code's stream; the contract
test at the end checks it against so101.platform_info once pyproject.toml has it.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

from so101 import platform_info as pi

ROOT = Path(__file__).resolve().parents[1]
INSTALL_SH = ROOT / "install" / "install.sh"
BASH = shutil.which("bash")

needs_bash = pytest.mark.skipif(
    sys.platform == "win32" or BASH is None, reason="install.sh needs a POSIX shell"
)

# Runs as the fake interpreter. FAKE_PYTHON_VERSION is like 3.12.3, with a
# trailing "t" for a free-threaded build; FAKE_BASE_PREFIX is sys.base_prefix.
FAKE_PYTHON = """\
import os
import sys
import sysconfig

raw = os.environ["FAKE_PYTHON_VERSION"]
free_threaded = raw.endswith("t")
version = tuple(int(part) for part in raw.rstrip("t").split("."))
args = sys.argv[1:]
while args[:1] in (["-P"], ["-I"], ["-S"]):
    args = args[1:]
if args[:1] != ["-c"]:
    raise SystemExit("fake python: only isolation flags plus -c are supported, got %r" % (args,))
code = args[1]
if "platform.python_version()" in code:
    print(".".join(map(str, version)) + ("t" if free_threaded else ""))
    raise SystemExit(0)
if "import so101; print(so101.__version__)" in code:
    print(os.environ["FAKE_SUITE_VERSION"])
    raise SystemExit(0)
sys.argv = ["-c", *args[2:]]
# FAKE_DISTS (name=version,...): the PyTorch packages this venv has; none by default.
import importlib.metadata

dists = dict(item.split("=", 1) for item in os.environ.get("FAKE_DISTS", "").split(",") if item)
real_version = importlib.metadata.version


def fake_version(name):
    if name not in ("torch", "torchvision", "torchcodec"):
        return real_version(name)
    if name in dists:
        return dists[name]
    raise importlib.metadata.PackageNotFoundError(name)


importlib.metadata.version = fake_version
sys.version_info = version + ("final", 0)
sys.base_prefix = os.environ.get("FAKE_BASE_PREFIX", "/opt/python")
real_get_config_var = sysconfig.get_config_var
sysconfig.get_config_var = lambda name: (
    int(free_threaded) if name == "Py_GIL_DISABLED" else real_get_config_var(name)
)
exec(compile(code, "<string>", "exec"), {"__name__": "__main__"})
"""


def _files_under(root: Path) -> dict[str, bytes]:
    return {
        file.relative_to(root).as_posix(): file.read_bytes()
        for file in root.rglob("*")
        if file.is_file()
        and "__pycache__" not in file.parts
        and file.suffix != ".pyc"
        and file.name != ".DS_Store"
    }


def _ui_sources_digest(frontend: Path) -> str:
    candidates = [
        *(file for name in ("src", "public") for file in (frontend / name).rglob("*")),
        *(
            frontend / name
            for name in (
                "index.html",
                "package.json",
                "package-lock.json",
                "vite.config.ts",
                "build-stamp.ts",
                "tsconfig.json",
                "tsconfig.node.json",
            )
        ),
    ]
    files = sorted(
        (
            file.relative_to(frontend).as_posix(),
            hashlib.sha256(file.read_bytes()).hexdigest(),
        )
        for file in candidates
        if file.is_file() and file.name != ".DS_Store"
    )
    digest = hashlib.sha256()
    for name, sha256 in files:
        digest.update(f"{name}\0{sha256}\n".encode())
    return digest.hexdigest()


def _git_head(source: Path) -> str | None:
    if not (source / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def make_wheel(
    path: Path,
    version: str,
    *,
    source: Path = ROOT,
    ui: bool = True,
    ui_sources_digest: str | None = None,
    package_override: dict[str, bytes] | None = None,
    metadata_extra: tuple[str, ...] = (),
    dist_info_extra: dict[str, bytes] | None = None,
    omit_extra: str | None = None,
    extra_script: tuple[str, str] | None = None,
    wheel_metadata_extra: str = "",
    record_override: bytes | None = None,
) -> Path:
    """A small suite wheel matching ``source``, optionally with a stamped UI."""

    dist = f"so101-{version}.dist-info"
    package = {
        f"so101/{name}": data for name, data in _files_under(source / "src" / "so101").items()
    }
    package.update(package_override or {})
    project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    requires = [f"Requires-Dist: {item}" for item in project.get("dependencies", [])]
    for extra, dependencies in project.get("optional-dependencies", {}).items():
        if extra == omit_extra:
            continue
        requires.append(f"Provides-Extra: {extra}")
        for item in dependencies:
            requirement, separator, marker = item.partition(";")
            combined = f"({marker.strip()}) and " if separator else ""
            requires.append(f"Requires-Dist: {requirement.strip()}; {combined}extra == '{extra}'")
    if project.get("requires-python"):
        requires.append(f"Requires-Python: {project['requires-python']}")
    requires.extend(metadata_extra)
    metadata = (
        f"Metadata-Version: 2.4\nName: so101\nVersion: {version}\n" + "\n".join(requires) + "\n"
    )
    files = dict(package)
    if ui:
        built = {
            "index.html": b"<!doctype html><title>SO-101 test UI</title>\n",
            "assets/app.js": b"export const testBuild = true;\n",
        }
        files.update({f"so101/web/{name}": data for name, data in built.items()})
        stamp = {
            "format": 1,
            "sources": ui_sources_digest or _ui_sources_digest(source / "frontend"),
            "git_head": _git_head(source),
            "files": {name: hashlib.sha256(data).hexdigest() for name, data in built.items()},
        }
        files["so101/web/so101-ui-stamp.json"] = json.dumps(stamp).encode()
    files[f"{dist}/METADATA"] = metadata.encode()
    scripts = project.get("scripts", {})
    if scripts or extra_script:
        entries = "[console_scripts]\n" + "".join(
            f"{name} = {target}\n" for name, target in scripts.items()
        )
        if extra_script:
            entries += f"{extra_script[0]} = {extra_script[1]}\n"
        files[f"{dist}/entry_points.txt"] = entries.encode()
    files[f"{dist}/WHEEL"] = (
        "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n" + wheel_metadata_extra
    ).encode()
    files.update({f"{dist}/{name}": data for name, data in (dist_info_extra or {}).items()})
    record_name = f"{dist}/RECORD"
    record_lines = []
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record_lines.append(f"{name},sha256={digest},{len(data)}\n")
    record_lines.append(f"{record_name},,\n")
    files[record_name] = record_override or "".join(record_lines).encode()
    with zipfile.ZipFile(path, "w") as wheel:
        for name, data in files.items():
            wheel.writestr(name, data)
    return path


def shim(folder: Path, name: str, body: str) -> None:
    path = folder / name
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
    path.chmod(0o755)


def fake_python(folder: Path, version: str, base_prefix: str = "/opt/python") -> Path:
    """A ``python`` that says it is ``version``; really the interpreter running the tests."""

    helper = folder / f"fake_python_{version}.py"
    helper.write_text(FAKE_PYTHON, encoding="utf-8")
    (folder / "fake_dists").touch()
    suite_version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    (folder / "fake_suite_version").write_text(suite_version, encoding="utf-8")
    path = folder / f"python{version}-{base_prefix.strip('/').replace('/', '_')}"
    path.write_text(
        "#!/bin/sh\n"
        f"HERE={shlex.quote(str(folder))}\n"
        'FAKE_DISTS="$(cat "$HERE/fake_dists" 2>/dev/null || true)"\n'
        'FAKE_SUITE_VERSION="$(cat "$HERE/fake_suite_version")"\n'
        f"export FAKE_DISTS FAKE_SUITE_VERSION FAKE_PYTHON_VERSION={version} "
        f"FAKE_BASE_PREFIX={base_prefix}\n"
        f'if [ "${{1-}}" = "-I" ]; then\n'
        "  shift\n"
        f'  if [ "${{1-}}" = "-S" ]; then shift; exec "{sys.executable}" -I -S '
        f'"{helper}" -I -S "$@"; fi\n'
        f'  exec "{sys.executable}" -I "{helper}" -I "$@"\n'
        "fi\n"
        f'exec "{sys.executable}" "{helper}" "$@"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def dry_run(
    tmp_path: Path,
    *args: str,
    os_name: str = "Linux",
    arch: str = "x86_64",
    macos: str = "15.5",
    python: str | None = "3.12.3",
    base_prefix: str = "/opt/python",
    profile: str | None = "cpu",
    uv: bool = False,
    uv_body: str | None = None,
    groups: str = "ada dialout",
    installed: tuple[str, ...] | None = None,
    nvidia: str | None = "",
    nvidia_body: str | None = None,
    dpkg_body: str | None = None,
    uid: int = 1000,
    dists: str = "",
    dry: bool = True,
    launcher: bool = False,
    extra_env: dict[str, str] | None = None,
    script: Path = INSTALL_SH,
    wheel_arg: bool = True,
) -> subprocess.CompletedProcess[str]:
    """``python=None``: the installer picks one. ``installed``: the dpkg packages, all
    by default. ``nvidia``: nvidia-smi's driver version; "" for one that doesn't answer.
    ``nvidia_body`` and ``dpkg_body``: those shims' whole script instead. ``dists``:
    the torch builds a reused venv has, like ``torch=2.11.0+cpu,torchvision=0.26.0+cpu``.
    ``dry=False``: a real run (see :func:`full_run`). ``script``: another copy of
    install.sh (see :func:`fake_repo`); ``wheel_arg=False``: no --wheel, so the
    installer picks a wheel from the copy's dist/ or uses its source folder."""

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    shim(bin_dir, "uname", f'case "$1" in -m) echo {arch} ;; *) echo {os_name} ;; esac')
    shim(bin_dir, "sw_vers", f"echo {macos}")
    shim(bin_dir, "sysctl", "echo 0")
    shim(
        bin_dir,
        "id",
        f'case "$1" in -u) echo {uid} ;; -un) echo ada ;; *) echo "{groups}" ;; esac',
    )
    if dpkg_body is not None:
        shim(bin_dir, "dpkg-query", dpkg_body)
    elif installed is None:
        shim(bin_dir, "dpkg-query", "echo 'install ok installed'")
    else:
        names = " ".join(installed)
        shim(
            bin_dir,
            "dpkg-query",
            f'for p in {names}; do [ "$p" = "$3" ] && echo "install ok installed" && exit 0; done\n'
            "exit 1",
        )
    if nvidia_body is not None:
        shim(bin_dir, "nvidia-smi", nvidia_body)
    elif nvidia is not None:
        answer = f"echo {nvidia}" if nvidia else "exit 9"
        shim(bin_dir, "nvidia-smi", answer)
    if uv:
        shim(bin_dir, "uv", uv_body or "exit 0")
    (bin_dir / "fake_dists").write_text(dists, encoding="utf-8")
    (tmp_path / "fake_dists").write_text(dists, encoding="utf-8")
    project_version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    wheel = tmp_path / f"so101-{project_version}-py3-none-any.whl"
    if not wheel.exists():
        make_wheel(wheel, project_version)
    if profile == "cpu" and os_name == "Darwin":
        profile = "mps"
    command = [
        BASH or "bash",
        str(script),
        *(["--dry-run"] if dry else []),
        *([] if launcher else ["--no-launcher"]),
        "--prefix",
        str(tmp_path / "app"),
        *(["--wheel", str(wheel)] if wheel_arg else []),
    ]
    if profile is not None:
        command += ["--profile", profile]
    if python is not None:
        command += ["--python", str(fake_python(bin_dir, python, base_prefix))]
    command += args
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    env.pop("SO101_HOME", None)
    env.pop("XDG_DATA_HOME", None)
    env["USER"] = "ada"
    env["HOME"] = str(tmp_path / "home")
    env.update(extra_env or {})
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,  # never a terminal: nothing is asked, sudo never runs
        timeout=60,
        check=False,
    )


def output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def make_venv(tmp_path: Path, version: str) -> Path:
    venv = tmp_path / "app" / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (venv / "bin" / "python").symlink_to(fake_python(tmp_path, version))
    return venv


# --- PyTorch profile --------------------------------------------------------------


@pytest.mark.parametrize(
    ("driver", "profile", "step1", "note"),
    [
        ("580.95.05", "cu130", "--index-url https://pypi.org/simple torch==2.11.0 "
         "torchvision==0.26.0\n", "torch from PyPI, built for CUDA 13.0"),
        ("575.64.03", "cu128", "--index-url https://download.pytorch.org/whl/cu128 "
         "torch==2.11.0+cu128 torchvision==0.26.0+cu128\n", "built for CUDA 12.8"),
        ("565.57.01", "cu126", "--index-url https://download.pytorch.org/whl/cu126 "
         "torch==2.11.0+cu126 torchvision==0.26.0+cu126\n", "built for CUDA 12.6"),
        ("535.183.01", "cu126", "torch==2.11.0+cu126", "minor-version compatibility mode"),
        ("580.65.06", "cu130", "torch==2.11.0 torchvision==0.26.0\n", "older than 580.95.05"),
        ("470.256.02", "cpu", "--index-url https://download.pytorch.org/whl/cpu "
         "torch==2.11.0+cpu torchvision==0.26.0+cpu\n", "too old for any CUDA build"),
    ],
)  # fmt: skip
@needs_bash
def test_auto_profile_follows_the_nvidia_driver(tmp_path, driver, profile, step1, note):
    result = dry_run(tmp_path, profile=None, nvidia=driver)
    text = output(result)
    assert result.returncode == 0, text
    assert f"==> profile:  {profile} (" in text
    assert step1 in result.stdout
    assert note in text
    assert "--extra-index-url" not in text
    # torchcodec: always its own resolve, from PyPI.
    assert "+ " + str(tmp_path / "app" / "venv" / "bin" / "python") in result.stdout
    assert (
        "pip install --disable-pip-version-check --index-url https://pypi.org/simple "
        "torchcodec==0.11.1\n"
    ) in result.stdout


@needs_bash
def test_auto_profile_without_an_answering_driver_is_cpu(tmp_path):
    result = dry_run(tmp_path, profile=None, nvidia="")
    text = output(result)
    assert result.returncode == 0, text
    assert "no NVIDIA driver answered: the cpu build" in text
    assert "MOK enrollment" in result.stderr  # nvidia-smi is there but not answering


@needs_bash
def test_profile_cuda_needs_a_driver_and_explicit_builds_are_kept(tmp_path):
    refused = dry_run(tmp_path, profile="cuda", nvidia="")
    assert refused.returncode == 1 and "nvidia-smi doesn't answer" in refused.stderr
    picked = dry_run(tmp_path, profile="cuda", nvidia="580.95.05")
    assert picked.returncode == 0 and "the cu130 build" in picked.stdout
    explicit = dry_run(tmp_path, profile="cu128", nvidia="580.95.05")
    assert explicit.returncode == 0, output(explicit)
    assert "torch==2.11.0+cu128" in explicit.stdout
    assert dry_run(tmp_path, profile="mps").returncode == 1  # Linux


@needs_bash
def test_macos_takes_torch_and_torchcodec_from_pypi(tmp_path):
    result = dry_run(tmp_path, os_name="Darwin", arch="arm64")
    assert result.returncode == 0, output(result)
    assert "--index-url https://pypi.org/simple torch==2.11.0 torchvision==0.26.0\n" in (
        result.stdout
    )
    assert "torchcodec==0.11.1\n" in result.stdout
    assert "download.pytorch.org" not in result.stdout
    assert "apt-get" not in result.stdout and "udevadm" not in result.stdout


# --- Python ----------------------------------------------------------------------


@pytest.mark.parametrize("realsense", [(), ("--realsense",)])
@pytest.mark.parametrize("version", ["3.11.9", "3.14.0", "3.13.1t"])
@needs_bash
def test_linux_refuses_a_python_outside_312_313_or_free_threaded(tmp_path, version, realsense):
    result = dry_run(tmp_path, *realsense, python=version)
    assert result.returncode == 1
    assert f"is Python {version}, but this install needs Python 3.12 or 3.13" in result.stderr
    assert "not free-threaded" in result.stderr
    assert "+ " not in result.stdout  # refused before doing anything


@pytest.mark.parametrize("version", ["3.12.3", "3.13.1"])
@needs_bash
def test_linux_accepts_312_and_313(tmp_path, version):
    result = dry_run(tmp_path, python=version)
    text = output(result)
    assert result.returncode == 0, text
    assert "[robot,sim,train]" in text
    assert f"-m venv {tmp_path / 'app' / 'venv'}" in result.stdout
    assert "realsense:" not in text and "RealSense" not in text


@needs_bash
def test_macos_takes_python_312_or_313_and_refuses_free_threaded(tmp_path):
    mac = {"os_name": "Darwin", "arch": "arm64"}
    ok = dry_run(tmp_path, python="3.13.1", **mac)
    assert ok.returncode == 0, output(ok)
    assert "python:   Python 3.12 or 3.13" in ok.stdout
    newer = dry_run(tmp_path, python="3.14.0", **mac)  # requires-python is <3.14
    assert newer.returncode == 1 and "this install needs Python 3.12 or 3.13" in newer.stderr
    free = dry_run(tmp_path, python="3.13.1t", **mac)
    assert free.returncode == 1 and "is Python 3.13.1t" in free.stderr


@needs_bash
def test_uv_is_used_on_macos_only(tmp_path):
    linux = dry_run(tmp_path, python=None, uv=True)
    assert "uv venv" not in linux.stdout and "uv pip" not in linux.stdout
    mac = dry_run(tmp_path, python=None, uv=True, os_name="Darwin", arch="arm64")
    assert mac.returncode == 0, output(mac)
    assert (
        f"+ {tmp_path / 'bin' / 'uv'} venv --seed --managed-python --python 3.12 "
        f"{tmp_path / 'app' / 'venv'}"
    ) in (mac.stdout)


@needs_bash
def test_uv_subprocess_uses_the_same_credential_scrubbed_environment(tmp_path):
    dump = tmp_path / "uv-environment.log"
    venv = tmp_path / "app" / "venv"
    result = full_run(
        tmp_path,
        os_name="Darwin",
        arch="arm64",
        profile="mps",
        python=None,
        uv=True,
        uv_body=f'env | sort >>"{dump}"\nprintf "%s\\n" -- >>"{dump}"\nexit 0',
        extra_env={
            "UV_INDEX_URL": "https://uv-user:uv-secret@example.invalid/simple",
            "UV_CONFIG_FILE": "/tmp/host-uv.toml",
            "PYTHONPATH": "/tmp/python-injection",
            "GITHUB_TOKEN": "github-secret",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
        },
    )
    assert result.returncode == 0, output(result)
    retained = dump.read_text(encoding="utf-8")
    assert "UV_NO_CONFIG=1" in retained
    assert "HOME=/tmp/so101-install-private." in retained
    for forbidden in (
        "UV_INDEX_URL",
        "UV_CONFIG_FILE=/tmp/host-uv.toml",
        "PYTHONPATH",
        "github-secret",
        "aws-secret",
    ):
        assert forbidden not in retained
    assert f"+ {tmp_path / 'bin' / 'uv'} pip install --python {venv / 'bin' / 'python'}" in (
        result.stdout
    )


@needs_bash
def test_a_reused_venv_with_an_unsupported_python_needs_fresh(tmp_path):
    venv = make_venv(tmp_path, "3.14.0")

    result = dry_run(tmp_path)
    assert result.returncode == 1
    assert "has Python 3.14.0, but this install needs Python 3.12 or 3.13" in result.stderr
    assert "Re-run with --fresh" in result.stderr and "the data folder is kept" in result.stderr

    fresh = dry_run(tmp_path, "--fresh")
    assert fresh.returncode == 0, output(fresh)
    assert f"+ rm -rf {venv}" in fresh.stdout


@needs_bash
def test_a_reused_venv_is_kept_unless_realsense_needs_an_older_python(tmp_path):
    make_venv(tmp_path, "3.13.1")
    mac14 = {"os_name": "Darwin", "arch": "arm64", "macos": "14.6", "python": "3.12.9"}

    plain = dry_run(tmp_path, **mac14)
    assert plain.returncode == 0, output(plain)
    assert "reusing" in plain.stdout and "(Python 3.13.1)" in plain.stdout

    realsense = dry_run(tmp_path, "--realsense", **mac14)
    assert realsense.returncode == 1
    assert "has Python 3.13.1, but this install needs Python 3.12" in realsense.stderr
    assert "(--realsense before macOS 15" in realsense.stderr


# --- Linux system packages, groups and udev ------------------------------------------


@needs_bash
def test_running_a_live_installer_as_root_is_refused_before_changes(tmp_path):
    result = dry_run(
        tmp_path,
        uid=0,
        dry=False,
        extra_env={"USER": "root", "SUDO_USER": "ada"},
    )

    assert result.returncode == 1
    assert "run install.sh as your normal user, without sudo" in result.stderr
    assert "asks for sudo only for the individual system changes" in result.stderr
    assert "+ " not in result.stdout


@needs_bash
def test_root_may_audit_a_dry_run_without_making_changes(tmp_path):
    result = dry_run(
        tmp_path,
        uid=0,
        extra_env={"USER": "root", "SUDO_USER": "ada"},
    )

    assert result.returncode == 0, output(result)
    assert "run install.sh as your normal user" not in result.stderr
    assert not (tmp_path / "app").exists()
    assert not (tmp_path / "home").exists()


@needs_bash
def test_missing_apt_packages_are_printed_as_one_command_and_never_run(tmp_path):
    result = dry_run(tmp_path, python="3.12.3", base_prefix="/usr", installed=("git", "libgl1"))
    text = output(result)
    assert result.returncode == 0, text
    expected = (
        "sudo apt-get update && sudo apt-get install python3.12-venv ffmpeg libusb-1.0-0 libegl1"
    )
    assert f"    {expected}\n" in result.stdout
    assert "doesn't run sudo on its own" in result.stdout
    assert "+ sudo" not in result.stdout and "[y/N]" not in text


@needs_bash
def test_installed_apt_packages_and_a_non_system_python_need_nothing(tmp_path):
    result = dry_run(tmp_path, python="3.12.3")  # base prefix /opt/python: no venv package
    assert result.returncode == 0, output(result)
    assert "system packages: all installed (ffmpeg libusb-1.0-0 git libegl1 libgl1)" in (
        result.stdout
    )


@needs_bash
def test_the_dialout_group_is_offered_only_when_missing(tmp_path):
    member = dry_run(tmp_path)
    assert "serial ports: ada is in the dialout group" in member.stdout
    missing = dry_run(tmp_path, groups="ada sudo")
    assert "    sudo usermod -aG dialout ada\n" in missing.stdout
    assert "+ sudo" not in missing.stdout


def test_installers_provision_stable_motor_device_state() -> None:
    shell = INSTALL_SH.read_text(encoding="utf-8")
    powershell = (ROOT / "install" / "install.ps1").read_text(encoding="utf-8")

    assert 'LINUX_DEVICE_STATE_ROOT="/var/lib/so101/device-ownership-v2"' in shell
    assert (
        'MACOS_DEVICE_STATE_ROOT="/Library/Application Support/SO101/device-ownership-v2"' in shell
    )
    assert "sudo install -d -o root -g" in shell and "-m 3770" in shell
    assert '"$root/intent-events/aliases"' in shell
    assert shell.index("device_ownership_state") < shell.rindex("settings_file\n")

    assert "LOCALAPPDATA is required for stable motor-device locks" in powershell
    assert "SO101\\device-ownership-v2" in powershell
    start = powershell.index("$DeviceState =")
    device_block = powershell[start : powershell.index("if ($DryRun) {", start)]
    assert "$env:TEMP" not in device_block


@needs_bash
def test_arm_udev_rules_name_the_adapters_by_serial(tmp_path):
    result = dry_run(tmp_path, "--follower-serial", "585A0077531", "--leader-serial", "585A0077541")
    text = output(result)
    assert result.returncode == 0, text
    assert (
        '    | SUBSYSTEM=="tty", ATTRS{serial}=="585A0077531", SYMLINK+="so101-follower", '
        'ENV{ID_MM_DEVICE_IGNORE}="1"\n'
    ) in result.stdout
    assert 'ATTRS{serial}=="585A0077541", SYMLINK+="so101-leader"' in result.stdout
    assert "/etc/udev/rules.d/99-so101-arms.rules" in result.stdout
    assert "SO101_FOLLOWER_PORT=/dev/so101-follower" in result.stdout
    assert "+ sudo" not in result.stdout

    hint = dry_run(tmp_path)
    assert "re-run with --follower-serial and --leader-serial" in hint.stdout


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (("--follower-serial", "A1"), "give both"),
        (("--follower-serial", "A1", "--leader-serial", "A1"), "the same adapter"),
        (("--follower-serial", 'A1"; rm', "--leader-serial", "B2"), "only letters, digits"),
    ],
)
@needs_bash
def test_bad_serial_numbers_are_refused(tmp_path, args, error):
    result = dry_run(tmp_path, *args)
    assert result.returncode == 1 and error in result.stderr
    mac = dry_run(
        tmp_path, "--follower-serial", "A1", "--leader-serial", "B2", os_name="Darwin", arch="arm64"
    )
    assert mac.returncode == 1 and "are for Linux" in mac.stderr


# --- RealSense ---------------------------------------------------------------------


@needs_bash
def test_linux_realsense_adds_the_extra_and_prints_the_udev_commands(tmp_path):
    result = dry_run(tmp_path, "--realsense")

    text = output(result)
    assert result.returncode == 0, text
    assert "[robot,sim,train,realsense]" in text
    assert "realsense: yes" in text
    assert "python:   Python 3.12 or 3.13\n" in text
    rules = (pi.REALSENSE_UDEV_RULES, *pi.REALSENSE_UDEV_RULES_SYSTEM)
    if any(pi.realsense_udev_rules_valid(path) for path in rules):
        assert "RealSense udev rules: already installed" in text
    else:
        assert "-m so101.platform_info --realsense-udev-rules)" in text
        assert f'sudo cp "$({tmp_path / "app" / "venv" / "bin" / "python"}' in text
        assert "sudo udevadm control --reload-rules && sudo udevadm trigger" in text
    assert "[y/N]" not in text  # a dry run never asks
    assert "plug the camera into a USB 3 port" in text


@needs_bash
def test_a_realsense_extra_is_the_same_as_the_flag(tmp_path):
    result = dry_run(tmp_path, "--extras", "robot, realsense")
    assert result.returncode == 0, output(result)
    assert "[robot,realsense]" in result.stdout
    assert "realsense: yes" in result.stdout

    alone = dry_run(tmp_path, "--extras", "", "--realsense")
    assert alone.returncode == 0, output(alone)
    assert "[realsense]" in alone.stdout


@needs_bash
def test_linux_arm64_has_no_realsense_wheel(tmp_path):
    result = dry_run(tmp_path, "--realsense", arch="aarch64")
    assert result.returncode == 1
    assert "no Linux aarch64 wheel" in result.stderr
    assert "+ " not in result.stdout  # refused before doing anything


@needs_bash
def test_macos_before_15_needs_python_312_for_realsense(tmp_path):
    old_mac = dry_run(
        tmp_path, "--realsense", os_name="Darwin", arch="arm64", macos="14.6", python="3.12.9"
    )
    assert old_mac.returncode == 0, output(old_mac)
    assert "python:   Python 3.12 (--realsense before macOS 15: pyrealsense2-macosx 2.54.2" in (
        old_mac.stdout
    )

    refused = dry_run(
        tmp_path, "--realsense", os_name="Darwin", arch="arm64", macos="14.6", python="3.13.1"
    )
    assert refused.returncode == 1
    assert "is Python 3.13.1, but this install needs Python 3.12 (a standard" in refused.stderr

    new_mac = dry_run(
        tmp_path, "--realsense", os_name="Darwin", arch="arm64", macos="15.5", python="3.13.1"
    )
    text = output(new_mac)
    assert new_mac.returncode == 0, text
    assert "librealsense usually needs sudo" in text and "UVC index" in text
    assert "udevadm" not in text  # udev is Linux only


# --- help and the end of the run ---------------------------------------------------------


@needs_bash
def test_help_shows_the_whole_header_and_the_options(tmp_path):
    result = subprocess.run(
        [BASH or "bash", str(INSTALL_SH), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout.startswith("Install or upgrade the SO-101 suite")
    assert "Behind a TLS-inspecting proxy" in result.stdout
    for option in ("--realsense ", "--follower-serial S", "--leader-serial S", "cu130"):
        assert option in result.stdout
    assert "set -euo pipefail" not in result.stdout


@needs_bash
def test_linux_puts_so101_on_path_through_local_bin(tmp_path):
    result = dry_run(tmp_path)
    assert result.returncode == 0, output(result)
    venv_bin = tmp_path / "app" / "venv" / "bin"
    local_bin = tmp_path / "home" / ".local" / "bin"
    for name in ("so101", "so101-suite"):
        assert f"+ ln -sfn {venv_bin / name} {local_bin / name}\n" in result.stdout
    assert "on PATH after you log out and back in" in result.stdout

    local_bin.mkdir(parents=True)
    (local_bin / "so101").write_text("someone else's so101", encoding="utf-8")
    kept = dry_run(tmp_path)
    assert f"{local_bin / 'so101'} exists and isn't a link" in kept.stderr
    assert f"+ ln -sfn {venv_bin / 'so101'} " not in kept.stdout


@needs_bash
def test_linux_run_ends_with_doctor_and_the_mujoco_note(tmp_path):
    result = dry_run(tmp_path)
    assert result.returncode == 0, output(result)
    assert f"+ {tmp_path / 'app' / 'venv' / 'bin' / 'so101'} doctor" in result.stdout
    assert "MUJOCO_GL=egl" in result.stdout


# --- a real run against a faked venv: doctor, the launcher and late failures --------------

# The reused venv's python in a full run: pip and so101.platform_info answer here,
# anything run with -c goes to the fake interpreter (FAKE_PYTHON).
FULL_VENV_PYTHON = """\
#!/bin/sh
FAKE_LOG=__FAKE_LOG__
FAKE_DATA_DIR=__FAKE_DATA_DIR__
FAKE_C_PYTHON=__FAKE_C_PYTHON__
FAKE_C_DISTS=__FAKE_C_DISTS__
FAKE_SETTINGS_EXIT=__FAKE_SETTINGS_EXIT__
FAKE_MUTATE_STAGED_AFTER_HASH=__FAKE_MUTATE_STAGED_AFTER_HASH__
FAKE_MUTATE_SOURCE_AFTER_CHECK=__FAKE_MUTATE_SOURCE_AFTER_CHECK__
FAKE_MUTATE_SOURCE_SNAPSHOT_AFTER_HASH=__FAKE_MUTATE_SOURCE_SNAPSHOT_AFTER_HASH__
FAKE_FINAL_DISTS=__FAKE_FINAL_DISTS__
export FAKE_LOG FAKE_DATA_DIR FAKE_C_PYTHON FAKE_SETTINGS_EXIT
printf 'python %s\\n' "$*" >>"$FAKE_LOG"
isolated=0
no_site=0
case "${1-}" in
    -I) isolated=1; shift ;;
    -P) shift ;;
esac
if [ "${1-}" = "-S" ]; then no_site=1; shift; fi
case "$*" in
    *runpy.run_path* | *shutil.copyfileobj* | *hashlib.sha256* | *SO101_INPUT_FINAL_HASH*)
        printf 'helper-flags isolated=%s no_site=%s\n' "$isolated" "$no_site" >>"$FAKE_LOG"
        ;;
esac
case "$1 $2" in
    "-m pip")
        format='pip-env HOME=%s XDG_CONFIG_HOME=%s PIP_CONFIG_FILE=%s '
        format=$format'NETRC=%s PIP_KEYRING_PROVIDER=%s PIP_INDEX_URL=%s PIP_EXTRA_INDEX_URL=%s '
        format=$format'PYTHONPATH=%s UV_INDEX_URL=%s SO101_BACKEND=%s HF_TOKEN=%s '
        format=$format'GITHUB_TOKEN=%s AWS_SECRET_ACCESS_KEY=%s SSH_AUTH_SOCK=%s '
        format=$format'UNRELATED_SAFE=%s PIP_CERT=%s\\n'
        printf "$format" \
            "$HOME" "${XDG_CONFIG_HOME-}" "${PIP_CONFIG_FILE-}" "${NETRC-}" \
            "${PIP_KEYRING_PROVIDER-}" "${PIP_INDEX_URL-}" "${PIP_EXTRA_INDEX_URL-}" \
            "${PYTHONPATH-}" "${UV_INDEX_URL-}" "${SO101_BACKEND-}" "${HF_TOKEN-}" \
            "${GITHUB_TOKEN-}" "${AWS_SECRET_ACCESS_KEY-}" "${SSH_AUTH_SOCK-}" \
            "${UNRELATED_SAFE-}" "${PIP_CERT-}" >>"$FAKE_LOG"
        for argument do
            case "$argument" in
                *"/source["*)
                    snapshot="${argument%%[[]*}"
                    mode_check='import os, stat, sys; root=sys.argv[1]; '
                    mode_check=$mode_check'file=os.path.join(root, "pyproject.toml"); '
                    mode_check=$mode_check'print("source-modes %o %o" % ('
                    mode_check=$mode_check'stat.S_IMODE(os.stat(root).st_mode), '
                    mode_check=$mode_check'stat.S_IMODE(os.stat(file).st_mode)))'
                    "$FAKE_C_PYTHON" -I -S -c "$mode_check" "$snapshot" >>"$FAKE_LOG"
                    ;;
            esac
        done
        exit 0 ;;
    "-m so101.platform_info")
        case "${3-}" in
            --data-dir) echo "$FAKE_DATA_DIR" ;;
            --init-env | --set-env)
                echo "settings file: $FAKE_DATA_DIR/.env (unchanged)"
                exit "${FAKE_SETTINGS_EXIT:-0}"
                ;;
            *) echo "fake platform report" ;;
        esac
        exit 0 ;;
esac
case "$*" in
    *SO101_SELECTED_PACKAGE_CHECK*)
        printf '%s' "$FAKE_FINAL_DISTS" >"$FAKE_C_DISTS"
        ;;
esac
if [ "$isolated" -eq 1 ]; then
    if [ "$no_site" -eq 1 ]; then
        "$FAKE_C_PYTHON" -I -S "$@"
    else
        "$FAKE_C_PYTHON" -I "$@"
    fi
else
    "$FAKE_C_PYTHON" "$@"
fi
status=$?
if [ "$status" -eq 0 ] && [ -n "${FAKE_MUTATE_SOURCE_AFTER_CHECK:-}" ]; then
    case "$*" in
        *runpy.run_path*)
            printf 'mutated after verification\n' >"$FAKE_MUTATE_SOURCE_AFTER_CHECK"
            ;;
    esac
fi
if [ "$status" -eq 0 ] && [ -n "${FAKE_MUTATE_SOURCE_SNAPSHOT_AFTER_HASH:-}" ]; then
    case "$*" in
        *SO101_SOURCE_TREE_SHA256*SO101_INPUT_FINAL_HASH*)
            for last do :; done
            chmod u+w "$last/pyproject.toml"
            printf '\n# changed after validation\n' >>"$last/pyproject.toml"
            ;;
    esac
fi
if [ "$status" -eq 0 ] && [ -n "${FAKE_MUTATE_STAGED_AFTER_HASH:-}" ]; then
    case "$*" in
        *SO101_INPUT_FINAL_HASH*)
            for last do :; done
            chmod u+w "$last"
            printf 'replaced after validation' >"$last"
            ;;
    esac
fi
exit "$status"
"""

# The venv's so101 command: --help works unless FAKE_SO101_HELP_EXIT says otherwise;
# doctor prints a FAIL row and exits FAKE_DOCTOR_EXIT.
FULL_VENV_SO101 = """\
#!/bin/sh
printf 'so101 %s\\n' "$*" >>"$FAKE_LOG"
case "$1" in
    --help) exit "${FAKE_SO101_HELP_EXIT:-0}" ;;
    doctor)
        echo "FAIL  Serial access  this login session started before you joined 'dialout'"
        exit "${FAKE_DOCTOR_EXIT:-0}" ;;
esac
exit 0
"""


def full_run(
    tmp_path: Path,
    *args: str,
    doctor_exit: int = 0,
    settings_exit: int = 0,
    so101: str = "works",
    data_dir: Path | None = None,
    mutate_staged_after_hash: bool = False,
    mutate_source_after_check: Path | None = None,
    mutate_source_snapshot_after_hash: bool = False,
    final_dists: str | None = None,
    launcher: bool = True,
    **kwargs: object,
) -> subprocess.CompletedProcess[str]:
    """install.sh for real (not --dry-run), launcher included, on a faked Ubuntu whose
    venv already exists. ``so101``: "works", "missing" or "fails" (its --help exits 1).
    Everything the venv runs is logged to ``tmp_path / "fake.log"``."""

    venv = tmp_path / "app" / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    python = venv / "bin" / "python"
    selected_data_dir = data_dir or tmp_path / "data"
    selected_profile = str(kwargs.get("profile") or "cpu")
    label = selected_profile if selected_profile in {"cpu", "cu126", "cu128"} else ""
    suffix = f"+{label}" if label else ""
    expected_dists = f"torch=2.11.0{suffix},torchvision=0.26.0{suffix},torchcodec=0.11.1"
    fake_c_python = fake_python(tmp_path, "3.12.3")
    fake_c_dists = tmp_path / "fake_dists"
    fake_c_dists.write_text(str(kwargs.get("dists", "")), encoding="utf-8")
    selected_script = Path(kwargs.get("script", INSTALL_SH))
    selected_source = selected_script.parents[1]
    selected_version = tomllib.loads(
        (selected_source / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    (tmp_path / "fake_suite_version").write_text(str(selected_version), encoding="utf-8")
    final_versions = final_dists if final_dists is not None else expected_dists
    replacements = {
        "__FAKE_LOG__": shlex.quote(str(tmp_path / "fake.log")),
        "__FAKE_DATA_DIR__": shlex.quote(str(selected_data_dir)),
        "__FAKE_C_PYTHON__": shlex.quote(str(fake_c_python)),
        "__FAKE_C_DISTS__": shlex.quote(str(fake_c_dists)),
        "__FAKE_SETTINGS_EXIT__": shlex.quote(str(settings_exit)),
        "__FAKE_MUTATE_STAGED_AFTER_HASH__": shlex.quote("1" if mutate_staged_after_hash else ""),
        "__FAKE_MUTATE_SOURCE_AFTER_CHECK__": shlex.quote(str(mutate_source_after_check or "")),
        "__FAKE_MUTATE_SOURCE_SNAPSHOT_AFTER_HASH__": shlex.quote(
            "1" if mutate_source_snapshot_after_hash else ""
        ),
        "__FAKE_FINAL_DISTS__": shlex.quote(final_versions),
    }
    python_text = FULL_VENV_PYTHON
    for marker, value in replacements.items():
        python_text = python_text.replace(marker, value)
    python.write_text(python_text, encoding="utf-8")
    python.chmod(0o755)
    if so101 != "missing":
        command = venv / "bin" / "so101"
        command.write_text(FULL_VENV_SO101, encoding="utf-8")
        command.chmod(0o755)
    caller_env = kwargs.pop("extra_env", None)
    extra_env = {
        "FAKE_LOG": str(tmp_path / "fake.log"),
        "FAKE_DATA_DIR": str(selected_data_dir),
        "FAKE_C_PYTHON": str(fake_c_python),
        "FAKE_DOCTOR_EXIT": str(doctor_exit),
        "FAKE_SETTINGS_EXIT": str(settings_exit),
        "FAKE_SO101_HELP_EXIT": "1" if so101 == "fails" else "0",
        "SO101_HOME": str(selected_data_dir),
        "FAKE_MUTATE_STAGED_AFTER_HASH": "1" if mutate_staged_after_hash else "",
        "FAKE_MUTATE_SOURCE_AFTER_CHECK": str(mutate_source_after_check or ""),
        "FAKE_MUTATE_SOURCE_SNAPSHOT_AFTER_HASH": (
            "1" if mutate_source_snapshot_after_hash else ""
        ),
        "FAKE_FINAL_DISTS": final_versions,
    }
    if caller_env is not None:
        extra_env.update(caller_env)  # type: ignore[arg-type]
    return dry_run(tmp_path, *args, dry=False, launcher=launcher, extra_env=extra_env, **kwargs)  # type: ignore[arg-type]


def fake_log(tmp_path: Path) -> str:
    log = tmp_path / "fake.log"
    return log.read_text(encoding="utf-8") if log.exists() else ""


def desktop_file(tmp_path: Path) -> Path:
    return tmp_path / "home" / ".local" / "share" / "applications" / "so101-suite.desktop"


@needs_bash
def test_a_failing_doctor_doesnt_stop_the_launcher_or_the_run(tmp_path):
    """A first install can't pass the checks that need a fresh login (dialout, PATH)."""

    result = full_run(tmp_path, doctor_exit=1)
    text = output(result)
    assert result.returncode == 0, text
    desktop = desktop_file(tmp_path)
    assert desktop.is_file()
    desktop_text = desktop.read_text(encoding="utf-8")
    assert "-m so101.launcher --home " in desktop_text
    assert " --pause-on-error" in desktop_text
    assert "so101 doctor\n" in fake_log(tmp_path)
    # The launcher is written before the final check, and the run still ends with the summary.
    assert text.index("==> launcher: SO-101 Suite in the app menu") < text.index("==> so101 doctor")
    assert "so101 doctor reported a FAIL" in text
    assert "the dialout group" in text and "~/.local/bin on PATH" in text
    assert "/dev/so101-follower" in text
    assert "log out and back in, then run so101 doctor again" in text
    assert "==> done" in result.stdout
    assert "error:" not in result.stderr


@needs_bash
def test_a_passing_doctor_gets_no_login_advice(tmp_path):
    result = full_run(tmp_path, doctor_exit=0)
    text = output(result)
    assert result.returncode == 0, text
    assert desktop_file(tmp_path).is_file()
    assert "so101 doctor reported a FAIL" not in text and "log out and back in, then" not in text
    assert "==> done" in result.stdout


@needs_bash
def test_live_installer_uses_the_exact_default_dependency_plan(tmp_path):
    """Hermetic integration contract: exercise the real script and capture every pip call."""

    result = full_run(tmp_path, profile="cpu")
    assert result.returncode == 0, output(result)
    wheel = next(tmp_path.glob("so101-*.whl"))
    calls = [
        line
        for line in fake_log(tmp_path).splitlines()
        if line.startswith("python -I -m pip install ")
    ]
    assert calls[:3] == [
        "python -I -m pip install --disable-pip-version-check --index-url "
        "https://pypi.org/simple --upgrade pip",
        "python -I -m pip install --disable-pip-version-check --index-url "
        "https://download.pytorch.org/whl/cpu torch==2.11.0+cpu "
        "torchvision==0.26.0+cpu",
        "python -I -m pip install --disable-pip-version-check --index-url "
        "https://pypi.org/simple torchcodec==0.11.1",
    ]
    suite_prefix = (
        "python -I -m pip install --disable-pip-version-check --index-url https://pypi.org/simple "
    )
    refresh_prefix = (
        "python -I -m pip install --disable-pip-version-check --index-url "
        "https://pypi.org/simple --no-deps --force-reinstall "
    )
    assert calls[3].startswith(suite_prefix) and calls[3].endswith("[robot,sim,train]")
    staged = calls[3][len(suite_prefix) : -len("[robot,sim,train]")]
    assert calls[4] == refresh_prefix + staged
    assert Path(staged).name == wheel.name
    assert Path(staged).parent.name == "wheel"
    assert Path(staged).parent.parent.name.startswith("so101-install-private.")
    assert str(wheel) not in "\n".join(calls[3:])
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert set(("robot", "sim", "train")) <= extras.keys()


@needs_bash
def test_a_verified_wheel_mutated_before_pip_is_refused(tmp_path):
    result = full_run(tmp_path, mutate_staged_after_hash=True)
    assert result.returncode == 1, output(result)
    assert "private verified wheel changed after validation" in result.stderr
    log = fake_log(tmp_path)
    assert "[robot,sim,train]" not in log
    assert "--no-deps --force-reinstall" not in log


@needs_bash
def test_package_build_environment_is_an_explicit_transport_only_allowlist(tmp_path):
    hostile = tmp_path / "host"
    result = full_run(
        tmp_path,
        profile="cpu",
        extra_env={
            "HOME": str(hostile),
            "XDG_CONFIG_HOME": str(hostile / "xdg"),
            "PIP_CONFIG_FILE": str(hostile / "pip.conf"),
            "NETRC": str(hostile / "netrc"),
            "PIP_KEYRING_PROVIDER": "import",
            "PIP_INDEX_URL": "https://attacker.invalid/simple",
            "PIP_EXTRA_INDEX_URL": "https://also-attacker.invalid/simple",
            "PIP_CERT": "/tmp/corporate-ca.pem",
            "PYTHONPATH": "/tmp/sitecustomize-injection",
            "UV_INDEX_URL": "https://uv-attacker.invalid/simple",
            "SO101_BACKEND": "real",
            "HF_TOKEN": "hf-secret",
            "GITHUB_TOKEN": "github-secret",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "SSH_AUTH_SOCK": "/tmp/credential-agent.sock",
            "UNRELATED_SAFE": "must-also-be-scrubbed",
        },
    )
    assert result.returncode == 0, output(result)
    log = fake_log(tmp_path)
    env_lines = [line for line in log.splitlines() if line.startswith("pip-env ")]
    assert env_lines
    assert all("HOME=/tmp/so101-install-private." in line for line in env_lines)
    assert all(
        "/package-home XDG_CONFIG_HOME=/tmp/so101-install-private." in line for line in env_lines
    )
    assert all("PIP_CONFIG_FILE=/dev/null" in line for line in env_lines)
    assert all("NETRC=/dev/null" in line for line in env_lines)
    assert all("PIP_KEYRING_PROVIDER=disabled" in line for line in env_lines)
    assert all("PIP_INDEX_URL= PIP_EXTRA_INDEX_URL=" in line for line in env_lines)
    assert all("PIP_CERT=/tmp/corporate-ca.pem" in line for line in env_lines)
    for forbidden in (
        "attacker.invalid",
        "sitecustomize-injection",
        "SO101_BACKEND=real",
        "hf-secret",
        "github-secret",
        "aws-secret",
        "credential-agent.sock",
        "must-also-be-scrubbed",
    ):
        assert forbidden not in log
    installs = [line for line in log.splitlines() if line.startswith("python -I -m pip install ")]
    assert installs and all("--index-url " in line for line in installs)


@needs_bash
def test_live_install_verifies_torch_vision_and_codec_versions_after_suite_install(tmp_path):
    result = full_run(
        tmp_path,
        profile="cpu",
        final_dists="torch=2.11.0+cpu,torchvision=0.26.0+cu126,torchcodec=0.11.0",
    )
    assert result.returncode == 1, output(result)
    assert "installed PyTorch packages do not match the selected cpu profile" in result.stderr
    assert "torchvision is 0.26.0+cu126, expected 0.26.0+cpu" in result.stderr
    assert "torchcodec is 0.11.0, expected 0.11.1" in result.stderr
    assert not desktop_file(tmp_path).exists()


@needs_bash
def test_live_source_install_uses_one_private_read_only_snapshot(tmp_path):
    script = fake_repo(tmp_path)
    repo = script.parents[1]
    expected_head = "e" * 40
    (repo / ".git").mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    shim(bin_dir, "git", f"echo {expected_head}")
    source_file = repo / "src" / "so101" / "__init__.py"
    result = full_run(
        tmp_path,
        "--allow-no-ui",
        script=script,
        wheel_arg=False,
        launcher=False,
        mutate_source_after_check=source_file,
    )
    assert result.returncode == 0, output(result)
    assert source_file.read_text(encoding="utf-8") == "mutated after verification\n"
    suite_calls = [
        line
        for line in fake_log(tmp_path).splitlines()
        if line.startswith("python -I -m pip install ") and "[robot,sim,train]" in line
    ]
    assert len(suite_calls) == 1
    assert "/so101-install-private." in suite_calls[0]
    assert "/source[robot,sim,train]" in suite_calls[0]
    assert str(repo) not in suite_calls[0]
    assert "source-modes 500 400" in fake_log(tmp_path)
    assert f"--expected-git-head {expected_head}" in fake_log(tmp_path)


@needs_bash
def test_live_source_install_refuses_a_snapshot_mutated_after_validation(tmp_path):
    script = fake_repo(tmp_path)
    result = full_run(
        tmp_path,
        "--allow-no-ui",
        script=script,
        wheel_arg=False,
        launcher=False,
        mutate_source_snapshot_after_hash=True,
    )
    assert result.returncode == 1, output(result)
    assert "private verified source snapshot changed after validation" in result.stderr
    log = fake_log(tmp_path)
    assert "[robot,sim,train]" not in log
    assert "--no-deps --force-reinstall" not in log


@needs_bash
def test_linux_launcher_pins_the_custom_data_home(tmp_path):
    data_dir = tmp_path / 'custom "data" % with $dollars `ticks`'
    result = full_run(tmp_path, data_dir=data_dir)
    assert result.returncode == 0, output(result)
    desktop = desktop_file(tmp_path).read_text(encoding="utf-8")
    escaped = (
        str(data_dir)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("`", "\\`")
        .replace("$", "\\$")
        .replace("%", "%%")
    )
    assert f'--home "{escaped}"' in desktop


@needs_bash
def test_macos_launcher_pins_the_custom_data_home(tmp_path):
    data_dir = tmp_path / 'custom "data" with $dollars `ticks`'
    result = full_run(
        tmp_path,
        data_dir=data_dir,
        os_name="Darwin",
        arch="arm64",
        profile="mps",
    )
    assert result.returncode == 0, output(result)
    command = tmp_path / "home" / "Applications" / "SO-101 Suite.command"
    content = command.read_text(encoding="utf-8")
    quoted = subprocess.run(
        [BASH or "bash", "-c", 'printf %q "$1"', "quote", str(data_dir)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert f"--home {quoted}" in content


@needs_bash
def test_launcher_refuses_a_data_home_that_could_inject_another_line(tmp_path):
    data_dir = tmp_path / "data\nExec=malicious-command"
    result = full_run(tmp_path, data_dir=data_dir)
    assert result.returncode == 1, output(result)
    assert "cannot be encoded safely in a launcher" in result.stderr
    assert not desktop_file(tmp_path).exists()


@needs_bash
def test_an_unusable_data_folder_fails_instead_of_becoming_doctor_advice(tmp_path):
    data = tmp_path / "data"
    data.write_text("not a directory", encoding="utf-8")

    result = full_run(tmp_path, doctor_exit=1)

    assert result.returncode == 1
    assert f"the data folder {data} cannot be created or used" in result.stderr
    assert "so101 doctor reported a FAIL" not in output(result)
    assert "the install itself is complete" not in output(result)
    assert "so101 doctor\n" not in fake_log(tmp_path)


@needs_bash
def test_an_unusable_settings_path_fails_before_doctor(tmp_path):
    data = tmp_path / "data"
    env = data / ".env"
    env.mkdir(parents=True)
    marker = env / "keep"
    marker.write_text("unchanged", encoding="utf-8")

    result = full_run(tmp_path, settings_exit=1)

    assert result.returncode == 1
    assert f"the data folder {data} cannot be used for settings" in result.stderr
    assert env.is_dir() and marker.read_text(encoding="utf-8") == "unchanged"
    assert "so101 doctor\n" not in fake_log(tmp_path)
    assert "the install itself is complete" not in output(result)


@pytest.mark.parametrize("so101", ["missing", "fails"])
@needs_bash
def test_a_broken_so101_command_still_fails_the_install(tmp_path, so101):
    result = full_run(tmp_path, so101=so101, doctor_exit=0)
    assert result.returncode == 1, output(result)
    assert "error: " in result.stderr and "so101" in result.stderr
    assert "--fresh" in result.stderr
    assert not desktop_file(tmp_path).exists()  # the suite doesn't run: no launcher for it
    assert "so101 doctor" not in fake_log(tmp_path)


@needs_bash
def test_late_steps_that_fail_dont_stop_the_run_before_the_check_and_summary(tmp_path):
    """~/.local is a file: neither the PATH links nor the launcher can be written."""

    (tmp_path / "home").mkdir()
    (tmp_path / "home" / ".local").write_text("not a folder", encoding="utf-8")
    result = full_run(tmp_path, doctor_exit=1)
    text = output(result)
    assert result.returncode == 0, text
    assert "couldn't create" in result.stderr and ".local/bin" in result.stderr
    assert "couldn't write the launcher" in result.stderr
    assert "so101 doctor\n" in fake_log(tmp_path)
    assert "==> done" in result.stdout


@needs_bash
def test_a_dry_run_writes_the_launcher_before_running_doctor(tmp_path):
    result = dry_run(tmp_path, launcher=True)
    assert result.returncode == 0, output(result)
    out = result.stdout
    venv_bin = tmp_path / "app" / "venv" / "bin"
    assert f"+ {venv_bin / 'so101'} --help\n" in out
    assert out.index("+ write ") < out.index(f"+ {venv_bin / 'so101'} doctor\n")


# --- re-runs: another PyTorch build already in the venv ----------------------------------

CPU_BUILD = "torch=2.11.0+cpu,torchvision=0.26.0+cpu"
CU128_BUILD = "torch=2.11.0+cu128,torchvision=0.26.0+cu128"
CU130_BUILD = "torch=2.11.0,torchvision=0.26.0"


@pytest.mark.parametrize(
    ("installed", "profile", "replaced"),
    [
        (CPU_BUILD, "cu130", "torch 2.11.0+cpu, torchvision 0.26.0+cpu"),
        (CU128_BUILD, "cu130", "torch 2.11.0+cu128, torchvision 0.26.0+cu128"),
        (CU130_BUILD, "cu130", None),
        (CPU_BUILD, "cpu", None),
        (CU130_BUILD, "cpu", "torch 2.11.0, torchvision 0.26.0"),
        ("torch=2.11.0,torchvision=0.26.0+cpu", "cu130", "torchvision 0.26.0+cpu"),
        ("", "cu130", None),  # nothing installed yet
    ],
)
@needs_bash
def test_a_rerun_replaces_another_torch_build_pip_would_keep(
    tmp_path, installed, profile, replaced
):
    """pip and uv take 2.11.0+cpu as satisfying torch==2.11.0 (PEP 440 local versions)."""

    venv = make_venv(tmp_path, "3.12.3")
    result = dry_run(tmp_path, profile=profile, nvidia="580.95.05", dists=installed)
    text = output(result)
    assert result.returncode == 0, text
    python = venv / "bin" / "python"
    if replaced is None:
        assert "pip uninstall" not in result.stdout and "replacing" not in text
        return
    names = " ".join(item.split(" ")[0] for item in replaced.split(", "))
    uninstall = f"+ {python} -I -m pip uninstall --disable-pip-version-check -y {names}\n"
    assert uninstall in result.stdout
    assert f"replacing {replaced}: not the {profile} build" in text
    first_torch_install = result.stdout.index("torch==", result.stdout.index("step 1/3"))
    assert result.stdout.index(uninstall) < first_torch_install


@needs_bash
def test_a_rerun_on_macos_replaces_another_build_with_uv(tmp_path):
    venv = make_venv(tmp_path, "3.12.3")
    result = dry_run(
        tmp_path, os_name="Darwin", arch="arm64", python=None, uv=True, dists=CPU_BUILD
    )
    text = output(result)
    assert result.returncode == 0, text
    assert "using:    uv" in text
    uv = tmp_path / "bin" / "uv"
    uninstall = f"+ {uv} pip uninstall --python {venv / 'bin' / 'python'} torch torchvision\n"
    assert uninstall in result.stdout
    assert "not the mps build" in text
    assert result.stdout.index(uninstall) < result.stdout.index(f"+ {uv} pip install")


@needs_bash
def test_a_real_rerun_uninstalls_before_installing_the_new_build(tmp_path):
    result = full_run(tmp_path, profile="cu130", nvidia="580.95.05", dists=CPU_BUILD)
    assert result.returncode == 0, output(result)
    log = fake_log(tmp_path)
    uninstall = "python -I -m pip uninstall --disable-pip-version-check -y torch torchvision\n"
    install = (
        "python -I -m pip install --disable-pip-version-check --index-url "
        "https://pypi.org/simple torch==2.11.0 torchvision==0.26.0\n"
    )
    assert uninstall in log and install in log
    assert log.index(uninstall) < log.index(install)


# --- set -euo pipefail: what a fresh Ubuntu answers must not stop the script --------------


@needs_bash
def test_nvidia_smi_failing_with_a_message_counts_as_no_driver(tmp_path):
    """nvidia-smi prints its error on stdout when the kernel module isn't loaded yet."""

    body = (
        "echo \"NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA "
        'driver. Make sure that the latest NVIDIA driver is installed and running."\nexit 9'
    )
    result = dry_run(tmp_path, profile=None, nvidia_body=body)
    text = output(result)
    assert result.returncode == 0, text
    assert "no NVIDIA driver answered: the cpu build" in text
    assert "MOK enrollment" in result.stderr
    assert "too old" not in text and "NVIDIA-SMI" not in text
    refused = dry_run(tmp_path, profile="cuda", nvidia_body=body)
    assert refused.returncode == 1 and "nvidia-smi doesn't answer" in refused.stderr


@needs_bash
def test_a_failing_df_doesnt_stop_the_install(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim(bin_dir, "df", "echo 'df: Transport endpoint is not connected' >&2\nexit 1")
    result = dry_run(tmp_path)
    assert result.returncode == 0, output(result)
    assert "only " not in result.stderr  # no free-space warning from a failed df


@needs_bash
def test_a_package_installed_for_two_architectures_counts_as_installed(tmp_path):
    """dpkg-query prints one status per architecture (libgl1:amd64 and libgl1:i386)."""

    result = dry_run(tmp_path, dpkg_body="printf 'install ok installed\\ninstall ok installed\\n'")
    assert result.returncode == 0, output(result)
    assert "system packages: all installed (ffmpeg libusb-1.0-0 git libegl1 libgl1)" in (
        result.stdout
    )


# --- which wheel: dist/ must match the source folder, --wheel its own METADATA -------------


def fake_repo(tmp_path: Path, version: str = "0.2.1") -> Path:
    """A source folder with install.sh, a pyproject.toml of ``version``, sources and dist/;
    its install/install.sh (to pass as ``script``) is returned."""

    repo = tmp_path / "repo"
    (repo / "install").mkdir(parents=True)
    shutil.copy(INSTALL_SH, repo / "install" / "install.sh")
    shutil.copy(INSTALL_SH.with_name("verify_input.py"), repo / "install" / "verify_input.py")
    (repo / "pyproject.toml").write_text(
        f'[build-system]\nrequires = ["hatchling"]\n\n[project]\nname = "so101"\n'
        f'version = "{version}"\ndependencies = ["core-package>=1"]\n\n'
        '[project.optional-dependencies]\nrobot = ["robot-package==1"]\n'
        'sim = ["sim-package==1"]\ntrain = ["train-package==1"]\n'
        'realsense = ["camera-package==1"]\ncustom = ["custom-package==1"]\n\n'
        '[tool.other]\nversion = "9.9.9"\n',
        encoding="utf-8",
    )
    for source in ("src/so101/__init__.py", "frontend/src/main.tsx"):
        (repo / source).parent.mkdir(parents=True)
        (repo / source).write_text("# source\n", encoding="utf-8")
    (repo / "dist").mkdir()
    return repo / "install" / "install.sh"


def set_mtime(path: Path, seconds_ago: float) -> None:
    import time

    when = time.time() - seconds_ago
    os.utime(path, (when, when))


@needs_bash
def test_a_dist_wheel_older_than_the_source_is_refused(tmp_path):
    script = fake_repo(tmp_path)
    repo = script.parents[1]
    wheel = make_wheel(repo / "dist" / "so101-0.2.1-py3-none-any.whl", "0.2.1")
    set_mtime(wheel, 3600)  # built an hour ago; the sources changed since
    result = dry_run(tmp_path, script=script, wheel_arg=False)
    assert result.returncode == 1, output(result)
    assert f"error: {wheel} is older than the source folder" in result.stderr
    assert "make frontend-deps build-ui wheel" in result.stderr and "--wheel FILE" in result.stderr
    assert "+ " not in result.stdout  # refused before doing anything


@needs_bash
def test_a_fresh_dist_wheel_of_the_source_version_is_installed(tmp_path):
    script = fake_repo(tmp_path)
    repo = script.parents[1]
    for source in ("src/so101/__init__.py", "frontend/src/main.tsx"):
        set_mtime(repo / source, 3600)
    wheel = make_wheel(repo / "dist" / "so101-0.2.1-py3-none-any.whl", "0.2.1", source=repo)
    cache = repo / "src" / "so101" / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"")  # newer, but a cache
    (repo / "src" / "so101" / ".DS_Store").write_bytes(b"")
    result = dry_run(tmp_path, script=script, wheel_arg=False)
    text = output(result)
    assert result.returncode == 0, text
    assert f"==> suite:    wheel {wheel} [robot,sim,train]" in result.stdout
    assert "==> input:    so101 0.2.1; wheel matches this source" in result.stdout


@needs_bash
def test_a_dist_wheel_of_another_version_is_ignored(tmp_path):
    """dist/so101-0.2.0 left over from before a version bump: the source folder is 0.2.1."""

    script = fake_repo(tmp_path)
    repo = script.parents[1]
    old = make_wheel(repo / "dist" / "so101-0.2.0-py3-none-any.whl", "0.2.0", source=repo)
    result = dry_run(tmp_path, script=script, wheel_arg=False)
    text = output(result)
    assert result.returncode == 1, text
    assert f"ignoring {old}: this source folder is version 0.2.1" in result.stderr
    assert "has no built web UI" in result.stderr
    assert "make frontend-deps build-ui wheel" in result.stderr

    backend = dry_run(
        tmp_path,
        "--allow-no-ui",
        script=script,
        wheel_arg=False,
    )
    assert backend.returncode == 0, output(backend)
    assert f"==> suite:    source folder {repo} [robot,sim,train]" in backend.stdout


@needs_bash
def test_an_explicit_wheel_must_match_its_metadata(tmp_path):
    renamed = make_wheel(tmp_path / "so101-0.2.1-py3-none-any.whl", "0.2.0")
    result = dry_run(tmp_path, "--wheel", str(renamed))
    assert result.returncode == 1, output(result)
    assert f"error: {renamed} is named version 0.2.1, but its METADATA says 0.2.0" in result.stderr
    assert "+ " not in result.stdout


@pytest.mark.parametrize(
    ("name", "content", "error"),
    [
        ("so101-0.2.0-py3-none-any.whl", b"not a zip", "it is not a readable wheel"),
        (
            "so101-0.2.0-cp312-cp312-linux_x86_64.whl",
            None,
            "isn't the supported pure-Python suite wheel name",
        ),
        ("suite.whl", None, "isn't named like a suite wheel"),
    ],
)
@needs_bash
def test_an_explicit_wheel_that_isnt_a_suite_wheel_is_refused(tmp_path, name, content, error):
    wheel = tmp_path / "given" / name
    wheel.parent.mkdir()
    if content is None:
        make_wheel(wheel, "0.2.0")
    else:
        wheel.write_bytes(content)
    result = dry_run(tmp_path, "--wheel", str(wheel))
    assert result.returncode == 1, output(result)
    assert error in result.stderr


@needs_bash
def test_requirement_special_characters_in_an_install_path_are_refused_before_changes(tmp_path):
    folder = tmp_path / "wheel;python_version>='0'"
    folder.mkdir()
    project_version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["version"]
    wheel = folder / f"so101-{project_version}-py3-none-any.whl"
    make_wheel(wheel, project_version)
    result = dry_run(tmp_path, "--wheel", str(wheel))
    assert result.returncode == 1, output(result)
    assert "path contains a character pip can misparse" in result.stderr
    assert "+ " not in result.stdout


@needs_bash
def test_an_explicit_wheel_must_match_the_source_folder_version(tmp_path):
    """An explicit wheel cannot silently come from another source revision/version."""

    script = fake_repo(tmp_path, version="0.3.0")
    wheel = make_wheel(tmp_path / "so101-0.2.1-py3-none-any.whl", "0.2.1")
    set_mtime(wheel, 3600)
    result = dry_run(tmp_path, "--wheel", str(wheel), script=script, wheel_arg=False)
    assert result.returncode == 1, output(result)
    assert "source folder is version 0.3.0" in result.stderr


@needs_bash
def test_an_explicit_workstation_wheel_must_carry_a_current_ui(tmp_path):
    script = fake_repo(tmp_path)
    repo = script.parents[1]
    no_ui = tmp_path / "no-ui" / "so101-0.2.1-py3-none-any.whl"
    no_ui.parent.mkdir()
    make_wheel(no_ui, "0.2.1", source=repo, ui=False)
    refused = dry_run(tmp_path, "--wheel", str(no_ui), script=script, wheel_arg=False)
    assert refused.returncode == 1, output(refused)
    assert "has no built web UI" in refused.stderr

    allowed = dry_run(
        tmp_path,
        "--wheel",
        str(no_ui),
        "--allow-no-ui",
        script=script,
        wheel_arg=False,
    )
    assert allowed.returncode == 0, output(allowed)

    launcher_refused = dry_run(
        tmp_path,
        "--wheel",
        str(no_ui),
        "--allow-no-ui",
        script=script,
        wheel_arg=False,
        launcher=True,
    )
    assert launcher_refused.returncode == 1
    assert "--allow-no-ui requires --no-launcher" in launcher_refused.stderr


@needs_bash
def test_wheel_with_a_stale_or_altered_ui_is_refused(tmp_path):
    script = fake_repo(tmp_path)
    repo = script.parents[1]
    wheel = tmp_path / "so101-0.2.1-py3-none-any.whl"
    make_wheel(wheel, "0.2.1", source=repo, ui_sources_digest="0" * 64)
    result = dry_run(tmp_path, "--wheel", str(wheel), script=script, wheel_arg=False)
    assert result.returncode == 1, output(result)
    assert "web UI was built from other frontend sources" in result.stderr


@needs_bash
def test_wheel_code_must_match_the_source_folder(tmp_path):
    script = fake_repo(tmp_path)
    repo = script.parents[1]
    wheel = tmp_path / "so101-0.2.1-py3-none-any.whl"
    make_wheel(
        wheel,
        "0.2.1",
        source=repo,
        package_override={"so101/__init__.py": b'__version__ = "9.9.9"\n'},
    )
    result = dry_run(tmp_path, "--wheel", str(wheel), script=script, wheel_arg=False)
    assert result.returncode == 1, output(result)
    assert "does not match this source folder" in result.stderr


@needs_bash
def test_mismatched_wheel_is_rejected_without_running_inherited_sitecustomize(tmp_path):
    script = fake_repo(tmp_path)
    repo = script.parents[1]
    sentinel = tmp_path / "SITECUSTOMIZE_EXECUTED"
    injected = tmp_path / "python-injection"
    injected.mkdir()
    (injected / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n",
        encoding="utf-8",
    )
    wheel = tmp_path / "mismatch" / "so101-0.2.1-py3-none-any.whl"
    wheel.parent.mkdir()
    make_wheel(
        wheel,
        "0.2.1",
        source=repo,
        package_override={"so101/__init__.py": b"# mismatched bytes\n"},
    )

    result = full_run(
        tmp_path,
        "--wheel",
        str(wheel),
        script=script,
        wheel_arg=False,
        launcher=False,
        extra_env={"PYTHONPATH": str(injected), "PYTHONUSERBASE": str(injected)},
    )

    assert result.returncode == 1, output(result)
    assert "does not match this source folder" in result.stderr
    assert not sentinel.exists()
    helper_flags = [
        line for line in fake_log(tmp_path).splitlines() if line.startswith("helper-flags ")
    ]
    assert len(helper_flags) >= 3
    assert set(helper_flags) == {"helper-flags isolated=1 no_site=1"}


@needs_bash
def test_wheel_dependencies_must_match_the_source_default_extras(tmp_path):
    script = fake_repo(tmp_path)
    repo = script.parents[1]
    wheel = tmp_path / "so101-0.2.1-py3-none-any.whl"
    make_wheel(wheel, "0.2.1", source=repo)
    pyproject = repo / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text(encoding="utf-8").replace(
            'train = ["train-package==1"]',
            'train = ["train-package==1", "required-after-wheel==2"]',
        ),
        encoding="utf-8",
    )

    result = dry_run(tmp_path, "--wheel", str(wheel), script=script, wheel_arg=False)
    assert result.returncode == 1, output(result)
    assert "does not exactly match this source folder's dependencies" in result.stderr


@pytest.mark.parametrize(
    "metadata_extra",
    [
        (
            "Provides-Extra: bogus",
            "Requires-Dist: injected-package; extra == 'bogus' or python_version >= '0'",
        ),
        ("Requires-Dist: injected-package; python_version >= '0'",),
    ],
)
@needs_bash
def test_wheel_rejects_dependencies_hidden_behind_unknown_or_active_markers(
    tmp_path, metadata_extra
):
    script = fake_repo(tmp_path)
    repo = script.parents[1]
    wheel = tmp_path / "so101-0.2.1-py3-none-any.whl"
    make_wheel(wheel, "0.2.1", source=repo, metadata_extra=metadata_extra)
    result = dry_run(tmp_path, "--wheel", str(wheel), script=script, wheel_arg=False)
    assert result.returncode == 1, output(result)
    assert "does not exactly match this source folder's" in result.stderr


@needs_bash
def test_selected_realsense_and_custom_extras_must_exist_and_match_metadata(tmp_path):
    script = fake_repo(tmp_path)
    repo = script.parents[1]
    good = tmp_path / "so101-0.2.1-py3-none-any.whl"
    make_wheel(good, "0.2.1", source=repo)
    selected = dry_run(
        tmp_path,
        "--wheel",
        str(good),
        "--extras",
        "robot,realsense,custom",
        script=script,
        wheel_arg=False,
    )
    assert selected.returncode == 0, output(selected)

    missing = tmp_path / "missing" / good.name
    missing.parent.mkdir()
    make_wheel(missing, "0.2.1", source=repo, omit_extra="custom")
    refused = dry_run(
        tmp_path,
        "--wheel",
        str(missing),
        "--extras",
        "custom",
        script=script,
        wheel_arg=False,
    )
    assert refused.returncode == 1
    assert "provided extras" in refused.stderr or "dependencies" in refused.stderr

    unknown = dry_run(
        tmp_path,
        "--wheel",
        str(good),
        "--extras",
        "robot,bogus",
        script=script,
        wheel_arg=False,
    )
    assert unknown.returncode == 1
    assert "selected suite extras are not declared" in unknown.stderr


@pytest.mark.parametrize(
    ("wheel_kwargs", "message"),
    [
        ({"extra_script": ("backdoor", "evil.module:main")}, "executable entry points"),
        ({"wheel_metadata_extra": "Build: 7\n"}, "WHEEL metadata"),
        ({"dist_info_extra": {"top_level.txt": b"evil\n"}}, "unexpected dist-info"),
        ({"record_override": b"../../outside,,\n"}, "RECORD"),
    ],
)
@needs_bash
def test_wheel_rejects_unexpected_executable_or_dist_info_metadata(tmp_path, wheel_kwargs, message):
    script = fake_repo(tmp_path)
    repo = script.parents[1]
    wheel = tmp_path / "so101-0.2.1-py3-none-any.whl"
    make_wheel(wheel, "0.2.1", source=repo, **wheel_kwargs)
    result = dry_run(tmp_path, "--wheel", str(wheel), script=script, wheel_arg=False)
    assert result.returncode == 1, output(result)
    assert message in result.stderr


# --- the realsense extra in pyproject.toml ------------------------------------------------


def realsense_extra() -> list[str] | None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data.get("project", {}).get("optional-dependencies", {})
    found = extras.get("realsense")
    return None if found is None else list(found)


@pytest.mark.parametrize(
    ("sys_platform", "os_kind", "lower"),
    [("linux", "linux", "2.56.4"), ("win32", "windows", "2.56.4"), ("darwin", "macos", "2.54.2")],
)
def test_the_realsense_extra_matches_what_doctor_and_the_installers_expect(
    sys_platform, os_kind, lower
):
    extra = realsense_extra()
    if extra is None:
        pytest.skip("pyproject.toml has no realsense extra yet (the camera stream adds it)")
    from packaging.requirements import Requirement

    active = [
        requirement
        for requirement in map(Requirement, extra)
        if requirement.marker is None or requirement.marker.evaluate({"sys_platform": sys_platform})
    ]
    assert [r.name for r in active] == [pi.realsense_distribution(os_kind)]
    # Below 2.57: the installers' Python ranges are those of the 2.56 wheels.
    assert str(active[0].specifier) in {f">={lower},<2.57.0", f"<2.57.0,>={lower}"}


# --- the data folder's settings file (run for real, outside a dry run) -------------------


def _install_functions(*names: str) -> str:
    """These functions and constants from install.sh, to run on their own."""

    import re

    text = INSTALL_SH.read_text(encoding="utf-8")
    parts = [
        line
        for line in text.splitlines()
        if re.match(r"(FOLLOWER_LINK|LEADER_LINK|SET_ENV_KEPT)=", line)
    ]
    for name in names:
        block = re.search(rf"^{name}\(\) \{{\n.*?^\}}$", text, re.M | re.S)
        one_line = re.search(rf"^{name}\(\) \{{ .*\}}$", text, re.M)
        found = block or one_line
        assert found is not None, name
        parts.append(found.group(0))
    return "\n".join(parts)


def run_realsense_udev_step(
    tmp_path: Path, *, installed: str, system: str
) -> subprocess.CompletedProcess[str]:
    ours = tmp_path / "99-realsense-libusb.rules"
    vendor = tmp_path / "60-librealsense2-udev-rules.rules"
    ours.write_text(installed, encoding="utf-8")
    vendor.write_text(system, encoding="utf-8")
    script = "\n".join(
        [
            "set -euo pipefail",
            "DRY_RUN=1",
            "VPY=/fake/python",
            f"REALSENSE_RULES={shlex.quote(str(ours))}",
            f"REALSENSE_RULES_SYSTEM={shlex.quote(str(vendor))}",
            _install_functions(
                "say", "warn", "run", "confirm", "realsense_rules_valid", "realsense_udev"
            ),
            "realsense_udev",
        ]
    )
    return subprocess.run(
        [BASH or "bash", "-c", script],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=30,
        check=False,
    )


REALSENSE_TARGET_IDS = ("0ad3", "0b07", "0b3a", "0b5b", "0b5c")


def realsense_rules_fixture(*, vendor: bool = False, ids=REALSENSE_TARGET_IDS) -> str:
    if vendor:
        return "".join(
            f'SUBSYSTEM=="usb", ATTR{{idVendor}}=="8086", ATTR{{idProduct}}=="{product}", '
            'TAG+="uaccess"\n'
            for product in ids
        )
    return "".join(
        f'SUBSYSTEMS=="usb", ATTRS{{idVendor}}=="8086", '
        f'ATTRS{{idProduct}}=="{product}", MODE:="0666", GROUP:="plugdev"\n'
        for product in ids
    )


@pytest.mark.parametrize(
    ("installed", "system", "expected"),
    [
        (
            realsense_rules_fixture(),
            "# unused\n",
            "99-realsense-libusb.rules",
        ),
        (
            "# empty placeholder\n",
            realsense_rules_fixture(vendor=True),
            "60-librealsense2-udev-rules.rules",
        ),
    ],
    ids=["upstream-suite", "system-vendor"],
)
@needs_bash
def test_realsense_udev_skips_empty_rules_and_accepts_supported_variants(
    tmp_path, installed, system, expected
):
    result = run_realsense_udev_step(tmp_path, installed=installed, system=system)

    assert result.returncode == 0, output(result)
    assert f"RealSense udev rules: already installed ({tmp_path / expected})" in result.stdout


@needs_bash
@pytest.mark.parametrize(
    "installed",
    [
        realsense_rules_fixture(ids=REALSENSE_TARGET_IDS[:1]),
        realsense_rules_fixture(ids=REALSENSE_TARGET_IDS[:-1]),
        'SUBSYSTEMS=="usb", ATTRS{idVendor}=="8086", MODE:="0666"\n',
        realsense_rules_fixture().replace("SUBSYSTEMS", "XSUBSYSTEMS"),
        realsense_rules_fixture().lower(),
        (
            'this is not a rule # SUBSYSTEM=="usb", ATTR{idVendor}=="8086", '
            'ATTR{idProduct}=="0b5c", TAG+="uaccess"\n'
        ),
    ],
    ids=["single", "missing-d455", "no-product", "malformed-prefix", "lowercase", "comment"],
)
def test_realsense_udev_rejects_truncated_stale_or_malformed_files(tmp_path, installed):
    result = run_realsense_udev_step(tmp_path, installed=installed, system="# invalid too\n")

    assert result.returncode == 0, output(result)
    assert "RealSense udev rules: already installed" not in result.stdout
    assert "sudo cp" in result.stdout and "--realsense-udev-rules" in result.stdout


@needs_bash
def test_realsense_udev_checks_the_actual_vendored_rules_and_a_truncated_copy(tmp_path):
    source = ROOT / "src" / "so101" / "hardware" / "resources" / "99-realsense-libusb.rules"
    content = source.read_text(encoding="utf-8")

    complete = run_realsense_udev_step(tmp_path, installed=content, system="# unused\n")
    assert complete.returncode == 0, output(complete)
    assert f"already installed ({tmp_path / '99-realsense-libusb.rules'})" in complete.stdout

    missing_d455 = "\n".join(line for line in content.splitlines() if '0b5c"' not in line)
    stale = run_realsense_udev_step(tmp_path, installed=missing_d455, system="# invalid\n")
    assert stale.returncode == 0, output(stale)
    assert "RealSense udev rules: already installed" not in stale.stdout
    assert "sudo cp" in stale.stdout


def run_settings_file(tmp_path: Path, *, serials: bool) -> subprocess.CompletedProcess[str]:
    """install.sh's settings_file step with the real so101.platform_info (no terminal)."""

    data = tmp_path / "share" / "so101"
    script = "\n".join(
        [
            "set -euo pipefail",
            "DRY_RUN=0",
            f"VPY={sys.executable}",
            f"DATA_DIR={data}",
            f"FOLLOWER_SERIAL={'585A0077531' if serials else ''}",
            _install_functions("say", "warn", "ask", "show_settings", "settings_file"),
            "settings_file",
        ]
    )
    src = str(Path(pi.__file__).resolve().parents[1])
    env = {
        **os.environ,
        "SO101_HOME": str(data),
        "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])),
    }
    return subprocess.run(
        [BASH or "bash", "-c", script],
        capture_output=True,
        text=True,
        env=env,
        stdin=subprocess.DEVNULL,
        timeout=60,
        check=False,
    )


@needs_bash
def test_the_settings_file_gets_the_data_folders_and_the_arm_ports(tmp_path):
    result = run_settings_file(tmp_path, serials=True)
    text = output(result)
    assert result.returncode == 0, text
    env = tmp_path / "share" / "so101" / ".env"
    assert f"==> settings file: {env} (created)\n" in result.stdout
    assert f"    + SO101_DATA_DIR={env.parent / 'data'}\n" in result.stdout
    assert "    + SO101_FOLLOWER_PORT=/dev/so101-follower\n" in result.stdout
    values = pi.parse_dotenv(env.read_text(encoding="utf-8"))
    assert values == {
        **pi.storage_settings(env.parent),
        "SO101_FOLLOWER_PORT": "/dev/so101-follower",
        "SO101_LEADER_PORT": "/dev/so101-leader",
    }

    again = run_settings_file(tmp_path, serials=True)  # a re-run (upgrade) changes nothing
    assert again.returncode == 0, output(again)
    assert f"==> settings file: {env} (unchanged)\n" in again.stdout
    assert "+ " not in again.stdout


@needs_bash
def test_the_settings_file_keeps_values_the_user_set_without_a_terminal(tmp_path):
    env = tmp_path / "share" / "so101" / ".env"
    env.parent.mkdir(parents=True)
    env.write_text(
        "SO101_FOLLOWER_PORT=/dev/ttyACM0\nSO101_DATA_DIR=/mnt/big/so101\n", encoding="utf-8"
    )
    result = run_settings_file(tmp_path, serials=True)
    text = output(result)
    assert result.returncode == 0, text
    assert "SO101_FOLLOWER_PORT=/dev/ttyACM0 kept" in result.stdout
    assert "SO101_DATA_DIR=/mnt/big/so101 kept" in result.stdout
    assert "kept the folders the settings file already names" in result.stdout
    assert "not asked (no terminal): nothing was replaced" in result.stdout
    assert "kept the ports already set" in result.stdout
    values = pi.parse_dotenv(env.read_text(encoding="utf-8"))
    assert values["SO101_FOLLOWER_PORT"] == "/dev/ttyACM0"  # never clobbered unasked
    assert values["SO101_DATA_DIR"] == "/mnt/big/so101"
    assert values["SO101_LEADER_PORT"] == "/dev/so101-leader"  # missing ones are added
    assert values["SO101_OUTPUTS_DIR"] == str(env.parent / "outputs")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
@needs_bash
def test_the_settings_file_is_made_owner_only_and_the_installer_says_so(tmp_path):
    import stat

    env = tmp_path / "share" / "so101" / ".env"
    env.parent.mkdir(parents=True)
    env.write_text("SO101_HF_TOKEN=hf_secret\n", encoding="utf-8")
    env.chmod(0o644)
    result = run_settings_file(tmp_path, serials=False)
    assert result.returncode == 0, output(result)
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert "    permissions: was 0644 (others could read it); now 0600, owner only\n" in (
        result.stdout
    )
    assert pi.parse_dotenv(env.read_text(encoding="utf-8"))["SO101_HF_TOKEN"] == "hf_secret"


@needs_bash
def test_without_serials_only_the_data_folders_are_written(tmp_path):
    result = run_settings_file(tmp_path, serials=False)
    assert result.returncode == 0, output(result)
    values = pi.parse_dotenv((tmp_path / "share" / "so101" / ".env").read_text(encoding="utf-8"))
    assert set(values) == {"SO101_DATA_DIR", "SO101_DATASETS_DIR", "SO101_OUTPUTS_DIR"}


@needs_bash
def test_a_dry_run_shows_the_settings_edits_without_making_them(tmp_path):
    result = dry_run(tmp_path, "--follower-serial", "A1", "--leader-serial", "B2")
    assert result.returncode == 0, output(result)
    python = tmp_path / "app" / "venv" / "bin" / "python"
    assert f"+ {python} -P -m so101.platform_info --init-env\n" in result.stdout
    assert (
        f"+ {python} -P -m so101.platform_info --set-env SO101_FOLLOWER_PORT=/dev/so101-follower "
        "--set-env SO101_LEADER_PORT=/dev/so101-leader\n"
    ) in result.stdout
    assert not (tmp_path / "home" / ".local" / "share" / "so101").exists()
    mac = dry_run(tmp_path, os_name="Darwin", arch="arm64")
    assert "--init-env" in mac.stdout and "--set-env" not in mac.stdout
