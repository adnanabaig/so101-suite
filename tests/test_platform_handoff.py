"""scripts/make_handoff.py: the Zephyrus handoff folder, built from a throwaway repository."""

from __future__ import annotations

import errno
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which("git")

pytestmark = pytest.mark.skipif(GIT is None, reason="needs git")


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "make_handoff", ROOT / "scripts" / "make_handoff.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_handoff"] = module
    spec.loader.exec_module(module)
    return module


handoff = load_builder()


def load_g15_runner():
    spec = importlib.util.spec_from_file_location(
        "handoff_test_g15_runner", ROOT / "scripts" / "run_g15_ci.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


g15 = load_g15_runner()


def load_hatch_build():
    spec = importlib.util.spec_from_file_location(
        "handoff_test_hatch_build", ROOT / "hatch_build.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The UI-stamp helpers, for building test wheels the way make ci-wheel does.
hb = load_hatch_build()

GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def git(repo: Path, *args: str) -> str:
    env = {**os.environ, **GIT_ENV}
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env
    )
    return result.stdout.strip()


def run_fake_m0(repo: Path, evidence: Path, commit: str) -> int:
    """Exercise the real M0 source/freshness gate without running make."""

    assert GIT is not None

    def tools():
        paths = {
            "python": sys.executable,
            "node": "/usr/bin/node",
            "npm": "/usr/bin/npm",
            "git": GIT,
            "make": "/usr/bin/make",
        }
        return {
            name: g15.ToolInfo(
                name=name,
                invocation_path=path,
                resolved_path=path,
                sha256="0" * 64,
                mode="0755",
                owner_uid=0,
            )
            for name, path in paths.items()
        }

    def probe(argv, *, cwd, env):
        command = tuple(argv)
        executable = Path(command[0]).name
        versions = {
            "node": b"v24.11.1\n",
            "npm": b"11.6.1\n",
            "git": b"git version 2.45.2\n",
            "make": b"GNU Make 4.3\n",
        }
        if command[1:] == ("--version",) and executable in versions:
            return g15.ProbeResult(0, versions[executable])
        if executable == "git" and len(command) > 3 and command[1] == "-C":
            result = subprocess.run(command, cwd=cwd, env=dict(env), capture_output=True)
            return g15.ProbeResult(result.returncode, result.stdout, result.stderr)
        raise AssertionError(f"unexpected M0 probe: {command}")

    def execute(command_id, argv, *, log_path, **_kwargs):
        log_path.write_text(f"simulated {command_id} pass\n", encoding="utf-8")
        log_path.chmod(0o600)
        return g15.CommandResult(
            command_id=command_id,
            argv=tuple(argv),
            log_path=log_path.name,
            started_at="2026-10-08T12:00:00.000000Z",
            finished_at="2026-10-08T12:00:00.000001Z",
            duration_ns=1,
            exit_code=0,
        )

    host = g15.HostFacts(
        system="Linux",
        architecture="x86_64",
        kernel_release="6.8.0-test",
        python_version="3.12.9",
        python_implementation="CPython",
        python_major=3,
        python_minor=12,
        python_isolated=True,
        os_release={"ID": "ubuntu", "VERSION_ID": "24.04", "VERSION_CODENAME": "noble"},
        effective_uid=1000,
    )
    return g15.run_gate(
        commit,
        evidence,
        repo_root=repo,
        environ={"PATH": "/usr/bin"},
        host_facts=host,
        probe=probe,
        executor=execute,
        tool_resolver=tools,
        relay=g15.SignalRelay(install_handlers=False),
    )


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A two-commit repository on branch main, with ignored data and a .venv."""

    for key, value in GIT_ENV.items():
        monkeypatch.setenv(key, value)
    path = tmp_path / "repo"
    (path / "install").mkdir(parents=True)
    git(path.parent, "init", "--quiet", "--initial-branch=main", str(path))
    (path / "pyproject.toml").write_text('[project]\nname = "so101"\nversion = "0.2.0"\n')
    (path / "install" / "install.sh").write_text('#!/usr/bin/env bash\necho install "$@"\n')
    (path / "install" / "install.sh").chmod(0o755)
    shutil.copy(ROOT / "install" / "verify_input.py", path / "install" / "verify_input.py")
    (path / "src" / "so101").mkdir(parents=True)
    (path / "src" / "so101" / "__init__.py").write_text('__version__ = "0.2.0"\n')
    (path / "src" / "so101" / "cli.py").write_text("app = object()\n")
    (path / "frontend" / "src").mkdir(parents=True)
    (path / "frontend" / "src" / "main.tsx").write_text("export {}\n")
    (path / "frontend" / "index.html").write_text("<!doctype html>\n")
    (path / ".gitignore").write_text("data/\n.venv/\nnode_modules/\n")
    git(path, "add", ".")
    git(path, "commit", "--quiet", "-m", "first")
    (path / "README.md").write_text("SO-101\n")
    git(path, "add", ".")
    git(path, "commit", "--quiet", "-m", "second")
    (path / "data").mkdir()
    (path / "data" / "episode.parquet").write_text("ignored")
    (path / ".venv").mkdir()
    (path / ".venv" / "pyvenv.cfg").write_text("ignored")
    return path


BUILT_UI = {"index.html": b"<html></html>", "assets/index-abc123.js": b"console.log(1)"}


def make_wheel(
    folder: Path,
    version: str = "0.2.0",
    *,
    source: Path,
    ui: bool = True,
    stamp: bool = True,
    git_head: str | None = None,
    sources: dict[str, bytes] | None = None,
    extra: dict[str, bytes] | None = None,
) -> Path:
    """A wheel of ``source``'s HEAD, the way make ci-wheel builds one.

    ``so101/`` holds the commit's src/so101; with ``ui`` the UI and its build
    stamp (built at ``git_head``, default HEAD, from ``sources``, default the
    commit's frontend files). ``extra`` adds or replaces files afterwards.
    """
    commit = git(source, "rev-parse", "HEAD")
    archived = subprocess.run(
        ["git", "-C", str(source), "archive", "--format=tar", commit],
        capture_output=True,
        check=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archived)) as archive:
        tree = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    files = {
        "so101/" + name.removeprefix("src/so101/"): data
        for name, data in tree.items()
        if name.startswith("src/so101/")
    }
    files[f"so101-{version}.dist-info/METADATA"] = f"Name: so101\nVersion: {version}\n".encode()
    if ui:
        files.update({f"so101/web/{name}": data for name, data in BUILT_UI.items()})
        frontend = {
            name.removeprefix("frontend/"): data
            for name, data in tree.items()
            if name.startswith("frontend/")
        }
        if stamp:
            manifest = {name: hb.sha256_hex(data) for name, data in BUILT_UI.items()}
            files["so101/web/so101-ui-stamp.json"] = json.dumps(
                {
                    "format": 1,
                    "sources": hb.ui_sources_digest(frontend if sources is None else sources),
                    "git_head": git_head or commit,
                    "files": manifest,
                }
            ).encode()
    files.update(extra or {})
    folder.mkdir(parents=True, exist_ok=True)
    wheel = folder / f"so101-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return wheel


def test_handoff_folder_has_the_clone_the_wheel_the_runbook_and_checksums(repo, tmp_path, capsys):
    # An unreachable commit, like one left by `git commit --amend`, stays behind.
    dangling = git(repo, "commit-tree", "-m", "amended away", git(repo, "rev-parse", "HEAD^{tree}"))
    wheel = make_wheel(tmp_path / "built", source=repo)
    out = tmp_path / "usb" / "so101-handoff"

    code = handoff.main(["--repo", str(repo), "--out", str(out), "--wheel", str(wheel)])

    assert code == 0, capsys.readouterr().err
    printed = capsys.readouterr().out
    assert f"handoff folder: {out.resolve()}" in printed and "size: " in printed
    assert sorted(p.name for p in out.iterdir()) == [
        "SHA256SUMS",
        "START_HERE.md",
        "install.sh",
        "so101",
        "wheels",
    ]
    clone = out / "so101"
    assert git(clone, "rev-parse", "HEAD") == git(repo, "rev-parse", "HEAD")
    assert git(clone, "config", "--local", "--get", "core.filemode") == "false"
    assert git(clone, "rev-list", "--count", "HEAD") == "2"  # the full history
    assert git(clone, "remote") == ""
    leftover = subprocess.run(
        ["git", "-C", str(clone), "cat-file", "-e", dangling], capture_output=True
    )
    assert leftover.returncode != 0
    assert git(clone, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    assert not (clone / "data").exists() and not (clone / ".venv").exists()
    assert (clone / "install" / "verify_input.py").is_file()
    assert (out / "wheels" / wheel.name).read_bytes() == wheel.read_bytes()

    script = out / "install.sh"
    assert os.access(script, os.X_OK)
    assert (
        'exec bash "$HERE/so101/install/install.sh" '
        '--wheel "$HERE/wheels/so101-0.2.0-py3-none-any.whl" "$@"'
    ) in script.read_text()
    assert "so101/install/verify_input.py" in (out / "SHA256SUMS").read_text()
    runbook = (out / "START_HERE.md").read_text()
    commit = git(repo, "rev-parse", "HEAD")
    for text in (
        "SO-101 Suite 0.2.0",
        f"`{commit}`",
        "sudo ubuntu-drivers install",
        "g15-ci-evidence",
        "Enroll MOK",
        "bash install.sh --realsense",
        "bash install.sh --realsense --follower-serial SERIAL --leader-serial SERIAL",
        "so101 doctor",
        "so101 preflight",
        "so101 calibrate follower",
        "so101 rest-pose record",
        "so101 hil run",
        "http://localhost:8000",
        "so101 doctor --json > ~/doctor.json",
        "whatever folder you run it from",
        "/dev/so101-follower",
    ):
        assert text in runbook
    # The generated runbook is safety-first: establish the physical envelope
    # and identify the supplies before power, complete torque-off setup before
    # starting the server, then prove the stop paths before recording/training.
    physical_gate = runbook.index("## 4. Physical safety and device identity")
    bundle_checksum = runbook.index("sha256sum -c SHA256SUMS")
    m0 = runbook.index("## M0. One-shot software gate")
    m0_repo = runbook.index("cd so101", m0)
    m0_make = runbook.index("/usr/bin/make g15-ci-evidence", m0_repo)
    driver = runbook.index("sudo ubuntu-drivers install")
    installer_dry_run = runbook.index("bash install.sh --realsense --profile cuda --dry-run")
    first_power = runbook.index("Switch on the arm supplies only after")
    preflight = runbook.index("so101 preflight", first_power)
    midpoint = runbook.index("Place each arm at its midpoint before starting its calibration")
    calibration = runbook.index("so101 calibrate follower")
    server = runbook.index("## 8. Start the suite")
    hil = runbook.index("## 9. The HIL checklist")
    assert bundle_checksum < m0 < m0_repo < m0_make < driver < installer_dry_run
    assert physical_gate < first_power < preflight < midpoint < calibration < server < hil
    assert f"G15_CI_EXPECTED_COMMIT={commit}" in runbook
    assert 'G15_CI_EVIDENCE="$PWD/../../evidence/' in runbook
    assert "/usr/bin/make -C" not in runbook
    for warning in (
        "Keep both arm motor supplies switched off",
        "Confirm the follower and leader supply",
        "mechanical catch",
        "turn torque off",
        "hands-on sweep",
        "A second person",
    ):
        assert warning in runbook

    # Every HIL group, with the stop group before data/training verification.
    groups = [
        f'so101 hil run --operator "Your Name" --group {group}'
        for group in ("gate", "stop", "verification")
    ]
    positions = [runbook.find(command) for command in groups]
    assert -1 not in positions and positions == sorted(positions), positions
    assert "power-switch operator" in runbook
    assert "mechanical catch for the whole `stop` group" in runbook
    assert "does not check that a group passed" in runbook
    assert runbook.index("so101-suite") < positions[0]
    flat_runbook = " ".join(runbook.split())
    assert "Do not stop the server between HIL steps" in flat_runbook
    assert "collection/resume step deliberately restarts it" in flat_runbook
    assert "hold-timer setting" in flat_runbook and "--resume RUN_ID" in flat_runbook
    # A FAT32/exFAT stick drops the exec bit: no command runs ./install.sh.
    assert "\n    ./install.sh" not in runbook
    # Settings and data follow the data folder's .env, not the working folder.
    assert "cd ~/.local/share/so101" not in runbook
    assert "from the data folder" not in runbook.lower()

    listed = {}
    for line in (out / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split("  ", 1)
        listed[path] = digest
    assert "SHA256SUMS" not in listed
    assert not any(path.startswith("so101/.git/") for path in listed)
    for path in ("install.sh", "START_HERE.md", f"wheels/{wheel.name}", "so101/README.md"):
        assert listed[path] == hashlib.sha256((out / path).read_bytes()).hexdigest()
    files = {
        p.relative_to(out).as_posix()
        for p in out.rglob("*")
        if p.is_file() and not p.relative_to(out).as_posix().startswith("so101/.git/")
    }
    assert set(listed) == files - {"SHA256SUMS"}


@pytest.mark.skipif(shutil.which("sha256sum") is None, reason="needs sha256sum")
def test_sha256sum_accepts_the_checksums(repo, tmp_path):
    out = tmp_path / "out"
    assert (
        handoff.main(
            [
                "--repo",
                str(repo),
                "--out",
                str(out),
                "--wheel",
                str(make_wheel(tmp_path / "w", source=repo)),
            ]
        )
        == 0
    )
    result = subprocess.run(
        ["sha256sum", "--quiet", "-c", "SHA256SUMS"], cwd=out, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "dirty",
    [
        lambda repo: (repo / "README.md").write_text("changed\n"),
        lambda repo: (repo / "notes.txt").write_text("untracked\n"),
    ],
    ids=["modified", "untracked"],
)
def test_a_dirty_tree_is_refused(repo, tmp_path, capsys, dirty):
    dirty(repo)
    out = tmp_path / "out"
    code = handoff.main(
        [
            "--repo",
            str(repo),
            "--out",
            str(out),
            "--wheel",
            str(make_wheel(tmp_path / "w", source=repo)),
        ]
    )
    assert code == 1
    assert "the working tree isn't clean" in capsys.readouterr().err
    assert not out.exists()


def test_the_output_folder_must_be_new_or_empty_and_outside_the_repo(repo, tmp_path, capsys):
    wheel = str(make_wheel(tmp_path / "w", source=repo))
    inside = handoff.main(
        ["--repo", str(repo), "--out", str(repo / "data" / "handoff"), "--wheel", wheel]
    )
    assert inside == 1 and "is inside the repository" in capsys.readouterr().err

    full = tmp_path / "full"
    full.mkdir()
    (full / "old.txt").write_text("x")
    assert handoff.main(["--repo", str(repo), "--out", str(full), "--wheel", wheel]) == 1
    assert "isn't an empty folder" in capsys.readouterr().err
    assert (full / "old.txt").read_text() == "x"

    empty = tmp_path / "empty"
    empty.mkdir()
    assert handoff.main(["--repo", str(repo), "--out", str(empty), "--wheel", wheel]) == 0
    assert git(repo, "status", "--porcelain") == ""  # nothing was written into the repo


def test_a_wheel_must_match_the_commit_and_carry_the_ui(repo, tmp_path, capsys):
    out = tmp_path / "out"
    wrong = make_wheel(tmp_path / "old", "0.1.0", source=repo)
    assert handoff.main(["--repo", str(repo), "--out", str(out), "--wheel", str(wrong)]) == 1
    assert "isn't so101-0.2.0-py3-none-any.whl" in capsys.readouterr().err

    no_ui = make_wheel(tmp_path / "plain", source=repo, ui=False)
    assert handoff.main(["--repo", str(repo), "--out", str(out), "--wheel", str(no_ui)]) == 1
    assert "has no so101/web/index.html" in capsys.readouterr().err
    allowed = ["--repo", str(repo), "--out", str(out), "--wheel", str(no_ui), "--allow-no-ui"]
    assert handoff.main(allowed) == 0
    wrapper = (out / "install.sh").read_text(encoding="utf-8")
    assert "--no-launcher --allow-no-ui" in wrapper
    launched = subprocess.run(
        ["bash", "install.sh", "--realsense"],
        cwd=out,
        capture_output=True,
        text=True,
        check=False,
    )
    assert launched.returncode == 0, launched.stderr
    assert "--no-launcher --allow-no-ui --realsense" in launched.stdout

    broken = tmp_path / "broken" / "so101-0.2.0-py3-none-any.whl"
    broken.parent.mkdir()
    broken.write_text("not a zip")
    assert (
        handoff.main(["--repo", str(repo), "--out", str(tmp_path / "o2"), "--wheel", str(broken)])
        == 1
    )
    assert "isn't a wheel" in capsys.readouterr().err


def test_a_detached_head_is_refused(repo, tmp_path, capsys):
    git(repo, "checkout", "--quiet", "--detach")
    code = handoff.main(
        [
            "--repo",
            str(repo),
            "--out",
            str(tmp_path / "out"),
            "--wheel",
            str(make_wheel(tmp_path / "w", source=repo)),
        ]
    )
    assert code == 1 and "HEAD is detached" in capsys.readouterr().err


def test_without_a_wheel_it_builds_one_with_the_ui_from_a_temporary_clone(repo, tmp_path):
    calls: list[tuple[list[str], dict]] = []

    def run(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        if cmd[:3] == [sys.executable, "-m", "build"]:
            assert kwargs["env"]["SO101_REQUIRE_UI"] == "1"
            make_wheel(Path(cmd[cmd.index("--outdir") + 1]), source=Path(cmd[-1]))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == "npm":
            assert Path(kwargs["cwd"]).name == "frontend"
            assert repo not in Path(kwargs["cwd"]).parents  # never in the repository
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return handoff._run(cmd, **kwargs)

    out = handoff.make_handoff(tmp_path / "out", repo=repo, run=run)

    commands = [cmd for cmd, _ in calls]
    assert ["npm", "ci"] in commands and ["npm", "run", "build"] in commands
    assert (out / "wheels" / "so101-0.2.0-py3-none-any.whl").is_file()
    assert git(repo, "status", "--porcelain") == ""


def test_a_build_that_makes_the_wrong_wheel_is_an_error(repo, tmp_path):
    def run(cmd, **kwargs):
        if cmd[:3] == [sys.executable, "-m", "build"]:
            make_wheel(Path(cmd[cmd.index("--outdir") + 1]), "9.9.9", source=Path(cmd[-1]))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == "npm":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return handoff._run(cmd, **kwargs)

    with pytest.raises(handoff.HandoffError, match="made so101-9.9.9-py3-none-any.whl"):
        handoff.make_handoff(tmp_path / "out", repo=repo, run=run)


def test_a_failing_command_is_reported_with_its_last_error_line(
    repo, tmp_path, capsys, monkeypatch
):
    real = handoff._run

    def failing(cmd, **kwargs):
        if cmd[0] == "npm":
            raise subprocess.CalledProcessError(1, cmd, "", "npm warn\nnpm error network")
        return real(cmd, **kwargs)

    monkeypatch.setattr(handoff, "_run", failing)
    out = tmp_path / "out"
    code = handoff.main(["--repo", str(repo), "--out", str(out)])
    assert code == 1
    assert "error: npm ci exited with code 1: npm error network" in capsys.readouterr().err
    assert not out.exists()

    wheel = make_wheel(tmp_path / "w", source=repo)
    assert handoff.main(["--repo", str(repo), "--out", str(out), "--wheel", str(wheel)]) == 0


def test_a_failed_build_leaves_a_preexisting_empty_output_empty(repo, tmp_path, monkeypatch):
    real = handoff._run

    def failing(cmd, **kwargs):
        if cmd[0] == "npm":
            raise subprocess.CalledProcessError(1, cmd, "", "npm error network")
        return real(cmd, **kwargs)

    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(handoff, "_run", failing)

    assert handoff.main(["--repo", str(repo), "--out", str(out)]) == 1
    assert out.is_dir() and list(out.iterdir()) == []


@pytest.mark.parametrize("failure", [OSError("read-only filesystem"), KeyboardInterrupt()])
def test_a_final_publish_failure_preserves_a_preexisting_empty_output(
    repo, tmp_path, monkeypatch, failure
):
    out = tmp_path / "out"
    out.mkdir(mode=0o751)
    before = out.stat()
    wheel = make_wheel(tmp_path / "w", source=repo)
    real_replace = handoff.os.replace

    def fail_publish(source, destination):
        if Path(destination) == out:
            raise failure
        return real_replace(source, destination)

    monkeypatch.setattr(handoff.os, "replace", fail_publish)
    if isinstance(failure, OSError):
        with pytest.raises(handoff.HandoffError, match="could not publish"):
            handoff.make_handoff(out, repo=repo, wheel=wheel)
    else:
        with pytest.raises(KeyboardInterrupt):
            handoff.make_handoff(out, repo=repo, wheel=wheel)

    after = out.stat()
    assert (after.st_dev, after.st_ino, after.st_mode) == (
        before.st_dev,
        before.st_ino,
        before.st_mode,
    )
    assert list(out.iterdir()) == []
    assert not list(tmp_path.glob(".out.building-*"))


def test_an_output_populated_while_building_is_left_untouched(repo, tmp_path):
    out = tmp_path / "out"
    sentinel = out / "keep.txt"
    wheel = make_wheel(tmp_path / "w", source=repo)

    def run(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"] and "--single-branch" in cmd:
            out.mkdir()
            sentinel.write_text("mine", encoding="utf-8")
        return handoff._run(cmd, **kwargs)

    with pytest.raises(handoff.HandoffError, match="isn't an empty folder"):
        handoff.make_handoff(out, repo=repo, wheel=wheel, run=run)

    assert sentinel.read_text(encoding="utf-8") == "mine"


def test_handoff_tolerates_chmod_rejection_on_fat(repo, tmp_path, monkeypatch):
    out = tmp_path / "out"
    wheel = make_wheel(tmp_path / "w", source=repo)
    real_chmod = Path.chmod

    def reject_chmod(path, mode, *, follow_symlinks=True):
        if path.name == "install.sh":
            raise PermissionError("FAT showexec mount")
        return real_chmod(path, mode, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "chmod", reject_chmod)

    assert handoff.make_handoff(out, repo=repo, wheel=wheel) == out
    assert (out / "install.sh").is_file()


def test_a_supplied_wheel_changed_during_the_build_is_rejected(repo, tmp_path):
    out = tmp_path / "out"
    wheel = make_wheel(tmp_path / "given", source=repo)
    changed = make_wheel(
        tmp_path / "changed", source=repo, extra={"so101/cli.py": b"app = 'changed'\n"}
    )

    def run(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"] and "--single-branch" in cmd:
            wheel.write_bytes(changed.read_bytes())
        return handoff._run(cmd, **kwargs)

    with pytest.raises(handoff.HandoffError, match="differ from the commit"):
        handoff.make_handoff(out, repo=repo, wheel=wheel, run=run)

    assert not out.exists()


# --- The wheel is the commit (review findings L11, L12) --------------------------------


def _handoff_with(repo: Path, out: Path, wheel: Path) -> int:
    return handoff.main(["--repo", str(repo), "--out", str(out), "--wheel", str(wheel)])


def _bare_wheel(folder: Path, files: dict[str, str]) -> Path:
    folder.mkdir(parents=True)
    wheel = folder / "so101-0.2.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return wheel


def test_a_wheel_without_the_suites_code_or_version_is_refused(repo, tmp_path, capsys):
    # The reviewer's repro: a zip with no code and Version 0.1.0 inside was accepted.
    ui = {"so101/web/index.html": "<html></html>"}
    empty = _bare_wheel(
        tmp_path / "a", {"so101-0.1.0.dist-info/METADATA": "Version: 0.1.0\n", **ui}
    )
    assert _handoff_with(repo, tmp_path / "out", empty) == 1
    assert "METADATA says (no Name) 0.1.0, not so101 0.2.0" in capsys.readouterr().err

    no_code = _bare_wheel(
        tmp_path / "b", {"so101-0.2.0.dist-info/METADATA": "Name: so101\nVersion: 0.2.0\n", **ui}
    )
    assert _handoff_with(repo, tmp_path / "out", no_code) == 1
    assert "has no so101/__init__.py, so it isn't the suite" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()  # refused before anything was written

    planted = make_wheel(tmp_path / "c", source=repo, extra={"sitecustomize.py": b"import os\n"})
    assert _handoff_with(repo, tmp_path / "out", planted) == 1
    assert "holds more than so101: sitecustomize.py" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"so101/zz_local_hack.py": b"print('hack')\n"}, "doesn't have: so101/zz_local_hack.py"),
        ({"so101/cli.py": b"app = 'edited'\n"}, "differ from the commit: so101/cli.py"),
    ],
    ids=["untracked-file", "dirty-file"],
)
def test_a_wheel_whose_code_is_not_the_commits_is_refused(repo, tmp_path, capsys, extra, message):
    wheel = make_wheel(tmp_path / "w", source=repo, extra=extra)
    assert _handoff_with(repo, tmp_path / "out", wheel) == 1
    err = capsys.readouterr().err
    assert "isn't the suite at" in err and message in err


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"stamp": False}, "no so101-ui-stamp.json"),
        ({"git_head": "f" * 40}, f"its UI was built at {'f' * 40}"),
        (
            {"sources": {"src/main.tsx": b"export const old = 1\n"}},
            "built from other frontend sources",
        ),
        (
            {"extra": {"so101/web/assets/old-chunk-DEADBEEF.js": b"old"}},
            "assets/old-chunk-DEADBEEF.js is not in the build's manifest",
        ),
        (
            {"extra": {"so101/web/index.html": b"<html>stale</html>"}},
            "index.html differs from the build's manifest",
        ),
    ],
    ids=["no-stamp", "other-commit", "other-sources", "extra-file", "stale-index"],
)
def test_the_wheels_ui_must_be_the_stamped_build_of_this_commit(
    repo, tmp_path, capsys, options, message
):
    wheel = make_wheel(tmp_path / "w", source=repo, **options)
    assert _handoff_with(repo, tmp_path / "out", wheel) == 1
    err = capsys.readouterr().err
    assert "the web UI isn't the build of" in err and message in err


def test_the_wheel_is_built_at_the_exact_commit_even_if_the_branch_moves(repo, tmp_path):
    commit = git(repo, "rev-parse", "HEAD")
    out = tmp_path / "out"
    built_from: list[str] = []

    def run(cmd, **kwargs):
        if cmd[:2] == ["git", "clone"] and "--no-checkout" in cmd and not built_from:
            # A commit lands on the branch while the handoff is building.
            (repo / "README.md").write_text("moved on\n")
            git(repo, "commit", "--quiet", "-am", "third")
            built_from.append("")
        if cmd == ["npm", "ci"]:
            built_from.append(git(Path(kwargs["cwd"]).parent, "rev-parse", "HEAD"))
        if cmd[0] == "npm":
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[:3] == [sys.executable, "-m", "build"]:
            make_wheel(Path(cmd[cmd.index("--outdir") + 1]), source=Path(cmd[-1]))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return handoff._run(cmd, **kwargs)

    handoff.make_handoff(out, repo=repo, run=run)
    assert built_from == ["", commit]  # not the commit that landed meanwhile
    assert git(out / "so101", "rev-parse", "HEAD") == commit
    assert f"`{commit}`" in (out / "START_HERE.md").read_text()


# --- A FAT32 or exFAT USB stick drops the exec bit -------------------------------------


def test_install_sh_runs_from_a_stick_without_the_exec_bit(repo, tmp_path):
    out = tmp_path / "usb"
    assert _handoff_with(repo, out, make_wheel(tmp_path / "w", source=repo)) == 0
    # udisks mounts FAT32/exFAT with showexec: no file keeps its exec bit.
    for script in (out / "install.sh", out / "so101" / "install" / "install.sh"):
        script.chmod(0o644)
    result = subprocess.run(
        ["bash", "install.sh", "--realsense"], cwd=out, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("install --wheel ")
    assert result.stdout.rstrip().endswith("/wheels/so101-0.2.0-py3-none-any.whl --realsense")


@pytest.mark.skipif(shutil.which("sha256sum") is None, reason="needs sha256sum")
def test_exact_commit_archive_extracts_verifies_and_dry_runs_without_exec_bits(repo, tmp_path):
    commit = git(repo, "rev-parse", "HEAD")
    wheel = make_wheel(tmp_path / "wheel", source=repo)
    archive = tmp_path / "release" / f"so101-handoff-{commit[:12]}.tar.gz"

    code = handoff.main(
        [
            "--repo",
            str(repo),
            "--archive",
            str(archive),
            "--expected-commit",
            commit,
            "--wheel",
            str(wheel),
        ]
    )

    assert code == 0
    sidecar = Path(str(archive) + ".sha256")
    checked = subprocess.run(
        ["sha256sum", "--check", sidecar.name],
        cwd=archive.parent,
        capture_output=True,
        text=True,
    )
    assert checked.returncode == 0, checked.stdout + checked.stderr
    with tarfile.open(archive) as packed:
        names = packed.getnames()
        assert names and all(
            name == handoff.ARCHIVE_ROOT or name.startswith(handoff.ARCHIVE_ROOT + "/")
            for name in names
        )
        extracted = tmp_path / "extracted"
        packed.extractall(extracted)
    bundle = extracted / handoff.ARCHIVE_ROOT
    inner = subprocess.run(
        ["sha256sum", "--check", "SHA256SUMS"],
        cwd=bundle,
        capture_output=True,
        text=True,
    )
    assert inner.returncode == 0, inner.stdout + inner.stderr
    clone = bundle / "so101"
    assert "100755" in git(clone, "ls-files", "--stage", "install/install.sh")
    for path in bundle.rglob("*"):
        if path.is_file():
            path.chmod(0o644)
    after_mode_loss = subprocess.run(
        ["sha256sum", "--check", "SHA256SUMS"],
        cwd=bundle,
        capture_output=True,
        text=True,
    )
    assert after_mode_loss.returncode == 0, after_mode_loss.stdout + after_mode_loss.stderr
    assert git(clone, "config", "--local", "--get", "core.filemode") == "false"
    assert git(clone, "status", "--porcelain=v1", "--untracked-files=all") == ""
    assert g15._fresh_checkout_blockers(clone) == []
    assert run_fake_m0(clone, tmp_path / "m0-evidence", commit) == 0
    dry_run = subprocess.run(
        ["bash", "install.sh", "--dry-run"],
        cwd=bundle,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert dry_run.returncode == 0, dry_run.stdout + dry_run.stderr
    assert dry_run.stdout.startswith("install --wheel ")

    readme = clone / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "changed bytes\n", encoding="utf-8")
    assert git(clone, "status", "--porcelain=v1", "--untracked-files=all") != ""
    changed = subprocess.run(
        ["sha256sum", "--check", "SHA256SUMS"],
        cwd=bundle,
        capture_output=True,
        text=True,
    )
    assert changed.returncode != 0


def test_archive_refuses_the_wrong_approved_head_before_writing(repo, tmp_path, capsys):
    archive = tmp_path / "release.tar.gz"
    code = handoff.main(
        [
            "--repo",
            str(repo),
            "--archive",
            str(archive),
            "--expected-commit",
            "a" * 40,
            "--wheel",
            str(make_wheel(tmp_path / "wheel", source=repo)),
        ]
    )
    assert code == 1
    assert "not the approved commit" in capsys.readouterr().err
    assert not archive.exists() and not Path(str(archive) + ".sha256").exists()


def test_archive_refuses_head_drift_during_construction(repo, tmp_path, monkeypatch):
    approved = git(repo, "rev-parse", "HEAD")
    wheel = make_wheel(tmp_path / "wheel", source=repo)
    archive = tmp_path / "release.tar.gz"
    real_check = handoff.check_wheel
    calls = 0

    def moving_check(*args, **kwargs):
        nonlocal calls
        answer = real_check(*args, **kwargs)
        calls += 1
        if calls == 2:
            (repo / "README.md").write_text("new candidate\n", encoding="utf-8")
            git(repo, "commit", "--quiet", "-am", "candidate moved")
        return answer

    monkeypatch.setattr(handoff, "check_wheel", moving_check)
    with pytest.raises(handoff.HandoffError, match="not the approved commit"):
        handoff.make_handoff_archive(
            archive,
            expected_commit=approved,
            repo=repo,
            wheel=wheel,
        )
    assert not archive.exists() and not Path(str(archive) + ".sha256").exists()


@pytest.mark.parametrize("race_target", ["archive", "sidecar"])
def test_archive_publication_never_overwrites_a_racing_sentinel(
    repo, tmp_path, monkeypatch, race_target
):
    approved = git(repo, "rev-parse", "HEAD")
    wheel = make_wheel(tmp_path / "wheel", source=repo)
    archive = tmp_path / "release.tar.gz"
    sidecar = Path(str(archive) + ".sha256")
    target = archive if race_target == "archive" else sidecar
    real_publish = handoff._publish_exclusive

    def race(source, destination):
        if Path(destination) == target and not target.exists():
            target.write_text("operator sentinel\n", encoding="utf-8")
        return real_publish(source, destination)

    monkeypatch.setattr(handoff, "_publish_exclusive", race)
    with pytest.raises(handoff.HandoffError, match="appeared while building"):
        handoff.make_handoff_archive(
            archive,
            expected_commit=approved,
            repo=repo,
            wheel=wheel,
        )

    assert target.read_text(encoding="utf-8") == "operator sentinel\n"
    other = sidecar if target == archive else archive
    assert not other.exists()


def test_sidecar_publish_failure_rolls_back_and_the_same_output_can_retry(
    repo, tmp_path, monkeypatch
):
    approved = git(repo, "rev-parse", "HEAD")
    wheel = make_wheel(tmp_path / "wheel", source=repo)
    archive = tmp_path / "release.tar.gz"
    sidecar = Path(str(archive) + ".sha256")
    real_publish = handoff._publish_exclusive

    def fail_sidecar(source, destination):
        if Path(destination) == sidecar:
            raise OSError(errno.EIO, "injected sidecar publication failure")
        return real_publish(source, destination)

    monkeypatch.setattr(handoff, "_publish_exclusive", fail_sidecar)
    with pytest.raises(handoff.HandoffError, match="could not publish"):
        handoff.make_handoff_archive(
            archive,
            expected_commit=approved,
            repo=repo,
            wheel=wheel,
        )
    assert not archive.exists() and not sidecar.exists()

    monkeypatch.setattr(handoff, "_publish_exclusive", real_publish)
    published, checksum = handoff.make_handoff_archive(
        archive,
        expected_commit=approved,
        repo=repo,
        wheel=wheel,
    )
    assert published == archive and checksum == sidecar
    assert archive.is_file() and sidecar.is_file()


def test_generated_runbook_verifies_the_copied_bundle_before_dry_run(repo, tmp_path):
    out = tmp_path / "out"
    assert _handoff_with(repo, out, make_wheel(tmp_path / "wheel", source=repo)) == 0
    runbook = (out / "START_HERE.md").read_text(encoding="utf-8")
    copied = runbook.index("copy or extract this whole folder")
    checksum = runbook.index("sha256sum -c SHA256SUMS")
    fresh = runbook.index("freshly extracted `so101/` repository")
    repo_cwd = runbook.index("cd so101", fresh)
    m0 = runbook.index("/usr/bin/make g15-ci-evidence", repo_cwd)
    driver = runbook.index("sudo ubuntu-drivers install")
    dry_run = runbook.index("bash install.sh --realsense --profile cuda --dry-run")
    assert copied < checksum < fresh < repo_cwd < m0 < driver < dry_run
    assert "/usr/bin/make g15-ci-evidence \\\n" in runbook
    assert 'G15_CI_EVIDENCE="$PWD/../../evidence/' in runbook
    assert "another fresh\nextraction" in runbook


def test_makefile_exposes_the_approved_commit_archive_target():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "handoff-archive:" in makefile
    assert "HANDOFF_APPROVED_COMMIT" in makefile
    assert '--archive "$(OUT)"' in makefile


def test_documented_external_checksum_runs_from_the_archive_directory():
    install_doc = (ROOT / "docs" / "install.md").read_text(encoding="utf-8")
    section = install_doc.split("## The handoff folder", 1)[1].split(
        "## Behind a TLS-inspecting proxy", 1
    )[0]
    checksum = section.index('sha256sum --check "$(basename "$archive").sha256"')
    checksum_cwd = section.rfind('cd "$(dirname "$archive")"', 0, checksum)
    extract = section.index('tar -xzf "$(basename "$archive")"')
    extract_cwd = section.rfind('cd "$(dirname "$archive")"', 0, extract)
    assert checksum_cwd != -1 and checksum_cwd < checksum
    assert extract_cwd != -1 and extract_cwd < extract
    assert "sha256sum --check ../" not in section
    assert "inside `so101/` before any install" in section


def test_wrapper_and_generic_installer_keep_the_verifier_inside_the_clone(repo, tmp_path):
    out = tmp_path / "usb"
    assert _handoff_with(repo, out, make_wheel(tmp_path / "w", source=repo)) == 0
    wrapper = (out / "install.sh").read_text(encoding="utf-8")
    assert 'bash "$HERE/so101/install/install.sh"' in wrapper
    bundled_verifier = out / "so101" / "install" / "verify_input.py"
    assert bundled_verifier.read_bytes() == (ROOT / "install" / "verify_input.py").read_bytes()

    generic = (ROOT / "install" / "install.sh").read_text(encoding="utf-8")
    assert 'INPUT_VERIFIER="$ROOT/install/verify_input.py"' in generic
