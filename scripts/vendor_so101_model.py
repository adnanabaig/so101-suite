#!/usr/bin/env python3
"""Vendor the MuJoCo Menagerie SO-101 model into ``so101.simulation.assets``.

The suite ships the Menagerie ``robotstudio_so101`` description so the physics
simulator works offline.  This script is the only way those files are made:

* This base conversion copies ``so101.xml``, ``scene.xml``, ``scene_box.xml`` and
  ``LICENSE`` byte-exact. The documented G15 operating-range adaptation is a
  separate subsequent step in ``adapt_so101_g15_model.py``. Runtime changes such
  as the timestep are applied in memory through ``mujoco.MjSpec``.
* Meshes that can influence dynamics are copied byte-exact:
  - every mesh used by a collision-enabled geom (``contype``/``conaffinity``);
  - every mesh on a body without an explicit ``<inertial>``, because MuJoCo
    then infers that body's mass properties from the mesh shape.  In this
    model that is the wrist camera mount.
* Every other mesh is visual-only and is decimated (quadric edge collapse) so
  the vendored package stays under about 3 MB.  Each decimated STL carries a
  modification notice in its 80-byte header (Apache-2.0 section 4(b)).
* ``PROVENANCE.md`` records the source URL and commit, per-file input/output
  SHA-256, every modification, and a dynamics-equivalence proof: the vendored
  model and the source model produce bitwise-identical ``qpos``/``qvel``
  trajectories over thousands of steps under the same control sequence.

Dev-only requirements (never runtime dependencies of the suite):
``mujoco>=3.3,<4``, ``pymeshlab`` and ``numpy``::

    git clone --filter=blob:none https://github.com/google-deepmind/mujoco_menagerie ../menagerie
    git -C ../menagerie checkout 4d038b3feae26ec82b46a4d586379114012a8ac7
    python scripts/vendor_so101_model.py --source ../menagerie/robotstudio_so101
    python scripts/vendor_so101_model.py --source ../menagerie/robotstudio_so101 --check
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "src" / "so101" / "simulation" / "assets" / "robotstudio_so101"
UPSTREAM_REPO = "https://github.com/google-deepmind/mujoco_menagerie"
UPSTREAM_SUBDIR = "robotstudio_so101"
MJCF_FILES = ("so101.xml", "scene.xml", "scene_box.xml")
COPIED_FILES = (*MJCF_FILES, "LICENSE")
STL_RECORD = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])


@dataclass
class MeshRecord:
    name: str
    file: str
    role: str
    reason: str
    faces_in: int
    faces_out: int
    bytes_in: int
    bytes_out: int
    sha_in: str
    sha_out: str


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_commit(source: Path) -> tuple[str, str, bool]:
    """Return ``(commit, commit_date, clean)`` for the checkout containing ``source``."""

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(source), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    commit = run("rev-parse", "HEAD")
    date = run("log", "-1", "--format=%cI", "HEAD")
    clean = run("status", "--porcelain", "--", ".") == ""
    return commit, date, clean


def read_stl(data: bytes) -> np.ndarray:
    """Return the ``(n, 3, 3)`` float32 triangle soup of a binary STL."""

    if len(data) < 84:
        raise ValueError("STL file is truncated")
    count = struct.unpack_from("<I", data, 80)[0]
    if 84 + 50 * count != len(data):
        raise ValueError("only binary STL files are supported")
    records = np.frombuffer(data, dtype=STL_RECORD, count=count, offset=84)
    return np.array(records["v"], dtype=np.float32)


def write_stl(triangles: np.ndarray, header: bytes) -> bytes:
    if header.lower().startswith(b"solid"):
        raise ValueError("a binary STL header must not start with 'solid'")
    tris = np.asarray(triangles, dtype=np.float32)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    records = np.zeros(len(tris), dtype=STL_RECORD)
    records["normal"] = normals
    records["v"] = tris
    return header.ljust(80, b"\0")[:80] + struct.pack("<I", len(tris)) + records.tobytes()


def decimate(triangles: np.ndarray, target_faces: int) -> np.ndarray:
    """Quadric-edge-collapse a triangle soup to ``target_faces`` faces (MeshLab).

    ``fast-simplification`` was tried first; it stalls at about half the faces
    on these CAD meshes because it never changes topology.  MeshLab's filter may
    close tiny holes (``preservetopology=False``), which is invisible at robot
    scale and reaches the target exactly.
    """

    import pymeshlab

    points, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int32)
    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(pymeshlab.Mesh(vertex_matrix=points.astype(np.float64), face_matrix=faces))
    meshset.meshing_decimation_quadric_edge_collapse(
        targetfacenum=int(target_faces),
        qualitythr=0.3,
        preservenormal=True,
        preservetopology=False,
        optimalplacement=True,
        planarquadric=True,
        autoclean=True,
    )
    mesh = meshset.current_mesh()
    out_points = np.asarray(mesh.vertex_matrix(), dtype=np.float64)
    out_faces = np.asarray(mesh.face_matrix(), dtype=np.int64)
    tris = out_points[out_faces]
    doubled_area = np.linalg.norm(
        np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1
    )
    distinct = (
        (out_faces[:, 0] != out_faces[:, 1])
        & (out_faces[:, 1] != out_faces[:, 2])
        & (out_faces[:, 0] != out_faces[:, 2])
    )
    return np.asarray(tris[distinct & (doubled_area > 0.0)], dtype=np.float32)


def classify_meshes(so101_xml: Path) -> dict[str, tuple[str, str, str]]:
    """Map mesh name -> (file, role, reason) using MuJoCo's own default resolution."""

    import mujoco

    spec = mujoco.MjSpec.from_file(str(so101_xml))
    # MJCF names a mesh after its file stem when no name is given; MjSpec leaves it empty.
    names = {mesh.name or Path(mesh.file).stem: Path(mesh.file).name for mesh in spec.meshes}
    uses: dict[str, list[Any]] = {name: [] for name in names}
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            uses.setdefault(geom.meshname, []).append(geom)
    result: dict[str, tuple[str, str, str]] = {}
    for name, file in names.items():
        geoms = uses.get(name, [])
        collision = [g for g in geoms if g.contype != 0 or g.conaffinity != 0]
        inferred = [g for g in geoms if not g.parent.explicitinertial]
        if collision:
            result[name] = (file, "collision", "used by a collision-enabled geom")
        elif inferred:
            bodies = sorted({g.parent.name for g in inferred})
            result[name] = (
                file,
                "mass-bearing",
                f"body {', '.join(bodies)} has no explicit <inertial>; "
                "MuJoCo infers it from this mesh",
            )
        elif geoms:
            result[name] = (file, "visual", "visual-only geoms on bodies with explicit inertia")
        else:
            result[name] = (file, "unused", "declared in <asset> but referenced by no geom")
    return result


def build(
    source: Path, out: Path, commit: str, *, ratio: float, min_faces: int
) -> list[MeshRecord]:
    """Write the vendored files into ``out`` (which must be empty or absent)."""

    out.mkdir(parents=True, exist_ok=True)
    (out / "assets").mkdir(exist_ok=True)
    for name in COPIED_FILES:
        shutil.copyfile(source / name, out / name)
    header = f"SO-101 Suite: decimated visual mesh from mujoco_menagerie@{commit[:7]}".encode()
    records: list[MeshRecord] = []
    for name, (file, role, reason) in sorted(classify_meshes(source / "so101.xml").items()):
        data = (source / "assets" / file).read_bytes()
        triangles = read_stl(data)
        if role in ("collision", "mass-bearing"):
            out_data = data
            faces_out = len(triangles)
        else:
            target = max(min_faces, int(round(len(triangles) * ratio)))
            if target >= len(triangles):
                out_data = data
                faces_out = len(triangles)
            else:
                decimated = decimate(triangles, target)
                out_data = write_stl(decimated, header)
                faces_out = len(decimated)
        (out / "assets" / file).write_bytes(out_data)
        records.append(
            MeshRecord(
                name=name,
                file=file,
                role=role,
                reason=reason,
                faces_in=len(triangles),
                faces_out=faces_out,
                bytes_in=len(data),
                bytes_out=len(out_data),
                sha_in=sha256(data),
                sha_out=sha256(out_data),
            )
        )
    return records


def _smooth_controls(model: Any, steps: int, dt: float, seed: int) -> np.ndarray:
    """Seeded sum-of-sinusoids control sequence inside every actuator's ctrlrange."""

    rng = np.random.default_rng(seed)
    t = np.arange(steps)[:, None] * dt
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    signal = np.zeros((steps, model.nu))
    for _ in range(3):
        freq = rng.uniform(0.05, 0.6, size=model.nu)
        phase = rng.uniform(0.0, 2.0 * np.pi, size=model.nu)
        signal += np.sin(2.0 * np.pi * freq * t + phase) / 3.0
    return np.asarray(mid + 0.9 * half * signal)


def _pickup_controls(model: Any, steps: int, dt: float) -> np.ndarray:
    """Close on the keyframe box, lift it, swing it, and drop it."""

    key = model.key("pickup").id
    base = np.array(model.key_ctrl[key], dtype=float)
    ctrl = np.repeat(base[None, :], steps, axis=0)
    t = np.arange(steps) * dt
    gripper = model.actuator("gripper").id
    lift = model.actuator("shoulder_lift").id
    pan = model.actuator("shoulder_pan").id
    close = np.clip((t - 0.5) / 1.0, 0.0, 1.0)
    raise_ = np.clip((t - 2.0) / 2.0, 0.0, 1.0)
    swing = np.clip((t - 5.0) / 3.0, 0.0, 1.0)
    release = t > 12.0
    ctrl[:, gripper] = base[gripper] + close * (-0.17453 - base[gripper])
    ctrl[release, gripper] = base[gripper]
    ctrl[:, lift] = base[lift] - 0.6 * raise_
    ctrl[:, pan] = base[pan] + 0.8 * swing
    return ctrl


def _rollout(model: Any, ctrl: np.ndarray, keyframe: str | None) -> dict[str, np.ndarray]:
    import mujoco

    data = mujoco.MjData(model)
    if keyframe is None:
        mujoco.mj_resetData(model, data)
    else:
        mujoco.mj_resetDataKeyframe(model, data, model.key(keyframe).id)
    qpos = np.empty((len(ctrl), model.nq))
    qvel = np.empty((len(ctrl), model.nv))
    ncon = np.empty(len(ctrl), dtype=np.int64)
    for i, row in enumerate(ctrl):
        data.ctrl[:] = row
        mujoco.mj_step(model, data)
        qpos[i] = data.qpos
        qvel[i] = data.qvel
        ncon[i] = data.ncon
    return {"qpos": qpos, "qvel": qvel, "ncon": ncon}


def verify_dynamics(source: Path, out: Path) -> dict[str, Any]:
    """Prove the vendored model is dynamically identical to the source model."""

    import mujoco

    src = mujoco.MjModel.from_xml_path(str(source / "scene_box.xml"))
    dst = mujoco.MjModel.from_xml_path(str(out / "scene_box.xml"))
    inertial_fields = (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "body_subtreemass",
        "jnt_range",
        "dof_damping",
        "dof_armature",
        "dof_frictionloss",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_forcerange",
        "actuator_ctrlrange",
        "site_pos",
        "site_quat",
        "cam_pos",
        "cam_quat",
    )
    mismatched = [
        f for f in inertial_fields if not np.array_equal(getattr(src, f), getattr(dst, f))
    ]
    collision = (src.geom_contype != 0) | (src.geom_conaffinity != 0)
    geom_fields = (
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_friction",
        "geom_solref",
        "geom_rbound",
    )
    mismatched += [
        f
        for f in geom_fields
        if not np.array_equal(getattr(src, f)[collision], getattr(dst, f)[collision])
    ]
    if mismatched:
        raise SystemExit(f"vendored model differs from source in: {', '.join(mismatched)}")

    scenarios: list[dict[str, Any]] = []
    for label, timestep, steps, keyframe in (
        ("random smooth controls, native timestep", None, 6000, None),
        ("random smooth controls, suite timestep 1/240 s", 1.0 / 240.0, 7200, None),
        ("pickup keyframe: close, lift, swing, release, 1/240 s", 1.0 / 240.0, 4800, "pickup"),
    ):
        for model in (src, dst):
            model.opt.timestep = timestep if timestep is not None else 0.005
        dt = float(src.opt.timestep)
        if keyframe is None:
            ctrl = _smooth_controls(src, steps, dt, seed=101)
        else:
            ctrl = _pickup_controls(src, steps, dt)
        a = _rollout(src, ctrl, keyframe)
        b = _rollout(dst, ctrl, keyframe)
        identical = all(np.array_equal(a[k], b[k]) for k in a)
        box_qpos = src.jnt_qposadr[_free_joint_id(src)]
        max_box_z = float(np.max(a["qpos"][:, box_qpos + 2]))
        scenarios.append(
            {
                "label": label,
                "timestep_s": dt,
                "steps": steps,
                "simulated_s": steps * dt,
                "identical": identical,
                "max_abs_qpos_diff": float(np.max(np.abs(a["qpos"] - b["qpos"]))),
                "max_abs_qvel_diff": float(np.max(np.abs(a["qvel"] - b["qvel"]))),
                "steps_with_contacts": int(np.count_nonzero(a["ncon"])),
                "max_contacts": int(np.max(a["ncon"])),
                "max_box_height_m": max_box_z,
            }
        )
        if not identical:
            raise SystemExit(f"dynamics differ for scenario: {label}")
    return {"fields_checked": len(inertial_fields) + len(geom_fields), "scenarios": scenarios}


def _free_joint_id(model: Any) -> int:
    import mujoco

    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return jid
    raise SystemExit("scene_box.xml has no free joint")


def compile_check(out: Path) -> dict[str, int]:
    import mujoco

    counts: dict[str, int] = {}
    for name in MJCF_FILES:
        model = mujoco.MjModel.from_xml_path(str(out / name))
        counts[name] = int(model.nmesh)
    return counts


def write_provenance(
    out: Path,
    *,
    commit: str,
    commit_date: str,
    clean: bool,
    records: list[MeshRecord],
    proof: dict[str, Any],
    ratio: float,
    min_faces: int,
    source: Path,
) -> None:
    import mujoco

    copied = []
    for name in COPIED_FILES:
        data = (source / name).read_bytes()
        copied.append((name, len(data), sha256(data)))
    total_in = sum(r.bytes_in for r in records) + sum(c[1] for c in copied)
    total_out = sum(r.bytes_out for r in records) + sum(c[1] for c in copied)
    decimated = [r for r in records if r.sha_in != r.sha_out]
    exact = [r for r in records if r.sha_in == r.sha_out]
    source_files = {p.name for p in (source / "assets").glob("*.stl")}
    dirty_note = "" if clean else " (checkout had local changes)"
    unreferenced = sorted(source_files - {r.file for r in records})
    lines = [
        "# Provenance: `robotstudio_so101`",
        "",
        "Generated by `scripts/vendor_so101_model.py`. Do not edit these files by hand;",
        "re-run the script (it is deterministic) and commit its output.",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Upstream | MuJoCo Menagerie, `{UPSTREAM_SUBDIR}` |",
        f"| URL | {UPSTREAM_REPO}/tree/{commit}/{UPSTREAM_SUBDIR} |",
        f"| Commit | `{commit}` ({commit_date}){dirty_note} |",
        "| License | Apache-2.0; `LICENSE` is copied verbatim from the upstream directory |",
        "| Original description | The Robot Studio SO-ARM100 "
        "`Simulation/SO101/so101_new_calib.xml`, per the Menagerie README |",
        "",
        "## Modifications",
        "",
        "1. `so101.xml`, `scene.xml`, `scene_box.xml` and `LICENSE` are byte-exact copies.",
        "   The suite applies runtime changes (timestep, added scene bodies and cameras) in",
        "   memory through `mujoco.MjSpec`. A subsequent G15 range adaptation is separate",
        "   from this base conversion; see `scripts/adapt_so101_g15_model.py`.",
        f"2. {len(exact)} meshes are byte-exact: the gripper collision meshes and the wrist",
        "   camera mount, whose body has no explicit `<inertial>` (MuJoCo infers that body's",
        "   mass properties from the mesh, so decimating it would change dynamics).",
        f"3. {len(decimated)} visual-only meshes were decimated with quadric edge collapse",
        f"   (`pymeshlab`, target {ratio:.0%} of the faces, at least {min_faces} faces).",
        "   Decimated STL headers read `SO-101 Suite: decimated visual mesh from",
        f"   mujoco_menagerie@{commit[:7]}` as the Apache-2.0 section 4(b) change notice.",
        "4. Not vendored: `README.md`, `CHANGELOG.md`, `so101.png`"
        + "".join(f", `assets/{name}`" for name in unreferenced)
        + " (not referenced by `so101.xml`).",
        "",
        f"Total size: {total_in:,} bytes upstream, {total_out:,} bytes vendored "
        "(excluding this file).",
        "",
        "## Files",
        "",
        "| File | Role | Faces in -> out | Bytes in -> out | SHA-256 in | SHA-256 out |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, size, digest in copied:
        lines.append(f"| `{name}` | copied | - | {size:,} -> {size:,} | `{digest}` | `{digest}` |")
    for r in sorted(records, key=lambda rec: rec.file):
        lines.append(
            f"| `assets/{r.file}` | {r.role} | {r.faces_in:,} -> {r.faces_out:,} | "
            f"{r.bytes_in:,} -> {r.bytes_out:,} | `{r.sha_in}` | `{r.sha_out}` |"
        )
    lines += ["", "Roles:", ""]
    for role in ("collision", "mass-bearing", "visual", "unused"):
        reasons = sorted({r.reason for r in records if r.role == role})
        if reasons:
            lines.append(f"- `{role}`: {'; '.join(reasons)}.")
    lines += [
        "",
        "## Dynamics equivalence",
        "",
        f"`verify_dynamics()` compiled `scene_box.xml` from both directories and compared "
        f"{proof['fields_checked']} model arrays (body mass, inertia and frames, joint, dof,",
        "actuator, site and camera parameters, and every collision geom). All were",
        "bitwise identical. It then ran the same control sequence through both models:",
        "",
        "| Scenario | Timestep (s) | Steps | Steps with contacts | Max box height (m) | "
        "Max abs qpos diff | Max abs qvel diff | Bitwise identical |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in proof["scenarios"]:
        lines.append(
            f"| {s['label']} | {s['timestep_s']:.6f} | {s['steps']:,} | "
            f"{s['steps_with_contacts']:,} | {s['max_box_height_m']:.4f} | "
            f"{s['max_abs_qpos_diff']:.1e} | {s['max_abs_qvel_diff']:.1e} | "
            f"{'yes' if s['identical'] else 'NO'} |"
        )
    lines += [
        "",
        "## Tools",
        "",
        f"- Python {platform.python_version()}, numpy {np.__version__}, "
        f"pymeshlab {importlib.metadata.version('pymeshlab')}, "
        f"mujoco {mujoco.__version__}.",
        "- Reproduce: `python scripts/vendor_so101_model.py --source "
        "<menagerie>/robotstudio_so101`; verify with `--check`.",
        "",
    ]
    (out / "PROVENANCE.md").write_text("\n".join(lines), encoding="utf-8")


def tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): sha256(p.read_bytes())
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source", type=Path, required=True, help="menagerie robotstudio_so101 dir"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--commit", help="override the upstream commit (default: git HEAD)")
    parser.add_argument("--ratio", type=float, default=0.12, help="visual mesh face ratio")
    parser.add_argument("--min-faces", type=int, default=1200)
    parser.add_argument("--budget-bytes", type=int, default=3_000_000)
    parser.add_argument("--check", action="store_true", help="rebuild in a temp dir and diff")
    parser.add_argument("--g15-operating-ranges", action="store_true")
    args = parser.parse_args(argv)

    source = args.source.resolve()
    if args.commit:
        commit, commit_date, clean = args.commit, "date not recorded", True
    else:
        commit, commit_date, clean = git_commit(source)

    with tempfile.TemporaryDirectory(prefix="so101_vendor_") as tmp:
        staging = Path(tmp) / "robotstudio_so101"
        records = build(source, staging, commit, ratio=args.ratio, min_faces=args.min_faces)
        meshes = compile_check(staging)
        proof = verify_dynamics(source, staging)
        write_provenance(
            staging,
            commit=commit,
            commit_date=commit_date,
            clean=clean,
            records=records,
            proof=proof,
            ratio=args.ratio,
            min_faces=args.min_faces,
            source=source,
        )
        if args.g15_operating_ranges:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("adapt_so101_g15_model.py")),
                    "--model",
                    str(staging / "so101.xml"),
                ],
                check=True,
            )
        total = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
        if total > args.budget_bytes:
            raise SystemExit(f"vendored size {total:,} exceeds budget {args.budget_bytes:,}")
        if args.check:
            expected, actual = tree_digest(staging), tree_digest(args.out)
            if expected != actual:
                changed = sorted(set(expected) ^ set(actual)) + sorted(
                    k for k in set(expected) & set(actual) if expected[k] != actual[k]
                )
                print("vendored files differ from a fresh build:", *changed, sep="\n  ")
                return 1
            print(f"OK: {len(actual)} vendored files match a fresh build ({total:,} bytes)")
            return 0
        if args.out.exists():
            shutil.rmtree(args.out)
        shutil.copytree(staging, args.out)
    print(f"vendored {len(records)} meshes into {args.out} ({total:,} bytes); meshes: {meshes}")
    for s in proof["scenarios"]:
        print(f"  {s['label']}: {s['steps']} steps identical={s['identical']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
