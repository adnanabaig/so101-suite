"""``GET /api/simulation/model`` and ``GET /api/simulation/assets/{file}``.

Both endpoints are served from the vendored MJCF and need no MuJoCo. The
MuJoCo cross-checks skip when it is not installed.
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np
import pytest
from fastapi.testclient import TestClient

from so101 import JOINT_NAMES
from so101.services.app import create_app
from so101.simulation.kinematics import JOINT_LIMITS, forward_kinematics
from so101.simulation.mjcf import MODEL_COMMIT, MODEL_DIR, so101_model

MESH_DIR = MODEL_DIR / "assets"


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def model_json(client) -> dict:
    response = client.get("/api/simulation/model")
    assert response.status_code == 200
    return response.json()


def _mat(quat) -> np.ndarray:
    w, x, y, z = (float(value) for value in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    half = angle / 2.0
    return _mat([math.cos(half), *(axis * math.sin(half))])


def test_model_endpoint_has_the_published_shape(model_json):
    assert model_json["name"] == "so101"
    assert model_json["source"]["commit"] == MODEL_COMMIT
    assert model_json["source"]["license"] == "Apache-2.0"
    assert model_json["source"]["repo"].endswith("mujoco_menagerie")
    assert model_json["units"] == {"length": "m", "angle": "rad"}
    assert model_json["up_axis"] == "z"
    bodies = model_json["bodies"]
    assert [body["name"] for body in bodies][:6] == [
        "base",
        "shoulder",
        "upper_arm",
        "lower_arm",
        "wrist",
        "gripper",
    ]
    assert bodies[0]["parent"] is None and bodies[0]["joint"] is None
    seen = set()
    for body in bodies:
        assert set(body) == {"name", "parent", "pos", "quat", "joint", "geoms"}
        assert body["parent"] is None or body["parent"] in seen  # parents come first
        seen.add(body["name"])
        assert len(body["pos"]) == 3 and len(body["quat"]) == 4
        if body["joint"] is not None:
            assert set(body["joint"]) == {"name", "axis", "range"}
        for geom in body["geoms"]:
            assert len(geom["pos"]) == 3 and len(geom["quat"]) == 4 and len(geom["rgba"]) == 4
            if geom["kind"] == "mesh":
                assert geom["mesh"] in model_json["meshes"]
            else:
                assert geom["kind"] in {"box", "capsule", "cylinder", "sphere"}
                assert geom["size"]
    joints = [body["joint"]["name"] for body in bodies if body["joint"]]
    assert joints == list(JOINT_NAMES)
    assert set(model_json["joint_map"]) == set(JOINT_NAMES)
    for suite_joint, entry in model_json["joint_map"].items():
        assert entry["mjcf_joint"] == suite_joint
        expected = "gripper_linear" if suite_joint == "gripper" else "revolute_deg"
        assert entry["kind"] == expected
    assert model_json["sites"]["gripperframe"]["body"] == "gripper"
    for url in model_json["meshes"].values():
        assert re.fullmatch(r"/api/simulation/assets/[a-z0-9_]+\.stl", url)


def test_model_json_alone_reproduces_forward_kinematics(model_json):
    """A client applying the documented convention gets the server's FK."""

    joint_map = model_json["joint_map"]
    site = model_json["sites"]["gripperframe"]
    rng = np.random.default_rng(11)
    for _ in range(20):
        suite = {joint: float(rng.uniform(*JOINT_LIMITS[joint])) for joint in JOINT_NAMES}
        world: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for body in model_json["bodies"]:
            parent_pos, parent_rot = world.get(body["parent"], (np.zeros(3), np.eye(3)))
            pos = parent_pos + parent_rot @ np.asarray(body["pos"])
            rot = parent_rot @ _mat(body["quat"])
            joint = body["joint"]
            if joint is not None:
                entry = next(e for e in joint_map.values() if e["mjcf_joint"] == joint["name"])
                suite_joint = next(k for k, v in joint_map.items() if v is entry)
                angle = entry["scale"] * suite[suite_joint] + entry["offset"]
                rot = rot @ _axis_angle(joint["axis"], angle)
            world[body["name"]] = (pos, rot)
        body_pos, body_rot = world[site["body"]]
        tool = body_pos + body_rot @ np.asarray(site["pos"])
        assert tool == pytest.approx(forward_kinematics(suite)["position_m"], abs=1e-9)


def test_visual_mesh_placement_matches_compiled_mujoco(model_json):
    """Raw STL + published geom pos/quat == MuJoCo's recentred mesh + its geom frame."""

    mujoco = pytest.importorskip("mujoco")
    compiled = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "so101.xml"))
    stl = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    checked = 0
    for body in model_json["bodies"]:
        body_id = compiled.body(body["name"]).id
        visual_ids = [
            g
            for g in range(compiled.ngeom)
            if compiled.geom_bodyid[g] == body_id
            and compiled.geom_contype[g] == 0
            and compiled.geom_conaffinity[g] == 0
        ]
        assert len(visual_ids) == len(body["geoms"])
        for geom_id, geom in zip(visual_ids, body["geoms"], strict=True):
            if geom["kind"] != "mesh":
                continue
            raw = (MESH_DIR / model_json["meshes"][geom["mesh"]].rsplit("/", 1)[1]).read_bytes()
            count = int(np.frombuffer(raw, "<u4", 1, 80)[0])
            vertices = np.frombuffer(raw, stl, count, 84)["v"].reshape(-1, 3).astype(float)
            ours = vertices @ _mat(geom["quat"]).T + geom["pos"]
            mesh_id = compiled.geom_dataid[geom_id]
            start, size = compiled.mesh_vertadr[mesh_id], compiled.mesh_vertnum[mesh_id]
            theirs = compiled.mesh_vert[start : start + size].astype(float)
            theirs = theirs @ _mat(compiled.geom_quat[geom_id]).T + compiled.geom_pos[geom_id]
            assert ours.min(axis=0) == pytest.approx(theirs.min(axis=0), abs=2e-6)
            assert ours.max(axis=0) == pytest.approx(theirs.max(axis=0), abs=2e-6)
            checked += 1
    assert checked == len(
        [g for body in model_json["bodies"] for g in body["geoms"] if g["kind"] == "mesh"]
    )
    assert checked >= 14


def test_model_endpoint_supports_etag_revalidation(client):
    first = client.get("/api/simulation/model")
    etag = first.headers["etag"]
    assert first.headers["cache-control"] == "no-cache"
    again = client.get("/api/simulation/model", headers={"If-None-Match": etag})
    assert again.status_code == 304


def test_asset_endpoint_serves_every_listed_mesh(client, model_json):
    for url in model_json["meshes"].values():
        response = client.get(url)
        assert response.status_code == 200, url
        assert response.headers["content-type"] == "model/stl"
        # Revalidated on every use: the URL stays the same when a mesh is re-vendored.
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.content == (MESH_DIR / url.rsplit("/", 1)[1]).read_bytes()
        etag = response.headers["etag"]
        assert etag.startswith('"') and etag.endswith('"')  # strong, not W/"..."
        cached = client.get(url, headers={"If-None-Match": etag})
        assert cached.status_code == 304
        assert cached.content == b""
        assert cached.headers["etag"] == etag and cached.headers["cache-control"] == "no-cache"
        changed = client.get(url, headers={"If-None-Match": '"a-mesh-before-a-re-vendor"'})
        assert changed.status_code == 200 and changed.content == response.content


@pytest.mark.parametrize(
    "path",
    [
        "so101.xml",
        "LICENSE",
        "PROVENANCE.md",
        "moving_jaw_so101_gripper_part0_v1.stl",  # collision mesh: not a visual asset
        "moving_jaw_so101_gripper_v1.stl",  # vendored but unused
        "BASE_SO101_V2.STL",  # case variant (case-insensitive filesystems)
        "base_so101_v2.stl.",
        "base_so101_v2",
        "..%2Fso101.xml",
        "%2e%2e%2fso101.xml",
        "..%5Cso101.xml",
        "assets%2Fbase_so101_v2.stl",
        "%2Fetc%2Fpasswd",
        "..",
        "base_so101_v2.stl%00",
    ],
)
def test_asset_endpoint_rejects_anything_off_the_allowlist(client, path):
    response = client.get(f"/api/simulation/assets/{path}", follow_redirects=False)
    assert response.status_code == 404, path
    assert not response.content.startswith(b"<mujoco")


def test_raw_traversal_paths_never_reach_vendored_files(client):
    for path in ("/api/simulation/assets/../assets/../so101.xml", "/api/simulation/assets/a/b.stl"):
        response = client.get(path)
        assert response.status_code == 404
        assert b"<mujoco" not in response.content
    # A trailing slash never serves another file. Without a built UI, Starlette
    # redirects to the canonical, allowlisted URL. With one (a checkout's
    # frontend/dist), the SPA fallback answers backend paths with a JSON 404.
    slash = client.get("/api/simulation/assets/base_so101_v2.stl%2F", follow_redirects=False)
    if slash.status_code == 307:
        assert slash.headers["location"].endswith("/api/simulation/assets/base_so101_v2.stl")
    else:
        assert slash.status_code == 404
        assert slash.json() == {"detail": "Not Found"}


def test_vendored_files_match_provenance_hashes():
    """Byte-for-byte integrity of the vendored model against PROVENANCE.md."""

    table = (MODEL_DIR / "PROVENANCE.md").read_text(encoding="utf-8")
    rows = re.findall(r"^\| `([^`]+)` \| (\w[\w-]*) \|.*\| `([0-9a-f]{64})` \|$", table, re.M)
    assert len(rows) == 22
    for relative, _role, sha_out in rows:
        digest = hashlib.sha256((MODEL_DIR / relative).read_bytes()).hexdigest()
        assert digest == sha_out, relative
    vendored = {p.relative_to(MODEL_DIR).as_posix() for p in MODEL_DIR.rglob("*") if p.is_file()}
    assert vendored == {row[0] for row in rows} | {"PROVENANCE.md"}
    assert set(so101_model().meshes.values()) <= {row[0].removeprefix("assets/") for row in rows}
