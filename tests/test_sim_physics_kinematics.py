"""One kinematic truth (audit B2): numpy FK from the vendored MJCF.

These tests run without MuJoCo. Tests that compare against MuJoCo directly
call ``pytest.importorskip("mujoco")`` themselves.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from so101 import JOINT_NAMES
from so101.hardware.safety import DEFAULT_LIMITS
from so101.simulation import units
from so101.simulation.kinematics import JOINT_LIMITS, MODEL_ID, forward_kinematics
from so101.simulation.mjcf import (
    MODEL_COMMIT,
    MODEL_DIR,
    MjcfError,
    mat_to_quat,
    parse_mjcf,
    quat_to_mat,
    so101_model,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "fixtures" / "so101_fk_golden.json"
SRC = ROOT / "src"


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _quat_close(a, b, atol=1e-9) -> bool:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return bool(np.allclose(a, b, atol=atol) or np.allclose(a, -b, atol=atol))


# -- golden table -------------------------------------------------------------
def test_numpy_fk_matches_mujoco_golden_table_within_a_tenth_of_a_millimetre():
    golden = _golden()
    assert golden["model_commit"] == MODEL_COMMIT
    assert golden["joint_order"] == list(JOINT_NAMES)
    assert len(golden["poses"]) >= 50
    model = so101_model()
    worst = 0.0
    for entry in golden["poses"]:
        qpos = units.suite_to_mjcf(entry["suite"])
        assert [qpos[joint] for joint in JOINT_NAMES] == pytest.approx(entry["qpos_rad"], abs=1e-12)
        poses = model.body_poses(qpos)
        site = model.site_pose("gripperframe", qpos)
        worst = max(worst, float(np.abs(site.pos - entry["gripperframe"]["pos"]).max()))
        assert _quat_close(mat_to_quat(site.rot), entry["gripperframe"]["quat_wxyz"], atol=1e-7)
        for body, expected in entry["bodies"].items():
            worst = max(worst, float(np.abs(poses[body].pos - expected).max()))
        for geom_name, expected in entry["jaw_tips"].items():
            body, geom = model.geom(geom_name)
            worst = max(worst, float(np.abs(poses[body.name].apply(geom.pos).pos - expected).max()))
        fk = forward_kinematics(entry["suite"])
        assert fk["position_m"] == pytest.approx(entry["gripperframe"]["pos"], abs=1e-4)
    assert worst <= golden["tolerance_m"] == 1e-4
    # In practice the two agree to floating-point precision.
    assert worst < 1e-9


def test_golden_table_is_regenerable_from_mujoco():
    pytest.importorskip("mujoco")
    spec = importlib.util.spec_from_file_location(
        "generate_fk_golden", ROOT / "scripts" / "generate_fk_golden.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fresh = json.loads(json.dumps(module.build()))
    assert module.equivalent(fresh, _golden())


def test_golden_equivalence_only_tolerates_numeric_fk_roundoff():
    spec = importlib.util.spec_from_file_location(
        "generate_fk_golden", ROOT / "scripts" / "generate_fk_golden.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = _golden()
    generated = copy.deepcopy(original)
    generated["mujoco_version"] = "another supported version"
    generated["poses"][0]["gripperframe"]["pos"][0] += 5e-13
    assert module.equivalent(original, generated)
    generated["poses"][0]["gripperframe"]["pos"][0] += 2e-9
    assert not module.equivalent(original, generated)
    for field in ("suite", "qpos_rad", "tolerance_m", "body_structure", "nonfinite"):
        changed = copy.deepcopy(original)
        if field == "suite":
            changed["poses"][0]["suite"]["shoulder_pan"] += 1e-13
        elif field == "qpos_rad":
            changed["poses"][0]["qpos_rad"][0] += 1e-13
        elif field == "tolerance_m":
            changed["tolerance_m"] *= 2
        elif field == "body_structure":
            changed["poses"][0]["bodies"].pop("base")
        else:
            changed["poses"][0]["gripperframe"]["pos"][0] = float("nan")
        assert not module.equivalent(original, changed), field


def test_fk_engine_and_units_import_without_mujoco():
    code = (
        "import sys\n"
        "sys.modules['mujoco'] = None\n"  # any `import mujoco` now raises ImportError
        "from so101.simulation import SimulationEngine\n"
        "from so101.simulation.kinematics import forward_kinematics, model_description\n"
        "engine = SimulationEngine()\n"
        "engine.step(3, targets={'shoulder_lift': 20.0})\n"
        "fk = forward_kinematics(engine.state()['telemetry']['measured'])\n"
        "assert len(model_description()['bodies']) == 8\n"
        "print(fk['model'])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PYTHONPATH": str(SRC), "PATH": "/usr/bin:/bin", "SO101_MOCK": "1"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == MODEL_ID


def test_forward_kinematics_contract_and_tool_frame():
    neutral = {joint: 0.0 for joint in JOINT_NAMES} | {"gripper": 50.0}
    fk = forward_kinematics(neutral)
    assert fk["model"] == MODEL_ID == "mjcf:robotstudio_so101@4d038b3"
    assert fk["tool_frame"] == "gripperframe"
    assert fk["position_m"] == pytest.approx([0.3913615, -0.00097947, 0.24634518], abs=1e-7)
    assert len(fk["orientation_rpy_deg"]) == 3
    assert len(fk["orientation_quat_wxyz"]) == 4
    assert fk["frames"][0]["name"] == "base"
    assert fk["frames"][-1] == {"name": "gripperframe", "position_m": fk["position_m"]}
    rot = quat_to_mat(fk["orientation_quat_wxyz"])
    roll, pitch, yaw = (math.radians(value) for value in fk["orientation_rpy_deg"])
    rz = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]
    )
    ry = np.array(
        [[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]]
    )
    rx = np.array(
        [[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]]
    )
    assert np.allclose(rz @ ry @ rx, rot, atol=1e-7)

    apertures = [
        forward_kinematics(neutral | {"gripper": value})["gripper_aperture_m"]
        for value in (0.0, 25.0, 50.0, 75.0, 100.0)
    ]
    assert apertures == sorted(apertures)
    assert apertures[0] < 0.005 < 0.1 < apertures[-1]

    # Shoulder lift pitches the arm about the shoulder: the tool height changes.
    raised = forward_kinematics(neutral | {"shoulder_lift": -45.0})
    assert raised["position_m"] != pytest.approx(fk["position_m"], abs=1e-3)


# -- units and limits ---------------------------------------------------------
def test_suite_mjcf_round_trip_and_gripper_mapping():
    maps = units.joint_maps()
    assert maps["shoulder_pan"].kind == "revolute_deg"
    assert maps["shoulder_pan"].scale == pytest.approx(math.pi / 180)
    assert maps["gripper"].kind == "gripper_linear"
    assert units.suite_to_mjcf({"gripper": 0.0})["gripper"] == pytest.approx(-0.17453)
    assert units.suite_to_mjcf({"gripper": 100.0})["gripper"] == pytest.approx(1.74533)
    assert units.suite_to_mjcf({"elbow_flex": 90.0})["elbow_flex"] == pytest.approx(math.pi / 2)

    rng = np.random.default_rng(7)
    for _ in range(25):
        pose = {joint: float(rng.uniform(*JOINT_LIMITS[joint])) for joint in JOINT_NAMES}
        back = units.mjcf_to_suite(units.suite_to_mjcf(pose))
        assert back == pytest.approx(pose, abs=1e-9)
        vector = np.array([pose[joint] for joint in JOINT_NAMES])
        rad = units.suite_to_mjcf_array(vector)
        assert rad == pytest.approx([units.suite_to_mjcf(pose)[j] for j in JOINT_NAMES])
        assert units.mjcf_to_suite_array(rad) == pytest.approx(vector, abs=1e-9)
    assert units.velocity_mjcf_to_suite("shoulder_pan", math.pi) == pytest.approx(180.0)
    assert units.velocity_suite_to_mjcf("gripper", 100.0) == pytest.approx(1.91986)


def test_effective_limits_are_the_safety_envelope_intersected_with_the_mjcf():
    model_limits = units.mjcf_limits()
    assert model_limits["elbow_flex"] == (-96.83, 99.0)
    assert model_limits["wrist_roll"] == (-157.21, 162.79)
    assert model_limits["gripper"] == (0.0, 100.0)
    for joint in JOINT_NAMES:
        low = max(DEFAULT_LIMITS[joint][0], model_limits[joint][0])
        high = min(DEFAULT_LIMITS[joint][1], model_limits[joint][1])
        assert JOINT_LIMITS[joint] == units.effective_limits()[joint] == (low, high)
        # Rounding to 0.01 can widen a model bound by at most 0.005 suite units.
        joint_range = so101_model().joint(joint).range
        assert joint_range is not None
        model_low, model_high = (units.joint_maps()[joint].to_suite(v) for v in joint_range)
        assert JOINT_LIMITS[joint][0] >= model_low - 0.005
        assert JOINT_LIMITS[joint][1] <= model_high + 0.005


def test_joint_order_is_asserted_not_assumed():
    model = so101_model()
    assert [joint.name for joint in model.joints] == list(JOINT_NAMES)
    assert [actuator.name for actuator in model.actuators] == list(JOINT_NAMES)
    units.assert_joint_order(JOINT_NAMES, JOINT_NAMES, JOINT_NAMES)
    swapped = list(JOINT_NAMES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(units.JointOrderError, match="joints"):
        units.assert_joint_order(swapped, JOINT_NAMES)
    with pytest.raises(units.JointOrderError, match="actuators"):
        units.assert_joint_order(JOINT_NAMES, swapped)
    with pytest.raises(units.JointOrderError, match="drive"):
        units.assert_joint_order(JOINT_NAMES, JOINT_NAMES, swapped)


# -- MJCF reader ----------------------------------------------------------------
_ORIENTATION_MJCF = """
<mujoco model="frames">
  <compiler angle="{angle}" eulerseq="{seq}"/>
  <default>
    <joint damping="1"/>
    <default class="arm"><joint axis="0 1 0" range="-30 45"/></default>
  </default>
  <worldbody>
    <body name="a" pos="0.1 0 0.2" euler="{e1}">
      <joint name="j1" class="arm"/>
      <geom type="box" size="0.01 0.02 0.03"/>
      <body name="b" pos="0 0.1 0" axisangle="1 1 0 {aa}">
        <joint name="j2" axis="1 0 0" pos="0.01 0.02 0" ref="{ref}"/>
        <geom type="sphere" size="0.01"/>
        <body name="c" pos="0.05 0 0" xyaxes="0 1 0 -1 0.2 0">
          <joint name="j3" axis="0 0 1"/>
          <geom type="sphere" size="0.01"/>
          <body name="d" pos="0 0 0.05" zaxis="0.3 -0.4 0.8">
            <geom type="sphere" size="0.01"/>
            <site name="tip" pos="0.01 0 0.02" quat="0.9 0.1 0.2 0.3"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.parametrize(
    ("angle", "seq", "e1", "aa", "ref"),
    [
        ("degree", "xyz", "10 20 30", "40", "5"),
        ("radian", "XYZ", "0.3 -0.2 0.9", "1.1", "0.2"),
        ("radian", "zyx", "1.0 0.5 -0.7", "-0.4", "0"),
        ("degree", "ZXz", "-15 70 5", "120", "-10"),
    ],
)
def test_reader_orientation_forms_match_mujoco(tmp_path, angle, seq, e1, aa, ref):
    mujoco = pytest.importorskip("mujoco")
    path = tmp_path / "frames.xml"
    path.write_text(
        _ORIENTATION_MJCF.format(angle=angle, seq=seq, e1=e1, aa=aa, ref=ref), encoding="utf-8"
    )
    parsed = parse_mjcf(path)
    compiled = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(compiled)
    for body in parsed.bodies:
        body_id = compiled.body(body.name).id
        assert np.allclose(body.pos, compiled.body_pos[body_id])
        assert _quat_close(body.quat, compiled.body_quat[body_id], atol=1e-9), body.name
    j1 = parsed.joint("j1")
    assert j1.axis == pytest.approx((0.0, 1.0, 0.0))
    assert j1.range == pytest.approx(tuple(compiled.jnt_range[compiled.joint("j1").id]))
    rng = np.random.default_rng(3)
    for _ in range(10):
        qpos = rng.uniform(-1.0, 1.0, size=3)
        data.qpos[:] = qpos
        mujoco.mj_kinematics(compiled, data)
        pose = parsed.site_pose("tip", dict(zip(("j1", "j2", "j3"), qpos, strict=True)))
        site_id = compiled.site("tip").id
        assert np.allclose(pose.pos, data.site_xpos[site_id], atol=1e-12)
        assert np.allclose(pose.rot, data.site_xmat[site_id].reshape(3, 3), atol=1e-12)


@pytest.mark.parametrize(
    ("snippet", "message"),
    [
        ('<body name="a"><joint name="s" type="slide"/></body>', "only hinge"),
        ('<body name="a"><frame/></body>', "not supported"),
        ('<body name="a"><freejoint/></body>', "not supported"),
        ('<body name="a"><geom type="capsule" fromto="0 0 0 1 0 0" size="0.1"/></body>', "fromto"),
        ('<body name="a"><joint name="x"/><joint name="y"/></body>', "at most one"),
        ('<body name="a"><geom class="missing" size="0.1"/></body>', "unknown default class"),
        ('<body name="a" quat="1 0 0 0" euler="0 0 0"/>', "more than one orientation"),
    ],
)
def test_reader_refuses_mjcf_it_does_not_model(tmp_path, snippet, message):
    path = tmp_path / "bad.xml"
    path.write_text(
        f'<mujoco model="bad"><worldbody>{snippet}</worldbody></mujoco>', encoding="utf-8"
    )
    with pytest.raises(MjcfError, match=message):
        parse_mjcf(path)


def test_vendored_model_tree_matches_the_compiled_mujoco_model():
    mujoco = pytest.importorskip("mujoco")
    parsed = so101_model()
    compiled = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "so101.xml"))
    assert [compiled.body(i).name for i in range(1, compiled.nbody)] == [
        body.name for body in parsed.bodies
    ]
    checked = 0
    for body in parsed.bodies:
        body_id = compiled.body(body.name).id
        geom_ids = [g for g in range(compiled.ngeom) if compiled.geom_bodyid[g] == body_id]
        assert len(geom_ids) == len(body.geoms)
        for geom_id, geom in zip(geom_ids, body.geoms, strict=True):
            assert int(compiled.geom_contype[geom_id]) == geom.contype
            assert int(compiled.geom_conaffinity[geom_id]) == geom.conaffinity
            assert int(compiled.geom_group[geom_id]) == geom.group
            material = compiled.geom_matid[geom_id]
            rgba = compiled.geom_rgba[geom_id]
            if material >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
                rgba = compiled.mat_rgba[material]
            assert np.allclose(rgba, geom.rgba, atol=1e-6)
            if geom.kind != "mesh":  # MuJoCo recentres meshes but not primitives
                assert np.allclose(geom.pos, compiled.geom_pos[geom_id])
                assert _quat_close(geom.quat, compiled.geom_quat[geom_id])
            checked += 1
    assert checked == compiled.ngeom
    for actuator in parsed.actuators:
        actuator_id = compiled.actuator(actuator.name).id
        assert actuator.ctrlrange == pytest.approx(tuple(compiled.actuator_ctrlrange[actuator_id]))
