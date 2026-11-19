"""MuJoCo-only SO101 motion worker; JSON telemetry on stdout, no hardware access."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request

from so101.simulation.mjcf import MODEL_DIR, mat_to_quat, so101_model
from so101.simulation.physics_engine import PhysicsSimulationEngine
from so101.simulation.units import suite_to_mjcf


def targets(seconds: float) -> dict[str, float]:
    return {
        "shoulder_pan": 25 * math.sin(seconds * 0.35),
        "shoulder_lift": -20 + 15 * math.sin(seconds * 0.3),
        "elbow_flex": 30 + 20 * math.sin(seconds * 0.4),
        "wrist_flex": 25 * math.sin(seconds * 0.5),
        "wrist_roll": 35 * math.sin(seconds * 0.3),
        "gripper": 50 + 30 * math.sin(seconds * 0.5),
    }


def model_parts(model):
    return [
        {
            "id": f"{body.name}-{index}",
            "body": body.name,
            "mesh": str(MODEL_DIR / "assets" / model.meshes[geom.mesh]),
            "rgba": list(model.materials.get(geom.material, geom.rgba)),
            "geom": geom,
        }
        for body in model.bodies
        for index, geom in enumerate(body.geoms)
        if geom.visual and geom.kind == "mesh"
    ]


def snapshot(engine, model, parts, seconds):
    state = engine.step(3, targets=targets(seconds))
    measured = state["telemetry"]["measured"]
    poses = model.body_poses(suite_to_mjcf(measured))
    models = []
    for part in parts:
        pose = poses[part["body"]].apply(part["geom"].pos, part["geom"].quat)
        models.append(
            {
                "id": part["id"],
                "mesh": part["mesh"],
                "rgba": part["rgba"],
                "position": list(pose.pos),
                "quaternion_wxyz": list(mat_to_quat(pose.rot)),
            }
        )
    return {
        "source": "mujoco_simulation",
        "physical_arm_connected": False,
        "elapsed_seconds": seconds,
        "measured": measured,
        "target": state["telemetry"]["target"],
        "models": models,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--source", choices=("simulation", "encoders"), default="simulation")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.source == "simulation":
        model, engine = so101_model(), PhysicsSimulationEngine()
        parts = model_parts(model)
    index = 0
    while not args.frames or index < args.frames:
        started = time.monotonic()
        if args.source == "simulation":
            data = snapshot(engine, model, parts, index / 20)
        else:
            try:
                with urllib.request.urlopen(
                    args.api_base + "/api/mcap/encoder-model", timeout=1
                ) as response:
                    data = json.load(response)
            except (OSError, ValueError) as exc:
                data = {"source": "motor_encoders", "live": False, "error": str(exc), "models": []}
        print(json.dumps(data), flush=True)
        index += 1
        time.sleep(max(0, 0.05 - (time.monotonic() - started)))


if __name__ == "__main__":
    main()
