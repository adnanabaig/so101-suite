"""Convert the vendored model's STL assets to locally served GLB meshes."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import trimesh

    source = ROOT / "src/so101/simulation/assets/robotstudio_so101"
    destination = ROOT / ".artifacts/foxglove-models"
    destination.mkdir(parents=True, exist_ok=True)
    index = {}
    for path in sorted((source / "assets").glob("*.stl")):
        output = destination / (path.stem + ".glb")
        trimesh.load(path, force="mesh").export(output)
        index[path.name] = {
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "glb_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    shutil.copyfile(source / "LICENSE", destination / "LICENSE")
    (destination / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"Prepared {len(index)} GLB models in {destination}")


if __name__ == "__main__":
    main()
