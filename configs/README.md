# Reference presets

These YAML files describe robot layouts, camera presets, simulation scenarios,
tracking and agent skills. Their comments identify which values are reference
documentation versus settings consumed by a specific subsystem.

Real motor configuration comes through the application settings and installed
calibration, not by treating a reference YAML range as a fresh measurement.
See [`src/so101/config.py`](../src/so101/config.py) and
[`.env.example`](../.env.example).

Service definitions, local rig profiles and external-viewer layouts are in
[`config/`](../config). Existing paths remain stable for deployment compatibility.
