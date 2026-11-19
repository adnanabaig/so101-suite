# Workstation integration configuration

- `services/`: service unit templates for the motor API, cameras and recorder.
- `rigs/`: example machine settings and ignored private local storage profiles.
- `visualization/`: optional external-viewer layouts and OBS integration files.
- `launch-*.sh`: launch helpers retained for existing workflows.

Runtime settings are defined by [`src/so101/config.py`](../src/so101/config.py)
and documented in [`.env.example`](../.env.example). The workstation's `.env`
contains private, machine-specific values and is excluded from Git.

Reference YAML presets are in [`configs/`](../configs), a separate purpose
from these workstation integration files. See the [codebase tour](../docs/codebase-tour.md).
