# Workstation and engineering scripts

The application lives in [`src/so101`](../src/so101). These are process entry
points and engineering utilities. Use root operator commands where available;
their paths are intentionally stable for existing service installations.

| Group | Main files |
| --- | --- |
| Service lifecycle | `suite_session.py` ([foreground runbook](../docs/suite-session.md)), `service_suite.py`, `start_local.py`, `motor_service.py` |
| Shared cameras and MCAP | `arm_camera.py`, `record_mcap.py`, `recording_storage.py`, `review_mcap_episodes.py` |
| Camera-to-table alignment | `camera_alignment.py`, `camera_alignment.html` — [phone target guide](../docs/camera-alignment.md); reads existing D435i relay, no motor access |
| Manual calibration and evidence | `calibrate_arm_manual.py`, `calibrate_leader.py`, `verify_arm_manual.py`, `install_arm_calibration.py` |
| Reference poses | `record_wrist_roll.py`, `check_shoulder_reference.py`, `derive_rest_pose.py` |
| Validation and packaging | `run_g15_ci.py`, `installed_e2e.py`, `run_smoke.py`, `check_coverage.py`, `check_dist.py`, `make_handoff.py` |
| Simulation experiments | `measure_sim_*.py`, `measure_health_latency.py`, `bc_baseline_measure.py` |
| Model provenance | `vendor_so101_model.py` (base conversion), `adapt_so101_g15_model.py` (documented G15 operating ranges) |
| Optional external viewer | `foxglove_viewer.py`, `foxglove_simulation.py`, `build_foxglove_layout.py`, `prepare_foxglove_models.py` |

Names such as `setup_motors.py` describe tools, not permission to run them on
the live rig. Calibration and motion tools have ownership and physical setup
requirements. See the [codebase tour](../docs/codebase-tour.md).
