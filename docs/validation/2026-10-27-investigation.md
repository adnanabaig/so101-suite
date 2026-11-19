# G15 investigation: manipulation pipeline

Date: 2026-10-27. This supplements the historical
[G15 validation checkpoint](g15.md). The action plan and research proposal are
in the [manipulation roadmap](../plans/2026-10-27-manipulation-roadmap.md).

## Conclusions supported by inspection

- Real calibration evidence exists and must be preserved. Follower has ten
  accepted sweeps per joint. Leader has six initial accepted sweeps per joint
  and ten gripper sweeps; its later wrist-roll overflow remains unresolved.
- Repeated out-of-model rest captures indicate an unresolved shoulder
  reference/rest-selection issue. Repeating the same folded capture is not
  evidence that the software limit should be widened.
- The teleop loop opens the native camera manager, while the real configuration
  names the same USB devices already owned by the camera relays. Shared UI
  preview integration does not resolve control-loop capture ownership.
- Raw MCAP interval annotations exist, but an independently sampled telemetry
  stream is not proof of complete control-tick training data. Native
  `TeleopStep` retains desired/accepted actions; the manager snapshot currently
  retains only the accepted action and observation.
- The simulation expert obtains object coordinates from simulator state.
  Real RGB-D localization and physical task execution are separate work.
- No RealSense was enumerated and the current application venv lacked
  `pyrealsense2` at the initial inspection. Model identification precedes
  SDK/mount compatibility work; later rig updates may supersede this inventory.

## Selected existing test failures

Seven failures from the previous full run were selected for investigation;
all seven reproduced. This is a targeted diagnostic selection, not a new full
test run or an estimate of the remaining total failures.

| Test module / case | Observed failure | Next diagnostic action |
| --- | --- | --- |
| `test_bringup_buses.py` / `test_open_feetech_bus_skips_the_handshake` | Test does not provide the required adopted canonical device lock | Update test setup to exercise the actual ownership contract |
| `test_control_plane.py` / `test_disarm_cancels_slow_in_progress_arm` | Disarm encounters bus-busy Halt fallback | Trace arming cancellation/stop ordering with deterministic timing |
| `test_control_faults.py` / `test_server_marker_write_failure_prevents_enable_and_latches_fault` | Expected fault field missing (`KeyError`) | Inspect fault persistence, returned state and injected failure boundary |
| `test_control_collapse.py` / `test_estop_preempted_fault_collapse_does_not_make_halt_report_collapsing` | Stop result reports `applied=True` where test expects false | Resolve E-stop precedence and observable stop-result semantics |
| `test_hardware.py` / `test_modern_lerobot_adapter_preserves_order_and_units` | Fake bus lacks required torque-off write/read-back support | Bring fake adapter contract up to date; retain production confirmation |
| `test_data_eval_reports.py` / `test_teardown_error_stays_primary_when_terminal_journal_fsync_also_fails` | Expected primary failure differs under fsync injection | Check error precedence and whether injection targets the intended write |
| `test_sim_physics_kinematics.py` / `test_golden_table_is_regenerable_from_mujoco` | Exact float equality fails between MuJoCo 3.14 and 3.16 | Use an explicit numeric comparison while preserving exact metadata/pose inputs; measured differences are below physical significance |

These observations identify investigation directions, not completed root-cause
fixes. Stop/fault cases remain priority gates for physical operation. The old
complete backend result remains 3,561 passes, 67 failures, ten skips; it must
not be presented as passing CI.

The FK comparison was subsequently quantified across all 54 poses. There were
no structural differences; the largest numeric difference was 3.61×10^-16
(quaternion component), and the largest position-component difference was
1.67×10^-16 metres. This is numerical precision drift, not evidence of a
physical model correction. Neither the golden fixture nor its tolerance was
changed by this investigation. Physical encoder-to-model alignment remains
a separate measurement.

## New storage verification

The storage, MCAP-session and service tests pass together: 22 cases. Ruff
checks/formatting pass on the two recorder scripts and storage tests; mypy
passes on the two scripts. The first shell invocation inherited ROS's Python
3.10 plugin path and failed during pytest startup; running the application
Python with `-I` and without inherited `PYTHONPATH` isolates the intended venv.
No retry changed a failing assertion into a pass.

A real eight-second observer recording to an external validation folder,
with five-second rotation and local status metadata, produced two completed
MCAPs. Both decode with CRC validation and contain all three camera image
topics, telemetry and encoder topics. This proves the new destination and
rotation path for that pilot, not motor tracking or training synchronization.

During restart recovery, the observation/API services were restarted without
arming. The API reported real backend, idle, disarmed, torque off and both
motor buses closed. Measured relay rates near 20:46 UTC were approximately
30/17/10 fps (OBSBOT/camera 1/camera 2). All requested 30 fps; only OBSBOT met
the target. These short measurements do not replace a sustained capture test.

USB descriptors were also read without opening another video stream. The
`camera_1` alias resolves to product 9005, which advertises 640×480 YUY2 with
30/25/20/15/10/5 fps intervals; no MJPEG format was advertised. `camera_2`
resolves to product 9221, which advertises MJPEG modes including 640×480 at
30 fps, and a 640×480 uncompressed mode. No 60 fps video mode appeared in
either descriptor inventory. A requested MJPEG setting cannot create MJPEG
support in camera 1; verify negotiated format instead of assuming the setter
succeeded. The slower delivered rates remain unexplained by descriptors alone;
exposure, negotiated capture format, USB allocation and capture/encode timing
need controlled measurements. No exposure or stream setting was changed here.

Migration completion and production recorder recovery are tracked separately
in the local migration receipt/verification artifacts. At this document's
initial creation, the migration was still running and the production recorder
was intentionally stopped.

Completion update, 21:02 UTC: all 1,365 original files (423,468,310,174 bytes)
are accounted for by destination read-back hash receipts, matching final sizes
and verified legacy symlinks. No incomplete transfers remain. About 394 GiB
was free internally and 504 GiB externally at audit. A newly closed production
segment on SigmaDrive decoded with CRC validation, all three raw camera image
topics and 421 telemetry messages. The recorder is active again on the external
path with fresh local status metadata. This establishes storage recovery, not
clearance for physical teleop.

Raw test logs, runtime snapshots, calibration files and MCAPs remain local.
No physical motion, calibration write, fault reset, limit widening, Git commit
or publication was performed by this investigation.
