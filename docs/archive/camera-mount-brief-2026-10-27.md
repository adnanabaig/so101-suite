# Historical camera mounting brief — 2026-10-27

The maintained guide is [hardware/camera-mount-brief.md](../hardware/camera-mount-brief.md).
This earlier draft is retained for reference.

Plan date: 2026-10-27. Confirmed camera: RealSense D435i. The operator deferred
3D printing and spare-camera design. Current work uses available cameras and
existing rigid supports. CAD requirements below are future reference only.

## Proposed rig

Mount the D435i rigidly to the table/rig overlooking the picking area and tray.
Start with an adjustable height and tilt, then lock both before calibration.
Keep its support outside the follower's reachable envelope and the leader's
manual operating space. A fixed mount makes the camera-to-base transform
constant and keeps camera mass off the arm.

```text
                rigid support outside arm travel
                    [D435i]
                       \  RGB + depth
                        \___________
     [follower base] --> picking area --> tray

     [leader base]        operator space
     OBSBOT: side review view; spare camera: role after identification
```

The D435i has two M3 mounting points and a tripod attachment. Use the official
mechanical drawing for hole spacing and screw insertion limits, then measure
the actual unit before choosing fasteners. Do not copy D455 mount dimensions.
The [D400 datasheet](https://realsenseai.com/wp-content/uploads/2025/09/Intel-RealSense-D400-Series-Datasheet-October-2025.pdf)
contains the D435/D435i drawing; the
[D435i product page](https://www.realsenseai.com/products/depth-camera-d435i/)
lists its mounting interfaces. Screw length must account for bracket and
washer thickness and must not bottom out inside the camera.

## Inputs needed before CAD

| Input | Why it matters |
| --- | --- |
| D435i mounting interface and measured hole spacing | Fastener placement and insertion clearance |
| Table thickness or extrusion dimensions | Clamp/support geometry |
| Arm bases, task area and tray positions | Coverage and collision clearance |
| Camera-to-table distance at candidate modes | Valid depth and RGB coverage across object height |
| USB plug envelope and cable bend radius | Strain relief without pulling the camera |
| Printer build volume, material and hardware available | Part split, tolerances and fastening |
| Spare camera model, mass, dimensions and connector | Fixed versus wrist mounting feasibility |

Use a temporary rigid tripod to establish field of view before printing.
Check the actual depth profile's minimum range and the closest raised object,
not just distance to the empty table. Select height from measured coverage and
valid depth; there is no universal mounting height for this workspace.

## CAD deliverables

Create an editable parametric source, STEP assembly and printable STL files,
with a dimensioned drawing, parts/fastener list and version identifier. Favor
a table clamp or existing rigid support, a camera plate, adjustable tilt and
a separate strain-relief feature. Test a small fit coupon before a full print.
Use positive locking for the final angle; check creep/flex under cable load.

Keep all lenses, projector, ventilation and connector access unobstructed.
Check the whole arm envelope, wrist-roll cable travel and rest transition.
Do not load thin cosmetic covers or cover motor joints with the bracket.
Printer/material-specific settings follow the actual printer and fit test.

## Hardware and software acceptance

1. Enumerate the D435i serial, firmware, USB link and supported simultaneous
   profiles. The Suite adapter exists, but physical streaming is unverified.
   Install the pinned RealSense extra; firmware updates are a separate action.
2. Test color and depth together on USB 3. Request 30 fps first. Measure capture
   rate, frame age, dropped frames and valid depth over the task region for ten
   minutes while the other cameras, recorder and UI are running. A requested
   frame rate is not a measured rate. Consider 60 fps only after checking modes,
   bandwidth, exposure and whether the task benefits.
3. Preserve raw metric depth, scale, intrinsics, distortion, color/depth
   extrinsics, timestamp domains and serial/profile metadata. Publish an
   explicitly labeled preview separately. IMU publication is optional scope;
   existing RGB-D support does not establish IMU recording.
4. Calibrate the fixed camera to the arm base using a measured target. Record
   transform direction and units. Validate on held-out target locations across
   the workspace, including raised points; report RMS/max error and choose an
   admission tolerance from gripper clearance and object size before grasps.
5. Tighten the mount, apply light expected cable load, then repeat held-out
   measurements. Recalibrate after moving the camera, base or mount. Keep
   OBSBOT tracking/gimbal fixed if using it for geometric comparison.

The motor-tag overlay currently detects 2D ArUco centers. Camera calibration
and measured tag-to-pivot offsets are still needed to compare projected
encoder pivots with computer vision. A green line following tags does not
establish the arm's 3D pose or authorize motion.

See [RealSense integration](../realsense.md) and the
[manipulation roadmap](../plans/2026-10-27-manipulation-roadmap.md).
