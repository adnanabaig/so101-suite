# Guided phone target alignment

This camera-only tool reads the running D435i RGB relay. It never opens motor
buses, commands motion, writes factory camera calibration, or installs robot
extrinsics. Start the Suite's existing camera services first, then run:

```bash
cd /home/adnan/so101-suite
.venv/bin/python scripts/camera_alignment.py
```

Open <http://127.0.0.1:8782> on the workstation. The live overlay outlines marker
42 and checks visibility, image size, fresh sequences, stable corners and fit.
The guide is currently for the fixed D435i only, serial 347622075852, using its
saved factory VGA color intrinsics. Other cameras need their own intrinsics;
the robot-mounted camera also needs camera-to-link alignment.

## Simple workflow: no measurement form

Scan the QR on the workstation with the iPhone 14 Pro Max, on the same Wi-Fi.
Leave that Safari target page open, place the phone face-up in view, and press
**Capture alignment** once steady. The target page uses a fixed 340 CSS pixel
black marker. It checks portrait 430×932 CSS display, device pixel ratio 3,
100% visual zoom and matching layout width. With Apple's nominal 460 ppi this
is approximately **56.32 mm**. Unexpected display zoom, orientation or viewport
scaling leaves a visible preview with an explanation and rejects automatic
sizing rather than guessing. Capture stays unavailable until sizing is confirmed.

This controlled page replaces assumptions about how Drive or Photos fit images.
The old PNG remains usable with manually measured sizing in Advanced settings.
Apple's nominal specification is a scale assumption, not a physical measurement
of this particular display. Physical accuracy still needs independent checking.

The workstation guide and camera routes bind to loopback only. The separate
phone server binds to the LAN IP on port 8783 and serves only the generated
target plus a small authenticated display-report endpoint. It exposes no camera
footage, stored calibration, files or robot controls. `--phone-host IP` can set
the LAN address explicitly; otherwise it is discovered from the local route.

The operator supplied table height **29⅛ inches = 739.775 mm** and floor to
camera-bottom **53½ inches = 1358.9 mm**. Their difference is **619.125 mm**.
These values are preserved as approximate physical context. The bottom of the
housing is not the optical center and is not used as an exact camera transform.

Quick mode estimates a **phone-screen reference frame** without requiring X/Y,
yaw or screen height. It leaves the table transform and screen-to-table offset
unset. Thus a tilted phone, camera bump, case or support is never silently
flattened onto the tabletop. The nominal phone body thickness (7.85 mm) is
recorded separately and never treated as its actual supported screen height.
Quick capture is useful alignment evidence, not a qualified robot-base transform.
Use the optional Advanced settings below to establish measured table geometry.

[Apple iPhone 14 Pro Max specifications](https://support.apple.com/en-ie/111846)

## Advanced: measured table setup

Open `SO101-iPhone-calibration-target.png` from Google Drive on the phone and
display it fullscreen in portrait. Keep the entire black square and white quiet
zone visible. Avoid image resizing after measuring. Temporarily disable display
auto-lock, True Tone and Night Shift; use steady brightness without glare.

Measure the actual displayed **black square width**, excluding the white margin,
with a ruler. Never infer that measurement from the iPhone model, image pixel
dimensions or nominal screen diagonal. The PNG uses the iPhone 14 Pro Max pixel
aspect ratio, but the photo viewer controls its actual displayed size.

Support the phone face-up with its screen level. A case or camera bump can tilt
it: use a stable shim, measure table-to-screen height at both ends, and confirm
they agree. Enter the screen-plane height including all support material.
The transformation places the marker at that height rather than on the table.
A tilted screen is not supported by this initial guide; do not confirm level if
it is tilted. A larger matte printed target is preferable if phone pixels are
too small or glare prevents reliable detection.

## Table coordinates

Mark a reproducible table origin. Looking from the operator's position, choose
+X right, +Y away, +Z upward. Align the target TOP toward +Y and its rightward
direction along +X (yaw 0 degrees). Measure table X and Y to the **top-left
black corner**, not a phone corner. The page includes a diagram. If using a
different yaw, measure it counterclockwise from table +X to marker rightward.

Put the target inside the useful camera field; the blue box is a framing aid,
not a known physical location. A single marker needs all four corners, edge
clearance and at least 40 pixels along its shortest side. Hold it still while
the guide gathers at least two seconds and 15 distinct relay sequences. Apply
the measured values and save the candidate.

Move the phone at least 100 mm to another independently measured position,
keeping the screen level. Update X/Y/height/yaw as appropriate, apply and save
again. Compare the two placements on the page. Differences below 10 mm and
2 degrees are a provisional consistency check, not a measured accuracy claim.
Planar pose ambiguity is explicitly retained in candidate files. Check known
distances elsewhere on the table and measure the robot base's relationship to
the table frame before using camera coordinates for robot commands.

Candidates and annotated images go into timestamped private folders under
`data/real/camera_alignment/`. They include the target, measurements, intrinsics
hash, camera identity, fit errors, pose alternatives and transform convention.
They are marked `candidate_only`; this tool never activates them automatically.

## Technical references

- [OpenCV planar pose estimation](https://docs.opencv.org/4.10.0/d5/d1f/calib3d_solvePnP.html)
- [RealSense projection and distortion](https://dev.realsenseai.com/docs/projection-in-realsense-sdk-2-0/)

The current D435i color profile has zero distortion coefficients. The guide
refuses nonzero distortion rather than interpreting inverse Brown-Conrady
coefficients as OpenCV's forward model.
