# Troubleshooting and recovery

Start with the **Diagnostics** page. It reports the active backend and gives a
specific remediation for each failed check. Do not work around a blocking
safety or calibration check just to make the UI turn green.

## Arm is not discovered

1. Disconnect motor power before changing cables or bus jumpers.
2. Confirm the serial path in `.env`. On Linux prefer a stable
   `/dev/serial/by-id/...` path; numeric camera and USB indices can change.
3. Confirm the Waveshare adapter is in the correct USB/bus mode and only one
   application owns the port.
4. Check permissions, then scan motor IDs and baud rate with the official
   LeRobot setup tools.
5. Verify IDs are unique and map in order: shoulder pan 1, shoulder lift 2,
   elbow 3, wrist flex 4, wrist roll 5, gripper 6.
6. Mixed servo firmware, supply mismatch and a duplicate motor ID can look like
   intermittent packet corruption. Resolve those before increasing retries.

## Commands are rejected

The API rejection is intentional and includes a reason. Check, in order:

- E-stop is latched: explicitly reset it only after the scene is safe.
- No lease / wrong owner: acquire an exclusive lease for the selected source.
- Disarmed: arm while the lease is still valid.
- Deadman false: hold/re-engage the source deadman.
- Stale timestamp or expired TTL: fix clocking or sender latency.
- Replayed sequence: increase the source sequence monotonically.
- Non-finite/out-of-range target: fix the producer; the safety layer will not
  accept NaN or infinity and will clamp configured physical limits.
- Tracking confidence: recenter/reacquire and deliberately re-engage.

## Robot stops when a tab sleeps or closes

That is expected. The control lease requires heartbeats, and the watchdog
disarms a stale source. Return to the operator page, acquire a new lease, check
the workspace and arm again. Do not lengthen the watchdog to hide network or
browser-lifecycle problems.

## Camera tile is blank

- Close other camera users (video conferencing, browser tabs and capture
  tools), confirm OS camera permission and run `make find-cameras`.
- An Intel RealSense: check the message against the table in
  [realsense.md](realsense.md#troubleshooting) (udev rules on Linux, USB 3
  link, one program per stream). Do not configure it both as
  `realsense:<serial>` and as its OpenCV index.
- Check requested resolution/FPS against the device's negotiated format.
- Use a stable device identifier where the platform supports it.
- A configured camera is not proof of a live fresh frame. Diagnostics should
  show frame age, capture errors and the backend actually in use.

## Calibration looks wrong

Stop and disable motion. Do not compensate with larger software limits.
Calibration must match the exact leader/follower device identity and the
current middle-zero SO-101 convention. Verify joint sign one joint at a time in
a small range. Recalibrate if hardware, motor IDs, firmware, homing offset or
mechanical assembly changed.

## Training or rollout fails

- Confirm the dataset schema and ordered joint names match the policy.
- A fallback development dataset is not automatically a valid input for every
  LeRobot trainer; convert/validate before training.
- Ensure the checkpoint has its saved preprocessing/postprocessing pipeline.
- In real mode, a missing or incompatible policy dependency is a hard error.
  The platform will not substitute a moving mock policy on real hardware.

## Collect a support snapshot

Use `GET /api/diagnostics` and `GET /api/events`. Redact host paths or device
identifiers before sharing. Include the platform version, LeRobot version,
backend type, arm calibration identity, exact failure time and the few events
immediately before and after it. Never include tokens from `.env`.
