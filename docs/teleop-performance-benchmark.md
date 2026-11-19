# Teleoperation performance benchmark

## Power and motor evidence

The operator reports a K651-1205000D supply marked **12 V, 5000 mA**
(2026-10-27): 5 A and 60 W nominal output. This is label evidence, not a
measurement of transient capacity or voltage at the motor bus. The operator
confirms this supply powers the follower only, with a separate supply for the
leader. The exact installed motor SKU remains unverified.

The [official SO-101 parts list](https://github.com/TheRobotStudio/SO-ARM100)
includes different motor variants. The
[Waveshare ST3215 specifications](https://docs.waveshare.com/ST3215_Servo)
list the 12 V, 1:345 variant at 0.222 s per 60 degrees without load, 900 mA
rated current, 2.7 A stall current, 10 kg.cm rated load and 30 kg.cm stall
torque. These figures apply conditionally to that variant; they do not prove
the installed motor identity. Stall torque is not a continuous operating target.

Six such motors would total 5.4 A at their listed rated currents or 16.2 A
at simultaneous stall. These sums illustrate why a 5 A label does not justify
maximum effort on every joint. Actual task load, adapter and daisy-chain
wiring, supply protection and transient droop determine the usable margin.
Reported servo current scaling is still unverified; do not interpret its sum
as a calibrated supply-current measurement. No torque/current protection,
voltage profile or servo gain is increased by this benchmark preparation.

## Existing observations

The original pilot's mean absolute leader/follower angular errors were about
24.9 degrees at the shoulder lift and 23.4 degrees at the elbow. With the
native present-position window removed only during bounded leader teleop,
and a 60 degrees/s command slew, pilot 2 measured about 8.4 and 9.1 degrees.
Different operator trajectories prevent treating these as a controlled speed
benchmark, but multi-second stationary demand intervals largely disappeared.

Pilot 2 averaged 29.97 control samples/s, with one 140.9 ms gap. Its motor-step
work peaked at 10.7 ms; the long gap occurred between steps. A newer 21.65 s
run passed the timing integrity gate, while the 275.66 s run averaged
29.98 samples/s with a 177.5 ms maximum gap. The latter also has a 12.43 s
gripper stationary-demand interval. The saved OBSBOT frame at approximately
185 s shows the gripper holding a bottle. Measured gripper opening stays near
14.6 percent while requested/accepted opening is about 1.2–3.4 percent. Health
reports a current warning followed by raw status 0x20 during that interval;
its bit interpretation and current scaling remain unverified. Object contact
explains why free-motion tracking is the wrong comparison here, but does not
qualify the grasp effort. Saved diagnostic episodes are not automatically
accepted task demonstrations.

Episode reports now summarize motor health advisories with arm/joint identity,
severity, reasons, first/last recorded occurrence and health-message count.
Repeated bridge messages can contain the same hardware sample; their count
is not an independent sensor-sample count. Warnings remain explicit review
items even when capture integrity checks pass.

The fifth tuning run (`c957b1ff`, 206.16 s) contains 6,164 control ticks at
29.90 Hz, with a 278.4 ms maximum gap. Mean raw leader/follower errors were
3.62 degrees at shoulder lift and 5.03 degrees at elbow; mean errors against
accepted commands were 2.01 and 2.31 degrees respectively. Its recorded
profile is still 60 degrees/s, without the new phase diagnostics. It has
gripper status/current warnings and shoulder-lift current warnings; current
scaling and status-bit meanings remain unverified. OBSBOT averaged 29.08 fps,
D435i 29.66 fps, and the icSpring cameras 6.43 and 5.04 fps. The report requires
review for the control gap. The operator explicitly classifies all five saved
runs as tuning trials, not the ten clean task demonstrations.

## Next measured increment

The workstation override prepares **90 degrees/s** command slew for the next
restart, retaining 30 Hz control and existing torque/gains. This is an
experimental command limit, not a hardware speed qualification. Absolute
joint limits and all stop paths remain enforced.

New control ticks include current cycle interval, motor-bus lock wait and motor
step duration, plus the previous cycle's telemetry/recording tail duration,
requested sleep, deadline lateness and resynchronization flag. Previous-cycle
fields are explicitly labeled; the first tick has no previous-cycle values.
MCAP source hashes include the manager and trace implementation.

The operator performs each powered trial with the same task, object, payload,
camera placement and trajectories. Review tracking error, clipping, control
gaps, camera frame timing, bus voltage, raw current, temperature and protection
events. Compare repeated trials before changing one further parameter.
Increasing control frequency and increasing effort are separate experiments.
Stop qualification on a protection event, unexpected motion, voltage droop or
tracking degradation; a brownout is a failure, not a target operating point.

## Camera evidence

Read-only checks on 2026-10-27 confirm both icSpring devices report a 1/30 s
interval and automatic exposure. Neither exposes an exposure-auto-priority
control through its V4L2 control inventory. Current observed capture rates
are about 6.3 and 5.1 fps, while OBSBOT and D435i color sustain about 30 fps.

Prior direct-device tests with each relay stopped also missed 30 fps:
camera 1 reached 16.6 fps; camera 2 reached roughly 5–8 fps across the tested
MJPEG/YUYV modes and sizes. Shorter manual exposure darkened images without
reaching 30 fps. Thus faster UI polling alone cannot fix the deficit. USB,
sensor/firmware and illumination remain candidate causes; no single cause has
been established. Original camera controls were restored after those trials.

A subsequent test stopped all four camera relays while the recorder was idle
and opened one icSpring device at a time. Camera 1 remained at 6.27 fps and
camera 2 at 4.99 fps. Competing camera streams therefore do not explain the
observed low rates in that test. With temporary manual 10 ms exposure under
the same isolated conditions, rates rose to 16.57 and 10.05 fps, but images
became unusably dark. Exposure/illumination contributes to the deficit; the
remaining shortfall is not yet isolated to sensor, firmware or USB path.
Both devices were verified back at automatic exposure 3 / absolute exposure
313, and all four relays were confirmed live after the test.

After the operator brightened the workspace, four passive measurements over
30 s showed camera 1 at 9.39–9.41 fps and camera 2 at 10.10–10.10 fps with
usable automatic-exposure images. Brighter lighting helps, but neither
icSpring camera is qualified at 30 fps. OBSBOT and D435i color remain near
30 fps. No unique explanation for the remaining camera-rate ceiling is proven.

## First 90 degrees/s pilot

Run `3fd2cb5e` records the new profile and phase diagnostics: 2,203 ticks over
73.51 s, averaging 29.97 Hz with one 147.7 ms maximum gap. The outlier includes
a 122.2 ms control step; the preceding sleep lateness was only 0.16 ms.
Telemetry/recording tail duration stayed below 0.19 ms. Another 62.0 ms gap
included 27.7 ms waiting for the bus lock. These observations distinguish
different delay locations without proving serial timeout versus thread
preemption inside the step. The pinned native driver config allows two read
retries; no retry or timeout setting has been changed on this evidence alone.
Gripper current/status warnings still require review. Different trajectories
prevent using this run as proof that 90 degrees/s improves tracking over 60.
