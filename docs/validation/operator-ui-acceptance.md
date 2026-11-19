# Operate dashboard acceptance — 2026-10-27

This is an acceptance record for the current UI, camera and recording changes.
It is not a hardware qualification or permission to start motion.

| Requested behavior | Current evidence | Remaining acceptance |
| --- | --- | --- |
| One compact Operate page | Icon sidebar, cameras and encoder view on the left; teleop, readiness and calibration controls in the right rail; browser inspected | Operator layout review |
| Halt and E-stop always accessible | Global buttons remain above inspection dialogs; stop/fault/latch regression group passes | On-kit operator stop validation |
| Start/stop teleop opens/closes an episode | Existing automatic episode workflow retained; activity and session tests pass; idle recorder verified ready without writing | Next operator tuning run |
| Inspect everything needed without Foxglove | Live control traces, measured/accepted/leader values, timing, health, profiles and raw evidence are available in inspection | Confirm usefulness during a real run |
| Watch recorded videos and arm motion | Real browser plays four original camera streams and recorded encoder-driven 3D; health populates at the playback cursor | Operator visual correspondence check |
| Simulation mathematical correspondence | Current-limit 54-pose reference regenerated; 81 simulation/model tests and independent 24-case review pass; exact pose inputs and model provenance enforced | Physical encoder zero/direction alignment remains an operator check |
| Open selected recording in Foxglove | Button prepares a bounded finalized export; local executable received the correct file path | Native Foxglove window rendering not independently observed |
| No misleading stale pose | No sample, idle sampling paused, stale telemetry and transport errors have distinct states | Live operator sampling check |
| Calibration dropdown is readable | State controls expand inline; motor labels no longer bleed through inspection dialog | Operator viewport review |
| Persistent camera controls | Four capture owners expose supported controls; write/readback and restart restoration exercised; automatic/manual dependency tests pass | Common neutral-reference exposure/color matching |
| Cameras 1 and 2 at delivered 30 fps | Advertised modes verified; isolated raw-driver tests still reach only about 22.59/19.95 fps with short manual exposure; selected USB endpoint payload ceilings exceed the 30-fps requirement | Exact SKU/advertising conditions and administrator-only UVC statistics; root cause unresolved |
| Motor 5 responds across its intended range | Recorded leader demand exceeds ±160-degree follower bounds and accepted commands clip there | Supported leader/follower reference and cable-travel verification before changing mapping or bounds |
| Automatic home/calibration | Installed-reference checks and the existing bounded recorded-rest return are exposed | Powered endpoint discovery is not implemented or qualified; manual endpoint data alone does not establish reliable safe contact detection |
| Clean, analyzable MCAP | New segments include activity/episode IDs, source hashes, task/profile provenance; quality review separates control integrity from all-camera targets | All nine saved tuning runs remain review material; zero are qualified four-camera 30-fps task demonstrations |
| Understandable repository | [Codebase tour](../codebase-tour.md), directory guides and archived build artifacts preserve runnable entry points | Operator source review |
| Independent review | No blocking defects in scoped final review; 765 frontend tests, lint/build and focused 223-case backend group pass; latest idle recovery test also passes | Full CI result and physical acceptance remain separate gates |
| Publish latest source | Changes remain local for review | User approval before Git commit/push |

## Next physical checks

1. Supply the camera-specific read-only UVC statistics requested in chat and,
   when available, the exact camera seller/model links. Neither the USB ID nor
   a negotiated 30-fps interval identifies a guaranteed sensor mode.
2. Under the intended task lighting, present the same neutral white/gray
   reference to each camera. Tune using **Inspect live data → Camera settings**
   while teleop and recording are stopped. Numeric values need not match
   between sensors; useful exposure, color and delivered rate are the criteria.
3. With the arms supported and torque off, use **Home & reference checks** to
   compare installed motor-5 calibration, encoder values and physical neutral
   alignment. Resolve the saved leader range/overflow interpretation before
   choosing a range change or a different leader-to-follower mapping.
4. Perform the next supervised tuning episode and validate stops, tracking,
   frame timing and recorded playback. This is a tuning acceptance step, not
   an accepted training demonstration by default.

See [G15 validation](g15.md), the [camera timing investigation](../hardware/camera-timing-investigation.md)
and the [teleop benchmark](../teleop-performance-benchmark.md) for evidence scope.
