# SO-101 v1 end-to-end execution plan

Status: active

Primary target: ASUS ROG Zephyrus G15, RTX 3070 Laptop GPU, Ubuntu 24.04 LTS

Hardware: SO-101 follower and leader, Feetech STS3215 servos, Intel RealSense
Baseline: v0.2.1 (`9f179b1b02d5655dacf7576c0edc0254741849cc`)

## Outcome

v1 is a repeatable, evidence-backed workflow on the target laptop:

1. install a verified handoff on a clean Ubuntu 24.04 system;
2. diagnose and bring up the follower, leader, GPU, MuJoCo and RealSense;
3. pass the real-arm gate, stop and verification HIL groups, in that order;
4. teleoperate and record a real LeRobot dataset with outcome labels;
5. inspect and select episodes, train with LeRobot 0.6.1 on the NVIDIA GPU;
6. load the resulting checkpoint, run a bounded rollout and cancel it safely;
7. evaluate trials into a durable report that names the real backend; and
8. repeat the workflow from the generated runbook without source-tree surgery.

Passing software tests alone is not v1. Passing on mock or simulation alone is
not v1. The target machine's doctor output and signed HIL reports are release
evidence, not optional follow-up.

The G15 is the only v1 robot host and HIL evidence source. A Jetson Thor may be
used as an optional batch compute worker for training, simulation and evaluation
when compatibility and benchmarks show an advantage. It receives immutable,
hashed artifacts and never owns motor buses, runs HIL steps or substitutes its
diagnostics for G15 evidence. Remote Thor inference is a later capability gate:
the G15 remains the sole command/safety authority and must Halt on a stale,
late, invalid or disconnected worker. The Mac is a development/operator client,
preferably through an SSH tunnel; it is not a motor-bus host.

The release deliverable is one copyable G15 handoff folder. Its exact-commit
`so101/` clone contains the optional `thor-worker/` qualification profile; the
top-level installer, wheel and runbook remain G15-specific. The G15 path must
install, operate and complete the v1 workflow with Thor absent.

## Non-negotiable invariants

- E-stop cuts torque on every motor and latches; Halt holds in place; Collapse
  lowers at the configured cap and then turns torque off. Their meanings do not
  change to make a test pass.
- The robot manager is the server-side owner of motor buses. A standalone
  command must hold `bus.lock`, honor every durable latch and persist any
  unconfirmed torque-off before releasing ownership.
- Unknown torque or position fails closed. A software success response never
  claims torque is off unless read-back confirmed it.
- No retries for flaky tests, no raised skip budget, and no lowered coverage
  floor. Each defect starts with a regression test that is red on the prior
  revision.
- Hardware motion is performed only by the operator-approved HIL steps, with a
  hand at the physical power switch. CI, mock and simulation never touch a bus.

## Working loop

Every tranche follows the same loop:

1. **Plan:** state the invariant, failure scenario, affected paths, acceptance
   test and evidence required.
2. **Design:** choose the smallest change that preserves lock order, ownership,
   durability and public contracts.
3. **Red test:** demonstrate the failure deterministically with injected clocks,
   events or faults; never timing sleeps.
4. **Execute:** implement the scoped change and update truthful documentation.
5. **Verify:** run focused tests, then the complete `make ci` gate once.
6. **Independent review:** a reviewer not responsible for the implementation
   traces the path end to end and checks backlog overlap and failure behavior.
7. **Fix and repeat:** resolve every substantiated review/CI issue and rerun the
   gate. Do not waive it.
8. **Promote evidence:** only after software is green, build a clean handoff and
   run the next G15/HIL gate. Archive the exact commit, checksums, doctor JSON,
   HIL reports and measurements.

## Ownership

The lead integrator owns scope, merge order, the full gate and the evidence
ledger. Specialist agents own a tranche through red test, implementation,
focused verification and handoff notes; a different agent reviews it.

| Workstream | Accountable agent | Deliverables |
| --- | --- | --- |
| Control safety | control-safety agent | RobotManager stops, resets, torque truth, latch persistence, deterministic concurrency/fault tests |
| Standalone ownership | standalone-safety agent | durable latch parity, bus-lock ownership, teardown truth, CLI/library-path tests |
| Linux release | linux-release agent | clean Ubuntu install, NVIDIA/PyTorch selection, udev/dialout, atomic handoff, installer tests |
| Data integrity | data-integrity agent | recording formats, labels, locks, loaders, checkpoint durability, training-selection tests |
| Simulation/evaluation | sim-eval agent | deterministic MuJoCo workflows, EGL, demos, policy rollout/eval parity and metrics |
| Hardware bring-up | hil agent plus human operator | preflight, calibration, rest pose, real-arm stop measurements, signed HIL evidence |
| Operator workflow | UI agent | install-to-operate UX, safety visibility, record/train/run/eval critical path, accessibility checks |
| Independent audit | rotating review agent | adversarial review against safety/security contracts and the known-backlog exclusion list |

At most three specialist agents run concurrently so file ownership stays clear.
The integrator resolves overlaps and is the only role that declares a gate
passed or promotes a build to the G15.

## Milestones and gates

### M0 — Reproducible Linux development gate

Goal: one supported Linux environment can build and test the archive from
scratch.

- Ubuntu 24.04 (native or a faithful clean image), Python 3.12 and Node 24.
- `make bootstrap` succeeds with the documented dependency sources.
- `make ci` passes in full: lint, mypy, branch coverage and floors, skip budget,
  frontend lint/tests/build/audit, HTTP smoke and UI-bearing wheel smoke.
- Run both exactly once through the standard-library-only
  [`g15-ci-evidence`](../g15-ci.md) gate. It refuses the wrong host, source,
  inherited SO-101 configuration or reusable build state and atomically seals
  private, hashed evidence without touching hardware.
- Record versions and complete logs. Any host-policy/network failure is an
  infrastructure blocker, not a green result.

Exit evidence: clean commit, environment manifest, bootstrap log and one full
green CI log.

### M1 — Hardware-blocking safety corrections

Goal: close every confirmed high-severity review finding before connecting an
arm.

- Runtime faults cut torque before potentially blocking persistence.
- Fault/E-stop resets cannot erase newer latches.
- Halt and Collapse preserve aggregate unconfirmed-torque truth.
- Loss of a halted bus faults and attempts category-0 cleanup; it never silently
  drops supervision.
- Standalone access honors the fallback latch marker, persists teardown
  failures, and cannot be shared by unrelated threads.
- LeRobot joint feature order is validated/canonicalized before a policy that
  can run on hardware is trained.

Exit evidence: red-before/green-after regression tests, full CI, independent
safety review with no unresolved high finding.

### M2 — Data and installer integrity

Goal: make installation and the record/train artifacts trustworthy.

- Semantic label corruption fails closed; sim label failures stop publication.
- CLI and API training hold the same cross-process dataset read lock.
- Fallback and LeRobot layouts never mix, and fallback appends validate FPS and
  feature layouts.
- Baseline checkpoint directory publication is power-loss durable.
- Installer refuses an unusable data home and whole-script sudo, validates udev
  rule content, and leaves a failed handoff output safely retryable.

Exit evidence: fault-injection/durability tests and full CI.

### M3 — Software-only end-to-end rehearsal

Goal: exercise the entire product without hardware before risking the arm.

- Mock: install wheel, serve built UI, operate, record, inspect, label, train a
  baseline, register/load it, run, cancel and evaluate.
- Simulation: preflight, calibrate/rehearse, scripted demos, labels, training,
  rollout, stop behavior and evaluation on deterministic seeds.
- Verify E-stop, Halt, Collapse, lease expiry, deadman and shutdown behavior in
  fault-injected tests and MuJoCo.
- Produce a diagnostic bundle and reports that clearly say mock/sim, never real.

Exit evidence: scripted E2E logs, artifacts and green CI from the exact commit.

### M4 — Verified Ubuntu handoff

Goal: install the exact green commit on a clean Ubuntu 24.04 G15.

- Build `make handoff OUT=<new-dir>` from a clean commit.
- Verify `SHA256SUMS`; copy to the G15; run installer as the normal user.
- Reboot/login as required, then rerun the installer to prove idempotence.
- Verify NVIDIA driver, selected torch/torchvision/torchcodec builds, CUDA
  availability, app launcher, storage paths and uninstall/recovery notes.

Exit evidence: handoff checksums, install transcript, package/version manifest,
launcher result and first `so101 doctor --json`.

### M5 — G15 bring-up with motors initially unpowered

Goal: resolve host/device integration before motion.

- Triage every doctor FAIL and every unexpected WARN.
- Verify both stable serial aliases, permissions, ModemManager exclusion and bus
  exclusivity.
- Verify RealSense color/depth, USB 3 and timestamps.
- Verify MuJoCo EGL rendering on the RTX 3070.
- Run read-only preflight; configure health thresholds, then calibrate one arm at
  a time only when the workspace gate is satisfied.
- Record and validate a collision-free Collapse rest pose.

Exit evidence: clean doctor JSON, real-backend preflight report, calibration
identities, rest-pose record, camera samples and EGL result.

### M5a — Optional Thor batch-compute qualification

Goal: use Thor where it measurably outperforms or fits a model the G15 cannot,
without adding a dependency to robot operation.

- Record the exact Thor SKU, JetPack/L4T, kernel, Python, CUDA, cuDNN, TensorRT,
  PyTorch and LeRobot 0.6.1 compatibility tuple.
- Use a digest-pinned NVIDIA ARM64/JetPack container; mount no USB serial device,
  install no robot/RealSense extra and grant no control-plane credential.
- Transfer only immutable dataset/checkpoint bundles with canonical manifests
  and SHA-256 verification before atomic publication.
- Benchmark the same commit, dataset digest, seed, policy, precision and batch
  size on G15 and Thor. Promote Thor for a workload only when correctness is
  equal and it fits otherwise-impossible work or improves representative
  throughput by at least 1.5x without thermal/OOM instability.
- Keep remote inference disabled for v1 hardware acceptance. A later gate must
  prove observation/model/epoch identity, G15-local monotonic deadlines,
  bounded finite actions, zero stale actions, disconnect Halt and stop-latency
  independence before it may command a real-arm proposal.

Exit evidence: compatibility manifest, container digest, dependency report,
artifact round-trip hashes, benchmark matrix and a 60-minute thermal/memory
soak. Failure leaves the G15-only path unchanged.

### M6 — Powered real-arm safety HIL

Goal: prove the exact stop build on the exact hardware.

Run HIL groups in order with the physical power switch in reach:

1. `gate`
2. `stop`
3. `verification`

Measure and retain E-stop response/torque removal, Halt overshoot and drift,
supervision-loss and hold expiry, Collapse duration/speed/rest accuracy/fallback,
servo-health read latency and torque read-back under disconnect/fault injection.
Any failed or ambiguous stop test blocks all autonomous rollout work.
Stop HIL passes before collection, training or rollout verification begins.

Exit evidence: sealed Markdown/JSON HIL reports and raw measurement files.

### M7 — Real data, training, rollout and evaluation

Goal: complete the learning loop on the G15 and real arm.

- Record at least two short RealSense + robot-state LeRobot episodes.
- Verify durability, timestamps, feature order and outcome sidecar; label one
  success and one excluded failure/partial take.
- Run dataset QC and prove `kept` selects exactly the intended episodes.
- Run a small, bounded LeRobot 0.6.1 ACT integration job on G15 CUDA (batch 1
  first) to prove the local path. Larger training may run on a qualified Thor
  batch worker when its benchmark wins; preserve dataset/container/run manifests
  and verify the returned checkpoint hashes on the G15.
- Preflight the checkpoint, run a short deadman-supervised rollout, exercise
  Halt and cancel, then evaluate multiple trials and save the report.

Exit evidence: dataset manifest/QC, label selection, GPU training log, run
manifest, checkpoint hash, rollout safety events and real-backend eval report.

### Measurable v1.0 acceptance bar

These limits are fixed before the first powered HIL run. A missing measurement,
an incompatible clock, a skipped probe or a result reported only by observation
is a failure, not a pass. Raw samples and the script or report that calculated
each value stay in the evidence ledger.

| Area | Release threshold |
| --- | --- |
| E-stop | In each of 3 trials, every motor on every follower and leader bus currently owned by RobotManager reports torque off and the latch is durable no later than 1.0 s after the G15 accepts the stop request. No motor on an owned bus may receive a goal or torque-enable write from request ingress until a deliberate E-stop reset succeeds. |
| Halt | In each of 3 trials while moving at 45–50 °/s, worst overshoot is at most 3°, then worst drift is at most 1° over 10 s; the gripper never opens farther. Supervision loss starts Collapse within the configured grace plus 3 s, and hold expiry does so within the configured hold plus 3 s. |
| Collapse | Every joint stays at or below 110% of the configured 15 °/s cap in 0.25 s windows, finishes within `min(max_distance / speed + 2 s, 12 s)` plus one telemetry interval, ends within 3° of rest with torque reported off, and emits no fallback event. Pass all 3 trials. |
| Fault/read-back | A disconnected or silent servo never produces a confirmed-off result. Within 3.0 s the operation reports a visible fault plus unconfirmed torque, and subsequent arming remains blocked across a process restart. |
| Health/control timing | Across a 10-minute 30 Hz motion-safe HIL sample, servo-health bus-lock hold p99 is at most 25 ms, control-loop interval p99 is at most 66.7 ms, no interval reaches the 300 ms command TTL, and no false watchdog or supervision stop occurs. |
| Local command latency | Across at least 1,000 G15-local commands, API receipt to completed validated bus write is at most 100 ms at p95 and below the 300 ms TTL at maximum. The measurement uses the manager's monotonic timestamps, not browser wall clocks. |
| Reliability | M4–M7 complete twice consecutively from fresh handoffs with no source edits, unhandled exception, unexplained warning, skipped mandatory HIL probe or unresolved critical/high finding. One run includes a 60-minute powered, supervised soak with no bus-ownership violation or unintended stop. |
| Task success | Freeze one real-arm task, start-state distribution and evaluation procedure before collection. The final checkpoint succeeds in at least 16 of 20 operator-scored trials (80% observed; Wilson 95% interval reported), with zero collision, limit, stale-command or stop failures. Interrupted trials are reported and rerun only when the report records the predeclared non-policy cause. |

A threshold may change only in a reviewed commit made before the evidence run it
grades. A result cannot be promoted by editing this table after it is known.

### M8 — v1 release acceptance

Goal: make the full path repeatable rather than a one-off lab success.

- Repeat M4-M7 from a fresh handoff without source edits on the laptop.
- `make ci` passes at the release commit and the wheel serves its stamped UI.
- No open critical/high safety finding; medium findings have an owner and do not
  undermine the v1 acceptance workflow.
- Documentation and generated `START_HERE.md` match observed commands and
  evidence.
- Tag the release only after the evidence set names the exact commit and hashes.

## Evidence ledger

Each promoted run gets one folder outside the repository:

```text
evidence/<date>-<commit>/
  environment.txt
  bootstrap.log
  ci.log
  SHA256SUMS
  install.log
  doctor.json
  preflight.json
  hil-gate/
  hil-stop/
  hil-verification/
  datasets/
  training/
  eval/
  review.md
```

Secrets, tokens and camera credentials must be redacted before the folder is
shared. Raw evidence is never edited in place; a triage note points to follow-up
commits and reruns.

## Scope discipline

Quest/hand teleoperation, generalized agent autonomy and broad new task
libraries are not on the v1 critical path. They remain capability-gated and may
advance in parallel only when they cannot delay or weaken install, stop safety,
the real data-learning loop or its evidence. The first v1 release favors one
well-measured G15 + SO-101 + RealSense workflow over breadth.
