// Typed API client for the SO-101 backend.
// Base is "" because Vite (dev) and FastAPI (prod) both serve /api on the same origin.

import { getOperatorId } from "./lib/operatorSession";

// ---------------------------------------------------------------------------
// Schemas — these mirror the backend contract exactly.
// ---------------------------------------------------------------------------

export interface Ack {
  ok: boolean;
  detail: string;
}

export interface Health {
  ok: boolean;
  version: string;
  mode: string;
  follower_connected: boolean;
  leader_connected: boolean;
  estopped: boolean;
  /** Kept for older clients: false under sim too. `backend` tells them apart. */
  mock: boolean;
  /** mock, sim (MuJoCo) or real. Absent on servers before v0.2. */
  backend?: RobotBackend;
  cameras: string[];
  live_cameras: string[];
  camera_frame_age_ms: Record<string, number>;
  operation_owner: string | null;
  fps: number;
}

/** C11 stop categories. Servers that omit `stop_kinds` only implement E-stop. */
export type StopKind = "estop" | "halt" | "collapse";
export type TorqueState = "on" | "off" | "unconfirmed";
export type StopState = "none" | "halted" | "collapsing";

/** C11: present while the arm is halted and holding position. */
export interface HaltInfo {
  since: string;
  reason: string;
  /** Server clock. Countdowns subtract `server_time`, never the client clock. */
  hold_expires_at: string;
  server_time: string;
}

/** C11: present while the arm is lowering to its rest pose. */
export interface CollapseInfo {
  since: string;
  reason: string;
  /** Fraction complete, 0..1. */
  progress: number;
  eta_s: number;
}

/**
 * Operator supervision as the hold timer sees it. While no operator browser
 * has a fresh heartbeat, a halt holds only `grace_s`, counted from `lost_at`
 * (or from the halt, if later). `lost_at` is null while supervised or when no
 * operator ever sent a heartbeat. Status only; older servers omit it.
 */
export interface SupervisionInfo {
  supervised: boolean;
  lost_at: string | null;
  grace_s: number;
}

/** Fields shared by status and telemetry once the backend implements C2/C11. */
export interface StopFields {
  /** C11; absent on wave-1 backends, which is treated as unknown. */
  torque_state?: TorqueState | string;
  /** C11; absent means "none". E-stop and fault keep their own fields. */
  stop_state?: StopState | string;
  halt?: HaltInfo | null;
  collapse?: CollapseInfo | null;
  supervision?: SupervisionInfo | null;
}

export interface RobotStatus extends StopFields {
  mode: string;
  follower_connected: boolean;
  leader_connected: boolean;
  estopped: boolean;
  /** Kept for older clients: false under sim too. `backend` tells them apart. */
  mock: boolean;
  /** mock, sim (MuJoCo) or real. Absent on servers before v0.2. */
  backend?: RobotBackend;
  cameras: string[];
  live_cameras: string[];
  camera_frame_age_ms: Record<string, number>;
  operation_owner: string | null;
  fps: number;
  /** C2: torque-off could not be read back; cut physical power. Absent = false. */
  torque_unconfirmed?: boolean;
  /** Whether the current safety latches will survive a server restart. */
  latch_persistence?: "saved" | "marker" | "unsaved" | string;
  /** C7: model path of the running rollout, null when none. */
  active_model?: string | null;
  /** C11: stops this server implements. Absent means ["estop"]. */
  stop_kinds?: string[];
}

export interface Telemetry extends StopFields {
  t: number;
  mode: string;
  estopped: boolean;
  fps: number;
  /** May be `{}` when `observation_live` is false (C6). */
  observation: Record<string, number>;
  action: Record<string, number>;
  recording: boolean;
  recording_saving: boolean;
  episode_frames: number;
  sample_age_ms: number;
  sample_seq: number;
  /**
   * C6: true only when joints were sampled by an active loop or armed session
   * within 500 ms. Transport freshness is a separate signal. Absent on
   * pre-C6 backends.
   */
  observation_live?: boolean;
}

export interface TeleopRequest {
  fps: number;
}

export interface CalibrateRequest {
  arm: "follower" | "leader";
  sample_seconds?: number;
  joint?: string;
}

export interface CalibrationLiveStatus {
  joint?: string | null;
  baseline_only?: boolean;
  restored?: boolean;
  phase: string;
  sequence: number;
  sample_count: number;
  elapsed_seconds: number;
  age_seconds: number;
  frozen: boolean;
  attempt_id?: string | null;
  error?: string | null;
  positions?: Record<string, number>;
  starting_positions?: Record<string, number>;
  range_mins?: Record<string, number>;
  range_maxes?: Record<string, number>;
  homing_offsets_requested?: Record<string, number>;
  homing_readback?: Record<string, {
    homing_offset: number | null; range_min: number | null; range_max: number | null;
  }>;
  affected_motors?: string[];
}

export interface CalibrationStatus {
  live?: CalibrationLiveStatus | null;
  joint_baselines?: Record<string, {
    created_at?: number;
    backend?: string;
    device_id?: string;
    error?: string;
    diagnostics?: CalibrationLiveStatus;
  }>;
  arm: string;
  in_progress: boolean;
  fraction: number;
  message: string;
  has_saved: boolean;
  device_id: string;
  created_at: number | null;
  ranges: Record<string, [number, number]>;
}

export interface CalibrationPose {
  arm: string;
  backend: string;
  captured_at: number;
  calibration_sha256: string;
  complete: boolean;
  observation: Record<string, number>;
  joints: Record<string, {
    motor_id: number; ticks: number | null; range_min: number | null;
    range_max: number | null; issues: string[];
  }>;
}

export interface CalibrationAttempt {
  joint?: string | null;
  id: string;
  event_id: number;
  arm: string;
  backend: string | null;
  device_id: string | null;
  sample_seconds: number | null;
  started_at: string | null;
  finished_at: string | null;
  result: "pass" | "fail" | "running" | "incomplete";
  saved: boolean | null;
  affected_motors: string[];
  error: string | null;
  diagnostics: Record<string, unknown>;
  log: { timestamp: string; level: string; message: string }[];
}

export interface CalibrationHistory {
  attempts: CalibrationAttempt[];
  limit: number;
}

export interface StartRecordingRequest {
  repo_id: string;
  task?: string;
  fps?: number;
}

export interface EpisodeResult {
  episode: number;
  frames: number;
  /** Sim backend only: the task predicate's verdict (null or absent elsewhere). */
  outcome?: "success" | "failure" | null;
  /** Sim backend only: how the verdict was stored, e.g. "auto-labeled". */
  label?: string | null;
}

export interface DatasetSummary {
  repo_id: string;
  path: string;
  format: string;
  num_episodes: number;
  num_frames: number;
  fps: number;
  task: string;
  camera_keys: string[];
}

export interface DatasetCapability {
  id: string;
  available: boolean;
  level: "full" | "partial" | "unavailable";
  detail: string;
}

export type EpisodeOutcome =
  | "unlabeled"
  | "success"
  | "failure"
  | "partial"
  | "discarded";

export interface EpisodeManifest {
  episode: number;
  name: string;
  frames: number | null;
  duration_seconds: number | null;
  started_at: number | null;
  ended_at: number | null;
  outcome: EpisodeOutcome | string;
  notes: string;
  complete: boolean;
  camera_chunks: Record<string, number>;
  camera_frames: Record<string, number>;
  issues: string[];
}

export interface IncompleteCapture {
  id: string;
  frames: number;
  duration_seconds: number | null;
  reason: string;
  captured_at: number | null;
  camera_chunks: Record<string, number>;
}

export interface DatasetDetail extends DatasetSummary {
  duration_seconds: number;
  episodes: EpisodeManifest[];
  incomplete_captures: IncompleteCapture[];
  outcome_counts: Record<string, number>;
  capabilities: DatasetCapability[];
}

export interface TimelinePoint {
  frame_index: number;
  t: number | null;
  relative_time_seconds: number;
  observation: number[];
  action: number[];
  desired_action: number[];
}

export interface EpisodeTimeline {
  repo_id: string;
  episode: number;
  fps: number;
  total_frames: number | null;
  offset: number;
  stride: number;
  returned_frames: number;
  has_more: boolean;
  joint_names: string[];
  actuates_hardware: false;
  capability: DatasetCapability;
  points: TimelinePoint[];
}

export interface DatasetQualityCheck {
  id: string;
  label: string;
  status: "pass" | "warn" | "fail" | "skip";
  detail: string;
  remediation: string;
  affected: number;
  examples: string[];
}

export interface DatasetQualityReport {
  repo_id: string;
  generated_at: number;
  overall: "pass" | "warn" | "fail";
  checked_frames: number;
  checks: DatasetQualityCheck[];
  capabilities: DatasetCapability[];
}

export interface TrashedDatasetSummary {
  id: string;
  repo_id: string;
  format: string;
  num_episodes: number;
  num_frames: number;
  deleted_at: number;
  size_bytes: number;
}

export interface PolicyPresetOut {
  name: string;
  policy_type: string;
  description: string;
  default_steps: number;
  default_batch_size: number;
  notes: string;
  // Additive (v0.2); absent on older servers.
  /** "lerobot" (lerobot-train subprocess) or "numpy_in_process" (torch-free baseline). */
  trainer?: string;
  requires_torch?: boolean;
}

export interface StartTrainingRequest {
  repo_id: string;
  policy: string;
  steps: number;
  batch_size: number;
  extra_args?: Record<string, string>;
}

/**
 * training/jobs.py JobStatus. A finished job is "succeeded" (F17: the UI used
 * to expect "completed", which the server never sends). "queued" (v0.2) waits
 * for the single trainer slot. "interrupted": the server stopped before the
 * job finished; "corrupt": its record could not be read after a restart.
 */
export type JobStatus =
  | "pending"
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "interrupted"
  | "corrupt"
  | string;

export interface JobSummary {
  id: string;
  repo_id: string;
  policy: string;
  status: JobStatus;
  step: number;
  steps: number;
  loss: number | null;
  output_dir: string;
  error: string | null;
  /** Latest trainer progress-bar line (v0.2; absent on older servers). */
  progress?: string | null;
}

export interface JobLogs {
  id: string;
  status: JobStatus;
  log_tail: string[];
}

export interface ModelSummary {
  name: string;
  path: string;
  policy_type: string;
  is_mock: boolean;
  checkpoint: string;
  // Additive (v0.2 data audit); absent on older servers.
  /** "lerobot" | "numpy_bc_mlp" | "mock" */
  kind?: string;
  /** Operator-facing label, e.g. "privileged-state sim baseline". */
  label?: string;
  privileged_state?: boolean;
  /** True when the checkpoint consumes a language task (including custom/tokenizer VLAs). */
  needs_task?: boolean;
  /** Language instruction the checkpoint was trained with (VLA policies). */
  task?: string | null;
  /** FPS of the checkpoint's training data. */
  fps?: number | null;
  dataset?: string | null;
}

/** Policies that take a language instruction; the Task field is shown for these. */
export const LANGUAGE_POLICIES: readonly string[] = [
  "smolvla",
  "pi0",
  "pi0_fast",
  "pi05",
  "xvla",
  "multi_task_dit",
];

export const isLanguagePolicy = (policyType: string | undefined | null) =>
  !!policyType && LANGUAGE_POLICIES.includes(policyType.toLowerCase());

/** New servers inspect tokenizer metadata; the type list keeps older servers usable. */
export const modelNeedsTask = (
  model: Pick<ModelSummary, "needs_task" | "policy_type"> | null | undefined,
) =>
  model?.needs_task ?? isLanguagePolicy(model?.policy_type);

export interface RolloutRequest {
  model_path: string;
  /** Omitted: the server uses the checkpoint's training fps, else 30. */
  fps?: number;
  /** Omitted: the server uses the training task. */
  task?: string;
}

export interface StartEvalRequest {
  model_path: string;
  n_episodes: number;
  max_seconds: number;
  /** C3: 1-240. Omitted uses the checkpoint's training rate, else 30. */
  fps?: number;
  /** Omitted or blank uses the checkpoint's training task. */
  task?: string;
}

/** C3 phases. `done`, `aborted` and `failed` are terminal. */
export type EvalPhase =
  | "idle"
  | "rollout"
  | "awaiting_score"
  | "awaiting_reset"
  | "preparing"
  | "finalizing"
  | "done"
  | "aborted"
  | "failed"
  | string;

export const EVAL_TERMINAL_PHASES: readonly string[] = ["done", "aborted", "failed"];

export interface EvalState {
  active: boolean;
  model_path: string;
  /** One-based current episode, or the completed count once finished. */
  episode: number;
  /** Requested trials. */
  n_episodes: number;
  phase: EvalPhase;
  successes: number;
  /** successes / scored; interrupted trials are excluded. */
  success_rate: number;
  /** Written when the evaluation ends; "" until then (null on old servers). */
  report_path?: string | null;
  /** C3: trials the operator scored. */
  scored?: number;
  /** C3: trials cut short (E-stop, fault, stop, lost heartbeat) and never scored. */
  interrupted?: number;
  task?: string | null;
  fps?: number | null;
  /** C3: why the session aborted or failed, or why its report was not saved. */
  error?: string;
}

// ---------------------------------------------------------------------------
// Workstation capabilities, control plane, simulation, and operations.
// ---------------------------------------------------------------------------

export type CapabilityMaturity =
  | "stable"
  | "beta"
  | "preview"
  | "experimental"
  | string;

export interface Capability {
  id: string;
  label: string;
  available: boolean;
  maturity: CapabilityMaturity;
  detail: string;
}

export interface CapabilitiesResponse {
  system_version: string;
  /** False under sim too; `backend` tells mock, sim and real apart. */
  mock: boolean;
  /** Absent on servers before v0.2. */
  backend?: RobotBackend;
  capabilities: Capability[];
}

/** Returned only to the lease owner, by POST /lease and the heartbeat (C4). */
export interface ControlLease {
  id: string;
  source: string;
  expires_at?: string | null;
  ttl_seconds?: number;
  server_time: string;
}

/** The public view of the current lease in GET /api/control/state: no id (C4). */
export interface ControlStateLease {
  source: string;
  expires_at: string;
  ttl_seconds: number;
  server_time: string;
}

export interface ControlState {
  armed: boolean;
  safety_state: string;
  lease: ControlStateLease | null;
  watchdog_ms: number;
  last_command_at: string | null;
}

export interface ControlCommandRequest {
  lease_id: string;
  source: string;
  sequence: number;
  sent_at: string;
  ttl_ms: number;
  deadman: boolean;
  targets: Record<string, number>;
}

export interface ControlCommandResponse extends Ack {
  accepted: boolean;
  targets: Record<string, number>;
}

export interface ControlObservation {
  observation: Record<string, number>;
  captured_at: string;
}

export type DiagnosticStatus =
  | "pass"
  | "ok"
  | "warn"
  | "warning"
  | "fail"
  | "error"
  | "unknown"
  | string;

export interface DiagnosticCheck {
  id: string;
  label: string;
  status: DiagnosticStatus;
  detail: string;
  remediation: string;
}

export interface DiagnosticPort {
  id?: string;
  path?: string;
  label?: string;
  type?: string;
  connected?: boolean;
  detail?: string;
  [key: string]: unknown;
}

export interface AuditEvent {
  id?: string | number;
  timestamp?: string;
  created_at?: string;
  level?: string;
  type?: string;
  source?: string;
  message?: string;
  detail?: string | Record<string, unknown>;
  [key: string]: unknown;
}

export interface DiagnosticsResponse {
  generated_at: string;
  overall: DiagnosticStatus;
  checks: DiagnosticCheck[];
  ports: Array<DiagnosticPort | string>;
  events: AuditEvent[];
}

// Preflight (bring-up kit). Shapes from so101.bringup.schemas; the server
// changes them additively only.

/** Which backend the server drives. Mock and sim runs are rehearsals. */
export type RobotBackend = "real" | "mock" | "sim";
export type PreflightCheckStatus = "pass" | "warn" | "fail" | "skip";
export type PreflightOverall = "pass" | "warn" | "fail" | "incomplete";

export interface PreflightCheckResult {
  id: string;
  /** Unique within a report: "<id>" or "<id>.<arm>". */
  key: string;
  arm: "follower" | "leader" | null;
  title: string;
  status: PreflightCheckStatus;
  message: string;
  remediation: string;
  measured: Record<string, unknown>;
  /** A skipped blocking check leaves hardware readiness unproven. */
  blocking: boolean;
  rehearsal: boolean;
  source: "real" | "mock" | "sim" | "host" | "server" | "none";
  references: string[];
  duration_ms: number;
}

export interface PreflightReport {
  kind: "so101.preflight";
  schema_version: number;
  /** ISO 8601 with a UTC offset. */
  created_at: string;
  so101_version: string;
  /** "cli" for `so101 preflight` (it reads the bus), "api" for the server's run. */
  origin: "cli" | "api";
  backend: RobotBackend;
  /** A mock or sim run: never evidence about a real arm. */
  rehearsal: boolean;
  arms: Array<"follower" | "leader">;
  /** opened | refused | deferred | synthetic | unavailable */
  bus_access: string;
  bus_access_detail: string;
  overall: PreflightOverall;
  /** True only for a real-backend run with no failure and no blocking skip. */
  ready_for_hardware: boolean;
  exit_code: number;
  counts: Record<string, number>;
  checks: PreflightCheckResult[];
  notes: string[];
}

export interface SimulationScenario {
  id: string;
  name?: string;
  label?: string;
  description?: string;
  detail?: string;
  available?: boolean;
  active?: boolean;
  backend?: string;
  physics?: boolean;
  surface?: SimulationSurface;
  objects?: SimulationSceneObject[];
  supports_domain_randomization?: boolean;
}

export interface SimulationSurface {
  size_m: number[];
  height_m: number;
  color?: string;
}

export interface SimulationSceneObject {
  id: string;
  shape: "box" | "cylinder" | "sphere" | "mesh" | string;
  size_m: number[];
  pose: number[];
  color?: string;
  role?: string;
  randomize?: boolean;
  nominal_pose?: number[];
  nominal_size_m?: number[];
}

export interface SimulationRandomizationSample {
  position_offset_m: number[];
  yaw_offset_deg: number;
  scale: number;
  applied: boolean;
}

export interface SimulationScene {
  scenario_id: string;
  label: string;
  description: string;
  seed: number;
  randomized: boolean;
  surface: SimulationSurface;
  objects: SimulationSceneObject[];
  randomization: {
    applied: boolean;
    seed: number;
    settings: Record<string, unknown>;
    samples: Record<string, SimulationRandomizationSample>;
  };
}

export interface SimulationJointTelemetry {
  source: "simulation" | string;
  position_units: string;
  velocity_units: string;
  measured: Record<string, number>;
  target: Record<string, number>;
  error: Record<string, number>;
  velocity: Record<string, number>;
}

export interface SimulationKinematicFrame {
  name: string;
  position_m: number[];
}

export interface SimulationKinematics {
  model: string;
  position_m: number[];
  orientation_rpy_deg: number[];
  gripper_aperture_m: number;
  frames: SimulationKinematicFrame[];
}

export interface SimulationState {
  backend: string;
  backend_kind: string;
  physics: boolean;
  truth_model: string;
  hardware_isolated: boolean;
  writes_motor_commands: boolean;
  status: "playing" | "paused" | string;
  scheduler: string;
  playback_rate: number;
  fixed_dt_s: number;
  fixed_hz: number;
  step_count: number;
  sim_time_s: number;
  generation: number;
  updated_at: string;
  joint_order: string[];
  joint_limits_deg: Record<string, number[]>;
  telemetry: SimulationJointTelemetry;
  kinematics: SimulationKinematics;
  scene: SimulationScene;
  /**
   * v0.2, MuJoCo engine only: contacts, effort, rigid-body objects and the
   * servo profile. null on the kinematic engine, absent on older servers.
   */
  physics_state?: SimulationPhysicsState | null;
}

/** One free rigid body in the MuJoCo scene (`physics_state.objects.<id>`). */
export interface SimulationPhysicsObject {
  position_m: number[];
  quat_wxyz: number[];
  speed_m_s: number;
  grasped: boolean;
  /** Contact counts with the table, each jaw, the arm and other objects. */
  contacts: Record<string, number>;
}

/** `SimulationState.physics_state` as docs/simulation.md describes it. */
export interface SimulationPhysicsState {
  engine: string;
  mujoco_version?: string;
  model?: string;
  timestep_s: number;
  substeps_per_tick: number;
  physics_steps: number;
  physics_time_s: number;
  /**
   * Ticks skipped by the catch-up limit: during play nobody polled the
   * simulator for more than 2 s. `step_count` counts only simulated ticks.
   */
  dropped_ticks: number;
  torque_enabled: boolean;
  servo: {
    mode: string;
    /** null in `step` mode. */
    acceleration_deg_s2: number | null;
    max_speed: Record<string, number>;
    speed_units: string;
    reference: Record<string, number>;
  };
  /** Signed actuator force as a percentage of the force limit (servo load). */
  effort_pct: Record<string, number>;
  contacts: {
    total: number;
    arm_self: number;
    arm_table: number;
    max_penetration_m: number;
  };
  objects: Record<string, SimulationPhysicsObject>;
  markers: Record<string, number[]>;
}

export interface SimulationResetRequest {
  scenario_id?: string;
  seed?: number | null;
  randomize?: boolean;
}

export interface SimulationResetResponse extends Ack {
  state: SimulationState;
}

export interface SimulationStepRequest {
  steps?: number;
  targets?: Record<string, number>;
}

export interface SimulationTrajectoryWaypoint {
  time_s: number;
  targets: Record<string, number>;
}

export interface SimulationTrajectoryPreviewRequest {
  target?: Record<string, number>;
  duration_s?: number;
  waypoints?: SimulationTrajectoryWaypoint[];
  sample_hz?: number;
}

export interface SimulationTrajectoryPoint {
  step: number;
  time_s: number;
  measured: Record<string, number>;
  target: Record<string, number>;
  error: Record<string, number>;
  end_effector: SimulationKinematics;
}

export interface SimulationTrajectoryPreview {
  backend: string;
  physics: boolean;
  mutates_state: boolean;
  source_step_count: number;
  fixed_dt_s: number;
  requested_duration_s: number;
  quantized_duration_s: number;
  requested_sample_hz: number;
  sample_hz: number;
  points: SimulationTrajectoryPoint[];
  final_reached: boolean;
  joint_limits_deg: Record<string, number[]>;
  starting_target: Record<string, number>;
  /** v0.2: how targets are applied over time, e.g. "step_targets_like_play". */
  semantics?: string | null;
  /** v0.2: the |target - measured| bound behind `final_reached`, in suite units. */
  reached_tolerance?: number | null;
}

export interface SimulationModelInfo {
  id: string;
  repo?: string;
  path?: string;
  commit?: string;
  license?: string;
  vendored?: boolean;
}

/** One engine in `/api/simulation/capabilities` `adapters`. */
export interface SimulationAdapter {
  /** null: not probed (SO101_SIM_ENGINE=kinematic never imports MuJoCo). */
  available?: boolean | null;
  active: boolean;
  kind: string;
  detail: string;
  model?: string | SimulationModelInfo;
}

/** The MuJoCo adapter: installed, usable, active, and why (v0.2). */
export interface SimulationMujocoAdapter extends SimulationAdapter {
  package_installed?: boolean;
  package_version?: string | null;
  adapter_implemented?: boolean;
  /** SO101_SIM_ENGINE as configured: auto, kinematic or mujoco. */
  engine_setting?: string;
  /** Why the active engine was chosen, e.g. why physics is not running. */
  selection_reason?: string;
  error?: string | null;
  /** Physics only. null while the server's GL probe is still running ("checking", not unavailable). */
  render_available?: boolean | null;
  render_error?: string | null;
  /** "probing", "available" or "unavailable: <reason>"; null when not probed. */
  render_status?: string | null;
  fixed_hz?: number | null;
  timestep_s?: number | null;
  substeps_per_tick?: number | null;
  servo_mode?: string | null;
  cameras?: string[] | null;
  model?: SimulationModelInfo;
}

export interface SimulationCapabilities {
  active_backend: string;
  backend_kind: string;
  physics_active: boolean;
  fixed_step: boolean;
  hardware_isolated: boolean;
  writes_motor_commands: boolean;
  adapters: SimulationAdapters;
}

/** The engines the UI reads; the server may list others, which are ignored. */
export interface SimulationAdapters {
  kinematic?: SimulationAdapter;
  mujoco?: SimulationMujocoAdapter;
}

/** One camera frame from `GET /api/simulation/render` (v0.2, MuJoCo only). */
export interface SimulationFrame {
  blob: Blob;
  /** From the X-SO101-Sim-* headers: the instant the frame shows. */
  camera: string | null;
  step: number | null;
  generation: number | null;
  simTimeS: number | null;
}

export interface SimulationRenderRequest {
  camera?: string;
  width?: number;
  height?: number;
  format?: "jpeg" | "png";
  quality?: number;
}

/**
 * Wrist pose in the normalised tracking frame. C9: `z` is null for monocular
 * camera tracking (no metric depth); the server then holds the elbow at its
 * engage-time value.
 */
export interface TrackingWrist {
  x?: number;
  y?: number;
  normalized_x?: number;
  normalized_y?: number;
  z?: number | null;
  pitch?: number;
  roll?: number;
  yaw?: number;
}

export interface TrackingPreviewRequest {
  source: string;
  confidence: number;
  intent?: Record<string, number | string | boolean | null>;
  wrist?: TrackingWrist;
  pinch?: number;
  execute?: boolean;
  lease_id?: string;
  sequence?: number;
  /** ISO-8601 in server time (see useMotionLease.timestampFor). */
  sent_at?: string;
  ttl_ms?: number;
  deadman?: boolean;
}

/** C9: executing requires a lease, a sequence, a TTL, the deadman and `sent_at`. */
export interface TrackingExecuteRequest extends TrackingPreviewRequest {
  execute: true;
  lease_id: string;
  sequence: number;
  sent_at: string;
  ttl_ms: number;
  deadman: true;
}

export interface TrackingPreviewResponse {
  accepted: boolean;
  reason: string;
  targets: Record<string, number>;
  desired_targets: Record<string, number>;
  commanded_targets: Record<string, number>;
  confidence: number;
}

export interface AgentProposal {
  id: string;
  title?: string;
  objective?: string;
  description?: string;
  skill?: string;
  status?: string;
  risk?: string;
  created_at?: string;
  updated_at?: string;
  targets?: Record<string, number>;
  constraints?: {
    max_duration_seconds?: number;
    require_deadman?: boolean;
    allow_motion?: boolean;
    planner?: {
      provider?: string;
      model?: string;
      rationale?: string;
    };
    [key: string]: unknown;
  };
  steps?: Array<string | { label?: string; detail?: string }>;
  /** C8: sha256 of the canonical JSON of `targets`. */
  target_digest?: string;
  /** C8: set once execution ends (or partially, after a cancel). */
  result?: AgentResult | null;
  [key: string]: unknown;
}

/**
 * C8 terminal statuses. "expired": a proposal older than 120 s that someone
 * tried to approve or execute; it must be proposed again.
 */
export const AGENT_TERMINAL_STATUSES: readonly string[] = [
  "executed",
  "failed",
  "cancelled",
  "timeout",
  "expired",
];

/**
 * C11 end states. "unknown": the server could not confirm the motor state
 * (a cancel with no live run, a restart, an unreadable follower). Values this
 * UI does not know are rendered verbatim.
 */
export type AgentEndState = "halted" | "deenergized" | "estopped" | "fault" | "unknown" | string;

/** C8 result, set when a run ends (only `started_at` while it runs). */
export interface AgentResult {
  reached?: boolean;
  /** Per joint: degrees, except `gripper`, which is in gripper units (0-100). */
  final_error_deg?: Record<string, number>;
  /** The largest per-joint value of `final_error_deg` (same units). */
  max_abs_error_deg?: number | null;
  duration_s?: number | null;
  steps?: number | null;
  end_state?: AgentEndState;
  status?: string;
  /**
   * Why the run ended: target_reached, max_duration, cancelled, deadman,
   * operator_heartbeat, estop, fault, rejected, arm, error, shutdown, restart.
   */
  stop_reason?: string;
  /** Server's explanation, shown verbatim; "" when the target was reached. */
  error?: string;
  targets?: Record<string, number>;
  tolerance?: { arm_deg?: number; gripper?: number };
  started_at?: string | null;
  finished_at?: string | null;
  /**
   * Set on "expired": how old the proposal was, on the server's monotonic
   * clock; null when it was created before a server restart (age unknown).
   */
  age_seconds?: number | null;
  cancelled?: boolean;
  [key: string]: unknown;
}

export interface AgentApproval {
  approval_token: string;
  expires_at: string;
  proposal: AgentProposal;
}

export interface CreateAgentProposalRequest {
  objective: string;
  skill: string;
  planner: "deterministic" | "configured";
  constraints: {
    max_duration_seconds: number;
    require_deadman: boolean;
    allow_motion: boolean;
  };
}

export interface AgentPlannerStatus {
  default_provider: string;
  configured_provider_available: boolean;
  configured_provider: string;
  /** Why the configured planner cannot run (sim-B13); null when it can. */
  unavailable_reason?: string | null;
  model: string;
  bounded_skills: string[];
  motion_authority: string;
}

export type SimulationScenariosResponse =
  | SimulationScenario[]
  | { scenarios: SimulationScenario[] };
export type AgentProposalsResponse =
  | AgentProposal[]
  | { proposals: AgentProposal[] };
export type AuditEventsResponse = AuditEvent[] | { events: AuditEvent[] };

// ---------------------------------------------------------------------------
// Fetch wrapper
// ---------------------------------------------------------------------------

const BASE = "";

export class ApiError extends Error {
  status: number;
  body: string;
  constructor(status: number, message: string, body: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

/** Pydantic's 422 `detail` entries (F16). */
interface ValidationIssue {
  loc?: Array<string | number>;
  msg?: string;
}

/**
 * Operator-facing text from a FastAPI error body: a string `detail`, or the
 * array form validation errors use (F16), e.g. "fps: Input should be <= 240".
 */
export function errorDetail(parsed: unknown): string | null {
  if (!parsed || typeof parsed !== "object") return null;
  const detail = (parsed as { detail?: unknown }).detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts = detail
      .map((item: ValidationIssue | string) => {
        if (typeof item === "string") return item;
        if (!item || typeof item !== "object" || typeof item.msg !== "string") return null;
        // The first loc entry is where the value came from (body, query, path).
        const field = (Array.isArray(item.loc) ? item.loc.slice(1) : []).join(".");
        return field ? `${field}: ${item.msg}` : item.msg;
      })
      .filter((part): part is string => !!part);
    return parts.length ? parts.join("; ") : null;
  }
  return null;
}

/**
 * The route itself is absent on this server (an older backend): FastAPI's
 * default 404, a 405 from a path that only exists for other methods, or a
 * non-JSON 2xx such as an SPA fallback page (F25).
 */
export function isMissingRoute(error: unknown): boolean {
  if (!(error instanceof ApiError)) return false;
  if (error.status === 405) return true;
  if (error.status === 404) return error.message === "Not Found";
  return error.status >= 200 && error.status < 300;
}

type BodyReader<T> = (res: Response, method: string, path: string) => Promise<T>;

async function readJson<T>(res: Response, method: string, path: string): Promise<T> {
  if (res.status === 204) return undefined as T;
  const ct = res.headers.get("content-type") ?? "";
  if (!ct.includes("application/json")) {
    // F25: an HTML fallback page (or any non-JSON body) is not an API
    // answer; surface it here instead of handing `undefined` to a page.
    let text = "";
    try {
      text = await res.text();
    } catch {
      /* ignore */
    }
    throw new ApiError(
      res.status,
      `${method} ${path} returned ${ct.split(";")[0] || "a body without a content type"} instead of JSON. The server may not have this API route.`,
      text.slice(0, 500),
    );
  }
  return (await res.json()) as T;
}

function numberHeader(res: Response, name: string): number | null {
  const value = res.headers.get(name);
  if (value === null || value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

async function readFrame(res: Response, method: string, path: string): Promise<SimulationFrame> {
  // JPEG needs Pillow or OpenCV on the server; without them the frame is PNG.
  const ct = res.headers.get("content-type") ?? "";
  if (!ct.startsWith("image/")) {
    throw new ApiError(
      res.status,
      `${method} ${path} returned ${ct.split(";")[0] || "a body without a content type"} instead of an image. The server may not have this API route.`,
      "",
    );
  }
  return {
    blob: await res.blob(),
    camera: res.headers.get("x-so101-sim-camera"),
    step: numberHeader(res, "x-so101-sim-step"),
    generation: numberHeader(res, "x-so101-sim-generation"),
    simTimeS: numberHeader(res, "x-so101-sim-time-s"),
  };
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  timeoutMs = 8000,
  read: BodyReader<T> = readJson,
): Promise<T> {
  const init: RequestInit = {
    method,
    headers: {},
  };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  const controller = new AbortController();
  init.signal = controller.signal;
  // The timeout covers reading the body too, not just the response headers.
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(BASE + path, init);
    if (!res.ok) {
      let text = "";
      try {
        text = await res.text();
      } catch {
        /* ignore */
      }
      let message = `${method} ${path} failed: ${res.status}`;
      try {
        message = errorDetail(JSON.parse(text)) ?? message;
      } catch {
        if (text) message = text;
      }
      throw new ApiError(res.status, message, text);
    }
    // Awaited here so the timeout also covers reading the body.
    return await read(res, method, path);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError(408, `${method} ${path} timed out`, "request timed out");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export const api = {
  get: <T>(path: string) => request<T>("GET", path),
  post: <T>(path: string, body?: unknown, timeoutMs?: number) =>
    request<T>("POST", path, body, timeoutMs),
  del: <T>(path: string) => request<T>("DELETE", path),
};

// Encode a repo_id that may contain "/" so it can live in a single path segment.
export function encodeRepoId(repoId: string): string {
  return encodeURIComponent(repoId);
}

// ---------------------------------------------------------------------------
// Endpoint functions
// ---------------------------------------------------------------------------

// Health & robot
export const getHealth = () => api.get<Health>("/api/health");

export interface CameraPreviewStatus {
  active: boolean;
  error: string | null;
  cameras: string[];
  devices: Record<string, {
    device: { streams?: { color?: { width: number; height: number; fps: number; fourcc?: string } } };
  }>;
}
export const getCameraPreviewStatus = () => api.get<CameraPreviewStatus>("/api/cameras/preview/status");
export const startCameraPreview = () => api.post<CameraPreviewStatus>("/api/cameras/preview/start");
export const stopCameraPreview = () => api.post<CameraPreviewStatus>("/api/cameras/preview/stop");
export const getRobotStatus = () => api.get<RobotStatus>("/api/robot/status");
// E-stop is exempt from the Origin guard (C1) and carries no body.
export const estop = () => api.post<Ack>("/api/robot/estop", undefined, 5000);
/** C2: a deliberate, attributed reset. */
export const resetEstop = () =>
  api.post<Ack>("/api/robot/estop/reset", {
    operator_id: getOperatorId(),
    confirm: true,
  });
export interface FaultResetOptions {
  /**
   * C2: required while `torque_unconfirmed` is true, after the operator has
   * cut physical power. The server answers 409 without it.
   */
  powerCutAcknowledged?: boolean;
}
export const resetFault = (options: FaultResetOptions = {}) =>
  api.post<Ack>("/api/robot/fault/reset", {
    operator_id: getOperatorId(),
    confirm: true,
    ...(options.powerCutAcknowledged ? { power_cut_acknowledged: true } : {}),
  });
/** Legacy stop. C11 makes it an alias of Halt; wave-1 servers soft-stop. */
export const stopRobot = () => api.post<Ack>("/api/robot/stop");
/** C11 Halt: freeze in place with motors on. A repeat renews the hold timer. */
export const haltRobot = (reason?: string) =>
  api.post<Ack>(
    "/api/robot/halt",
    { operator_id: getOperatorId(), ...(reason ? { reason } : {}) },
    5000,
  );
/** C11 Collapse: lower to rest at a capped speed, then motors off. */
export const collapseRobot = (reason?: string) =>
  api.post<Ack>("/api/robot/collapse", {
    operator_id: getOperatorId(),
    ...(reason ? { reason } : {}),
  });
/**
 * The server wants a heartbeat younger than 2 s. A short timeout frees the
 * supervision hook's in-flight slot quickly, so one lost request does not
 * silence the pulses that follow it.
 */
export const HEARTBEAT_TIMEOUT_MS = 1500;
export const heartbeatOperator = () =>
  api.post<Ack>(
    "/api/robot/operator-heartbeat",
    { operator_id: getOperatorId() },
    HEARTBEAT_TIMEOUT_MS,
  );

// Video stream URL helper (use directly as <img src>)
export const videoUrl = (cam: string) => `/api/video?cam=${encodeURIComponent(cam)}`;
/** C10: one fresh JPEG (`Cache-Control: no-store`) or a 503 JSON detail. */
export const snapshotUrl = (cam: string) =>
  `/api/video/snapshot?cam=${encodeURIComponent(cam)}`;

// Teleop
export const startTeleop = (req: TeleopRequest) =>
  api.post<Ack>("/api/teleop/start", { ...req, operator_id: getOperatorId() });
export const stopTeleop = () => api.post<Ack>("/api/teleop/stop");

export interface McapEpisode {
  id: string;
  task: string;
  source: string;
  started_at_ns: number;
  starting_segment?: string;
  ended_at_ns?: number;
  status: "starting" | "recording" | "saved" | "interrupted";
  end_reason?: string;
}

export interface McapStatus {
  recorder: {
    recording: boolean;
    service_ready?: boolean;
    telemetry_fresh: boolean;
    bytes?: number;
    path?: string;
    error?: string;
  };
  cameras: Record<string, { live: boolean; age_seconds?: number; actual_preview_fps?: number }>;
  active: McapEpisode | null;
  episodes: McapEpisode[];
}

export const getMcapStatus = () => api.get<McapStatus>("/api/mcap/status");

export interface ControlTick {
  seq: number;
  timestamp_ns: number;
  seconds?: number;
  observation: Record<string, number>;
  action: Record<string, number>;
  desired_action: Record<string, number>;
  timing?: { cycle_interval_seconds?: number | null; io_lock_wait_seconds?: number; motor_step_seconds?: number; previous_tail_seconds?: number | null; previous_pacing?: { deadline_lateness_seconds?: number } | null } | null;
  control_profile?: { limits?: Record<string, [number, number]>; max_velocity?: Record<string, number>; [key: string]: unknown } | null;
}
export interface ControlTickBatch {
  stream_id: string;
  last_seq: number;
  missing_before_buffer: number;
  ticks: ControlTick[];
}
export interface McapReviewItem {
  episode: McapEpisode;
  recording_available: boolean;
  review: { data_checks_passed?: boolean; issues?: string[]; health_review?: { attention_required: boolean; advisories: unknown[] }; [key: string]: unknown } | null;
}
export interface McapTimeline {
  episode: McapEpisode;
  duration_seconds: number;
  sample_count: number;
  samples_without_episode_identity: number;
  display_sample_count: number;
  measured_control_hz: number;
  max_control_gap_seconds: number;
  ticks: ControlTick[];
  profile: ControlTick["control_profile"];
  cameras: string[];
  topics: Record<string, number>;
  note: string;
}
export const getControlTicks = (after = 0, latestOnly = false) => api.get<ControlTickBatch>(`/api/mcap/control-ticks?after_seq=${after}&latest_only=${latestOnly}`);
export const getMcapReviews = () => api.get<{ recordings: McapReviewItem[] }>("/api/mcap/reviews");
export const getMcapTimeline = (id: string) => api.get<McapTimeline>(`/api/mcap/reviews/${encodeURIComponent(id)}/timeline`);
export const getMcapSnapshot = (id: string, topic: string, seconds: number) => api.get<{ topic: string; record: { seconds: number; data: unknown } | null; retained_messages: number }>(`/api/mcap/reviews/${encodeURIComponent(id)}/snapshot?topic=${encodeURIComponent(topic)}&seconds=${seconds}`);
export const startMcapPlayback = (id: string, seconds: number, speed: number) => api.post<{ playback_id: string; delay_seconds: number }>(`/api/mcap/reviews/${encodeURIComponent(id)}/playback`, { seconds, speed });
export const startMcapEpisode = (task: string) =>
  api.post<McapEpisode>("/api/mcap/episodes/start", { task, operator_id: getOperatorId() });
export const saveMcapEpisode = (episodeId: string, timeoutMs?: number) =>
  api.post<McapEpisode>("/api/mcap/episodes/save", {
    episode_id: episodeId,
    operator_id: getOperatorId(),
  }, timeoutMs);

// Calibration
export const startCalibration = (req: CalibrateRequest) =>
  api.post<Ack>("/api/calibration/start", req);
export const getCalibrationStatus = (arm: string) =>
  api.get<CalibrationStatus>(`/api/calibration/status/${arm}`);
export const readCalibrationPose = (arm: string) =>
  api.post<CalibrationPose>(`/api/calibration/pose/${arm}`, { confirm_supported_torque_off: true });
export const getCachedCalibrationPose = (arm: string) =>
  api.get<CalibrationPose | null>(`/api/calibration/pose/${arm}`);
export const getCalibrationHistory = () =>
  api.get<CalibrationHistory>("/api/calibration/history?limit=100");

// Recording
export const startRecording = (req: StartRecordingRequest) =>
  api.post<Ack>("/api/recording/start", {
    ...req,
    operator_id: getOperatorId(),
  });
export const startEpisode = () => api.post<Ack>("/api/recording/episode/start");
export const saveEpisode = () =>
  api.post<EpisodeResult>("/api/recording/episode/save");
export const stopRecording = () => api.post<Ack>("/api/recording/stop");

// Datasets
export const getDatasets = () => api.get<DatasetSummary[]>("/api/datasets");
export const getDataset = (repoId: string) =>
  api.get<DatasetDetail>(`/api/datasets/${encodeRepoId(repoId)}`);
export const getDatasetQuality = (repoId: string) =>
  api.get<DatasetQualityReport>(
    `/api/datasets/${encodeRepoId(repoId)}/qc`,
  );
export const getEpisodeTimeline = (
  repoId: string,
  episode: number,
  offset = 0,
  limit = 300,
  stride = 1,
) =>
  api.get<EpisodeTimeline>(
    `/api/datasets/${encodeRepoId(repoId)}/episodes/${episode}/timeline?offset=${offset}&limit=${limit}&stride=${stride}`,
  );
export const getReplayPreview = (
  repoId: string,
  episode: number,
  maxPoints = 240,
) =>
  api.get<EpisodeTimeline>(
    `/api/datasets/${encodeRepoId(repoId)}/episodes/${episode}/replay-preview?max_points=${maxPoints}`,
  );
export const setEpisodeOutcome = (
  repoId: string,
  episode: number,
  outcome: EpisodeOutcome,
  notes = "",
) =>
  api.post<DatasetDetail>(
    `/api/datasets/${encodeRepoId(repoId)}/episodes/${episode}/outcome`,
    { outcome, notes },
  );
export const getDatasetTrash = () =>
  api.get<TrashedDatasetSummary[]>("/api/datasets/trash");
export const restoreDataset = (trashId: string) =>
  api.post<DatasetSummary>("/api/datasets/trash/restore", {
    trash_id: trashId,
  });
export const deleteDataset = (repoId: string) =>
  api.del<Ack>(`/api/datasets/${encodeRepoId(repoId)}`);
export interface PushDatasetOptions {
  /** Defaults to true; a public push must be asked for explicitly. */
  private?: boolean;
}
/**
 * Data audit: an attributed JSON body replaces the `?private=` query. Older
 * servers ignore the body and default to private, which fails safe.
 */
export const pushDataset = (repoId: string, options: PushDatasetOptions = {}) =>
  api.post<Ack>(
    `/api/datasets/${encodeRepoId(repoId)}/push`,
    { operator_id: getOperatorId(), private: options.private ?? true },
    30 * 60 * 1000,
  );

// Training
export const getPresets = () =>
  api.get<PolicyPresetOut[]>("/api/training/presets");
export const startTraining = (req: StartTrainingRequest) =>
  api.post<JobSummary>("/api/training/jobs", req);
export const getJobs = () => api.get<JobSummary[]>("/api/training/jobs");
export const getJob = (id: string) =>
  api.get<JobSummary>(`/api/training/jobs/${id}`);
export const getJobLogs = (id: string) =>
  api.get<JobLogs>(`/api/training/jobs/${id}/logs`);
export const cancelJob = (id: string) =>
  api.post<Ack>(`/api/training/jobs/${id}/cancel`);

// Models
export const getModels = () => api.get<ModelSummary[]>("/api/models");
export const startRollout = (req: RolloutRequest) =>
  api.post<Ack>("/api/models/rollout/start", {
    ...req,
    operator_id: getOperatorId(),
  });
export const stopRollout = () => api.post<Ack>("/api/models/rollout/stop");

// Eval
/**
 * The server loads the policy inside this request (data-B13), which can take
 * far longer than the default timeout; a timeout here does not mean refusal.
 */
export const EVAL_START_TIMEOUT_MS = 120_000;
export const startEval = (req: StartEvalRequest) =>
  api.post<Ack>("/api/eval/start", { ...req, operator_id: getOperatorId() }, EVAL_START_TIMEOUT_MS);
export const getEvalState = () => api.get<EvalState>("/api/eval/state");
/** C3: attributed; only the session operator with a live heartbeat may score. */
export const scoreEval = (success: boolean) =>
  api.post<EvalState>("/api/eval/score", { operator_id: getOperatorId(), success });
/** C3: 409 unless this tab is the session operator with a fresh heartbeat. */
export const continueEval = () =>
  api.post<EvalState>("/api/eval/continue", { operator_id: getOperatorId() });
export const stopEval = () => api.post<Ack>("/api/eval/stop");

// Workstation capability discovery
export const getCapabilities = () =>
  api.get<CapabilitiesResponse>("/api/capabilities");

// Control plane. Motion-capable experiences acquire a short lease and keep it
// alive; the server remains the authoritative owner of armed/safety state.
export const getControlState = () =>
  api.get<ControlState>("/api/control/state");
export const createControlLease = (source: string, ttlSeconds = 15) =>
  api.post<ControlLease>("/api/control/lease", {
    source,
    ttl_seconds: ttlSeconds,
  });
export const heartbeatControlLease = (id: string) =>
  api.post<ControlLease>(`/api/control/lease/${encodeURIComponent(id)}/heartbeat`);
export const releaseControlLease = (id: string) =>
  api.del<Ack>(`/api/control/lease/${encodeURIComponent(id)}`);
export const armControl = (leaseId: string) =>
  api.post<ControlState>("/api/control/arm", { lease_id: leaseId });
export const observeControl = (leaseId: string) =>
  api.post<ControlObservation>("/api/control/observe", { lease_id: leaseId });
/** C5: only the lease owner may disarm; 409 for a wrong, missing or expired lease. */
export const disarmControl = (leaseId: string) =>
  api.post<ControlState>("/api/control/disarm", { lease_id: leaseId });
export const sendControlCommand = (command: ControlCommandRequest) =>
  api.post<ControlCommandResponse>("/api/control/command", command);
export const armAndSendControlCommand = (command: ControlCommandRequest) =>
  api.post<ControlCommandResponse>("/api/control/arm-command", command);

// Diagnostics and system audit trail
export const getDiagnostics = () =>
  api.get<DiagnosticsResponse>("/api/diagnostics");
export const getEvents = () =>
  api.get<AuditEventsResponse>("/api/events");

/**
 * The server backend's last saved preflight report, from the CLI or the
 * server; a 404 with a detail when there is none. Read-only. Running
 * preflight is POST /api/preflight (it reads the motor buses and saves a
 * report; a GET is answered 405 and runs nothing), and it belongs to the
 * wave-2 wizard, not here.
 */
export const getLatestPreflight = () =>
  api.get<PreflightReport>("/api/preflight/latest");

// Simulation and tracking intent preview
export const getSimulationScenarios = () =>
  api.get<SimulationScenariosResponse>("/api/simulation/scenarios");
export const getSimulationCapabilities = () =>
  api.get<SimulationCapabilities>("/api/simulation/capabilities");
export const getSimulationState = () =>
  api.get<SimulationState>("/api/simulation/state");
/**
 * One frame of the live MuJoCo scene. 503 while the kinematic engine runs or
 * without OpenGL; 422 for an unknown camera, size or format. There is no
 * stream: callers poll.
 */
export const getSimulationFrame = ({
  camera = "front",
  width = 320,
  height = 240,
  format = "jpeg",
  quality = 80,
}: SimulationRenderRequest = {}) => {
  const query = new URLSearchParams({
    camera,
    width: String(width),
    height: String(height),
    format,
    quality: String(quality),
  });
  return request("GET", `/api/simulation/render?${query.toString()}`, undefined, 8000, readFrame);
};
export const getSimulationTelemetry = () =>
  api.get<SimulationJointTelemetry>("/api/simulation/telemetry");
export const setSimulationTargets = (targets: Record<string, number>) =>
  api.post<SimulationState>("/api/simulation/targets", { targets });
export const stepSimulation = (request: SimulationStepRequest = {}) =>
  api.post<SimulationState>("/api/simulation/step", request);
export const playSimulation = (rate = 1) =>
  api.post<SimulationState>("/api/simulation/play", { rate });
export const pauseSimulation = () =>
  api.post<SimulationState>("/api/simulation/pause");
export const resetSimulation = (request: SimulationResetRequest = {}) =>
  api.post<SimulationResetResponse>("/api/simulation/reset", request);
export const previewSimulationTrajectory = (
  request: SimulationTrajectoryPreviewRequest,
) =>
  api.post<SimulationTrajectoryPreview>(
    "/api/simulation/trajectory/preview",
    request,
  );
export const previewTracking = (request: TrackingPreviewRequest) =>
  api.post<TrackingPreviewResponse>("/api/tracking/preview", request);
export const armAndExecuteTracking = (request: TrackingExecuteRequest) =>
  api.post<TrackingPreviewResponse>("/api/tracking/arm-execute", request);

// Agent proposal flow. Execution is deliberately a separate approval call.
export const getAgentProposals = () =>
  api.get<AgentProposalsResponse>("/api/agent/proposals");
export const getAgentPlannerStatus = () =>
  api.get<AgentPlannerStatus>("/api/agent/proposals/planner/status");
export const createAgentProposal = (request: CreateAgentProposalRequest) =>
  api.post<AgentProposal>("/api/agent/proposals", request);
const proposalPath = (id: string, action = "") =>
  `/api/agent/proposals/${encodeURIComponent(id)}${action ? `/${action}` : ""}`;
export const getAgentProposal = (id: string) => api.get<AgentProposal>(proposalPath(id));
/**
 * C8 step 1: approve exactly the targets on screen. The single-use token
 * (30 s) is bound to this proposal, digest, lease and operator.
 */
export const approveAgentProposal = (id: string, leaseId: string, targetDigest: string) =>
  api.post<AgentApproval>(proposalPath(id, "approve"), {
    lease_id: leaseId,
    operator_id: getOperatorId(),
    target_digest: targetDigest,
  });
/** C8 step 2: 202 with status "executing"; the server runs the motion loop. */
export const executeAgentProposal = (id: string, leaseId: string, approvalToken: string) =>
  api.post<AgentProposal>(proposalPath(id, "execute"), {
    lease_id: leaseId,
    approval_token: approvalToken,
  });
/** C8 deadman: at least every 300 ms while `require_deadman` motion runs. */
export const holdAgentProposal = (id: string, leaseId: string) =>
  api.post<Ack>(proposalPath(id, "hold"), { lease_id: leaseId }, 1000);
export const cancelAgentProposal = (id: string) =>
  api.post<AgentProposal>(proposalPath(id, "cancel"));

export interface MotorVisionStatus {
  running: boolean;
  live: boolean;
  error: string | null;
  detected: { id: number; name: string; pixel: [number, number] }[];
  camera_projection_calibrated: false;
}
export const getMotorVision = () => api.get<MotorVisionStatus>("/api/mcap/vision/status");
export const startMotorVision = () => api.post<MotorVisionStatus>("/api/mcap/vision/start");
export const stopMotorVision = () => api.post<MotorVisionStatus>("/api/mcap/vision/stop");
