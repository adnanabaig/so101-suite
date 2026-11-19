import { useCallback, useEffect, useRef, useState } from "react";
import { Hand, KeyRound, LockKeyhole } from "lucide-react";
import {
  armAndSendControlCommand,
  observeControl,
  sendControlCommand,
  getLatestPreflight,
} from "../api";
import { MAX_SYNC_ROUND_TRIP_MS, useMotionLease } from "../hooks/useMotionLease";
import { errorMessage } from "../lib/errors";
import { JOG_STEP, JOG_STEP_LARGE, jogActionForKey, jogTarget, nextJogJoint } from "../lib/jog";
import { isEditableTarget } from "../lib/keyboard";
import Button from "./Button";
import Card from "./Card";
import { useSafety } from "./safety/SafetyProvider";
import { StatusPill } from "./ui/StatusPill";
import ThumbstickControls from "./ThumbstickControls";
import { CENTERED_STICKS, stepThumbsticks, type StickAxes } from "../lib/thumbsticks";
import { usePolling } from "../hooks/usePolling";

export const JOINTS = [
  ["shoulder_pan", -110, 110, "deg"],
  ["shoulder_lift", -100, 100, "deg"],
  ["elbow_flex", -100, 100, "deg"],
  ["wrist_flex", -100, 100, "deg"],
  ["wrist_roll", -160, 160, "deg"],
  ["gripper", 0, 100, "% open"],
] as const;

type Targets = Record<string, number | undefined>;

const DEADMAN_KEYS = new Set([" ", "Enter"]);

/** Every joint's start position, or the names of the joints the robot did not report. */
export function readStartPose(observation: Record<string, number>): {
  targets: Targets;
  missing: string[];
} {
  const targets: Targets = {};
  const missing: string[] = [];
  for (const [joint] of JOINTS) {
    const value = observation[`${joint}.pos`] ?? observation[joint];
    if (typeof value === "number" && Number.isFinite(value)) targets[joint] = value;
    else missing.push(joint);
  }
  return { targets, missing };
}

/** Text for what Disarm does, from the server's advertised stops (C11). */
export function disarmHelpText(canCollapse: boolean): string {
  return canCollapse
    ? "Disarm lowers the arm to its rest pose at a capped speed, then turns the motors off."
    : "Disarm turns the motors off where the arm stands. The arm will drop under gravity, so support it first.";
}

export default function ManualControlPanel() {
  const lease = useMotionLease("manual-ui");
  const { view, status } = useSafety();
  const preflight = usePolling(getLatestPreflight, 5000, { enabled: status?.backend === "real" });
  const motionReady = status?.backend !== "real" || (preflight.data?.ready_for_hardware === true && !preflight.stale);
  const [controlMode, setControlMode] = useState<"jog" | "sticks">("jog");
  const stickAxes = useRef<StickAxes>(CENTERED_STICKS);
  const updateAxes = useCallback((axes: StickAxes) => { stickAxes.current = axes; }, []);
  const [targets, setTargetsState] = useState<Targets>(() =>
    Object.fromEntries(JOINTS.map(([joint]) => [joint, joint === "gripper" ? 50 : 0])),
  );
  // The command loop runs from an interval created when the hold began, so it
  // reads targets from this ref: a jog during the hold must reach the next
  // command, not the targets of the render that started the hold.
  const targetsRef = useRef(targets);
  const setTargets = (next: Targets) => {
    targetsRef.current = next;
    setTargetsState(next);
  };
  // The joint the arrow keys jog while the deadman is held.
  const [jogIndex, setJogIndex] = useState(0);
  const jogIndexRef = useRef(jogIndex);
  const [jogNotice, setJogNotice] = useState("");
  const [commandError, setCommandError] = useState<string | null>(null);
  // The lease whose complete start pose has been read; any other lease is not ready.
  const [poseLeaseId, setPoseLeaseId] = useState<string | null>(null);
  const [missingJoints, setMissingJoints] = useState<string[]>([]);
  const poseReady = lease.leaseId !== null && poseLeaseId === lease.leaseId;
  const observeGeneration = useRef(0);
  const holding = useRef(false);
  const timer = useRef<number | null>(null);
  const commandInFlight = useRef(false);
  const wasMotionActive = useRef(false);

  useEffect(() => {
    const generation = ++observeGeneration.current;
    const leaseId = lease.leaseId;
    if (!leaseId) return;
    void observeControl(leaseId)
      .then((result) => {
        if (generation !== observeGeneration.current) return;
        const { targets: pose, missing } = readStartPose(result.observation);
        targetsRef.current = pose;
        setTargetsState(pose);
        setMissingJoints(missing);
        // Streaming targets for unreported joints would command guesses.
        setPoseLeaseId(missing.length === 0 ? leaseId : null);
      })
      .catch((caught: unknown) => {
        if (generation !== observeGeneration.current) return;
        setCommandError(errorMessage(caught));
      });
  }, [lease.leaseId]);

  useEffect(() => {
    if ((wasMotionActive.current && !lease.motionActive) || !lease.leaseId) {
      holding.current = false;
      if (timer.current !== null) {
        window.clearInterval(timer.current);
        timer.current = null;
      }
    }
    wasMotionActive.current = lease.motionActive;
  }, [lease.motionActive, lease.leaseId]);

  const stop = async (reason?: string) => {
    holding.current = false;
    if (timer.current !== null) {
      window.clearInterval(timer.current);
      timer.current = null;
    }
    await lease.stopMotion(reason);
  };

  const completeTargets = (): Record<string, number> | null => {
    const current = targetsRef.current;
    const entries = JOINTS.map(([joint]) => [joint, current[joint]] as const);
    if (entries.some(([, value]) => typeof value !== "number")) return null;
    return Object.fromEntries(entries) as Record<string, number>;
  };

  const send = async () => {
    if (!holding.current || !lease.leaseId || !lease.canSend() || commandInFlight.current) return;
    if (controlMode === "sticks") {
      const limits = Object.fromEntries(JOINTS.map(([name, low, high]) => [name, [low, high] as const]));
      setTargets(stepThumbsticks(targetsRef.current, stickAxes.current, 0.1, limits));
    }
    const command = completeTargets();
    if (!command) return;
    commandInFlight.current = true;
    try {
      await sendControlCommand({
        lease_id: lease.leaseId,
        source: "manual-ui",
        sequence: lease.nextSequence(),
        sent_at: lease.timestampFor(),
        ttl_ms: 300,
        deadman: true,
        targets: command,
      });
      setCommandError(null);
    } catch (caught) {
      setCommandError(errorMessage(caught));
      await stop("command rejected");
    } finally {
      commandInFlight.current = false;
    }
  };

  const begin = async () => {
    if (holding.current || commandInFlight.current || !lease.leaseId || !poseReady || !motionReady) return;
    const command = completeTargets();
    if (!command) return;
    holding.current = true;
    const leaseId = lease.leaseId;
    commandInFlight.current = true;
    const armed = await lease.startMotion(() =>
      armAndSendControlCommand({
        lease_id: leaseId,
        source: "manual-ui",
        sequence: lease.nextSequence(),
        sent_at: lease.timestampFor(),
        ttl_ms: 300,
        deadman: true,
        targets: command,
      }),
    );
    commandInFlight.current = false;
    if (!armed || !holding.current) {
      await stop(armed ? "deadman released" : "arming failed");
      return;
    }
    if (holding.current) timer.current = window.setInterval(() => void send(), 100);
  };

  const stopRef = useRef(stop);
  useEffect(() => {
    stopRef.current = stop;
  });

  useEffect(() => {
    if (!motionReady && holding.current) void stopRef.current("preflight no longer ready");
  }, [motionReady]);

  // Jog keys, only while the deadman is held and motion is armed. They move
  // the streamed target inside the joint's range; the server still bounds
  // every step (joint limits, per-step cap, max_relative_target).
  useEffect(() => {
    if (!lease.motionActive || controlMode === "sticks") return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        !holding.current ||
        event.defaultPrevented ||
        event.ctrlKey ||
        event.metaKey ||
        event.altKey ||
        isEditableTarget(event.target)
      ) {
        return;
      }
      const action = jogActionForKey(event.key, event.shiftKey);
      if (!action) return;
      event.preventDefault();
      if (action.kind === "select") {
        const next = nextJogJoint(jogIndexRef.current, action.delta, JOINTS.length);
        jogIndexRef.current = next;
        setJogIndex(next);
        setJogNotice(`Jogging ${JOINTS[next][0]}`);
        return;
      }
      const [joint, minimum, maximum, unit] = JOINTS[jogIndexRef.current];
      const current = targetsRef.current[joint];
      if (typeof current !== "number") return;
      const value = jogTarget(current, action.delta, minimum, maximum);
      if (value === current) return;
      targetsRef.current = { ...targetsRef.current, [joint]: value };
      setTargetsState(targetsRef.current);
      if (!event.repeat) setJogNotice(`${joint} target ${value.toFixed(1)} ${unit}`);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [lease.motionActive, controlMode]);

  // A release anywhere ends a hold that pointer capture missed. Unrelated
  // pointer releases (no hold in progress) send nothing (F3).
  useEffect(() => {
    const release = () => {
      if (holding.current) void stopRef.current("deadman released");
    };
    window.addEventListener("pointerup", release);
    window.addEventListener("pointercancel", release);
    return () => {
      window.removeEventListener("pointerup", release);
      window.removeEventListener("pointercancel", release);
      if (timer.current !== null) window.clearInterval(timer.current);
    };
  }, []);

  const latencyTone =
    lease.latencyMs === null ? "neutral" : lease.latencyMs > MAX_SYNC_ROUND_TRIP_MS ? "warn" : "ok";

  return (
    <Card
      title="Manual joint control"
      subtitle="Set a preview target, then hold the deadman to stream bounded commands"
      actions={
        <>
          <StatusPill label="lease" value={lease.leaseId ? "owned" : "none"} tone={lease.leaseId ? "ok" : "neutral"} />
          <StatusPill label="motion" value={lease.motionActive ? "armed" : "off"} tone={lease.motionActive ? "warn" : "neutral"} />
          {lease.leaseId && (
            <StatusPill
              label="latency"
              value={lease.latencyMs === null ? "—" : `${lease.latencyMs.toFixed(0)} ms`}
              tone={latencyTone}
            />
          )}
        </>
      }
    >
      <div className="mb-4 flex gap-2">
        <Button variant={controlMode === "jog" ? "primary" : "secondary"} disabled={lease.motionActive || lease.busy}
          onClick={() => { if (!holding.current) setControlMode("jog"); }}>Joint jog</Button>
        <Button variant={controlMode === "sticks" ? "primary" : "secondary"} disabled={lease.motionActive || lease.busy}
          onClick={() => { if (!holding.current) setControlMode("sticks"); }}>Thumbsticks · WASD + arrows</Button>
      </div>
      {!motionReady && <p className="mb-3 text-sm text-amber-200">Complete the required preflight checks before powered manual control.</p>}
      {controlMode === "sticks" && <ThumbstickControls key={lease.motionActive ? "active" : "idle"} enabled={lease.motionActive} onAxesChange={updateAxes} />}
      <div className="grid gap-4 lg:grid-cols-[1fr_15rem]">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {JOINTS.map(([joint, minimum, maximum, unit], index) => {
            const value = targets[joint];
            const jogging = index === jogIndex;
            return (
              <label
                key={joint}
                data-jog-selected={jogging || undefined}
                className={`rounded-xl border bg-slate-950/35 p-3 text-xs text-slate-400 ${
                  jogging
                    ? lease.motionActive
                      ? "border-amber-300/60"
                      : "border-cyan-400/40"
                    : "border-white/5"
                }`}
              >
                <span className="flex justify-between gap-2">
                  <span className="flex items-center gap-1.5 font-mono">
                    {joint}
                    {jogging && (
                      <span className="rounded-sm border border-white/10 px-1 font-sans text-[9px] font-semibold uppercase tracking-wide text-slate-300">
                        jog
                      </span>
                    )}
                  </span>
                  <span className="font-mono text-cyan-200">
                    {typeof value === "number" ? `${value.toFixed(1)} ${unit}` : "—"}
                  </span>
                </span>
                <input
                  type="range"
                  min={minimum}
                  max={maximum}
                  step={1}
                  value={typeof value === "number" ? value : minimum}
                  disabled={lease.motionActive || typeof value !== "number"}
                  aria-label={`${joint} target`}
                  onFocus={() => {
                    jogIndexRef.current = index;
                    setJogIndex(index);
                  }}
                  onChange={(event) =>
                    setTargets({ ...targetsRef.current, [joint]: Number(event.target.value) })
                  }
                  className="mt-3 w-full accent-cyan-400"
                />
              </label>
            );
          })}
        </div>
        <div className="space-y-3">
          {!lease.leaseId ? (
            <Button className="w-full" onClick={() => void lease.acquire()} disabled={lease.busy}>
              <KeyRound aria-hidden="true" className="h-4 w-4" /> Acquire manual lease
            </Button>
          ) : (
            <>
              <button
                type="button"
                aria-pressed={lease.motionActive}
                onPointerDown={(event) => {
                  if (!event.isPrimary || event.button !== 0) return;
                  try {
                    event.currentTarget.setPointerCapture?.(event.pointerId);
                  } catch {
                    // Capture is best-effort; the window listener still sees the release.
                  }
                  void begin();
                }}
                onPointerUp={() => void stop("deadman released")}
                onPointerCancel={() => void stop("deadman released")}
                onLostPointerCapture={() => {
                  if (holding.current) void stop("deadman released");
                }}
                onKeyDown={(event) => {
                  if (!DEADMAN_KEYS.has(event.key)) return;
                  event.preventDefault();
                  if (!event.repeat) void begin();
                }}
                onKeyUp={(event) => {
                  if (DEADMAN_KEYS.has(event.key)) void stop("deadman released");
                }}
                onBlur={() => {
                  if (holding.current) void stop("deadman lost focus");
                }}
                disabled={!poseReady || !motionReady}
                className={`flex h-28 w-full touch-none select-none flex-col items-center justify-center rounded-2xl border-2 text-sm font-black uppercase tracking-wider transition focus-visible:outline-hidden focus-visible:ring-4 focus-visible:ring-cyan-300/50 ${lease.motionActive ? "border-amber-200 bg-amber-400 text-slate-950" : "border-cyan-300/30 bg-cyan-400/10 text-cyan-100"}`}
              >
                {lease.motionActive ? <Hand aria-hidden="true" className="mb-2 h-6 w-6" /> : <LockKeyhole aria-hidden="true" className="mb-2 h-6 w-6" />}
                {poseReady
                  ? "Hold to move"
                  : missingJoints.length > 0
                    ? "Start pose incomplete"
                    : "Reading safe start pose…"}
              </button>
              <p className="text-[10px] leading-relaxed text-muted">
                Hold with the pointer, or hold Space or Enter. While holding, Left and Right jog the
                marked joint by {JOG_STEP} (Shift: {JOG_STEP_LARGE}); Up and Down pick the joint.
              </p>
              <p className="sr-only" aria-live="polite" data-testid="jog-notice">
                {jogNotice}
              </p>
              <Button variant="secondary" className="w-full" onClick={() => void lease.release()}>
                Release lease
              </Button>
              <Button
                variant="secondary"
                className="w-full"
                disabled={lease.motionActive}
                onClick={() => void lease.disarm()}
              >
                Disarm
              </Button>
              <p className="text-[10px] leading-relaxed text-slate-400">{disarmHelpText(view.canCollapse)}</p>
            </>
          )}
          <p className="text-[10px] leading-relaxed text-muted">
            Releasing the deadman, hiding or leaving this tab, a stale command, or losing the lease
            stops motion:{" "}
            {view.canHalt
              ? "the arm halts and holds its position."
              : "this server turns torque off, so the arm drops under gravity."}{" "}
            Targets pass measured-relative and suite safety limits.
          </p>
          {missingJoints.length > 0 && (
            <p className="text-xs text-amber-200">
              The robot did not report {missingJoints.join(", ")}. Motion stays locked until every
              joint has a measured start position.
            </p>
          )}
          {(lease.error || commandError) && (
            <p role="alert" className="text-xs text-rose-300">
              {lease.error ?? commandError}
            </p>
          )}
        </div>
      </div>
    </Card>
  );
}
