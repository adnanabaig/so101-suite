import { useCallback, useEffect, useRef, useState } from "react";
import {
  CheckCircle2,
  Crosshair,
  Hand,
  Headset,
  LogIn,
  LogOut,
  RefreshCw,
  RotateCcw,
  ShieldCheck,
} from "lucide-react";
import {
  previewTracking,
  type TrackingPreviewResponse,
} from "../api";
import Button from "./Button";
import Card from "./Card";
import { StatusPill } from "./ui/StatusPill";
import { Notice, SectionLabel } from "./Workstation";
import {
  QUEST_CANONICAL_JOINT_COUNT,
  QuestWebXRSession,
  probeQuestWebXR,
  type QuestXRHandFrame,
  type QuestXRProbe,
  type QuestXRSessionStatus,
  type QuestXRVector3,
} from "../lib/questWebXR";
import { questOperatorWrist } from "../lib/operatorFrame";

const DEFAULT_STATUS: QuestXRSessionStatus = {
  phase: "idle",
  message: "Ready to request a headset session.",
  visibility: null,
  referenceSpace: null,
  handTrackingNegotiated: null,
  frameRate: null,
  startedAt: null,
  calibration: null,
};

function meters(vector: QuestXRVector3 | null): string {
  if (!vector) return "—";
  return `${vector.x.toFixed(3)}, ${vector.y.toFixed(3)}, ${vector.z.toFixed(3)} m`;
}

function phaseTone(
  phase: QuestXRSessionStatus["phase"],
): "ok" | "warn" | "danger" | "neutral" {
  if (phase === "active") return "ok";
  if (phase === "error") return "danger";
  if (phase === "starting" || phase === "ending") return "warn";
  return "neutral";
}

export default function QuestWebXRPanel({
  onSupportChange,
}: {
  onSupportChange?: (supported: boolean | null) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<QuestWebXRSession | null>(null);
  const mountedRef = useRef(true);
  const supportCallbackRef = useRef(onSupportChange);
  const previewInFlightRef = useRef(false);
  const lastPreviewAtRef = useRef(0);
  const trackingEpochRef = useRef(0);
  const [probe, setProbe] = useState<QuestXRProbe | null>(null);
  // The runtime is probed on mount, so the panel starts in the probing state.
  const [probing, setProbing] = useState(true);
  const [status, setStatus] = useState<QuestXRSessionStatus>(DEFAULT_STATUS);
  const [hands, setHands] = useState<QuestXRHandFrame | null>(null);
  const [trackingSeen, setTrackingSeen] = useState(false);
  const [lastTrackedAt, setLastTrackedAt] = useState<number | null>(null);
  const [preview, setPreview] = useState<TrackingPreviewResponse | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);

  useEffect(() => {
    supportCallbackRef.current = onSupportChange;
  });

  /** Callers set `probing` first; the mount probe relies on the initial state. */
  const runProbe = useCallback(async () => {
    supportCallbackRef.current?.(null);
    try {
      const result = await probeQuestWebXR();
      if (!mountedRef.current) return;
      setProbe(result);
      supportCallbackRef.current?.(result.immersiveVrSupported);
    } finally {
      if (mountedRef.current) setProbing(false);
    }
  }, []);

  const requestRetargetPreview = useCallback((frame: QuestXRHandFrame) => {
    const sample = frame.primaryHand;
    const relative = sample.relativeWrist;
    const now = Date.now();
    if (
      !relative ||
      previewInFlightRef.current ||
      now - lastPreviewAtRef.current < 200
    ) {
      return;
    }
    lastPreviewAtRef.current = now;
    previewInFlightRef.current = true;
    const epoch = trackingEpochRef.current;
    void previewTracking({
      source: "quest-webxr-preview",
      // B11: tracked-joint quality (inferred joints count half, an inferred
      // wrist caps it below the gate), never a handedness score.
      confidence: sample.trackingConfidence,
      // B10: the same operator frame as the camera path.
      wrist: questOperatorWrist(relative),
      pinch: sample.pinch,
      intent: {
        preview_only: true,
        canonical_joint_count: sample.joints.length,
        captured_at: frame.capturedAt,
      },
      execute: false,
      deadman: false,
      // C9: always stamped. Preview needs no lease, so this is client time.
      sent_at: new Date(frame.capturedAt).toISOString(),
    })
      .then((result) => {
        if (!mountedRef.current || trackingEpochRef.current !== epoch) return;
        setPreview(result);
        setPreviewError(null);
      })
      .catch((error: unknown) => {
        if (!mountedRef.current || trackingEpochRef.current !== epoch) return;
        setPreview(null);
        setPreviewError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        previewInFlightRef.current = false;
      });
  }, []);

  const handleHands = useCallback(
    (frame: QuestXRHandFrame | null) => {
      if (!mountedRef.current) return;
      setHands(frame);
      if (!frame) {
        trackingEpochRef.current += 1;
        setPreview(null);
        setPreviewError(null);
        return;
      }
      setTrackingSeen(true);
      setLastTrackedAt(frame.capturedAt);
      if (!frame.primaryHand.relativeWrist) {
        // No origin for this hand (not recentered, recentered on the other
        // hand, or the origin was reset): an older preview no longer applies.
        trackingEpochRef.current += 1;
        setPreview(null);
        return;
      }
      requestRetargetPreview(frame);
    },
    [requestRetargetPreview],
  );

  useEffect(() => {
    mountedRef.current = true;
    void runProbe();
    return () => {
      mountedRef.current = false;
      sessionRef.current?.dispose();
      sessionRef.current = null;
      trackingEpochRef.current += 1;
    };
  }, [runProbe]);

  const enterSession = async () => {
    if (!canvasRef.current) return;
    setTrackingSeen(false);
    setLastTrackedAt(null);
    setHands(null);
    setPreview(null);
    setPreviewError(null);
    trackingEpochRef.current += 1;
    const session =
      sessionRef.current ??
      new QuestWebXRSession(canvasRef.current, {
        onStatus: (next) => {
          if (mountedRef.current) setStatus(next);
        },
        onHands: handleHands,
      });
    sessionRef.current = session;
    try {
      await session.start();
    } catch {
      // The controller publishes a human-readable phase and message.
    }
  };

  const exitSession = async () => {
    trackingEpochRef.current += 1;
    await sessionRef.current?.stop();
  };

  const recenter = () => {
    if (!sessionRef.current?.recenterPreview()) return;
    setPreview(null);
    setPreviewError(null);
  };

  const primary = hands?.primaryHand ?? null;
  const isActive = status.phase === "active" || status.phase === "ending";
  const canEnter =
    probe?.secureContext === true &&
    probe.apiAvailable &&
    probe.immersiveVrSupported &&
    (status.phase === "idle" || status.phase === "error");
  const featureLabel = primary
    ? "Hand poses received"
    : status.handTrackingNegotiated === true
      ? "Negotiated · waiting for hands"
      : status.handTrackingNegotiated === false
        ? "Not negotiated"
        : "Requested · runtime unreported";

  return (
    <Card
      title="Quest 3 immersive hand preview"
      subtitle="Experimental WebXR · 26-anchor canonical hand frame · physical actuation disabled"
      actions={
        <StatusPill
          label="session"
          value={status.phase}
          tone={phaseTone(status.phase)}
        />
      }
    >
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        className="pointer-events-none fixed bottom-0 right-0 h-px w-px opacity-0"
        width={1}
        height={1}
      />
      <div className="space-y-4">
        <Notice tone="warn" title="Hardware-untested experimental path">
          The lifecycle and safety boundary are implemented against the WebXR hand-input
          specification, but this build has not been validated on a physical Quest 3. The
          session renders hand-joint points and requests backend retarget previews only;
          it never acquires a motion lease or sends an execution command.
        </Notice>

        <div className="grid gap-4 xl:grid-cols-[minmax(18rem,0.8fr)_minmax(0,1.2fr)]">
          <div className="space-y-4 rounded-xl border border-slate-800 bg-slate-950/35 p-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <SectionLabel>Session lifecycle</SectionLabel>
                <p className="mt-2 text-sm font-semibold text-slate-100">
                  {status.message}
                </p>
              </div>
              <Headset aria-hidden="true"
                className={`h-5 w-5 shrink-0 ${status.phase === "active" ? "text-emerald-300" : "text-muted"}`}
              />
            </div>

            <div className="grid grid-cols-2 gap-2 text-[11px]">
              {[
                ["Secure origin", probe?.secureContext ? "ready" : "blocked"],
                ["Immersive VR", probe?.immersiveVrSupported ? "available" : "unavailable"],
                ["Visibility", status.visibility ?? "outside XR"],
                ["Reference", status.referenceSpace ?? "not selected"],
                ["Hand feature", featureLabel],
                ["Refresh", status.frameRate ? `${status.frameRate} Hz` : "runtime default"],
              ].map(([label, value]) => (
                <div key={label} className="rounded-lg bg-slate-900/80 px-3 py-2">
                  <div className="text-[9px] font-bold uppercase tracking-wider text-muted">
                    {label}
                  </div>
                  <div className="mt-1 truncate text-slate-300" title={value}>
                    {value}
                  </div>
                </div>
              ))}
            </div>

            {probe && !probe.immersiveVrSupported && (
              <p className="text-xs leading-relaxed text-amber-300/80">{probe.detail}</p>
            )}

            <div className="flex flex-wrap gap-2">
              <Button
                size="sm"
                variant="secondary"
                onClick={() => {
                  setProbing(true);
                  void runProbe();
                }}
                disabled={probing || status.phase === "starting" || isActive}
              >
                <RefreshCw aria-hidden="true" className={`h-3.5 w-3.5 ${probing ? "motion-safe:animate-spin" : ""}`} />
                {probing ? "Probing…" : "Probe runtime"}
              </Button>
              {!isActive ? (
                <Button
                  size="sm"
                  onClick={() => void enterSession()}
                  disabled={!canEnter || status.phase === "starting"}
                >
                  <LogIn aria-hidden="true" className="h-3.5 w-3.5" />
                  {status.phase === "starting" ? "Entering…" : "Enter Quest preview"}
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant="danger"
                  onClick={() => void exitSession()}
                  disabled={status.phase === "ending"}
                >
                  <LogOut aria-hidden="true" className="h-3.5 w-3.5" />
                  {status.phase === "ending" ? "Exiting…" : "Exit immersive"}
                </Button>
              )}
            </div>
            <p className="text-[10px] leading-relaxed text-muted">
              Entry must be triggered here from Quest Browser. The session requests
              <span className="font-mono text-muted"> hand-tracking </span>
              and
              <span className="font-mono text-muted"> local-floor </span>
              as optional features, then falls back to local/viewer space.
            </p>
          </div>

          <div className="space-y-4 rounded-xl border border-slate-800 bg-[radial-gradient(circle_at_top_right,rgba(34,211,238,.07),transparent_52%)] p-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <SectionLabel>Live hand frame</SectionLabel>
                <div className="mt-2 flex items-center gap-2">
                  <span
                    className={`h-2 w-2 rounded-full ${primary ? "bg-emerald-300 shadow-[0_0_10px_rgba(110,231,183,.8)]" : "bg-slate-700"}`}
                  />
                  <span className="text-sm font-semibold text-slate-100">
                    {primary
                      ? `${primary.handedness} hand tracked`
                      : trackingSeen
                        ? "Tracking lost · stale pose cleared"
                        : "Waiting for a tracked hand"}
                  </span>
                </div>
              </div>
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="secondary"
                  onClick={recenter}
                  disabled={status.phase !== "active" || !primary?.wrist}
                >
                  <Crosshair aria-hidden="true" className="h-3.5 w-3.5" /> Recenter preview
                </Button>
                {status.calibration && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => sessionRef.current?.clearCalibration()}
                  >
                    <RotateCcw aria-hidden="true" className="h-3.5 w-3.5" /> Clear
                  </Button>
                )}
              </div>
            </div>

            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-lg border border-white/5 bg-slate-950/45 p-3">
                <div className="text-[9px] font-bold uppercase tracking-wider text-muted">
                  Canonical joints
                </div>
                <div className="mt-1 font-mono text-lg font-bold text-cyan-200">
                  {primary ? `${primary.joints.length}/${QUEST_CANONICAL_JOINT_COUNT}` : "—"}
                </div>
                <div className="text-[9px] text-muted">
                  {primary?.joints[0]?.derived ? "palm anchor derived" : "runtime poses"}
                </div>
              </div>
              <div className="rounded-lg border border-white/5 bg-slate-950/45 p-3">
                <div className="text-[9px] font-bold uppercase tracking-wider text-muted">
                  Tracking quality
                </div>
                <div className="mt-1 font-mono text-lg font-bold text-cyan-200">
                  {primary ? `${(primary.trackingConfidence * 100).toFixed(0)}%` : "—"}
                </div>
                <div className="text-[9px] text-muted">
                  {primary
                    ? `${primary.observedJointCount - primary.emulatedJointCount} tracked · ${primary.emulatedJointCount} inferred`
                    : "no sample"}
                </div>
              </div>
              <div className="rounded-lg border border-white/5 bg-slate-950/45 p-3">
                <div className="text-[9px] font-bold uppercase tracking-wider text-muted">
                  Thumb ↔ index
                </div>
                <div className="mt-1 font-mono text-lg font-bold text-amber-200">
                  {primary?.pinchDistanceM !== null && primary?.pinchDistanceM !== undefined
                    ? `${(primary.pinchDistanceM * 100).toFixed(1)} cm`
                    : "—"}
                </div>
                <div className="text-[9px] text-muted">
                  pinch {primary ? primary.pinch.toFixed(2) : "—"}
                </div>
              </div>
              <div className="rounded-lg border border-white/5 bg-slate-950/45 p-3">
                <div className="text-[9px] font-bold uppercase tracking-wider text-muted">
                  Preview origin
                </div>
                <div className="mt-1 flex items-center gap-1.5 text-sm font-bold text-slate-200">
                  {status.calibration ? (
                    <CheckCircle2 aria-hidden="true" className="h-3.5 w-3.5 text-emerald-300" />
                  ) : (
                    <Crosshair aria-hidden="true" className="h-3.5 w-3.5 text-muted" />
                  )}
                  {status.calibration ? "Calibrated" : "Not set"}
                </div>
                <div className="mt-1 text-[9px] text-muted">
                  {status.calibration
                    ? `${status.calibration.handedness} wrist`
                    : "capture a neutral wrist pose"}
                </div>
              </div>
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              {[
                ["Wrist (runtime)", meters(primary?.wrist?.position ?? null)],
                ["Wrist (centered)", meters(primary?.relativeWrist ?? null)],
                ["Thumb tip", meters(primary?.thumbTip?.position ?? null)],
                ["Index tip", meters(primary?.indexTip?.position ?? null)],
              ].map(([label, value]) => (
                <div
                  key={label}
                  className="flex items-center justify-between gap-3 rounded-lg bg-slate-950/40 px-3 py-2"
                >
                  <span className="text-[10px] text-muted">{label}</span>
                  <span className="font-mono text-[10px] text-slate-300">{value}</span>
                </div>
              ))}
            </div>

            {!status.calibration && primary && (
              <p className="text-[10px] leading-relaxed text-amber-300/75">
                Recenter from a neutral wrist pose to enable normalized backend retarget
                preview. This changes preview coordinates only; it does not reset Guardian or
                command the robot.
              </p>
            )}
            {lastTrackedAt && !primary && (
              <p className="text-[10px] text-muted">
                Last complete sample {new Date(lastTrackedAt).toLocaleTimeString()} · all
                displayed pose and retarget values were cleared on loss.
              </p>
            )}
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_minmax(18rem,0.55fr)]">
          <div className="rounded-xl border border-slate-800 bg-slate-950/30 p-4">
            <div className="flex items-center gap-2">
              <ShieldCheck aria-hidden="true" className="h-4 w-4 text-emerald-300" />
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
                Retarget preview only
              </span>
            </div>
            {preview ? (
              <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
                {Object.entries(preview.targets).slice(0, 6).map(([joint, value]) => (
                  <div key={joint} className="rounded-lg bg-slate-900/80 px-2.5 py-2">
                    <div className="truncate font-mono text-[9px] text-muted">{joint}</div>
                    <div className="mt-1 font-mono text-xs text-cyan-200">
                      {value.toFixed(2)}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <p className="mt-2 text-xs leading-relaxed text-muted">
                {status.calibration
                  ? "Waiting for a fresh calibrated hand pose."
                  : "Calibrate the preview origin to map wrist deltas into bounded normalized intent."}
              </p>
            )}
            {preview && (
              <p className="mt-2 text-[10px] text-muted">
                {preview.reason} · execute=false · deadman=false · commanded targets: 0
              </p>
            )}
            {previewError && (
              <p className="mt-2 text-[10px] text-rose-300">{previewError}</p>
            )}
          </div>
          <div className="rounded-xl border border-emerald-400/10 bg-emerald-400/[0.04] p-4">
            <div className="flex items-center gap-2 text-emerald-200">
              <Hand aria-hidden="true" className="h-4 w-4" />
              <span className="text-xs font-bold">Safety boundary</span>
            </div>
            <p className="mt-2 text-[10px] leading-relaxed text-muted">
              Quest input is intentionally disconnected from the execution endpoint. The
              existing browser-camera teleop remains the only hand path with an exclusive
              lease plus a continuously held DOM deadman.
            </p>
          </div>
        </div>
      </div>
    </Card>
  );
}
