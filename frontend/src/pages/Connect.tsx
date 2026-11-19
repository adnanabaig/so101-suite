import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  Camera,
  CheckCircle2,
  Crosshair,
  Hand,
  Headset,
  LockKeyhole,
  ShieldCheck,
  Wifi,
} from "lucide-react";
import {
  armAndExecuteTracking,
  getCapabilities,
  previewTracking,
  type TrackingPreviewResponse,
} from "../api";
import Button from "../components/Button";
import Card from "../components/Card";
import QuestWebXRPanel from "../components/QuestWebXRPanel";
import { StatusPill } from "../components/ui/StatusPill";
import {
  EmptyState,
  MetricTile,
  Notice,
  PageHeader,
  SectionLabel,
} from "../components/Workstation";
import { usePolling } from "../hooks/usePolling";
import { useMotionLease } from "../hooks/useMotionLease";
import {
  startBrowserHandTracking,
  type HandTrackingSample,
  type RunningHandTracker,
} from "../lib/handTracking";
import {
  cameraQualityHint,
  engageRoll,
  mirroredDisplayPoint,
  updateRoll,
  type RollReference,
} from "../lib/operatorFrame";
import {
  MIN_TRACKING_CONFIDENCE,
  cameraExecuteRequest,
  cameraPreviewRequest,
} from "../lib/trackingRequest";

type CameraState = "idle" | "requesting" | "ready" | "denied" | "unavailable";
type HandTrackerState = "idle" | "loading" | "running" | "error";
/**
 * A camera sample plus its roll relative to the current neutral (B6): the
 * neutral is captured when tracking (re)acquires a hand and again when the
 * deadman engages, so the first command of every hold has roll 0.
 */
type TrackedHand = HandTrackingSample & { roll: number };

export default function Connect({ embedded = false }: { embedded?: boolean }) {
  const handMotion = useMotionLease("browser-mediapipe");
  const { data: capabilities, error: capabilityError } = usePolling(
    getCapabilities,
    10000,
  );
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const handTrackerRef = useRef<RunningHandTracker | null>(null);
  const cameraGenerationRef = useRef(0);
  const trackerGenerationRef = useRef(0);
  const lastAutoPreviewRef = useRef(0);
  const autoPreviewInFlightRef = useRef(false);
  const executionInFlightRef = useRef(false);
  const handDeadmanRef = useRef(false);
  const handMotionReadyRef = useRef(false);
  const latestHandSampleRef = useRef<TrackedHand | null>(null);
  const rollReferenceRef = useRef<RollReference | null>(null);
  const sampleExpiryRef = useRef<number | null>(null);
  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [handTrackerState, setHandTrackerState] = useState<HandTrackerState>("idle");
  const [handTrackerError, setHandTrackerError] = useState<string | null>(null);
  const [handSample, setHandSample] = useState<TrackedHand | null>(null);
  const [handSampleFresh, setHandSampleFresh] = useState(false);
  const [xrSupported, setXrSupported] = useState<boolean | null>(null);
  const [intent, setIntent] = useState({ x: 0, y: 0, roll: 0, grip: 0.35 });
  const [confidence, setConfidence] = useState(0.82);
  const [preview, setPreview] = useState<TrackingPreviewResponse | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const secure = window.isSecureContext;
  const mediaSupported = !!navigator.mediaDevices?.getUserMedia;

  const trackingCapability = useMemo(
    () =>
      capabilities?.capabilities.find((capability) =>
        ["tracking", "hand_tracking", "quest", "webxr"].some((needle) =>
          capability.id.toLowerCase().includes(needle),
        ),
      ),
    [capabilities],
  );

  useEffect(
    () => () => {
      cameraGenerationRef.current += 1;
      trackerGenerationRef.current += 1;
      if (sampleExpiryRef.current !== null) window.clearTimeout(sampleExpiryRef.current);
      handTrackerRef.current?.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
    },
    [],
  );

  const startCamera = async () => {
    const generation = ++cameraGenerationRef.current;
    setCameraState("requesting");
    setCameraError(null);
    if (!mediaSupported) {
      setCameraState("unavailable");
      setCameraError("This browser does not expose MediaDevices.getUserMedia().");
      return;
    }
    try {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      if (cameraGenerationRef.current !== generation) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        try {
          await videoRef.current.play();
        } catch (error) {
          stream.getTracks().forEach((track) => track.stop());
          streamRef.current = null;
          videoRef.current.srcObject = null;
          throw error;
        }
      }
      if (cameraGenerationRef.current !== generation) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      setCameraState("ready");
    } catch (error) {
      if (cameraGenerationRef.current !== generation) return;
      setCameraState("denied");
      setCameraError(error instanceof Error ? error.message : String(error));
    }
  };

  const stopCamera = () => {
    cameraGenerationRef.current += 1;
    trackerGenerationRef.current += 1;
    if (sampleExpiryRef.current !== null) {
      window.clearTimeout(sampleExpiryRef.current);
      sampleExpiryRef.current = null;
    }
    handTrackerRef.current?.stop();
    handTrackerRef.current = null;
    latestHandSampleRef.current = null;
    rollReferenceRef.current = null;
    setHandTrackerState("idle");
    setHandSample(null);
    setHandSampleFresh(false);
    void handMotion.release();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraState("idle");
  };

  const endHandMotion = async () => {
    handDeadmanRef.current = false;
    handMotionReadyRef.current = false;
    await handMotion.stopMotion();
  };

  const sendTrackedExecution = async (
    sample: TrackedHand,
    armFirst = false,
  ): Promise<boolean> => {
    const leaseId = handMotion.currentLeaseId();
    if (
      !leaseId ||
      executionInFlightRef.current ||
      !handDeadmanRef.current ||
      (!armFirst && !handMotion.canSend())
    )
      return false;
    executionInFlightRef.current = true;
    try {
      const request = cameraExecuteRequest(sample, {
        leaseId,
        sequence: handMotion.nextSequence(),
        sentAt: handMotion.timestampFor(sample.capturedAt),
      });
      const result = await (armFirst
        ? armAndExecuteTracking(request)
        : previewTracking(request));
      if (!result.accepted) {
        throw new Error(result.reason);
      }
      setPreview(result);
      setPreviewError(null);
      return true;
    } catch (error) {
      setPreviewError(error instanceof Error ? error.message : String(error));
      if (armFirst) throw error;
      await endHandMotion();
      return false;
    } finally {
      executionInFlightRef.current = false;
    }
  };

  const beginHandMotion = async () => {
    const sample = latestHandSampleRef.current;
    if (
      !sample ||
      !handSampleFresh ||
      Date.now() - sample.capturedAt > 300 ||
      sample.confidence < MIN_TRACKING_CONFIDENCE ||
      executionInFlightRef.current ||
      handDeadmanRef.current
    )
      return;
    handDeadmanRef.current = true;
    // B6: roll is measured from the hand's orientation right now.
    rollReferenceRef.current = engageRoll(sample.rollRad);
    const firstSample: TrackedHand = { ...sample, roll: 0 };
    latestHandSampleRef.current = firstSample;
    const armed = await handMotion.startMotion(() =>
      sendTrackedExecution(firstSample, true),
    );
    if (!armed || !handDeadmanRef.current) {
      await endHandMotion();
      return;
    }
    handMotionReadyRef.current = true;
  };

  const onHandSample = (raw: HandTrackingSample | null) => {
    if (!raw) {
      latestHandSampleRef.current = null;
      rollReferenceRef.current = null;
      setHandSample(null);
      if (sampleExpiryRef.current !== null) {
        window.clearTimeout(sampleExpiryRef.current);
        sampleExpiryRef.current = null;
      }
      setHandSampleFresh(false);
      setConfidence(0);
      setPreview(null);
      if (handDeadmanRef.current) void endHandMotion();
      return;
    }
    const rolled = rollReferenceRef.current
      ? updateRoll(rollReferenceRef.current, raw.rollRad)
      : { reference: engageRoll(raw.rollRad), relative: 0 };
    rollReferenceRef.current = rolled.reference;
    const sample: TrackedHand = { ...raw, roll: rolled.relative };
    latestHandSampleRef.current = sample;
    setHandSample(sample);
    if (sampleExpiryRef.current !== null) window.clearTimeout(sampleExpiryRef.current);
    setHandSampleFresh(true);
    sampleExpiryRef.current = window.setTimeout(() => {
      latestHandSampleRef.current = null;
      rollReferenceRef.current = null;
      setHandSample(null);
      setHandSampleFresh(false);
      setConfidence(0);
      setPreview(null);
      if (handDeadmanRef.current) void endHandMotion();
    }, 350);
    setIntent({ x: sample.x, y: sample.y, roll: sample.roll, grip: sample.pinch });
    setConfidence(sample.confidence);
    if (sample.confidence < MIN_TRACKING_CONFIDENCE && handDeadmanRef.current) {
      void endHandMotion();
    } else if (handDeadmanRef.current && handMotionReadyRef.current) {
      void sendTrackedExecution(sample);
    }
    const now = Date.now();
    if (now - lastAutoPreviewRef.current < 200 || autoPreviewInFlightRef.current) return;
    lastAutoPreviewRef.current = now;
    autoPreviewInFlightRef.current = true;
    void previewTracking(
      cameraPreviewRequest(sample, handMotion.timestampFor(sample.capturedAt), sample.capturedAt),
    )
      .then(setPreview)
      .catch((error: unknown) =>
        setPreviewError(error instanceof Error ? error.message : String(error)),
      )
      .finally(() => {
        autoPreviewInFlightRef.current = false;
      });
  };

  useEffect(() => {
    if (!handMotion.motionActive) {
      handDeadmanRef.current = false;
      handMotionReadyRef.current = false;
    }
  }, [handMotion.motionActive]);

  // Latest-ref so the mount-once window listener always calls the current
  // endHandMotion (and therefore the current handMotion lease helpers).
  const endHandMotionRef = useRef(endHandMotion);
  useEffect(() => {
    endHandMotionRef.current = endHandMotion;
  });

  useEffect(() => {
    const release = () => {
      if (handDeadmanRef.current) void endHandMotionRef.current();
    };
    window.addEventListener("pointerup", release);
    window.addEventListener("pointercancel", release);
    return () => {
      window.removeEventListener("pointerup", release);
      window.removeEventListener("pointercancel", release);
    };
  }, []);

  const startHandTracker = async () => {
    if (!videoRef.current || cameraState !== "ready") return;
    const generation = ++trackerGenerationRef.current;
    setHandTrackerState("loading");
    setHandTrackerError(null);
    setPreviewError(null);
    try {
      handTrackerRef.current?.stop();
      const tracker = await startBrowserHandTracking(
        videoRef.current,
        onHandSample,
        (error) => {
          if (trackerGenerationRef.current !== generation) return;
          setHandTrackerState("error");
          setHandTrackerError(error instanceof Error ? error.message : String(error));
          handTrackerRef.current = null;
        },
      );
      if (trackerGenerationRef.current !== generation) {
        tracker.stop();
        return;
      }
      handTrackerRef.current = tracker;
      setHandTrackerState("running");
    } catch (error) {
      if (trackerGenerationRef.current !== generation) return;
      setHandTrackerState("error");
      setHandTrackerError(error instanceof Error ? error.message : String(error));
    }
  };

  const stopHandTracker = () => {
    trackerGenerationRef.current += 1;
    if (sampleExpiryRef.current !== null) {
      window.clearTimeout(sampleExpiryRef.current);
      sampleExpiryRef.current = null;
    }
    handTrackerRef.current?.stop();
    handTrackerRef.current = null;
    latestHandSampleRef.current = null;
    rollReferenceRef.current = null;
    setHandTrackerState("idle");
    setHandSample(null);
    setHandSampleFresh(false);
    void endHandMotion();
  };

  const updateIntent = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.buttons === 0 && event.pointerType === "mouse") return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    const y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
    setIntent((current) => ({
      ...current,
      x: Math.max(-1, Math.min(1, x)),
      y: Math.max(-1, Math.min(1, y)),
    }));
  };

  const requestPreview = async () => {
    setPreviewBusy(true);
    setPreviewError(null);
    setPreview(null);
    try {
      const result = await previewTracking({
        source: cameraState === "ready" ? "browser-camera-intent" : "manual-intent",
        confidence,
        intent: {
          normalized_x: intent.x,
          normalized_y: intent.y,
          grip: intent.grip,
          preview_only: true,
        },
        sent_at: handMotion.timestampFor(),
      });
      setPreview(result);
    } catch (error) {
      setPreviewError(error instanceof Error ? error.message : String(error));
    } finally {
      setPreviewBusy(false);
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        headingLevel={embedded ? 2 : 1}
        eyebrow="Embodied input"
        title="Quest 3 & hand tracking"
        description="Track a hand from a general-purpose browser camera, validate retargeted intent, and check Quest/WebXR readiness without bypassing the motion safety gate."
        icon={Headset}
      />

      {!secure && (
        <Notice tone="danger" title="A secure context is required">
          Quest Browser WebXR, camera access, and hand tracking require HTTPS.
          Serve this workstation through trusted HTTPS before opening it from a headset.
        </Notice>
      )}
      {capabilityError && (
        <Notice tone="warn" title="Capability discovery is unavailable">
          Browser checks still work, but backend tracking readiness cannot be verified: {capabilityError.message}
        </Notice>
      )}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Secure context"
          value={secure ? "Ready" : "Blocked"}
          detail={secure ? window.location.protocol : "HTTPS is mandatory on Quest"}
          tone={secure ? "green" : "red"}
          icon={LockKeyhole}
        />
        <MetricTile
          label="WebXR immersive"
          value={xrSupported === null ? "Not probed" : xrSupported ? "Supported" : "Unavailable"}
          detail="Quest Browser runtime capability"
          tone={xrSupported ? "green" : xrSupported === false ? "amber" : "neutral"}
          icon={Headset}
        />
        <MetricTile
          label="Camera"
          value={cameraState === "ready" ? "Streaming" : cameraState}
          detail={mediaSupported ? "MediaDevices available" : "Browser API unavailable"}
          tone={cameraState === "ready" ? "green" : cameraState === "denied" ? "red" : "neutral"}
          icon={Camera}
        />
        <MetricTile
          label="Tracking adapter"
          value={handTrackerState === "running" ? "Tracking" : trackingCapability?.available ? "Ready" : "Preview only"}
          detail={handSample ? `${handSample.handedness} hand · quality ${(handSample.confidence * 100).toFixed(0)}%` : trackingCapability?.detail ?? "Waiting for backend capability"}
          tone={handTrackerState === "running" ? "green" : trackingCapability?.available ? "green" : "amber"}
          icon={Hand}
        />
      </div>

      <QuestWebXRPanel onSupportChange={setXrSupported} />

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.45fr)_minmax(21rem,0.75fr)]">
        <Card
          title="Camera intent canvas"
          subtitle="MediaPipe hand landmarks or drag fallback → normalized wrist intent"
          actions={
            cameraState === "ready" ? (
              <>
                {handTrackerState === "running" ? (
                  <Button size="sm" variant="secondary" onClick={stopHandTracker}>Stop tracking</Button>
                ) : (
                  <Button size="sm" onClick={() => void startHandTracker()} disabled={handTrackerState === "loading"}>
                    <Hand aria-hidden="true" className="h-3.5 w-3.5" />
                    {handTrackerState === "loading" ? "Loading model…" : "Track hand"}
                  </Button>
                )}
                <Button size="sm" variant="secondary" onClick={stopCamera}>Stop camera</Button>
              </>
            ) : (
              <Button size="sm" onClick={() => void startCamera()} disabled={!secure || cameraState === "requesting"}>
                <Camera aria-hidden="true" className="h-3.5 w-3.5" />
                {cameraState === "requesting" ? "Requesting…" : "Enable camera"}
              </Button>
            )
          }
          bodyClassName="p-0"
          className="overflow-hidden"
        >
          <div
            className="relative aspect-video min-h-[24rem] touch-none overflow-hidden bg-[radial-gradient(circle_at_center,#122033_0,#070b14_68%)]"
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture?.(event.pointerId);
              updateIntent(event);
            }}
            onPointerMove={updateIntent}
          >
            {/* Shown mirrored (a selfie view) so the operator frame's +x, the
                operator's right, is also the right of the screen (B10). */}
            <video
              ref={videoRef}
              muted
              playsInline
              data-testid="camera-video"
              className={`h-full w-full -scale-x-100 object-cover transition-opacity ${cameraState === "ready" ? "opacity-65" : "opacity-0"}`}
            />
            <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(rgba(34,211,238,.05)_1px,transparent_1px),linear-gradient(90deg,rgba(34,211,238,.05)_1px,transparent_1px)] bg-[size:40px_40px]" />
            {handSample?.landmarks.map((landmark, index) => {
              const point = mirroredDisplayPoint(landmark);
              return (
                <span
                  key={index}
                  data-testid="hand-landmark"
                  className={`pointer-events-none absolute h-2 w-2 -translate-x-1/2 -translate-y-1/2 rounded-full border border-slate-950 ${index === 0 || index === 4 || index === 8 ? "bg-amber-300" : "bg-cyan-300/80"}`}
                  style={{ left: `${point.left * 100}%`, top: `${point.top * 100}%` }}
                />
              );
            })}
            {cameraState !== "ready" && (
              <div className="absolute inset-0 flex items-center justify-center p-6">
                <EmptyState
                  title="Camera is off"
                  detail="Enable a local camera for a visual intent preview. No frames are uploaded by this frontend."
                  icon={Camera}
                />
              </div>
            )}
            <div
              className="pointer-events-none absolute h-16 w-16 -translate-x-1/2 -translate-y-1/2 rounded-full border border-cyan-300/70 bg-cyan-300/10 shadow-[0_0_40px_rgba(34,211,238,.28)] transition-[left,top] duration-75"
              style={{
                left: `${((intent.x + 1) / 2) * 100}%`,
                top: `${((-intent.y + 1) / 2) * 100}%`,
              }}
            >
              <Crosshair aria-hidden="true" className="absolute left-1/2 top-1/2 h-6 w-6 -translate-x-1/2 -translate-y-1/2 text-cyan-200" />
            </div>
            <div className="pointer-events-none absolute bottom-4 left-4 rounded-lg border border-white/5 bg-slate-950/75 px-3 py-2 font-mono text-[10px] text-slate-400 backdrop-blur-sm">
              x {intent.x.toFixed(2)} · y {intent.y.toFixed(2)} · roll {intent.roll.toFixed(2)} · grip {intent.grip.toFixed(2)}
            </div>
            <div className="pointer-events-none absolute right-4 top-4 rounded-full border border-amber-300/20 bg-amber-400/10 px-2.5 py-1 text-[9px] font-bold uppercase tracking-wider text-amber-200">
              {handMotion.motionActive
                ? "Hand deadman held · bounded motion"
                : handTrackerState === "running"
                  ? handSample
                    ? "Live landmarks → preview"
                    : "Tracking lost · target cleared"
                  : "Preview only · no motion"}
            </div>
          </div>
        </Card>

        <div className="space-y-6">
          <Card title="Retarget preview" subtitle="Intent is validated by the backend safety adapter">
            <div className="space-y-5">
              {!handMotion.leaseId ? (
                <Button className="w-full" onClick={() => void handMotion.acquire()} disabled={handMotion.busy}>
                  <LockKeyhole aria-hidden="true" className="h-4 w-4" /> Acquire hand-teleop lease
                </Button>
              ) : (
                <>
                  <button
                    type="button"
                    disabled={handTrackerState !== "running" || !handSample || !handSampleFresh || handSample.confidence < MIN_TRACKING_CONFIDENCE}
                    onPointerDown={(event) => {
                      if (!event.isPrimary || event.button !== 0) return;
                      event.currentTarget.setPointerCapture?.(event.pointerId);
                      void beginHandMotion();
                    }}
                    onPointerUp={() => void endHandMotion()}
                    onPointerCancel={() => void endHandMotion()}
                    onLostPointerCapture={() => void endHandMotion()}
                    className={`flex h-20 w-full touch-none select-none items-center justify-center rounded-xl border-2 text-xs font-black uppercase tracking-wider disabled:cursor-not-allowed disabled:opacity-40 ${handMotion.motionActive ? "border-amber-200 bg-amber-400 text-slate-950" : "border-cyan-300/30 bg-cyan-400/10 text-cyan-100"}`}
                  >
                    <Hand aria-hidden="true" className="mr-2 h-5 w-5" /> Hold for tracked motion
                  </button>
                  {handTrackerState === "running" &&
                    handSample &&
                    handSample.confidence < MIN_TRACKING_CONFIDENCE && (
                      <p role="status" className="text-[11px] leading-relaxed text-amber-200/90">
                        Tracking quality {(handSample.confidence * 100).toFixed(0)}% is below the{" "}
                        {(MIN_TRACKING_CONFIDENCE * 100).toFixed(0)}% needed to move.{" "}
                        {cameraQualityHint(handSample.quality)}
                      </p>
                    )}
                  <Button variant="secondary" className="w-full" onClick={() => void handMotion.release()}>
                    Release hand-teleop lease
                  </Button>
                </>
              )}
              {handMotion.error && <Notice tone="danger" title="Motion lease rejected">{handMotion.error}</Notice>}
              <label className="block">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400">Grip / pinch</span>
                  <span className="font-mono text-cyan-300">{intent.grip.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={intent.grip}
                  onChange={(event) =>
                    setIntent((current) => ({ ...current, grip: Number(event.target.value) }))
                  }
                  className="mt-2 w-full accent-cyan-400"
                />
              </label>
              <label className="block">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400">Tracking quality</span>
                  <span className="font-mono text-cyan-300">{confidence.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={confidence}
                  onChange={(event) => setConfidence(Number(event.target.value))}
                  className="mt-2 w-full accent-cyan-400"
                />
              </label>
              <Button className="w-full" onClick={() => void requestPreview()} disabled={previewBusy}>
                <ShieldCheck aria-hidden="true" className="h-4 w-4" />
                {previewBusy ? "Validating intent…" : "Validate intent"}
              </Button>
              {previewError && <Notice tone="danger" title="Preview rejected">{previewError}</Notice>}
              {preview && (
                <div className={`rounded-xl border p-4 ${preview.accepted ? "border-emerald-400/20 bg-emerald-400/[0.05]" : "border-amber-400/20 bg-amber-400/[0.05]"}`}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-sm font-semibold text-slate-100">
                      {preview.accepted
                        ? Object.keys(preview.commanded_targets).length
                          ? "Command step accepted"
                          : "Intent preview accepted"
                        : "Intent gated"}
                    </span>
                    <StatusPill label="confidence" value={preview.confidence.toFixed(2)} tone={preview.accepted ? "ok" : "warn"} />
                  </div>
                  <p className="mt-2 text-xs leading-relaxed text-slate-400">{preview.reason}</p>
                  <div className="mt-3 grid grid-cols-2 gap-2">
                    {Object.entries(preview.targets).slice(0, 6).map(([key, value]) => (
                      <div key={key} className="rounded-lg bg-slate-950/50 px-2.5 py-2">
                        <div className="truncate font-mono text-[9px] text-muted">{key}</div>
                        <div className="mt-1 font-mono text-xs text-cyan-200">{value.toFixed(2)}</div>
                      </div>
                    ))}
                  </div>
                  {Object.keys(preview.commanded_targets).length > 0 && (
                    <p className="mt-2 text-[10px] text-muted">
                      Values shown are the actual driver-accepted command after suite and
                      measured-relative limiting; desired preview remains available in the audit payload.
                    </p>
                  )}
                </div>
              )}
            </div>
          </Card>

          <Card title="Quest connection path">
            <ol className="space-y-3" aria-label="Quest connection steps">
              {[
                [LockKeyhole, "Open this workstation over trusted HTTPS", secure],
                [Wifi, "Place Quest and robot host on a reachable network", null],
                [Headset, "Enter immersive-vr in Quest Browser", xrSupported],
                [Hand, "Verify backend retargeting capability", trackingCapability?.available ?? false],
              ].map(([IconValue, label, ready], index) => {
                const Icon = IconValue as typeof CheckCircle2;
                // The tick replaces the step number, so say the state in words too.
                const state = ready === null ? "not checked" : ready ? "done" : "not done";
                return (
                  <li key={label as string} className="flex items-center gap-3 rounded-lg bg-slate-950/40 px-3 py-2.5">
                    <div
                      aria-hidden="true"
                      className={`flex h-6 w-6 items-center justify-center rounded-full text-[10px] font-bold ${ready ? "bg-emerald-400/15 text-emerald-300" : "bg-slate-800 text-muted"}`}
                    >
                      {ready ? <CheckCircle2 className="h-3.5 w-3.5" /> : index + 1}
                    </div>
                    <Icon aria-hidden="true" className="h-4 w-4 text-muted" />
                    <span className="text-xs text-slate-300">
                      {label as string}
                      <span className="sr-only">: {state}</span>
                    </span>
                  </li>
                );
              })}
            </ol>
          </Card>
        </div>
      </div>

      {handTrackerError && (
        <Notice tone="danger" title="Hand tracker could not start">
          {handTrackerError}. The model/WASM URLs are configurable with
          VITE_HAND_LANDMARKER_MODEL_URL and VITE_MEDIAPIPE_WASM_URL.
        </Notice>
      )}

      <Card title="Safety contract" subtitle="What this page can and cannot do">
        <div className="grid gap-4 md:grid-cols-3">
          <div>
            <SectionLabel>Camera</SectionLabel>
            <p className="mt-2 text-xs leading-relaxed text-slate-400">
              Camera frames stay in the browser. The optional MediaPipe adapter
              downloads its WASM/model, infers 21 landmarks locally, and sends only
              normalized intent to the preview API.
            </p>
          </div>
          <div>
            <SectionLabel>Intent</SectionLabel>
            <p className="mt-2 text-xs leading-relaxed text-slate-400">
              Normalized intent stays preview-only until the operator acquires
              the hand lease and continuously holds the dedicated deadman button.
            </p>
          </div>
          <div>
            <SectionLabel>Teleoperation</SectionLabel>
            <p className="mt-2 text-xs leading-relaxed text-slate-400">
              Execution uses lease ownership, a {MIN_TRACKING_CONFIDENCE} tracking-quality
              gate, sequenced 300 ms intents, wrist roll measured from the pose at
              engage, measured-relative limits, and stop on release, tracking loss,
              blur, hide, lease loss, or disconnect.
            </p>
          </div>
        </div>
      </Card>
      {cameraError && <Notice tone="danger" title="Camera access failed">{cameraError}</Notice>}
    </div>
  );
}
