import { Link } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  Bot,
  Boxes,
  Camera,
  CheckCircle2,
  CircleGauge,
  Clock3,
  Database,
  Headset,
  Radio,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
  Wrench,
} from "lucide-react";
import { getCapabilities, getEvents, type AuditEvent } from "../api";
import CameraView from "../components/CameraView";
import Card from "../components/Card";
import JointBars from "../components/JointBars";
import SetupChecklist from "../components/onboarding/SetupChecklist";
import RobotTwin from "../components/RobotTwin";
import { hasJointData } from "../components/twin/types";
import { StatusPill } from "../components/ui/StatusPill";
import {
  EmptyState,
  MetricTile,
  Notice,
  PageHeader,
  SectionLabel,
} from "../components/Workstation";
import { useSafety } from "../components/safety/SafetyProvider";
import { usePolling } from "../hooks/usePolling";
import { useTelemetry, type SampleState } from "../hooks/useTelemetry";
import { backendInfo } from "../lib/backend";
import { currentMode } from "../lib/safetyView";

const WORKFLOWS = [
  {
    to: "/teleop",
    label: "Operate & capture",
    detail: "Teleoperate, monitor cameras, and record demonstrations.",
    icon: Radio,
    color: "text-cyan-300 bg-cyan-400/10",
  },
  {
    to: "/simulation",
    label: "Open digital twin",
    detail: "Inspect live kinematics and select scenario metadata.",
    icon: Boxes,
    color: "text-blue-300 bg-blue-400/10",
  },
  {
    to: "/connect",
    label: "Probe Quest & hands",
    detail: "Validate WebXR, camera, and hand intent capabilities.",
    icon: Headset,
    color: "text-violet-300 bg-violet-400/10",
  },
  {
    to: "/agent",
    label: "Agent lab",
    detail: "Create and review a deterministic bounded command proposal.",
    icon: Sparkles,
    color: "text-amber-300 bg-amber-400/10",
  },
];

function eventsFrom(value: Awaited<ReturnType<typeof getEvents>> | null): AuditEvent[] {
  if (!value) return [];
  return Array.isArray(value) ? value : value.events;
}

function eventLabel(event: AuditEvent): string {
  return (
    event.message ??
    (typeof event.detail === "string" ? event.detail : undefined) ??
    event.type ??
    "System event"
  );
}

function eventTime(event: AuditEvent): string {
  const raw = event.timestamp ?? event.created_at;
  if (!raw) return "—";
  const date = new Date(raw);
  return Number.isNaN(date.getTime()) ? raw : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/** What the twin's pose means right now (F14): never "live" unless it is. */
export function twinSubtitle(
  hasObservation: boolean,
  transportStale: boolean,
  sampleState: SampleState,
): string {
  if (!hasObservation) return "No joint sample yet, so the arm is a placeholder";
  if (transportStale) return "Link stale: the last known pose, not the live arm";
  if (sampleState === "stale") return "Joint samples overdue: the last known pose";
  if (sampleState === "idle") return "Idle: the last pose read; nothing is sampling the joints";
  return "Follows the live joint state";
}

/** Why the workstation is not ready, most urgent first; null when ready. */
export function attentionReason(inputs: {
  tripped: boolean;
  torqueUnconfirmed: boolean;
  stopState: string;
  transportStale: boolean;
  statusKnown: boolean;
  followerConnected: boolean;
  mode?: string;
  torqueState?: string;
}): string | null {
  if (inputs.tripped) return "The emergency-stop latch is engaged. Inspect the workspace before resetting it.";
  if (inputs.torqueUnconfirmed) {
    return "Torque-off could not be confirmed. Cut physical power to the arm, then clear the fault on Diagnostics.";
  }
  if (inputs.stopState === "halted") return "The arm is halted and holding position with motors on.";
  if (inputs.stopState === "collapsing") return "The arm is lowering to its rest pose.";
  if (inputs.transportStale) {
    return "Telemetry is not current. Motion workflows remain unsafe until the stream recovers.";
  }
  if (!inputs.statusKnown) return "Robot status has not been received yet.";
  if (!inputs.followerConnected && !(inputs.mode === "idle" && inputs.torqueState === "off")) {
    return "The follower arm is not connected. Use diagnostics to verify the motor bus.";
  }
  return null;
}

export default function Dashboard() {
  // F28: status and control come from the shell's single poll.
  const { view, status, control } = useSafety();
  const capabilityPoll = usePolling(getCapabilities, 10000);
  const { data: capabilityData, error: capabilityError } = capabilityPoll;
  const { data: eventResponse } = usePolling(getEvents, 5000);
  const { telemetry, state, transportStale, sampleState, stale } = useTelemetry();

  const linkLive = !transportStale && telemetry !== null;
  const fps = (linkLive ? telemetry.fps : status?.fps) ?? 0;
  // F4: the latch comes from the shared view, so a stale socket cannot outvote fresh status.
  const estopped = view.tripped;
  const mode = currentMode(telemetry, linkLive, view, status) ?? "offline";
  const cameras = status?.cameras ?? [];
  const backend = backendInfo(status);
  const capabilities = capabilityData?.capabilities ?? [];
  const readyCapabilities = capabilities.filter((capability) => capability.available).length;
  const motionOwner = control?.lease
    ? control.lease.source
    : status?.operation_owner
      ? status.operation_owner
      : "None";
  const events = eventsFrom(eventResponse).slice(0, 6);
  const attention = attentionReason({
    tripped: estopped,
    torqueUnconfirmed: view.torqueUnconfirmed,
    stopState: view.stopState,
    transportStale,
    statusKnown: status !== null,
    followerConnected: !!status?.follower_connected,
    mode: status?.mode,
    torqueState: status?.torque_state,
  });
  const ready = attention === null;
  const busIdle = status?.mode === "idle" && status.torque_state === "off" && !status.follower_connected;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Mission overview"
        title="Good to see you, operator."
        description="One surface for safe operation, demonstration capture, digital-twin inspection, training, evaluation, and bounded agent control."
        icon={CircleGauge}
        actions={
          <Link
            to="/diagnostics"
            className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-800 px-3.5 py-2 text-sm font-medium text-slate-100 transition hover:bg-slate-700"
          >
            <Wrench aria-hidden="true" className="h-4 w-4" /> Run diagnostics
          </Link>
        }
      />

      {capabilityError && (
        <Notice tone="warn" title="Some workstation services are unreachable">
          Live modules fall back to explicit unavailable states. {capabilityError.message}
        </Notice>
      )}

      <SetupChecklist capabilities={capabilityPoll} />

      <div className={`rounded-2xl border p-4 sm:p-5 ${ready ? "border-emerald-300/15 bg-linear-to-r/srgb from-emerald-400/[0.07] to-cyan-400/[0.03]" : "border-amber-300/15 bg-linear-to-r/srgb from-amber-400/[0.07] to-rose-400/[0.025]"}`}>
        <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">
          <div className="flex items-start gap-3">
            <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${ready ? "bg-emerald-400/15 text-emerald-300" : "bg-amber-400/15 text-amber-300"}`}>
              {ready ? <ShieldCheck aria-hidden="true" className="h-5 w-5" /> : <TriangleAlert aria-hidden="true" className="h-5 w-5" />}
            </div>
            <div>
              <div className="text-sm font-bold text-slate-100">
                {ready ? (busIdle ? "Idle · choose an operation" : "Workstation ready for supervised operation") : "Operator attention required"}
              </div>
              <p className="mt-1 text-xs leading-relaxed text-slate-400">
                {attention ??
                  (sampleState === "live"
                    ? "Follower state is live, the safety latch is clear, and commands remain backend bounded."
                    : "The link is live and the safety latch is clear. No loop is sampling joints right now; that is normal while idle.")}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <StatusPill label="mode" value={mode} tone="info" />
            <StatusPill
              label="safety"
              value={view.safetyState.replace(/_/g, " ")}
              tone={estopped || view.torqueUnconfirmed ? "danger" : view.safetyState === "unknown" ? "neutral" : view.safetyState === "disarmed" || view.safetyState === "armed" ? "ok" : "warn"}
            />
            <StatusPill
              label="control"
              value={control === null ? "unknown" : view.armed ? "armed" : "disarmed"}
              tone={view.armed ? "warn" : "neutral"}
            />
          </div>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricTile
          label="Follower arm"
          value={status?.follower_connected ? (backend?.kind === "sim" ? "Simulated" : "Online") : busIdle ? "Idle" : "Offline"}
          detail={
            backend?.kind === "sim"
              ? "MuJoCo simulation: no physical arm"
              : backend?.kind === "mock"
                ? "Mock hardware adapter"
                : busIdle ? "Bus opens when an operation starts" : "Physical motor bus"
          }
          tone={status?.follower_connected ? "green" : busIdle ? "neutral" : "red"}
          icon={Bot}
        />
        <MetricTile
          label="Telemetry"
          value={
            transportStale ? "Stale" : sampleState === "idle" ? "Idle" : sampleState === "stale" ? "No samples" : `${fps.toFixed(0)} Hz`
          }
          detail={
            transportStale
              ? state === "open"
                ? "Socket open, no recent frame"
                : `Socket ${state}`
              : sampleState === "idle"
                ? "Link live; no loop is sampling joints"
                : sampleState === "stale"
                  ? "A loop is running but joint samples are old"
                  : "Singleton realtime stream"
          }
          tone={transportStale || sampleState === "stale" ? "amber" : sampleState === "idle" ? "neutral" : "cyan"}
          icon={Activity}
        />
        <MetricTile
          label="Cameras"
          value={`${status?.live_cameras?.length ?? 0}/${cameras.length}`}
          detail={cameras.length ? "live / configured" : "No streams configured"}
          tone={status?.live_cameras?.length ? "green" : cameras.length ? "amber" : "neutral"}
          icon={Camera}
        />
        <MetricTile
          label="Capabilities"
          value={capabilities.length ? `${readyCapabilities}/${capabilities.length}` : "—"}
          detail="Backend adapters available"
          tone={capabilities.length && readyCapabilities === capabilities.length ? "green" : "neutral"}
          icon={CheckCircle2}
        />
        <MetricTile
          label="Motion owner"
          value={motionOwner}
          detail={control?.lease ? "External control lease" : status?.operation_owner ? `${mode} loop owner` : "Robot unclaimed"}
          tone={control?.lease || status?.operation_owner ? "amber" : "neutral"}
          icon={ShieldCheck}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.45fr)_minmax(22rem,0.75fr)]">
        <Card
          title="Digital twin"
          subtitle={twinSubtitle(hasJointData(telemetry?.observation), transportStale, sampleState)}
          actions={
            <Link to="/simulation" className="inline-flex items-center gap-1 text-xs font-semibold text-cyan-300 hover:text-cyan-200">
              Full workspace <ArrowRight aria-hidden="true" className="h-3.5 w-3.5" />
            </Link>
          }
          bodyClassName="p-0"
          className="overflow-hidden"
        >
          <RobotTwin
            observation={telemetry?.observation}
            action={telemetry?.action}
            mode={mode}
            stale={stale}
            idle={sampleState === "idle"}
            className="h-[30rem] w-full"
          />
        </Card>

        <div className="space-y-6">
          <Card title="Joint state" subtitle="Observed / commanded">
            <JointBars observation={telemetry?.observation} action={telemetry?.action} />
          </Card>

          <Card title="Recent activity" subtitle="Backend audit events">
            {events.length === 0 ? (
              <EmptyState title="No events yet" detail="Control and safety activity will appear when the event endpoint is available." icon={Clock3} />
            ) : (
              <div className="space-y-1">
                {events.map((event, index) => (
                  <div key={event.id ?? `${eventTime(event)}-${index}`} className="flex items-start gap-3 border-b border-white/5 py-2.5 last:border-0">
                    <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-400" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-xs text-slate-300">{eventLabel(event)}</div>
                      <div className="mt-1 text-[9px] uppercase tracking-wider text-muted">{event.source ?? event.type ?? "system"}</div>
                    </div>
                    <span className="font-mono text-[9px] text-muted">{eventTime(event)}</span>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>

      <div>
        <SectionLabel>Start a workflow</SectionLabel>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {WORKFLOWS.map((workflow) => {
            const Icon = workflow.icon;
            return (
              <Link
                key={workflow.to}
                to={workflow.to}
                className="group rounded-2xl border border-white/5 bg-slate-900/55 p-4 shadow-lg shadow-black/10 transition hover:-translate-y-0.5 hover:border-slate-700 hover:bg-slate-900"
              >
                <div className="flex items-center justify-between">
                  <div className={`flex h-9 w-9 items-center justify-center rounded-xl ${workflow.color}`}>
                    <Icon aria-hidden="true" className="h-4 w-4" />
                  </div>
                  <ArrowRight aria-hidden="true" className="h-4 w-4 text-slate-700 transition group-hover:translate-x-0.5 group-hover:text-slate-400" />
                </div>
                <div className="mt-4 text-sm font-semibold text-slate-100">{workflow.label}</div>
                <p className="mt-1.5 text-xs leading-relaxed text-muted">{workflow.detail}</p>
              </Link>
            );
          })}
        </div>
      </div>

      <Card title="Camera wall" subtitle={`${cameras.length} source${cameras.length === 1 ? "" : "s"}`}>
        {cameras.length === 0 ? (
          <EmptyState title="No cameras reported" detail="Configure a camera source, then run diagnostics to verify capture readiness." icon={Camera} />
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {cameras.map((camera) => (
              <CameraView
                key={camera}
                cam={camera}
                active={(status?.live_cameras ?? []).includes(camera)}
              />
            ))}
          </div>
        )}
      </Card>

      {capabilities.length > 0 && (
        <Card title="Capability registry" subtitle={`Platform ${capabilityData?.system_version ?? "unknown"}`}>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {capabilities.map((capability) => (
              <div key={capability.id} className="flex items-start gap-3 rounded-xl border border-white/5 bg-slate-950/35 p-3">
                <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${capability.available ? "bg-emerald-400" : "bg-slate-600"}`} />
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs font-semibold text-slate-300">{capability.label}</span>
                    <span className="font-mono text-[8px] uppercase tracking-wider text-muted">{capability.maturity}</span>
                  </div>
                  <p className="mt-1 line-clamp-2 text-[10px] leading-relaxed text-muted">{capability.detail}</p>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      <div className="flex items-center gap-2 text-[10px] text-muted">
        <Database aria-hidden="true" className="h-3.5 w-3.5" />
        Operational status is read from the backend; this page does not infer hardware readiness from local UI state.
      </div>
    </div>
  );
}
