import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Cable,
  CheckCircle2,
  CircleHelp,
  Clock3,
  Download,
  RefreshCw,
  ServerCog,
  ShieldAlert,
  XCircle,
} from "lucide-react";
import {
  getDiagnostics,
  getEvents,
  resetFault,
  type AuditEvent,
  type DiagnosticCheck,
  type DiagnosticPort,
  type DiagnosticsResponse,
  type DiagnosticStatus,
} from "../api";
import Button from "../components/Button";
import Card from "../components/Card";
import { StatusPill } from "../components/ui/StatusPill";
import {
  EmptyState,
  MetricTile,
  Notice,
  PageHeader,
  SectionLabel,
} from "../components/Workstation";
import { useSafety } from "../components/safety/SafetyProvider";
import { useConfirm } from "../components/ui/Dialog";
import { usePolling } from "../hooks/usePolling";
import { useTelemetry, type SampleState, type WsState } from "../hooks/useTelemetry";
import { errorMessage, needsPowerCutAcknowledgement } from "../lib/errors";

function statusKind(status: DiagnosticStatus): "ok" | "warn" | "danger" | "neutral" {
  const normalized = status.toLowerCase();
  if (["pass", "ok", "healthy", "ready"].includes(normalized)) return "ok";
  if (["warn", "warning", "degraded"].includes(normalized)) return "warn";
  if (["fail", "failed", "error", "critical"].includes(normalized)) return "danger";
  return "neutral";
}

function CheckIcon({ status }: { status: DiagnosticStatus }) {
  const kind = statusKind(status);
  if (kind === "ok") return <CheckCircle2 aria-hidden="true" className="h-5 w-5 text-emerald-400" />;
  if (kind === "warn") return <AlertTriangle aria-hidden="true" className="h-5 w-5 text-amber-400" />;
  if (kind === "danger") return <XCircle aria-hidden="true" className="h-5 w-5 text-rose-400" />;
  return <CircleHelp aria-hidden="true" className="h-5 w-5 text-muted" />;
}

function eventMillis(event: AuditEvent): number {
  const raw = event.timestamp ?? event.created_at;
  const ms = raw ? new Date(raw).getTime() : Number.NaN;
  return Number.isNaN(ms) ? Number.NEGATIVE_INFINITY : ms;
}

/**
 * F24: merge the diagnostics snapshot's events with the event stream,
 * drop duplicates and show newest first. Events without a parseable time go
 * last, in arrival order.
 */
export function mergeEvents(...sources: AuditEvent[][]): AuditEvent[] {
  const seen = new Set<string>();
  const merged: AuditEvent[] = [];
  for (const event of sources.flat()) {
    const key =
      event.id !== undefined && event.id !== null
        ? `id:${String(event.id)}`
        : `${event.timestamp ?? event.created_at ?? ""}|${event.source ?? event.type ?? ""}|${eventText(event)}`;
    if (seen.has(key)) continue;
    seen.add(key);
    merged.push(event);
  }
  return merged
    .map((event, index) => ({ event, index, ms: eventMillis(event) }))
    .sort((a, b) => (b.ms === a.ms ? a.index - b.index : b.ms - a.ms))
    .map(({ event }) => event);
}

function eventTime(event: AuditEvent): string {
  const raw = event.timestamp ?? event.created_at;
  if (!raw) return "time unavailable";
  const date = new Date(raw);
  return Number.isNaN(date.getTime()) ? raw : date.toLocaleString();
}

function eventText(event: AuditEvent): string {
  return (
    event.message ??
    (typeof event.detail === "string" ? event.detail : undefined) ??
    event.type ??
    "System event"
  );
}

function eventList(value: Awaited<ReturnType<typeof getEvents>> | null): AuditEvent[] {
  if (!value) return [];
  return Array.isArray(value) ? value : value.events;
}

function portLabel(port: DiagnosticPort | string): { title: string; detail: string } {
  if (typeof port === "string") return { title: port, detail: "Detected device path" };
  return {
    title: port.label ?? port.path ?? port.id ?? "Unknown port",
    detail: port.detail ?? port.type ?? (port.connected === false ? "Disconnected" : "Detected"),
  };
}

function safeExportTimestamp(value: unknown): string | null {
  if (typeof value !== "string") return null;
  const milliseconds = new Date(value).getTime();
  return Number.isFinite(milliseconds) ? new Date(milliseconds).toISOString() : null;
}

interface DiagnosticExportTelemetry {
  state: WsState;
  transport_stale: boolean;
  sample_state: SampleState;
  last_message_at: number | null;
}

/**
 * Build a support snapshot from an allowlist. Backend detail/remediation, port
 * identifiers, event messages/details, arbitrary extension fields and the
 * browser URL are deliberately excluded: all can contain camera/package URLs,
 * credentials, local paths or operator-entered text.
 */
export function buildDiagnosticExport(
  diagnostics: DiagnosticsResponse,
  events: AuditEvent[],
  telemetry: DiagnosticExportTelemetry,
  secureContext: boolean,
) {
  const checkStatusCounts = { ok: 0, warn: 0, danger: 0, neutral: 0 };
  for (const check of diagnostics.checks) checkStatusCounts[statusKind(check.status)] += 1;
  return {
    schema_version: 1,
    exported_at: new Date().toISOString(),
    page: "diagnostics",
    secure_context: secureContext,
    redaction_policy: "support-allowlist-v1",
    telemetry,
    diagnostics: {
      generated_at: safeExportTimestamp(diagnostics.generated_at),
      overall_kind: statusKind(diagnostics.overall),
      check_status_counts: checkStatusCounts,
      port_count: diagnostics.ports.length,
      event_count: events.length,
    },
    events: events.map((event) => ({
      timestamp: safeExportTimestamp(event.timestamp ?? event.created_at),
    })),
  };
}

export default function Diagnostics() {
  const {
    data: diagnostics,
    error,
    loading,
    refresh: refreshDiagnostics,
  } = usePolling(getDiagnostics, 5000);
  const { data: eventResponse, refresh: refreshEvents } = usePolling(getEvents, 5000);
  // F28: control state comes from the shell's single poll.
  const { control, view, refresh: refreshSafety } = useSafety();
  const { state, transportStale, sampleState, lastMessageAt } = useTelemetry();
  const [confirm, confirmDialog] = useConfirm();
  const [resettingFault, setResettingFault] = useState(false);
  const [resetError, setResetError] = useState<string | null>(null);
  // The server retried torque-off and still could not confirm it (C2).
  const [ackOffered, setAckOffered] = useState(false);
  const events = useMemo(
    () => mergeEvents(diagnostics?.events ?? [], eventList(eventResponse)).slice(0, 50),
    [diagnostics, eventResponse],
  );

  const checks = diagnostics?.checks ?? [];
  const passed = checks.filter((check) => statusKind(check.status) === "ok").length;
  const warnings = checks.filter((check) => statusKind(check.status) === "warn").length;
  const failures = checks.filter((check) => statusKind(check.status) === "danger").length;
  const refreshAll = () => {
    refreshDiagnostics();
    refreshEvents();
    refreshSafety();
  };

  const faultLatched = control?.safety_state === "fault";
  // C2: torque-off could not be read back. The server retries it on every fault
  // reset and asks for the power-cut acknowledgement only if the retry fails.
  const torqueUnconfirmed = view.torqueUnconfirmed;

  const runFaultReset = async (powerCutAcknowledged: boolean) => {
    setResettingFault(true);
    setResetError(null);
    try {
      await resetFault(powerCutAcknowledged ? { powerCutAcknowledged: true } : {});
    } catch (caught) {
      setResetError(errorMessage(caught));
      setAckOffered(!powerCutAcknowledged && needsPowerCutAcknowledgement(caught));
    } finally {
      setResettingFault(false);
      refreshAll();
    }
  };

  const clearFault = async () => {
    const confirmed = torqueUnconfirmed
      ? await confirm({
          title: "Retry torque-off and clear the fault?",
          body:
            "The server first retries the torque-off and reads it back on every bus. If the motors are " +
            "confirmed off, the fault clears. If not, the server refuses and you will be asked to cut physical power.",
          confirmLabel: "Retry and clear",
          tone: "danger",
        })
      : await confirm({
          title: "Clear the remediated fault?",
          body: "Clear the runtime fault only after completing its remediation and confirming the robot is idle.",
          confirmLabel: "Clear fault",
          tone: "warn",
        });
    if (!confirmed) return;
    await runFaultReset(false);
  };

  const acknowledgePowerCut = async () => {
    const confirmed = await confirm({
      title: "Clear the fault with torque-off unconfirmed?",
      body:
        "The server retried the torque-off and still could not confirm the motors are off, so they may still " +
        "be holding. Cut physical power to the arm first (switch off or unplug the motor supply); expect the arm " +
        "to drop when power goes. Clear the fault only once the power is off.",
      acknowledgement: "I have cut physical power to the arm",
      confirmLabel: "Clear fault",
      tone: "danger",
    });
    if (!confirmed) return;
    await runFaultReset(true);
  };

  const exportSnapshot = () => {
    if (!diagnostics) return;
    const payload = JSON.stringify(
      buildDiagnosticExport(
        diagnostics,
        events,
        {
          state,
          transport_stale: transportStale,
          sample_state: sampleState,
          last_message_at: lastMessageAt,
        },
        window.isSecureContext,
      ),
      null,
      2,
    );
    const blob = new Blob([payload], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `so101-diagnostics-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {confirmDialog}
      <PageHeader
        eyebrow="System health"
        title="Diagnostics & audit"
        description="Run readiness checks, inspect connected devices, and correlate safety and control-plane events."
        icon={ServerCog}
        actions={
          <>
            <Button variant="secondary" onClick={refreshAll}>
              <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" /> Refresh
            </Button>
            <Button variant="secondary" onClick={exportSnapshot} disabled={!diagnostics}>
              <Download aria-hidden="true" className="h-3.5 w-3.5" /> Export snapshot
            </Button>
            {(faultLatched || torqueUnconfirmed) && (
              <Button variant="danger" onClick={() => void clearFault()} disabled={resettingFault}>
                <ShieldAlert aria-hidden="true" className="h-3.5 w-3.5" />
                {resettingFault ? "Clearing…" : "Clear remediated fault"}
              </Button>
            )}
          </>
        }
      />

      {error && (
        <Notice tone="warn" title="Backend diagnostics are unavailable">
          Client connectivity checks remain visible. Guided hardware checks require
          the diagnostics endpoint: {error.message}
        </Notice>
      )}
      {torqueUnconfirmed && (
        <Notice tone="danger" title="Torque-off is unconfirmed">
          The motors may still be energised: cut physical power to the arm if anything moves. Clear
          remediated fault retries the torque-off first; if the server still cannot confirm it, it asks you
          to acknowledge that physical power is cut.
        </Notice>
      )}
      {resetError && (
        <Notice tone="danger" title="Fault reset rejected">
          <p>{resetError}</p>
          {ackOffered && (
            <div className="mt-2">
              <Button variant="danger" size="sm" onClick={() => void acknowledgePowerCut()} disabled={resettingFault}>
                I have cut physical power…
              </Button>
            </div>
          )}
        </Notice>
      )}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
        <MetricTile
          label="Overall"
          value={diagnostics?.overall ?? (loading ? "Checking" : "Unknown")}
          detail={diagnostics?.generated_at ? `Generated ${new Date(diagnostics.generated_at).toLocaleTimeString()}` : "No diagnostic snapshot"}
          tone={diagnostics ? (statusKind(diagnostics.overall) === "ok" ? "green" : statusKind(diagnostics.overall) === "danger" ? "red" : "amber") : "neutral"}
          icon={Activity}
        />
        <MetricTile label="Passed" value={passed} detail="Ready checks" tone="green" icon={CheckCircle2} />
        <MetricTile label="Warnings" value={warnings} detail="Attention recommended" tone={warnings ? "amber" : "neutral"} icon={AlertTriangle} />
        <MetricTile label="Failures" value={failures} detail="Blocking conditions" tone={failures ? "red" : "neutral"} icon={ShieldAlert} />
        <MetricTile
          label="Telemetry"
          value={transportStale ? "Stale" : sampleState === "live" ? "Live" : sampleState === "idle" ? "Idle" : "No samples"}
          detail={
            transportStale
              ? state === "open"
                ? "Socket open, no recent frame"
                : `Socket ${state}`
              : sampleState === "idle"
                ? "Link live; no loop is sampling joints"
                : "Realtime socket open"
          }
          tone={transportStale || sampleState === "stale" ? "amber" : sampleState === "idle" ? "neutral" : "green"}
          icon={Clock3}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(20rem,0.65fr)]">
        <Card title="Readiness checks" subtitle={`${checks.length} backend checks`}>
          {checks.length === 0 ? (
            <EmptyState
              title={loading ? "Running checks…" : "No checks reported"}
              detail="Start the backend diagnostics adapter to inspect motors, cameras, storage, and control safety."
              icon={ServerCog}
            />
          ) : (
            <div className="divide-y divide-white/5">
              {checks.map((check: DiagnosticCheck) => (
                <div key={check.id} className="grid gap-3 py-4 first:pt-0 last:pb-0 sm:grid-cols-[auto_1fr_auto]">
                  <CheckIcon status={check.status} />
                  <div className="min-w-0">
                    <div className="text-sm font-semibold text-slate-200">{check.label}</div>
                    <p className="mt-1 text-xs leading-relaxed text-muted">{check.detail}</p>
                    {check.remediation && statusKind(check.status) !== "ok" && (
                      <div className="mt-2 rounded-lg border border-amber-400/10 bg-amber-400/[0.04] px-3 py-2 text-xs text-amber-200/80">
                        {check.remediation}
                      </div>
                    )}
                  </div>
                  <StatusPill label="" value={check.status} tone={statusKind(check.status)} />
                </div>
              ))}
            </div>
          )}
        </Card>

        <div className="space-y-6">
          <Card title="Client readiness">
            <div className="space-y-3">
              {[
                {
                  label: "Secure browser context",
                  detail: window.isSecureContext ? window.location.protocol : "HTTPS required for XR/camera",
                  ready: window.isSecureContext,
                },
                {
                  label: "Realtime telemetry",
                  detail: transportStale ? "No recent telemetry frame" : "Receiving current robot state",
                  ready: !transportStale,
                },
                {
                  label: "Camera media API",
                  detail: navigator.mediaDevices ? "MediaDevices exposed" : "Browser API missing",
                  ready: !!navigator.mediaDevices,
                },
              ].map((check) => (
                <div key={check.label} className="flex items-start gap-3 rounded-xl bg-slate-950/40 p-3">
                  {check.ready ? (
                    <CheckCircle2 aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
                  ) : (
                    <AlertTriangle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
                  )}
                  <div>
                    <div className="text-xs font-semibold text-slate-300">{check.label}</div>
                    <div className="mt-1 text-[11px] text-muted">{check.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          </Card>

          <Card title="Detected ports" subtitle={`${diagnostics?.ports.length ?? 0} device paths`}>
            {!diagnostics?.ports.length ? (
              <EmptyState title="No ports reported" detail="Connect a motor bus or camera and rerun diagnostics." icon={Cable} />
            ) : (
              <div className="space-y-2">
                {diagnostics.ports.map((port, index) => {
                  const label = portLabel(port);
                  return (
                    <div key={`${label.title}-${index}`} className="flex items-center gap-3 rounded-lg border border-white/5 bg-slate-950/40 p-3">
                      <Cable aria-hidden="true" className="h-4 w-4 shrink-0 text-cyan-400" />
                      <div className="min-w-0">
                        <div className="truncate font-mono text-xs text-slate-200">{label.title}</div>
                        <div className="mt-1 truncate text-[10px] text-muted">{label.detail}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </Card>
        </div>
      </div>

      <Card title="Audit event timeline" subtitle="Newest safety, control, and service events">
        {events.length === 0 ? (
          <EmptyState
            title="No audit events reported"
            detail="Events appear here when the backend event stream is available. The UI does not fabricate operational history."
            icon={Clock3}
          />
        ) : (
          <div className="relative space-y-0 before:absolute before:bottom-3 before:left-[7px] before:top-3 before:w-px before:bg-slate-800">
            {events.map((event, index) => {
              const level = (event.level ?? "info").toLowerCase();
              const critical = ["error", "critical", "danger"].includes(level);
              const warning = ["warn", "warning"].includes(level);
              return (
                <div key={event.id ?? `${eventTime(event)}-${index}`} className="relative grid gap-3 py-3 pl-7 sm:grid-cols-[10rem_8rem_1fr]">
                  <span className={`absolute left-0 top-4 h-[15px] w-[15px] rounded-full border-[3px] border-slate-950 ${critical ? "bg-rose-400" : warning ? "bg-amber-400" : "bg-cyan-400"}`} />
                  <div className="font-mono text-[10px] text-muted">{eventTime(event)}</div>
                  <div className="text-[10px] font-bold uppercase tracking-wider text-muted">
                    {event.source ?? event.type ?? "system"}
                  </div>
                  <div className="text-xs leading-relaxed text-slate-300">{eventText(event)}</div>
                </div>
              );
            })}
          </div>
        )}
      </Card>

      <div className="flex items-center gap-2 text-[10px] text-muted">
        <SectionLabel>Privacy</SectionLabel>
        Exported snapshots contain only the data visible on this page; review before sharing.
      </div>
    </div>
  );
}
