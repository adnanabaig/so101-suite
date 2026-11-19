import { useCallback, useMemo, useState, type ReactNode } from "react";
import {
  Bot,
  BrainCircuit,
  ChevronRight,
  CircleStop,
  Fingerprint,
  Hand,
  KeyRound,
  LockKeyhole,
  Play,
  RefreshCw,
  Send,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import {
  ApiError,
  createAgentProposal,
  getAgentPlannerStatus,
  getAgentProposals,
  type AgentProposal,
  type AgentResult,
} from "../api";
import BlockerList from "../components/BlockerList";
import Button from "../components/Button";
import Card from "../components/Card";
import DeadmanButton from "../components/DeadmanButton";
import { StatusPill } from "../components/ui/StatusPill";
import { useSafety } from "../components/safety/SafetyProvider";
import { Field, INPUT_CLASS } from "../components/ui/Field";
import {
  EmptyState,
  MetricTile,
  Notice,
  PageHeader,
  SectionLabel,
} from "../components/Workstation";
import {
  PROPOSAL_MAX_AGE_MS,
  isApprovalUsable,
  proposalAgeDeadline,
  useAgentExecution,
  type AgentApprovalGrant,
  type AgentRun,
} from "../hooks/useAgentExecution";
import { useMotionLease } from "../hooks/useMotionLease";
import { useNow } from "../hooks/useNow";
import { useOperatorSupervision } from "../hooks/useOperatorSupervision";
import { usePolling } from "../hooks/usePolling";
import { useTelemetry } from "../hooks/useTelemetry";
import {
  agentExecutionBlockers,
  endStateHoldsTorque,
  endStateLabel,
  endStateUnknown,
  isCancellableStatus,
  isExecutableStatus,
  isTerminalStatus,
  jointUnit,
  proposalStatus,
  proposesMotion,
  requiresDeadman,
  stopReasonLabel,
} from "../lib/agentGates";
import { checkTargetDigest, type DigestCheck } from "../lib/canonicalJson";
import { errorMessage, joinSentences } from "../lib/errors";

const SKILLS = [
  {
    id: "inspect_scene",
    label: "Inspect scene",
    detail: "Analyze current observations and propose a bounded plan. No motion.",
    motion: false,
  },
  {
    id: "open_gripper",
    label: "Open gripper",
    detail: "Propose the backend's bounded open-gripper target.",
    motion: true,
  },
  {
    id: "close_gripper",
    label: "Close gripper",
    detail: "Propose the backend's bounded close-gripper target.",
    motion: true,
  },
  {
    id: "wrist_wave",
    label: "Wrist wave",
    detail: "Propose the backend's bounded wrist-roll target.",
    motion: true,
  },
  {
    id: "home",
    label: "Home pose",
    detail: "Propose the backend's bounded home pose: every arm joint to 0°, the gripper to 50.",
    motion: true,
  },
] as const;

/** gate's CreateAgentProposalRequest: objective is 1-500 characters. */
const OBJECTIVE_MAX_CHARS = 500;
const PROPOSAL_MAX_AGE_S = PROPOSAL_MAX_AGE_MS / 1000;

type SkillId = (typeof SKILLS)[number]["id"];
type Planner = "deterministic" | "configured";

export const OLD_SERVER_DEADMAN_HINT =
  "This server cannot run hold-to-run agent motion: it predates two-step approval (C8). Agent motion needs the v0.2 backend.";

function proposalList(
  value: Awaited<ReturnType<typeof getAgentProposals>> | null,
): AgentProposal[] {
  if (!value) return [];
  return Array.isArray(value) ? value : value.proposals;
}

function proposalTitle(proposal: AgentProposal): string {
  return proposal.title ?? proposal.objective ?? proposal.description ?? "Untitled proposal";
}

function shortDigest(digest: string): string {
  return digest.length > 16 ? `${digest.slice(0, 8)}…${digest.slice(-6)}` : digest;
}

function statusTone(status: string): "ok" | "warn" | "danger" | "info" {
  if (status === "executed" || status === "completed") return "ok";
  if (status === "failed" || status === "rejected") return "danger";
  if (status === "cancelled" || status === "expired" || status === "timeout") return "warn";
  return "info";
}

function endStateTone(endState: unknown): "ok" | "warn" | "danger" | "info" | "neutral" {
  if (endState === "halted" || endStateUnknown(endState)) return "warn";
  if (endState === "deenergized") return "neutral";
  if (endState === "estopped" || endState === "fault") return "danger";
  return "info";
}

/**
 * The digest check for every motion proposal, from the targets on screen.
 * Synchronous (see lib/sha256), so it can never lag the targets it describes.
 */
function digestChecks(proposals: AgentProposal[]): Record<string, DigestCheck> {
  const checks: Record<string, DigestCheck> = {};
  for (const proposal of proposals) {
    if (proposesMotion(proposal)) {
      checks[proposal.id] = checkTargetDigest(proposal.targets ?? {}, proposal.target_digest);
    }
  }
  return checks;
}

function DigestLine({ check }: { check: DigestCheck | undefined }) {
  if (!check) return null;
  if (check.status === "mismatch") {
    return (
      <p role="alert" className="mt-2 text-xs font-semibold leading-relaxed text-rose-300">
        Target digest mismatch: the server says{" "}
        <code title={check.server}>{shortDigest(check.server)}</code> but these targets hash to{" "}
        <code title={check.local}>{shortDigest(check.local)}</code>. Approval is blocked; refresh and
        review again.
      </p>
    );
  }
  return (
    <p className="mt-2 flex flex-wrap items-center gap-1.5 text-[10px] text-muted">
      <Fingerprint aria-hidden="true" className="h-3.5 w-3.5 text-cyan-300/70" />
      {check.status === "verified"
        ? "Targets identified by digest"
        : "Digest computed here (this server does not send one)"}
      <code className="font-mono text-slate-400" title={check.digest}>
        {shortDigest(check.digest)}
      </code>
    </p>
  );
}

/** A target or error in its unit: degrees, or gripper units for the gripper. */
function formatJoint(joint: string, value: number): string {
  return `${value.toFixed(2)}${jointUnit(joint)}`;
}

function finiteEntries(record: Record<string, unknown> | undefined | null): [string, number][] {
  return Object.entries(record ?? {}).filter(
    (entry): entry is [string, number] => typeof entry[1] === "number" && Number.isFinite(entry[1]),
  );
}

function toleranceFor(joint: string, tolerance: AgentResult["tolerance"]): number | null {
  const value = joint === "gripper" ? tolerance?.gripper : tolerance?.arm_deg;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** A server timestamp in local time, machine-readable in `dateTime`. */
function Timestamp({ iso }: { iso: unknown }) {
  if (typeof iso !== "string" || !iso) return <>—</>;
  const ms = Date.parse(iso);
  if (Number.isNaN(ms)) return <>{iso}</>;
  const date = new Date(ms);
  return <time dateTime={date.toISOString()}>{date.toLocaleTimeString()}</time>;
}

function ResultHeading({ status, children }: { status: string; children?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-[10px] font-bold uppercase tracking-[0.16em] text-slate-400">
        Result · {status}
      </span>
      {children}
    </div>
  );
}

const RESULT_CLASS = "mt-4 rounded-xl border border-white/5 bg-slate-900/60 p-3";

/**
 * A terminal C8 result as gate reports it: why the run ended (with the
 * server's own words), the measured error per joint against the tolerance,
 * any target the server limited, when it ran, and the motor state it left.
 */
export function AgentResultPanel({
  result,
  status,
  proposed,
  canCollapse,
}: {
  result: AgentResult;
  status: string;
  /** The targets that were approved, to show any the server limited. */
  proposed?: Record<string, number>;
  canCollapse: boolean;
}) {
  if (status === "expired") {
    const age = typeof result.age_seconds === "number" ? Math.round(result.age_seconds) : null;
    // The server measures a proposal's age on its own monotonic clock, so a
    // proposal created before a server restart has no known age (null).
    const createdBeforeRestart = result.age_seconds === null;
    return (
      <section aria-label="Run result" className={RESULT_CLASS}>
        <ResultHeading status={status} />
        <p className="mt-2 text-xs leading-relaxed text-amber-200">
          Expired before it ran; nothing moved. The server approves and runs a proposal only
          within {PROPOSAL_MAX_AGE_S / 60} minutes of its creation
          {age !== null ? ` (this one was ${age} s old)` : ""}.
          {createdBeforeRestart
            ? " This one was created before the server restarted, and a restart forgets proposal ages, so it cannot run."
            : ""}{" "}
          Create a fresh proposal to try again.
        </p>
      </section>
    );
  }
  if (status === "cancelled" && result.end_state == null && result.started_at == null) {
    return (
      <section aria-label="Run result" className={RESULT_CLASS}>
        <ResultHeading status={status} />
        <p className="mt-2 text-xs leading-relaxed text-slate-300">
          Cancelled before it ran. Nothing moved.
        </p>
      </section>
    );
  }

  const errors = finiteEntries(result.final_error_deg);
  const reached = result.reached;
  const reason = stopReasonLabel(result.stop_reason);
  const serverError = typeof result.error === "string" && result.error.trim() ? result.error : null;
  const maxError = typeof result.max_abs_error_deg === "number" ? result.max_abs_error_deg : null;
  const maxJoint = maxError === null ? undefined : errors.find(([, value]) => value === maxError)?.[0];
  const armTolerance = toleranceFor("shoulder_pan", result.tolerance);
  const gripperTolerance = toleranceFor("gripper", result.tolerance);
  const limited = proposed
    ? finiteEntries(result.targets).filter(
        ([joint, value]) => typeof proposed[joint] === "number" && Math.abs(proposed[joint] - value) > 1e-6,
      )
    : [];
  return (
    <section aria-label="Run result" className={RESULT_CLASS}>
      <ResultHeading status={status}>
        <StatusPill
          label=""
          value={
            reached === true
              ? "Reached target"
              : reached === false
                ? "Did not reach target"
                : "Reach not reported"
          }
          tone={reached === true ? "ok" : "warn"}
        />
        <StatusPill label="End state" value={endStateLabel(result.end_state)} tone={endStateTone(result.end_state)} />
      </ResultHeading>
      {(reason || serverError) && (
        <div className="mt-2 space-y-0.5 text-xs leading-relaxed">
          {reason && <p className="text-slate-300">Why it ended: {reason}.</p>}
          {serverError && (
            <p className="text-slate-400">
              Server: <span className="font-mono text-slate-200">{serverError}</span>
            </p>
          )}
        </div>
      )}
      <dl className="mt-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-3">
        <div className="rounded-lg bg-slate-950/60 px-2.5 py-2">
          <dt className="text-[10px] text-muted">Duration</dt>
          <dd className="font-mono text-slate-200">
            {typeof result.duration_s === "number" ? `${result.duration_s.toFixed(1)} s` : "—"}
          </dd>
        </div>
        <div className="rounded-lg bg-slate-950/60 px-2.5 py-2">
          <dt className="text-[10px] text-muted">Steps</dt>
          <dd className="font-mono text-slate-200">
            {typeof result.steps === "number" ? result.steps : "—"}
          </dd>
        </div>
        <div className="rounded-lg bg-slate-950/60 px-2.5 py-2">
          <dt className="text-[10px] text-muted">Largest final error</dt>
          <dd className="font-mono text-slate-200">
            {maxError === null
              ? "—"
              : maxJoint
                ? `${formatJoint(maxJoint, maxError)} (${maxJoint})`
                : maxError.toFixed(2)}
          </dd>
        </div>
        <div className="rounded-lg bg-slate-950/60 px-2.5 py-2">
          <dt className="text-[10px] text-muted">Started</dt>
          <dd className="font-mono text-slate-200">
            <Timestamp iso={result.started_at} />
          </dd>
        </div>
        <div className="rounded-lg bg-slate-950/60 px-2.5 py-2">
          <dt className="text-[10px] text-muted">Finished</dt>
          <dd className="font-mono text-slate-200">
            <Timestamp iso={result.finished_at} />
          </dd>
        </div>
      </dl>
      {errors.length > 0 && (
        <div className="mt-2">
          <div className="text-[10px] text-muted">
            Final error per joint
            {(armTolerance !== null || gripperTolerance !== null) &&
              ` · tolerance ${[
                armTolerance !== null ? `±${armTolerance}° per arm joint` : null,
                gripperTolerance !== null ? `±${gripperTolerance} gripper units` : null,
              ]
                .filter(Boolean)
                .join(", ")}`}
          </div>
          <ul className="mt-1 grid gap-1.5 sm:grid-cols-2 lg:grid-cols-3">
            {errors.map(([joint, value]) => {
              const tolerance = toleranceFor(joint, result.tolerance);
              const outside = tolerance !== null && value > tolerance;
              return (
                <li
                  key={joint}
                  className="flex justify-between gap-2 rounded-lg bg-slate-950/60 px-2.5 py-1.5 font-mono text-[11px]"
                >
                  <span className="truncate text-muted">{joint}</span>
                  <span className={outside ? "font-semibold text-amber-300" : "text-slate-200"}>
                    {formatJoint(joint, value)}
                    {outside && <span className="sr-only"> (outside tolerance)</span>}
                  </span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
      {limited.length > 0 && (
        <p className="mt-2 text-xs leading-relaxed text-amber-200">
          The server limited{" "}
          {limited.length === 1 ? "this target" : "these targets"} to the arm&apos;s calibrated
          range before running:{" "}
          {limited
            .map(([joint, value]) => `${joint} ${formatJoint(joint, proposed?.[joint] ?? value)} → ${formatJoint(joint, value)}`)
            .join("; ")}
          .
        </p>
      )}
      {endStateHoldsTorque(result.end_state) && (
        <p className="mt-2 text-xs leading-relaxed text-amber-200">
          The arm is holding position with its motors on.{" "}
          {canCollapse
            ? "Collapse (in the safety bar) lowers it to rest and turns the motors off."
            : "E-stop turns the motors off where the arm stands."}
        </p>
      )}
      {endStateUnknown(result.end_state) && (
        <p className="mt-2 text-xs font-semibold leading-relaxed text-amber-200">
          The server could not confirm whether the motors are on after this run. Check the arm
          before anything else; E-stop turns the motors off.
        </p>
      )}
      {result.end_state === "deenergized" && (
        <p className="mt-2 text-xs leading-relaxed text-slate-400">
          The motors are off, so the arm is not held in place.
        </p>
      )}
    </section>
  );
}

interface ExecutionControlsProps {
  proposal: AgentProposal;
  grant: AgentApprovalGrant | null;
  run: AgentRun | null;
  runActive: boolean;
  holding: boolean;
  blockers: string[];
  busy: string | null;
  reviewed: boolean;
  digest: DigestCheck | undefined;
  now: number;
  /** Client time after which the server refuses this proposal (120 s old). */
  ageDeadline: number | null;
  onReviewed: (value: boolean) => void;
  onApprove: () => void;
  onExecute: () => void;
  onPressHold: () => void;
  onReleaseHold: (reason: string) => void;
  onCancelRun: () => void;
}

function ExecutionControls({
  proposal,
  grant,
  run,
  runActive,
  holding,
  blockers,
  busy,
  reviewed,
  digest,
  now,
  ageDeadline,
  onReviewed,
  onApprove,
  onExecute,
  onPressHold,
  onReleaseHold,
  onCancelRun,
}: ExecutionControlsProps) {
  const deadman = requiresDeadman(proposal);
  const runHere = run !== null && run.proposalId === proposal.id && run.phase !== "ended";
  const canStart = grant !== null && blockers.length === 0 && !runActive && busy === null;
  const canApprove =
    reviewed &&
    blockers.length === 0 &&
    !runActive &&
    busy === null &&
    digest !== undefined &&
    digest.status !== "mismatch";
  const secondsLeft = grant ? Math.max(0, Math.ceil((grant.deadline - now) / 1000)) : 0;
  const reviewLeft =
    ageDeadline === null ? null : Math.max(0, Math.ceil((ageDeadline - now) / 1000));
  const hintId = `deadman-hint-${proposal.id}`;
  const deadmanLabel = !runHere
    ? "Hold to run"
    : run.phase === "starting"
      ? "Starting…"
      : run.phase === "cancelling" || !holding
        ? "Stopping…"
        : "Running · keep holding";

  return (
    <div className="mt-4 space-y-3 border-t border-white/5 pt-4">
      {!grant && !runHere && (
        <>
          {reviewLeft !== null && (
            <p
              className={`text-[11px] leading-relaxed ${reviewLeft > 0 ? "text-slate-400" : "font-semibold text-amber-200"}`}
            >
              {reviewLeft > 0
                ? `Review window: about ${reviewLeft} s left to approve and start. The server refuses proposals older than ${PROPOSAL_MAX_AGE_S / 60} minutes.`
                : `This proposal is older than ${PROPOSAL_MAX_AGE_S / 60} minutes, so the server will refuse it. Create a fresh proposal.`}
            </p>
          )}
          <label className="flex items-start gap-3">
            <input
              type="checkbox"
              checked={reviewed}
              onChange={(event) => onReviewed(event.target.checked)}
              className="mt-0.5 h-4 w-4 rounded-sm border-slate-600 bg-slate-900 accent-cyan-400"
            />
            <span className="text-xs leading-relaxed text-slate-400">
              I reviewed the objective, steps, exact targets and the physical workspace.
            </span>
          </label>
          <Button variant="success" className="w-full" onClick={onApprove} disabled={!canApprove}>
            {canApprove || busy === `approve:${proposal.id}` ? (
              <ShieldCheck aria-hidden="true" className="h-4 w-4" />
            ) : (
              <LockKeyhole aria-hidden="true" className="h-4 w-4" />
            )}
            {busy === `approve:${proposal.id}` ? "Approving…" : "Approve these targets"}
          </Button>
          <BlockerList title="Approval is blocked" reasons={blockers} />
        </>
      )}

      {grant && !runHere && (
        <p className="text-xs leading-relaxed text-emerald-200" role="status">
          Approved for this tab's lease · <span className="font-mono">{secondsLeft} s</span> left ·
          single use
        </p>
      )}

      {(grant || runHere) &&
        (deadman ? (
          <>
            <DeadmanButton
              active={holding && runHere}
              disabled={runHere ? !holding : !canStart}
              onHoldStart={onPressHold}
              onHoldEnd={onReleaseHold}
              aria-describedby={hintId}
              className={`flex h-20 w-full flex-col items-center justify-center rounded-2xl border-2 text-sm font-black uppercase tracking-wider transition focus-visible:outline-hidden focus-visible:ring-4 focus-visible:ring-cyan-300/50 disabled:cursor-not-allowed disabled:opacity-40 ${holding && runHere ? "border-amber-200 bg-amber-400 text-slate-950" : "border-cyan-300/30 bg-cyan-400/10 text-cyan-100"}`}
            >
              {holding && runHere ? (
                <Hand aria-hidden="true" className="mb-1 h-5 w-5" />
              ) : (
                <LockKeyhole aria-hidden="true" className="mb-1 h-5 w-5" />
              )}
              {deadmanLabel}
            </DeadmanButton>
            <p id={hintId} className="text-[10px] leading-relaxed text-muted">
              Hold with the pointer, or hold Space or Enter. Letting go cancels the run; E-stop
              (Escape) works at any time.
            </p>
          </>
        ) : runHere ? (
          <Button
            variant="danger"
            className="w-full"
            onClick={onCancelRun}
            disabled={run.phase === "cancelling"}
          >
            <CircleStop aria-hidden="true" className="h-4 w-4" />
            {run.phase === "cancelling" ? "Cancelling…" : "Cancel run"}
          </Button>
        ) : (
          <Button variant="success" className="w-full" onClick={onExecute} disabled={!canStart}>
            <Play aria-hidden="true" className="h-4 w-4" />
            {busy === `execute:${proposal.id}` ? "Starting…" : "Execute approved targets"}
          </Button>
        ))}

      {grant && !runHere && <BlockerList title="Execution is blocked" reasons={blockers} />}
    </div>
  );
}

export default function AgentLab() {
  const supervised = useOperatorSupervision();
  const { telemetry, transportStale } = useTelemetry();
  const { view, control, refresh: refreshSafety } = useSafety();
  const lease = useMotionLease("agent-lab");
  const { data: plannerStatus, refresh: refreshPlanner } = usePolling(getAgentPlannerStatus, 10000);
  const {
    data: response,
    error: proposalsError,
    refresh: refreshProposals,
  } = usePolling(getAgentProposals, 3000);
  const proposals = useMemo(() => proposalList(response), [response]);
  const digests = useMemo(() => digestChecks(proposals), [proposals]);
  const now = useNow(500);

  const onChanged = useCallback(() => {
    refreshProposals();
    refreshSafety();
  }, [refreshProposals, refreshSafety]);
  const exec = useAgentExecution({
    leaseId: lease.leaseId,
    timestampFor: lease.timestampFor,
    tripped: view.tripped,
    onChanged,
  });

  const [objective, setObjective] = useState("");
  const [planner, setPlanner] = useState<Planner>("deterministic");
  const [skill, setSkill] = useState<SkillId>("inspect_scene");
  const [maxDuration, setMaxDuration] = useState(30);
  const [allowMotion, setAllowMotion] = useState(false);
  const [holdToRun, setHoldToRun] = useState(true);
  const [reviewed, setReviewed] = useState<Record<string, boolean>>({});
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [createdNotice, setCreatedNotice] = useState(false);

  const selectedSkill = SKILLS.find((entry) => entry.id === skill) ?? SKILLS[0];
  const motionRequested =
    planner === "configured" ? allowMotion : selectedSkill.motion && allowMotion;
  const linkLive = !transportStale && telemetry !== null;
  const leaseOwned = lease.leaseId !== null;
  const blockers = agentExecutionBlockers({ leaseOwned, linkLive, supervised, view, control });
  const runActive = exec.run !== null && exec.run.phase !== "ended";
  const otherLease = !leaseOwned && control?.lease ? control.lease.source : null;

  const submit = async () => {
    const text = objective.trim();
    if (!text) return;
    setCreating(true);
    setCreateError(null);
    setCreatedNotice(false);
    try {
      await createAgentProposal({
        objective: text,
        skill,
        planner,
        constraints: {
          max_duration_seconds: maxDuration,
          require_deadman: holdToRun && motionRequested,
          allow_motion: motionRequested,
        },
      });
      setObjective("");
      setCreatedNotice(true);
      refreshProposals();
    } catch (caught) {
      const message = errorMessage(caught);
      const status = caught instanceof ApiError ? caught.status : null;
      if (status === 502) {
        setCreateError(`The planner answered with something unusable: ${message}`);
      } else if (status === 503) {
        setCreateError(`Planner unavailable: ${message}`);
        refreshPlanner();
      } else if (status === 422 && /deadman/i.test(message)) {
        setCreateError(joinSentences(message, OLD_SERVER_DEADMAN_HINT));
      } else {
        setCreateError(message);
      }
    } finally {
      setCreating(false);
    }
  };

  const robotTone =
    view.tripped || view.torqueUnconfirmed
      ? "red"
      : runActive
        ? "cyan"
        : blockers.length > 0
          ? "amber"
          : "green";

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Bounded skill builder"
        title="Agent contract lab"
        description="Use the offline router or an optional proposal-only model to select a named skill, inspect its exact targets, approve them, and watch a bounded server-side run end in a reported state."
        icon={Sparkles}
        actions={
          <Button variant="secondary" onClick={onChanged}>
            <RefreshCw aria-hidden="true" className="h-3.5 w-3.5" /> Refresh
          </Button>
        }
      />

      {proposalsError && (
        <Notice tone="warn" title="Agent proposals are unavailable">
          Proposal features stay disabled until the backend answers. {proposalsError.message}
        </Notice>
      )}
      {exec.error && (
        <Notice tone="danger" title="Agent action failed">
          {exec.error}
        </Notice>
      )}
      {lease.error && (
        <Notice tone="danger" title="Motion lease problem">
          {lease.error}
        </Notice>
      )}
      {exec.notice && (
        <Notice tone="success" title="Control plane update">
          {exec.notice}
        </Notice>
      )}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricTile
          label="Agent proposals"
          value={proposals.length}
          detail="Durable backend records"
          tone="cyan"
          icon={BrainCircuit}
        />
        <MetricTile
          label="Motion lease"
          value={leaseOwned ? "Owned" : otherLease ? "Busy" : "None"}
          detail={
            leaseOwned
              ? "Held by this tab"
              : otherLease
                ? `Held by ${otherLease}`
                : "No active owner"
          }
          tone={leaseOwned ? "green" : otherLease ? "amber" : "neutral"}
          icon={KeyRound}
        />
        <MetricTile
          label="Control plane"
          value={
            view.safetyState === "unknown" ? "Unknown" : view.armed ? "Armed" : "Disarmed"
          }
          detail={`Safety state: ${view.safetyState.replace(/_/g, " ")}`}
          tone={view.safetyState === "unknown" ? "neutral" : view.armed ? "amber" : "green"}
          icon={LockKeyhole}
        />
        <MetricTile
          label="Robot context"
          value={runActive ? "Running" : blockers.length > 0 ? "Blocked" : "Ready"}
          detail={
            runActive
              ? "An approved run is in progress"
              : (blockers[0] ?? "Approval and execution are allowed")
          }
          tone={robotTone}
          icon={Bot}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(22rem,0.78fr)_minmax(0,1.22fr)]">
        <div className="space-y-6">
          <Card title="1 · Define a bounded task" subtitle="The agent can only use the selected skill">
            <div className="space-y-4">
              <Field
                label="Planner"
                hint={
                  plannerStatus && !plannerStatus.configured_provider_available && plannerStatus.unavailable_reason
                    ? `Model planner unavailable: ${joinSentences(plannerStatus.unavailable_reason, "The deterministic router still works.")}`
                    : undefined
                }
              >
                {(controlProps) => (
                  <select
                    {...controlProps}
                    value={planner}
                    onChange={(event) => setPlanner(event.target.value as Planner)}
                    className={INPUT_CLASS}
                  >
                    <option value="deterministic">Deterministic named-skill router</option>
                    <option
                      value="configured"
                      disabled={!plannerStatus?.configured_provider_available}
                    >
                      {plannerStatus?.configured_provider_available
                        ? `${plannerStatus.configured_provider} · ${plannerStatus.model}`
                        : "Model planner unavailable"}
                    </option>
                  </select>
                )}
              </Field>
              <Field
                label="Objective"
                hint={`${objective.length}/${OBJECTIVE_MAX_CHARS} characters`}
              >
                {(controlProps) => (
                  <textarea
                    {...controlProps}
                    value={objective}
                    onChange={(event) => setObjective(event.target.value)}
                    maxLength={OBJECTIVE_MAX_CHARS}
                    rows={4}
                    placeholder="Inspect the scene and propose a safe path to the blue cube…"
                    className={`${INPUT_CLASS} resize-none`}
                  />
                )}
              </Field>

              <div>
                <SectionLabel>Skill boundary</SectionLabel>
                <div className="mt-2 grid gap-2 sm:grid-cols-2" role="group" aria-label="Skill">
                  {SKILLS.map((entry) => (
                    <button
                      key={entry.id}
                      type="button"
                      aria-pressed={skill === entry.id}
                      disabled={planner === "configured"}
                      onClick={() => {
                        setSkill(entry.id);
                        // A non-motion skill can never carry a motion grant.
                        if (!entry.motion) setAllowMotion(false);
                      }}
                      className={`rounded-xl border p-3 text-left transition focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-cyan-400/50 disabled:opacity-50 ${skill === entry.id ? "border-cyan-400/30 bg-cyan-400/10" : "border-white/5 bg-slate-950/30 hover:border-slate-700"}`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <span className="text-xs font-semibold text-slate-200">{entry.label}</span>
                        {entry.motion && <StatusPill label="" value="motion" tone="warn" />}
                      </div>
                      <p className="mt-1.5 text-[10px] leading-relaxed text-slate-400">{entry.detail}</p>
                    </button>
                  ))}
                </div>
              </div>

              <label className="block text-xs text-slate-400">
                Maximum duration · <span className="font-mono text-cyan-300">{maxDuration}s</span>
                <input
                  type="range"
                  min={5}
                  max={30}
                  step={5}
                  value={maxDuration}
                  onChange={(event) => setMaxDuration(Number(event.target.value))}
                  className="mt-2 w-full accent-cyan-400"
                />
              </label>

              {(selectedSkill.motion || planner === "configured") && (
                <div className="space-y-2 rounded-xl border border-amber-300/15 bg-amber-400/[0.05] p-3">
                  <label className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      checked={allowMotion}
                      onChange={(event) => setAllowMotion(event.target.checked)}
                      className="mt-0.5 h-4 w-4 rounded-sm border-slate-600 bg-slate-900 accent-amber-400"
                    />
                    <span className="text-xs leading-relaxed text-slate-400">
                      Allow the planner to propose one of the bounded motion skills. This does not
                      approve execution or expose a motor-write tool.
                    </span>
                  </label>
                  <label className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      checked={holdToRun}
                      onChange={(event) => setHoldToRun(event.target.checked)}
                      className="mt-0.5 h-4 w-4 rounded-sm border-slate-600 bg-slate-900 accent-amber-400"
                    />
                    <span className="text-xs leading-relaxed text-slate-400">
                      Require hold-to-run: the run continues only while you hold its control and
                      stops when you let go.
                      {!holdToRun &&
                        " Without it, an approved run continues until it reaches tolerance or its duration limit unless you cancel it, Halt or E-stop."}
                    </span>
                  </label>
                </div>
              )}

              {createError && (
                <Notice tone="danger" title="Proposal was not created">
                  {createError}
                </Notice>
              )}
              {createdNotice && !createError && (
                <p role="status" className="text-xs text-emerald-200">
                  Proposal created. Review every step and target before approving.
                </p>
              )}

              <Button
                className="w-full"
                onClick={() => void submit()}
                disabled={
                  creating ||
                  !objective.trim() ||
                  (planner === "deterministic" && selectedSkill.motion && !allowMotion)
                }
              >
                <Send aria-hidden="true" className="h-4 w-4" />
                {creating ? "Creating proposal…" : "Create proposal"}
              </Button>
            </div>
          </Card>

          <Card title="2 · Safety session" subtitle="Approval and execution both need this tab's lease">
            <div className="space-y-3">
              {!leaseOwned ? (
                <Button
                  className="w-full"
                  onClick={() => void lease.acquire()}
                  disabled={lease.busy || !!otherLease}
                >
                  <KeyRound aria-hidden="true" className="h-4 w-4" />
                  {otherLease ? `Lease held by ${otherLease}` : "Acquire the Agent Lab lease"}
                </Button>
              ) : (
                <Button variant="secondary" className="w-full" onClick={() => void lease.release()}>
                  Release lease
                </Button>
              )}
              <p className="text-[10px] leading-relaxed text-slate-400">
                The lease renews every 2.5 s while this tab is visible. Hiding or leaving the tab
                releases it, and releasing it cancels a run in progress. The server disarms when
                the lease is released.
              </p>
            </div>
          </Card>
        </div>

        <Card title="3 · Review, approve, run" subtitle="Execution never follows proposal creation automatically">
          {proposals.length === 0 ? (
            <EmptyState
              title="No proposals yet"
              detail="Create a bounded task. The agent's proposed steps and targets will appear here for human review."
              icon={BrainCircuit}
            />
          ) : (
            <div className="space-y-4">
              {proposals.map((listed) => {
                const proposal = exec.latest?.id === listed.id ? exec.latest : listed;
                const status = proposalStatus(proposal);
                const motion = proposesMotion(listed);
                const runHere =
                  exec.run !== null && exec.run.proposalId === listed.id && exec.run.phase !== "ended";
                const grant = isApprovalUsable(exec.approval, listed, lease.leaseId, now)
                  ? exec.approval
                  : null;
                const showControls = motion && (isExecutableStatus(proposalStatus(listed)) || runHere || grant !== null);
                return (
                  <article
                    key={listed.id}
                    aria-label={proposalTitle(listed)}
                    className="rounded-2xl border border-white/5 bg-slate-950/35 p-4"
                  >
                    <div className="flex flex-col justify-between gap-3 sm:flex-row sm:items-start">
                      <div>
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="text-sm font-semibold text-slate-100">{proposalTitle(listed)}</h3>
                          <StatusPill label="" value={status} tone={statusTone(status)} />
                          {listed.risk && (
                            <StatusPill label="risk" value={listed.risk} tone={listed.risk === "low" ? "ok" : "warn"} />
                          )}
                          {motion && requiresDeadman(listed) && (
                            <StatusPill label="" value="hold-to-run" tone="info" />
                          )}
                        </div>
                        <div className="mt-1 font-mono text-[9px] text-muted">{listed.id}</div>
                      </div>
                      <span className="rounded-lg bg-slate-900 px-2 py-1 font-mono text-[9px] text-cyan-300">
                        {listed.skill ?? "bounded-skill"}
                      </span>
                    </div>

                    {listed.description && listed.description !== proposalTitle(listed) && (
                      <p className="mt-3 text-xs leading-relaxed text-slate-400">{listed.description}</p>
                    )}

                    {listed.constraints?.planner && (
                      <div className="mt-3 rounded-lg border border-violet-300/10 bg-violet-400/[0.04] px-3 py-2.5 text-xs text-slate-400">
                        <span className="font-mono text-violet-200">
                          {listed.constraints.planner.provider}
                          {listed.constraints.planner.model ? ` · ${listed.constraints.planner.model}` : ""}
                        </span>
                        {listed.constraints.planner.rationale && (
                          <p className="mt-1 leading-relaxed">{listed.constraints.planner.rationale}</p>
                        )}
                      </div>
                    )}

                    {motion && (
                      <div className="mt-4 rounded-xl border border-amber-300/15 bg-amber-400/[0.05] p-3">
                        <div className="text-[10px] font-bold uppercase tracking-[0.16em] text-amber-200">
                          Exact proposed motor targets
                        </div>
                        <div className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                          {Object.entries(listed.targets ?? {}).map(([joint, value]) => (
                            <div key={joint} className="rounded-lg bg-slate-950/60 px-2.5 py-2">
                              <div className="truncate font-mono text-[9px] text-muted">{joint}</div>
                              <div className="mt-1 font-mono text-sm font-bold text-amber-100">
                                {formatJoint(joint, value)}
                              </div>
                            </div>
                          ))}
                        </div>
                        <p className="mt-2 text-[10px] text-muted">
                          Arm joints in degrees; the gripper in gripper units (0-100, higher is more
                          open).
                        </p>
                        <DigestLine check={digests[listed.id]} />
                        <p className="mt-2 text-[10px] leading-relaxed text-slate-400">
                          Approving authorizes exactly these targets, identified by digest, for 30
                          seconds and only for this tab's lease; the server accepts approval and
                          execution only within {PROPOSAL_MAX_AGE_S / 60} minutes of the
                          proposal's creation. The server then runs a rate-limited loop until the
                          arm is within tolerance or the{" "}
                          {listed.constraints?.max_duration_seconds ?? 30} s limit, and reports the
                          state it ended in. Calibration, position and rate limits still apply, so
                          the server may limit a target and measured motion may stop short of it.
                          Unless the result says the arm is holding position, the motors are off
                          when a run ends and the arm is not held.
                        </p>
                      </div>
                    )}

                    <div className="mt-4 space-y-2">
                      {(listed.steps ?? []).length ? (
                        listed.steps?.map((step, index) => {
                          const label =
                            typeof step === "string" ? step : (step.label ?? step.detail ?? `Step ${index + 1}`);
                          return (
                            <div
                              key={`${listed.id}-step-${index}`}
                              className="flex items-start gap-3 rounded-lg bg-slate-900/55 px-3 py-2.5"
                            >
                              <span className="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-cyan-400/10 font-mono text-[9px] text-cyan-300">
                                {index + 1}
                              </span>
                              <span className="text-xs leading-relaxed text-slate-300">{label}</span>
                              <ChevronRight aria-hidden="true" className="ml-auto mt-0.5 h-3.5 w-3.5 shrink-0 text-muted" />
                            </div>
                          );
                        })
                      ) : (
                        <div className="rounded-lg border border-dashed border-slate-800 px-3 py-4 text-center text-xs text-muted">
                          No structured steps were returned.
                        </div>
                      )}
                    </div>

                    {status === "executing" && (
                      <p role="status" className="mt-4 text-xs leading-relaxed text-cyan-200">
                        Executing on the server
                        {proposal.result?.started_at ? (
                          <>
                            {" "}since <Timestamp iso={proposal.result.started_at} />
                          </>
                        ) : null}
                        . It ends within tolerance, at its time limit, or when stopped.
                      </p>
                    )}
                    {proposal.result && isTerminalStatus(status) && (
                      <AgentResultPanel
                        result={proposal.result}
                        status={status}
                        proposed={listed.targets}
                        canCollapse={view.canCollapse}
                      />
                    )}

                    {showControls && (
                      <ExecutionControls
                        proposal={listed}
                        grant={grant}
                        run={exec.run}
                        runActive={runActive}
                        holding={exec.holding}
                        blockers={blockers}
                        busy={exec.busy}
                        reviewed={!!reviewed[listed.id]}
                        digest={digests[listed.id]}
                        now={now}
                        ageDeadline={
                          isExecutableStatus(proposalStatus(listed))
                            ? proposalAgeDeadline(listed.created_at, now, lease.timestampFor)
                            : null
                        }
                        onReviewed={(value) =>
                          setReviewed((current) => ({ ...current, [listed.id]: value }))
                        }
                        onApprove={() => void exec.approve(listed)}
                        onExecute={() => void exec.execute(listed)}
                        onPressHold={() => exec.pressHold(listed)}
                        onReleaseHold={exec.releaseHold}
                        onCancelRun={() => void exec.cancelProposal(listed)}
                      />
                    )}

                    {!motion && status === "proposed" && (
                      <div className="mt-4 rounded-lg border border-cyan-300/10 bg-cyan-400/[0.04] px-3 py-2.5 text-xs text-slate-400">
                        Observation-only proposal: no physical execution or motion lease is needed.
                      </div>
                    )}
                    {isCancellableStatus(proposalStatus(listed)) && !runHere && (
                      <Button
                        variant="secondary"
                        size="sm"
                        className="mt-3"
                        onClick={() => void exec.cancelProposal(listed)}
                        disabled={exec.busy !== null}
                      >
                        <CircleStop aria-hidden="true" className="h-3.5 w-3.5" />
                        {exec.busy === `cancel:${listed.id}` ? "Cancelling…" : "Cancel proposal"}
                      </Button>
                    )}
                  </article>
                );
              })}
            </div>
          )}
        </Card>
      </div>

      <Notice tone="info" title="Human authority is preserved">
        The agent proposes only named skills with explicit duration and motion bounds. Approval
        authorizes exactly the targets shown, by digest, for 30 seconds and for this tab's lease,
        within {PROPOSAL_MAX_AGE_S / 60} minutes of the proposal's creation. The server runs a
        rate-limited loop until the arm is within tolerance or the duration limit, then reports
        how it ended: motors off, holding position, E-stopped, fault, or unknown when it cannot
        confirm the motor state. Your operator heartbeat must stay live; hold-to-run proposals stop
        when you let go. E-stop (Escape) is global.
      </Notice>
    </div>
  );
}
