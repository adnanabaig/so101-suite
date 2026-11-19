import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";
import Card from "../components/Card";
import Button from "../components/Button";
import BlockerList from "../components/BlockerList";
import { StatusPill } from "../components/ui/StatusPill";
import { Notice } from "../components/Workstation";
import { useConfirm } from "../components/ui/Dialog";
import { Field, INPUT_CLASS } from "../components/ui/Field";
import {
  getPresets,
  getDatasets,
  getJobs,
  getJobLogs,
  startTraining,
  cancelJob,
  type JobSummary,
  type PolicyPresetOut,
} from "../api";
import { usePolling } from "../hooks/usePolling";

export interface LossPoint {
  step: number;
  loss: number;
}

/** Loss points kept per job; older history is thinned, never dropped (F20). */
export const MAX_LOSS_POINTS = 500;

/**
 * Append a point; past `max`, keep every other older point so the curve keeps
 * its whole history at half the resolution and memory stays bounded.
 */
export function appendLossPoint(points: readonly LossPoint[], point: LossPoint, max = MAX_LOSS_POINTS): LossPoint[] {
  const next = [...points, point];
  if (next.length <= max) return next;
  const thinned = next.slice(0, -1).filter((_, index) => index % 2 === 0);
  return [...thinned, point];
}

/** Tones for training/jobs.py statuses ("succeeded", not "completed": F17). */
export function jobTone(status: string): "info" | "ok" | "danger" | "warn" | "neutral" {
  switch (status) {
    case "running":
      return "info";
    case "succeeded":
      return "ok";
    case "failed":
      return "danger";
    case "cancelled":
    case "interrupted": // the server stopped before the job finished
      return "warn";
    case "corrupt": // the job record is unreadable; its outcome is unknown
      return "danger";
    default:
      return "neutral"; // pending, queued, unknown
  }
}

const CANCELLABLE = new Set(["pending", "queued", "running"]);

/** A required whole number of at least 1; `Number("")` would silently be 0 (F20). */
export function positiveIntegerError(raw: string, label: string): string | null {
  const text = raw.trim();
  if (!text) return `${label} is required.`;
  if (!/^\d+$/.test(text)) return `${label} must be a whole number.`;
  const value = Number(text);
  if (!Number.isSafeInteger(value)) return `${label} is too large.`;
  if (value < 1) return `${label} must be at least 1.`;
  return null;
}

export default function Training() {
  const { data: presets } = usePolling(getPresets, 0);
  const { data: datasets } = usePolling(getDatasets, 0);
  const { data: jobs, error: jobsError, refresh: refreshJobs } = usePolling(getJobs, 2000);
  const [confirm, confirmDialog] = useConfirm();

  // Form fields stay strings so an empty field is an error, never 0.
  const [repoId, setRepoId] = useState("");
  const [policy, setPolicy] = useState("");
  const [steps, setSteps] = useState("10000");
  const [batchSize, setBatchSize] = useState("64");
  const [starting, setStarting] = useState(false);
  const [formErr, setFormErr] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState<string | null>(null);
  const [cancelErr, setCancelErr] = useState<string | null>(null);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  const selectedPreset = presets?.find((p) => p.policy_type === policy) ?? null;
  const stepsError = positiveIntegerError(steps, "Steps");
  const batchError = positiveIntegerError(batchSize, "Batch size");
  const blockers = [
    !repoId && "Select a dataset.",
    !policy && "Select a policy.",
    stepsError,
    batchError,
  ].filter((reason): reason is string => !!reason);

  const applyPreset = (preset: PolicyPresetOut) => {
    setPolicy(preset.policy_type);
    setSteps(String(preset.default_steps));
    setBatchSize(String(preset.default_batch_size));
  };

  const choosePolicy = (policyType: string) => {
    const preset = presets?.find((p) => p.policy_type === policyType);
    if (preset) applyPreset(preset);
    else setPolicy(policyType);
  };

  const submit = async () => {
    if (blockers.length) return;
    setStarting(true);
    setFormErr(null);
    try {
      const job = await startTraining({
        repo_id: repoId,
        policy,
        steps: Number(steps.trim()),
        batch_size: Number(batchSize.trim()),
      });
      setSelectedJobId(job.id);
      refreshJobs();
    } catch (e) {
      setFormErr(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  };

  const cancel = async (job: JobSummary) => {
    const confirmed = await confirm({
      title: `Cancel training job ${job.id}?`,
      body: (
        <>
          The {job.policy} run on <span className="font-mono">{job.repo_id}</span> stops at step{" "}
          {job.step} of {job.steps}. Checkpoints already written stay in{" "}
          <span className="font-mono">{job.output_dir}</span>; the run cannot be resumed from here.
        </>
      ),
      confirmLabel: "Cancel job",
      cancelLabel: "Keep running",
      tone: "warn",
    });
    if (!confirmed) return;
    setCancelling(job.id);
    setCancelErr(null);
    try {
      await cancelJob(job.id);
      refreshJobs();
    } catch (err) {
      setCancelErr(`Could not cancel job ${job.id}: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setCancelling(null);
    }
  };

  const selectedJob = jobs?.find((j) => j.id === selectedJobId) ?? null;

  return (
    <div className="space-y-6">
      {confirmDialog}
      <h1 className="text-2xl font-bold tracking-tight">Training</h1>

      <Card title="Policy presets">
        {!presets ? (
          <p className="text-sm text-muted">Loading presets…</p>
        ) : presets.length === 0 ? (
          <p className="text-sm text-muted">The server offers no training presets.</p>
        ) : (
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-3">
            {presets.map((p) => (
              <button
                type="button"
                key={p.name}
                onClick={() => applyPreset(p)}
                aria-pressed={policy === p.policy_type}
                className={`rounded-xl border p-4 text-left transition-colors focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-cyan-400/60 ${
                  policy === p.policy_type
                    ? "border-cyan-500/50 bg-cyan-500/10"
                    : "border-slate-800 bg-slate-900/60 hover:border-slate-700"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold text-slate-100">{p.name}</span>
                  <StatusPill label="" value={p.policy_type} tone="info" />
                </div>
                <p className="mt-1 text-xs text-slate-400">{p.description}</p>
                <p className="mt-2 text-xs text-muted">{p.notes}</p>
                <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-muted">
                  <span>steps {p.default_steps}</span>
                  <span>batch {p.default_batch_size}</span>
                  {p.requires_torch === false && <span className="text-emerald-300/80">no PyTorch needed</span>}
                </div>
              </button>
            ))}
          </div>
        )}
      </Card>

      <Card title="Start training job">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Field label="Dataset">
            {(control) => (
              <select {...control} value={repoId} onChange={(e) => setRepoId(e.target.value)} className={INPUT_CLASS}>
                <option value="">— select dataset —</option>
                {(datasets ?? []).map((d) => (
                  <option key={d.repo_id} value={d.repo_id}>
                    {d.repo_id}
                  </option>
                ))}
              </select>
            )}
          </Field>

          <Field label="Policy" hint="Choosing a policy loads its preset budget.">
            {(control) => (
              <select {...control} value={policy} onChange={(e) => choosePolicy(e.target.value)} className={INPUT_CLASS}>
                <option value="">— select policy —</option>
                {(presets ?? []).map((p) => (
                  <option key={p.policy_type} value={p.policy_type}>
                    {p.policy_type}
                  </option>
                ))}
              </select>
            )}
          </Field>

          <Field
            label="Steps"
            hint={selectedPreset ? `Preset default ${selectedPreset.default_steps}` : undefined}
            error={stepsError}
          >
            {(control) => (
              <input
                {...control}
                type="text"
                inputMode="numeric"
                value={steps}
                onChange={(e) => setSteps(e.target.value)}
                className={INPUT_CLASS}
              />
            )}
          </Field>

          <Field
            label="Batch size"
            hint={selectedPreset ? `Preset default ${selectedPreset.default_batch_size}` : undefined}
            error={batchError}
          >
            {(control) => (
              <input
                {...control}
                type="text"
                inputMode="numeric"
                value={batchSize}
                onChange={(e) => setBatchSize(e.target.value)}
                className={INPUT_CLASS}
              />
            )}
          </Field>
        </div>

        <div className="mt-4 flex flex-wrap items-start gap-3">
          <Button onClick={() => void submit()} disabled={starting || blockers.length > 0}>
            {starting ? "Starting…" : "Start job"}
          </Button>
          <BlockerList reasons={blockers} className="pt-1.5" />
        </div>
        {formErr && (
          <div role="alert" className="mt-3">
            <Notice tone="danger" title="The server rejected the job">
              {formErr}
            </Notice>
          </div>
        )}
      </Card>

      <Card title="Jobs" bodyClassName="p-0">
        {cancelErr && (
          <div role="alert" className="px-5 pt-4">
            <Notice tone="danger" title="Cancel failed">
              {cancelErr}
            </Notice>
          </div>
        )}
        {jobsError && (
          <div className="px-5 pt-4">
            <Notice tone="warn" title="Job list is not updating">
              {jobsError.message}
              {jobs ? " The rows below are from the last successful poll." : ""}
            </Notice>
          </div>
        )}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-left text-xs uppercase tracking-wide text-muted">
                <th className="px-5 py-3 font-medium">id</th>
                <th className="px-5 py-3 font-medium">dataset</th>
                <th className="px-5 py-3 font-medium">policy</th>
                <th className="px-5 py-3 font-medium">status</th>
                <th className="px-5 py-3 font-medium">progress</th>
                <th className="px-5 py-3 font-medium text-right">loss</th>
                <th className="px-5 py-3 font-medium text-right">actions</th>
              </tr>
            </thead>
            <tbody>
              {(!jobs || jobs.length === 0) && (
                <tr>
                  <td colSpan={7} className="px-5 py-6 text-center text-muted">
                    {jobs ? "No jobs yet." : "Loading jobs…"}
                  </td>
                </tr>
              )}
              {(jobs ?? []).map((j) => (
                <JobRow
                  key={j.id}
                  job={j}
                  selected={j.id === selectedJobId}
                  busy={cancelling === j.id}
                  onSelect={() => setSelectedJobId(j.id)}
                  onCancel={() => void cancel(j)}
                />
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* F18: keyed so a newly selected job never shows the previous one's
          loss curve or logs, even for a moment. */}
      {selectedJob && <JobDetail key={selectedJob.id} job={selectedJob} />}
    </div>
  );
}

function JobRow({
  job,
  selected,
  busy,
  onSelect,
  onCancel,
}: {
  job: JobSummary;
  selected: boolean;
  busy: boolean;
  onSelect: () => void;
  onCancel: () => void;
}) {
  const pct = job.steps > 0 ? Math.round((job.step / job.steps) * 100) : 0;
  const cancellable = CANCELLABLE.has(job.status);

  return (
    <tr
      onClick={onSelect}
      className={`cursor-pointer border-b border-slate-800/60 hover:bg-slate-800/30 ${
        selected ? "bg-cyan-500/5" : ""
      }`}
    >
      <td className="px-5 py-3 font-mono text-xs text-slate-400">
        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            onSelect();
          }}
          className="rounded-sm hover:text-cyan-200 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-cyan-400/60"
          aria-label={`Show job ${job.id}`}
          aria-current={selected ? "true" : undefined}
        >
          {job.id}
        </button>
      </td>
      <td className="px-5 py-3 font-mono text-cyan-300">{job.repo_id}</td>
      <td className="px-5 py-3 text-slate-300">{job.policy}</td>
      <td className="px-5 py-3">
        <StatusPill label="" value={job.status} tone={jobTone(job.status)} />
      </td>
      <td className="px-5 py-3">
        <div className="flex items-center gap-2">
          <div
            className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-800"
            role="progressbar"
            aria-label={`Job ${job.id} progress`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={pct}
          >
            <div className="h-full rounded-full bg-cyan-400" style={{ width: `${pct}%` }} />
          </div>
          <span className="font-mono text-xs text-slate-400">
            {job.step}/{job.steps}
          </span>
        </div>
      </td>
      <td className="px-5 py-3 text-right font-mono text-slate-300">
        {job.loss === null ? "—" : job.loss.toFixed(4)}
      </td>
      <td className="px-5 py-3 text-right">
        <Button
          variant="secondary"
          size="sm"
          onClick={(e) => {
            e.stopPropagation();
            onCancel();
          }}
          disabled={busy || !cancellable}
          aria-label={`Cancel job ${job.id}`}
        >
          {busy ? "Cancelling…" : "Cancel"}
        </Button>
      </td>
    </tr>
  );
}

function JobDetail({ job }: { job: JobSummary }) {
  // Loss points accumulate from the polled job; adjusted during render when a
  // new step arrives (no effect, no extra render pass).
  const [series, setSeries] = useState<{ lastStep: number; points: LossPoint[] }>({
    lastStep: -1,
    points: [],
  });
  if (job.loss !== null && job.step !== series.lastStep) {
    setSeries({ lastStep: job.step, points: appendLossPoint(series.points, { step: job.step, loss: job.loss }) });
  }

  const { data: logs } = usePolling(() => getJobLogs(job.id), 2000, { key: job.id });
  const logTail = useMemo(() => logs?.log_tail ?? [], [logs]);

  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card title={`Loss — ${job.id}`} subtitle={`policy ${job.policy}`}>
        {series.points.length === 0 ? (
          <p className="text-sm text-muted">
            {CANCELLABLE.has(job.status) ? "Waiting for loss data…" : "This job reported no loss values."}
          </p>
        ) : (
          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={series.points} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                <XAxis dataKey="step" stroke="#64748b" tick={{ fontSize: 11 }} />
                <YAxis stroke="#64748b" tick={{ fontSize: 11 }} width={50} domain={["auto", "auto"]} />
                <Tooltip
                  contentStyle={{
                    background: "#0f172a",
                    border: "1px solid #1e293b",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  labelStyle={{ color: "#94a3b8" }}
                />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#22d3ee"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        {job.progress && (
          <p className="mt-3 truncate font-mono text-[11px] text-slate-400" title={job.progress}>
            {job.progress}
          </p>
        )}
        {job.error && (
          <div className="mt-3 rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-300">
            {job.error}
          </div>
        )}
      </Card>

      <Card title="Log tail" actions={<StatusPill label="status" value={job.status} tone={jobTone(job.status)} />}>
        <pre
          data-testid="job-log-tail"
          className="scroll-thin max-h-64 overflow-auto whitespace-pre-wrap rounded-lg bg-slate-950/80 p-3 font-mono text-xs text-slate-300"
        >
          {logs ? (logTail.length ? logTail.join("\n") : "No log output.") : "Loading logs…"}
        </pre>
      </Card>
    </div>
  );
}
