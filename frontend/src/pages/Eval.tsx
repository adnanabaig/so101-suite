import { useEffect, useRef, useState } from "react";
import Card from "../components/Card";
import Button from "../components/Button";
import { StatusPill } from "../components/ui/StatusPill";
import CameraView from "../components/CameraView";
import BlockerList from "../components/BlockerList";
import PolicyRunFields from "../components/PolicyRunFields";
import { useSafety } from "../components/safety/SafetyProvider";
import { useConfirm } from "../components/ui/Dialog";
import { Field, INPUT_CLASS } from "../components/ui/Field";
import { Notice } from "../components/Workstation";
import {
  ApiError,
  continueEval,
  EVAL_TERMINAL_PHASES,
  getEvalState,
  getModels,
  scoreEval,
  startEval,
  stopEval,
  type EvalState,
} from "../api";
import { usePolling } from "../hooks/usePolling";
import { useTelemetry } from "../hooks/useTelemetry";
import { useOperatorSupervision } from "../hooks/useOperatorSupervision";
import { errorMessage, joinSentences } from "../lib/errors";
import {
  EMPTY_RUN_FIELDS,
  initialRunFields,
  parseRunOverrides,
  policyLoopBlockers,
  type RunFields,
} from "../lib/policyRun";
import { currentMode } from "../lib/safetyView";

/** C3: why a score or continue was refused with 409 for ownership or supervision. */
export const EVAL_OWNER_HINT =
  "Only the operator tab that started this evaluation, with this page open and visible, can score or continue it.";

/** A score or continue that arrived after the session moved on ("not awaiting a score"). */
export const EVAL_MOVED_ON_HINT =
  "The session had already moved on, for example the trial was recorded as interrupted when the evaluation " +
  "stopped, so nothing was recorded from this click. Eval state below shows what the server recorded.";

/**
 * The hint for a 409 on score or continue, chosen from the server's detail:
 * the owner hint only when the refusal is about the operator (wrong tab, no
 * fresh heartbeat), never for a session that simply moved on.
 */
export function evalRefusalHint(detail: string): string | null {
  const text = detail.toLowerCase();
  if (text.includes("not awaiting")) return EVAL_MOVED_ON_HINT;
  if (text.includes("operator")) return EVAL_OWNER_HINT;
  return null;
}

/** Shown when /start outlives the client timeout: the server loads the policy inside the request. */
export const EVAL_START_TIMEOUT_HINT =
  "The server is still loading the policy. The evaluation may still start: watch Eval state below, and use " +
  "Stop evaluation to cancel it.";

const MAX_EPISODES = 1000;
const MAX_EPISODE_SECONDS = 3600;

type Tone = "ok" | "warn" | "danger" | "info" | "neutral";

function phaseTone(phase: string, active: boolean): Tone {
  if (phase === "done") return "ok";
  if (phase === "aborted") return "warn";
  if (phase === "failed") return "danger";
  if (phase === "awaiting_score" || phase === "awaiting_reset") return "warn";
  return active ? "info" : "neutral";
}

const SUMMARY_TITLE: Record<string, string> = {
  done: "Evaluation finished",
  aborted: "Evaluation aborted",
  failed: "Evaluation failed",
};

const SUMMARY_TONE: Record<string, "success" | "warn" | "danger"> = {
  done: "success",
  aborted: "warn",
  failed: "danger",
};

/** The terminal report of the last session: why it ended and what it measured (C3). */
function EvalSummary({ state }: { state: EvalState }) {
  const phase = state.phase;
  const reportFailure = phase === "done" && (!!state.error || !state.report_path);
  const scored = typeof state.scored === "number" ? state.scored : null;
  const interrupted = typeof state.interrupted === "number" ? state.interrupted : null;
  const percent = `${((state.success_rate ?? 0) * 100).toFixed(1)}%`;
  const title = reportFailure
    ? "Evaluation finished with errors"
    : (SUMMARY_TITLE[phase] ?? `Evaluation ${phase}`);
  const tone = reportFailure ? "danger" : (SUMMARY_TONE[phase] ?? "info");
  return (
    <Notice tone={tone} title={title}>
      <div className="space-y-1" data-testid="eval-summary">
        {state.error ? (
          <p>
            Reason: <span className="font-mono">{state.error}</span>
          </p>
        ) : null}
        {scored !== null ? (
          <p>
            {scored} of {state.n_episodes} requested episode{state.n_episodes === 1 ? "" : "s"} scored
            {interrupted !== null ? ` · ${interrupted} interrupted` : ""}. {state.successes} succeeded:{" "}
            {scored > 0 ? `${percent} of the scored episodes` : "no episode was scored"}.
          </p>
        ) : (
          <p>
            {state.successes} of {state.episode} episodes succeeded ({percent}).
          </p>
        )}
        {state.fps || state.task ? (
          <p>
            {state.fps ? `Ran at ${state.fps} fps.` : ""}
            {state.task ? ` Task: “${state.task}”.` : ""}
          </p>
        ) : null}
        <p>
          {state.report_path ? (
            <>
              Report saved to <span className="break-all font-mono">{state.report_path}</span>.
            </>
          ) : (
            "Report was not saved."
          )}
        </p>
        {phase === "aborted" && (
          <p>Resetting the E-stop or clearing a fault does not resume an evaluation. Start a new one.</p>
        )}
      </div>
    </Notice>
  );
}

function isEvalState(value: unknown): value is EvalState {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as EvalState).active === "boolean" &&
    typeof (value as EvalState).phase === "string"
  );
}

function parseEpisodes(text: string): number | null {
  const value = Number(text);
  return text.trim() && Number.isInteger(value) && value >= 1 && value <= MAX_EPISODES ? value : null;
}

function parseSeconds(text: string): number | null {
  const value = Number(text);
  return text.trim() && Number.isFinite(value) && value > 0 && value <= MAX_EPISODE_SECONDS ? value : null;
}

type BusyAction = "start" | "stop" | "score" | "continue";

export default function Eval() {
  const supervised = useOperatorSupervision();
  const { telemetry, transportStale } = useTelemetry();
  // F28: status and control come from the shell's single poll.
  const { view, status, control, refresh: refreshSafety } = useSafety();
  const { data: models } = usePolling(getModels, 0);
  const { data: state, refresh, mutate } = usePolling(getEvalState, 1000);
  const [confirm, confirmDialog] = useConfirm();
  const cameras = status?.cameras ?? [];

  const [modelPath, setModelPath] = useState("");
  const [episodesText, setEpisodesText] = useState("10");
  const [secondsText, setSecondsText] = useState("30");
  const [runFields, setRunFields] = useState<RunFields>(EMPTY_RUN_FIELDS);
  const [busy, setBusy] = useState<BusyAction | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const selectedModel = (models ?? []).find((model) => model.path === modelPath) ?? null;
  const validation = parseRunOverrides(selectedModel, runFields);
  const nEpisodes = parseEpisodes(episodesText);
  const maxSeconds = parseSeconds(secondsText);

  const active = state?.active ?? false;
  const phase = state?.phase ?? "idle";
  const awaitingScore = phase === "awaiting_score";
  const awaitingReset = phase === "awaiting_reset";
  const terminal = !active && EVAL_TERMINAL_PHASES.includes(phase);
  const terminalReportFailure =
    terminal && phase === "done" && (!!state?.error || !state?.report_path);
  const successRate = state?.success_rate ?? 0;

  const linkLive = !transportStale && telemetry !== null;
  const mode = currentMode(telemetry, linkLive, view, status);
  const safetyBlockers = policyLoopBlockers({ linkLive, supervised, view, control, mode, evalActive: active });
  const formBlockers = [
    ...(!modelPath ? ["Choose a model."] : []),
    ...(nEpisodes === null ? [`Episodes must be a whole number from 1 to ${MAX_EPISODES}.`] : []),
    ...(maxSeconds === null ? [`Seconds per episode must be above 0 and at most ${MAX_EPISODE_SECONDS}.`] : []),
    ...(!validation.ok ? [validation.error] : []),
  ];
  const blockers = [...safetyBlockers, ...formBlockers];
  // Re-checked after the confirmation dialog: the robot may have changed meanwhile.
  const blockersRef = useRef(blockers);
  useEffect(() => {
    blockersRef.current = blockers;
  });

  const run = async (action: BusyAction, fn: () => Promise<void>, ownerScoped = false) => {
    setBusy(action);
    setErr(null);
    try {
      await fn();
    } catch (caught) {
      const message = errorMessage(caught);
      if (ownerScoped && caught instanceof ApiError && caught.status === 409) {
        const hint = evalRefusalHint(caught.message);
        setErr(hint ? joinSentences(message, hint) : message);
        // The phase may have moved on (an interrupted trial): drop stale buttons.
        refresh();
      } else if (action === "start" && caught instanceof ApiError && caught.status === 408) {
        setErr(joinSentences(message, EVAL_START_TIMEOUT_HINT));
        refresh();
      } else {
        setErr(message);
      }
    } finally {
      setBusy(null);
      refreshSafety();
    }
  };

  /** C3/F19: show the EvalState the server answered with, not a stale poll. */
  const adopt = (next: unknown) => {
    if (isEvalState(next)) mutate(next);
    else refresh();
  };

  const onSelectModel = (path: string) => {
    setModelPath(path);
    setRunFields(initialRunFields((models ?? []).find((model) => model.path === path)));
  };

  const start = async () => {
    if (blockers.length > 0 || !validation.ok || nEpisodes === null || maxSeconds === null) return;
    const overrides = validation.overrides;
    const confirmed = await confirm({
      title: "Start autonomous evaluation?",
      body: (
        <>
          <span className="font-mono">{selectedModel?.name ?? modelPath}</span> will drive the arm for up to{" "}
          {nEpisodes} episode{nEpisodes === 1 ? "" : "s"} of {maxSeconds} s each. Confirm the workspace is clear
          and the physical stop is reachable.
        </>
      ),
      confirmLabel: "Start evaluation",
      tone: "warn",
    });
    if (!confirmed) return;
    if (blockersRef.current.length > 0) {
      setErr(`Not started: ${blockersRef.current[0]}`);
      return;
    }
    await run("start", async () => {
      await startEval({ model_path: modelPath, n_episodes: nEpisodes, max_seconds: maxSeconds, ...overrides });
      refresh();
    });
  };

  const stop = () =>
    run("stop", async () => {
      await stopEval();
      refresh();
    });

  const score = (success: boolean) =>
    run("score", async () => adopt(await scoreEval(success)), true);

  const next = () => run("continue", async () => adopt(await continueEval()), true);

  return (
    <div className="space-y-6">
      {confirmDialog}
      <h1 className="text-2xl font-bold tracking-tight">Evaluation</h1>

      <div className="rounded-xl border border-cyan-400/15 bg-cyan-400/[0.04] px-4 py-3 text-xs leading-relaxed text-slate-400">
        Evaluations run as an internally supervised, single-owner policy loop. They are blocked while an
        external browser or agent lease owns the arm or E-stop is latched. The loop connects the follower when
        it starts. Keep this page open and visible for the whole evaluation: this tab&apos;s heartbeat is what the
        server checks before it starts a trial or accepts a score or the next episode. Leaving the page or hiding
        the tab stops the heartbeat; the running trial then stops within about 2 s and the evaluation aborts.
        An E-stop or fault also aborts it, and resetting does not resume it.
      </div>

      <Card title="Start evaluation">
        <div className="grid gap-4 md:grid-cols-3">
          <Field label="Model">
            {(controlProps) => (
              <select
                {...controlProps}
                value={modelPath}
                onChange={(event) => onSelectModel(event.target.value)}
                className={INPUT_CLASS}
              >
                <option value="">— select model —</option>
                {(models ?? []).map((model) => (
                  <option key={model.path} value={model.path}>
                    {model.label ? `${model.name} · ${model.label}` : model.name}
                  </option>
                ))}
              </select>
            )}
          </Field>
          <Field
            label="Episodes"
            error={nEpisodes === null ? `A whole number from 1 to ${MAX_EPISODES}.` : null}
          >
            {(controlProps) => (
              <input
                {...controlProps}
                type="number"
                inputMode="numeric"
                min={1}
                max={MAX_EPISODES}
                step={1}
                value={episodesText}
                onChange={(event) => setEpisodesText(event.target.value)}
                className={INPUT_CLASS}
              />
            )}
          </Field>
          <Field
            label="Max seconds / episode"
            error={maxSeconds === null ? `Above 0 and at most ${MAX_EPISODE_SECONDS}.` : null}
          >
            {(controlProps) => (
              // StartEvalRequest.max_seconds: gt=0, le=3600, finite. HTML has
              // no exclusive minimum, so 0 is refused by parseSeconds instead.
              <input
                {...controlProps}
                type="number"
                min={0}
                max={MAX_EPISODE_SECONDS}
                step="any"
                value={secondsText}
                onChange={(event) => setSecondsText(event.target.value)}
                className={INPUT_CLASS}
              />
            )}
          </Field>
        </div>
        {selectedModel && (
          <div className="mt-4">
            <PolicyRunFields
              model={selectedModel}
              fields={runFields}
              onChange={setRunFields}
              validation={validation}
              taskMode="edit"
            />
          </div>
        )}
        <div className="mt-4 flex flex-wrap items-center gap-3">
          <Button onClick={() => void start()} disabled={busy !== null || blockers.length > 0}>
            {busy === "start" ? "Starting…" : "Start eval"}
          </Button>
          <Button variant="danger" onClick={() => void stop()} disabled={busy === "stop"}>
            {busy === "stop" ? "Stopping…" : "Stop evaluation"}
          </Button>
        </div>
        {!active && <BlockerList className="mt-3" reasons={blockers} />}
        {err && (
          <div className="mt-3">
            <Notice tone="danger" title="Evaluation request failed">
              {err}
            </Notice>
          </div>
        )}
      </Card>

      <Card
        title="Eval state"
        actions={
          <>
            <StatusPill
              label="phase"
              value={phase.replace(/_/g, " ")}
              tone={terminalReportFailure ? "danger" : phaseTone(phase, active)}
            />
            <StatusPill label="active" value={active ? "yes" : "no"} tone={active ? "ok" : "neutral"} />
          </>
        }
      >
        <div className="space-y-4">
          <div className="flex flex-wrap gap-2">
            <StatusPill label="episode" value={`${state?.episode ?? 0}/${state?.n_episodes ?? 0}`} tone="info" />
            <StatusPill label="successes" value={state?.successes ?? 0} tone="ok" />
            {typeof state?.scored === "number" && <StatusPill label="scored" value={state.scored} tone="neutral" />}
            {typeof state?.interrupted === "number" && (
              <StatusPill
                label="interrupted"
                value={state.interrupted}
                tone={state.interrupted > 0 ? "warn" : "neutral"}
              />
            )}
            <StatusPill
              label={typeof state?.scored === "number" ? "success rate (scored)" : "success rate"}
              value={`${(successRate * 100).toFixed(1)}%`}
              tone={successRate >= 0.5 ? "ok" : "warn"}
            />
          </div>

          <div
            className="h-2.5 overflow-hidden rounded-full bg-slate-800"
            role="progressbar"
            aria-label="Success rate"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(Math.max(0, Math.min(1, successRate)) * 100)}
          >
            <div
              className="h-full rounded-full bg-linear-to-r/srgb from-emerald-500 to-cyan-400 transition-all"
              style={{ width: `${Math.max(0, Math.min(1, successRate)) * 100}%` }}
            />
          </div>

          {terminal && state && <EvalSummary state={state} />}

          {awaitingScore && (
            <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 p-5">
              <p className="mb-4 text-center text-sm font-medium text-amber-200">
                Score episode {state?.episode ?? 0}
              </p>
              <div className="flex justify-center gap-4">
                <button
                  type="button"
                  onClick={() => void score(true)}
                  disabled={busy !== null}
                  className="rounded-xl bg-emerald-500 px-10 py-4 text-lg font-extrabold uppercase tracking-wide text-slate-950 shadow-lg shadow-emerald-900/40 transition-transform hover:bg-emerald-400 focus-visible:outline-hidden focus-visible:ring-4 focus-visible:ring-emerald-300/50 active:scale-95 disabled:opacity-50"
                >
                  Success
                </button>
                <button
                  type="button"
                  onClick={() => void score(false)}
                  disabled={busy !== null}
                  className="rounded-xl bg-red-600 px-10 py-4 text-lg font-extrabold uppercase tracking-wide text-white shadow-lg shadow-red-900/40 transition-transform hover:bg-red-700 focus-visible:outline-hidden focus-visible:ring-4 focus-visible:ring-red-300/50 active:scale-95 disabled:opacity-50"
                >
                  Fail
                </button>
              </div>
            </div>
          )}
          {awaitingReset && (
            <div className="rounded-xl border border-cyan-400/30 bg-cyan-400/[0.07] p-5">
              <p className="text-center text-sm font-bold text-cyan-100">
                Reset the physical workspace before episode {(state?.episode ?? 0) + 1}
              </p>
              <ul className="mx-auto mt-3 max-w-xl list-disc space-y-1 pl-5 text-xs leading-relaxed text-slate-400">
                <li>Remove failed grasps and return every object to its marked start pose.</li>
                <li>Confirm the arm path is clear and the physical stop is reachable.</li>
                <li>Check camera views and keep people outside the workspace.</li>
              </ul>
              <Button className="mx-auto mt-4" onClick={() => void next()} disabled={busy !== null}>
                Workspace reset — start next episode
              </Button>
            </div>
          )}
        </div>
      </Card>

      {active && phase === "rollout" && (
        <Card title="Live rollout" subtitle={state?.model_path ?? undefined}>
          {cameras.length === 0 ? (
            <p className="text-sm text-muted">No cameras reported.</p>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {cameras.map((cam) => (
                <CameraView key={cam} cam={cam} active={(status?.live_cameras ?? []).includes(cam)} />
              ))}
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
