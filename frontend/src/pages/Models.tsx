import { useEffect, useRef, useState } from "react";
import Card from "../components/Card";
import Button from "../components/Button";
import { StatusPill } from "../components/ui/StatusPill";
import JointBars from "../components/JointBars";
import CameraView from "../components/CameraView";
import BlockerList from "../components/BlockerList";
import PolicyRunFields from "../components/PolicyRunFields";
import { Notice } from "../components/Workstation";
import { useSafety } from "../components/safety/SafetyProvider";
import { Dialog } from "../components/ui/Dialog";
import {
  getEvalState,
  getModels,
  startRollout,
  stopEval,
  stopRollout,
  type ModelSummary,
} from "../api";
import { usePolling } from "../hooks/usePolling";
import { useTelemetry } from "../hooks/useTelemetry";
import { useOperatorSupervision } from "../hooks/useOperatorSupervision";
import { errorMessage } from "../lib/errors";
import {
  initialRunFields,
  parseRunOverrides,
  policyLoopBlockers,
  type RunFields,
  type RunOverrides,
} from "../lib/policyRun";
import { currentMode } from "../lib/safetyView";

/** An evaluation drives the arm in rollout mode; poll it to explain why. */
export const EVAL_STATE_POLL_MS = 3000;

function trimSlashes(path: string): string {
  return path.replace(/\/+$/, "");
}

/**
 * C7: does the server's `active_model` name this row? Paths may be reported
 * absolute or relative, so a match on a path-segment boundary counts.
 */
export function isActiveModel(model: ModelSummary, active: string | null | undefined): boolean {
  if (!active) return false;
  const a = trimSlashes(active);
  const p = trimSlashes(model.path);
  return a === p || a === model.name || a.endsWith(`/${p.replace(/^\.?\//, "")}`) || p.endsWith(`/${a}`);
}

function StartRolloutDialog({
  model,
  blockers,
  busy,
  onCancel,
  onStart,
}: {
  model: ModelSummary;
  blockers: string[];
  busy: boolean;
  onCancel: () => void;
  onStart: (overrides: RunOverrides) => void;
}) {
  const [fields, setFields] = useState<RunFields>(() => initialRunFields(model));
  const validation = parseRunOverrides(model, fields);
  const cancelRef = useRef<HTMLButtonElement>(null);
  return (
    <Dialog
      open
      role="alertdialog"
      tone="warn"
      title="Start autonomous rollout?"
      initialFocusRef={cancelRef}
      description={
        <>
          <span className="font-mono">{model.name}</span> will drive the arm until you stop it. Confirm the
          workspace is clear and the physical stop is reachable.
        </>
      }
      actions={
        <>
          <Button ref={cancelRef} variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
          <Button
            onClick={() => {
              if (validation.ok) onStart(validation.overrides);
            }}
            disabled={busy || !validation.ok || blockers.length > 0}
          >
            {busy ? "Starting…" : "Start rollout"}
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <PolicyRunFields model={model} fields={fields} onChange={setFields} validation={validation} disabled={busy} />
        <BlockerList reasons={blockers} />
      </div>
    </Dialog>
  );
}

type BusyAction = "start" | "stop";

export default function Models() {
  const supervised = useOperatorSupervision();
  const { data: models, error, loading, refresh } = usePolling(getModels, 0);
  const { data: evalState } = usePolling(getEvalState, EVAL_STATE_POLL_MS);
  const { telemetry, transportStale } = useTelemetry();
  // F28: status and control come from the shell's single poll.
  const { view, status, control, refresh: refreshSafety } = useSafety();
  const cameras = status?.cameras ?? [];

  const [dialogModel, setDialogModel] = useState<ModelSummary | null>(null);
  const [busy, setBusy] = useState<BusyAction | null>(null);
  const [err, setErr] = useState<string | null>(null);

  // F10: everything about "is a rollout running" comes from the server.
  const linkLive = !transportStale && telemetry !== null;
  const mode = currentMode(telemetry, linkLive, view, status);
  const modeKnown = mode !== undefined;
  const rollingOut = mode === "rollout";
  const evalActive = evalState?.active === true;
  const activeModel = evalActive ? evalState.model_path : status?.active_model;

  const blockers = policyLoopBlockers({ linkLive, supervised, view, control, mode, evalActive });
  // Re-read when Start is pressed: the dialog may have been open a while.
  const blockersRef = useRef(blockers);
  useEffect(() => {
    blockersRef.current = blockers;
  });

  const start = async (model: ModelSummary, overrides: RunOverrides) => {
    if (blockersRef.current.length > 0) return;
    setBusy("start");
    setErr(null);
    try {
      await startRollout({ model_path: model.path, ...overrides });
      setDialogModel(null);
    } catch (caught) {
      setErr(errorMessage(caught));
      setDialogModel(null);
    } finally {
      setBusy(null);
      refreshSafety();
    }
  };

  const stop = async () => {
    setBusy("stop");
    setErr(null);
    try {
      if (evalActive) await stopEval();
      else await stopRollout();
    } catch (caught) {
      setErr(errorMessage(caught));
    } finally {
      setBusy(null);
      refreshSafety();
    }
  };

  // Shown whenever the server says a loop runs, and while the mode is unknown
  // (stopping is always allowed; hiding it would be the unsafe failure).
  const showStop = rollingOut || evalActive || !modeKnown;
  const stopLabel = evalActive ? "Stop evaluation" : "Stop rollout";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h1 className="text-2xl font-bold tracking-tight">Models</h1>
        <div className="flex items-center gap-3">
          <Button variant="secondary" onClick={refresh}>
            Refresh
          </Button>
          {showStop && (
            <Button
              variant="danger"
              onClick={() => void stop()}
              disabled={busy === "stop"}
              title={modeKnown ? undefined : "The robot mode is unknown; stopping is always allowed."}
            >
              {busy === "stop" ? "Stopping…" : stopLabel}
            </Button>
          )}
        </div>
      </div>

      {error && (
        <Notice tone="danger" title="Could not load models">
          {errorMessage(error)}
        </Notice>
      )}
      {err && (
        <Notice tone="danger" title="Rollout request failed">
          {err}
        </Notice>
      )}
      {evalActive && (
        <Notice tone="info" title="An evaluation is driving the arm">
          {evalState.model_path ? <span className="font-mono">{evalState.model_path}</span> : "A model"} is
          running episode {evalState.episode} of {evalState.n_episodes}. Score and continue it on the Evaluation
          page; the stop button here ends the whole evaluation.
        </Notice>
      )}

      <div className="rounded-xl border border-cyan-400/15 bg-cyan-400/[0.04] px-4 py-3 text-xs leading-relaxed text-slate-400">
        Policy rollout is an internally supervised, single-owner robot loop. It connects the follower when
        started and is blocked when E-stop is latched or an external browser or agent lease owns the arm. FPS
        and, for language policies, the task can be set per run.
      </div>

      {!rollingOut && !evalActive && <BlockerList reasons={blockers} />}

      <Card bodyClassName="p-0">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-left text-xs uppercase tracking-wide text-muted">
                <th scope="col" className="px-5 py-3 font-medium">name</th>
                <th scope="col" className="px-5 py-3 font-medium">policy</th>
                <th scope="col" className="px-5 py-3 font-medium">mock</th>
                <th scope="col" className="px-5 py-3 font-medium">checkpoint</th>
                <th scope="col" className="px-5 py-3 font-medium">path</th>
                <th scope="col" className="px-5 py-3 text-right font-medium">actions</th>
              </tr>
            </thead>
            <tbody>
              {(!models || models.length === 0) && (
                <tr>
                  <td colSpan={6} className="px-5 py-6 text-center text-muted">
                    {loading && !models
                      ? "Loading models…"
                      : error && !models
                        ? "Models could not be loaded."
                        : "No models found. Checkpoints appear here when a training job finishes."}
                  </td>
                </tr>
              )}
              {(models ?? []).map((m) => {
                const isActive = (rollingOut || evalActive) && isActiveModel(m, activeModel);
                return (
                  <tr
                    key={m.path}
                    aria-current={isActive ? "true" : undefined}
                    className={`border-b border-slate-800/60 hover:bg-slate-800/30 ${isActive ? "bg-emerald-500/5" : ""}`}
                  >
                    <td className="px-5 py-3">
                      <div className="font-semibold text-slate-100">{m.name}</div>
                      {m.label && <div className="text-xs text-muted">{m.label}</div>}
                    </td>
                    <td className="px-5 py-3 text-slate-300">
                      {m.policy_type}
                      {m.privileged_state && (
                        <StatusPill className="ml-2" label="" value="privileged state" tone="warn" />
                      )}
                    </td>
                    <td className="px-5 py-3">
                      <StatusPill label="" value={m.is_mock ? "mock" : "real"} tone={m.is_mock ? "warn" : "ok"} />
                    </td>
                    <td className="px-5 py-3 font-mono text-xs text-slate-400">{m.checkpoint || "—"}</td>
                    <td className="px-5 py-3 font-mono text-xs text-muted">{m.path}</td>
                    <td className="px-5 py-3 text-right">
                      {isActive ? (
                        <StatusPill label="" value={evalActive ? "evaluating" : "running"} tone="ok" />
                      ) : (
                        <Button
                          size="sm"
                          onClick={() => {
                            setErr(null);
                            setDialogModel(m);
                          }}
                          disabled={busy !== null || blockers.length > 0}
                          aria-label={`Start rollout of ${m.name}`}
                        >
                          Start rollout
                        </Button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {dialogModel && (
        <StartRolloutDialog
          key={dialogModel.path}
          model={dialogModel}
          blockers={blockers}
          busy={busy === "start"}
          onCancel={() => setDialogModel(null)}
          onStart={(overrides) => void start(dialogModel, overrides)}
        />
      )}

      {(rollingOut || evalActive) && (
        <Card
          title={evalActive ? "Live evaluation" : "Live rollout"}
          subtitle={activeModel ?? "This server does not report which model is running."}
          actions={<StatusPill label="mode" value={mode ?? "—"} tone="info" />}
        >
          <div className="grid gap-6 lg:grid-cols-2">
            <JointBars observation={telemetry?.observation} action={telemetry?.action} />
            <div className="grid gap-3 sm:grid-cols-2">
              {cameras.length === 0 ? (
                <p className="text-sm text-muted">No cameras reported.</p>
              ) : (
                cameras.map((cam) => (
                  <CameraView key={cam} cam={cam} active={(status?.live_cameras ?? []).includes(cam)} />
                ))
              )}
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
