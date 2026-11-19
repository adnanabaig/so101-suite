import { modelNeedsTask, type ModelSummary } from "../api";
import {
  MAX_POLICY_FPS,
  MAX_TASK_LENGTH,
  MIN_POLICY_FPS,
  type RunFields,
  type RunOverridesResult,
} from "../lib/policyRun";
import { Field, INPUT_CLASS } from "./ui/Field";

interface PolicyRunFieldsProps {
  model: ModelSummary | null | undefined;
  fields: RunFields;
  onChange: (fields: RunFields) => void;
  /** Output of parseRunOverrides for `fields`, for inline errors. */
  validation: RunOverridesResult;
  disabled?: boolean;
  /**
   * "edit" (default) offers the task as an editable override. "checkpoint"
   * presents the checkpoint's own task read-only for a caller that does not
   * offer overrides.
   */
  taskMode?: "edit" | "checkpoint";
}

/**
 * Optional FPS and Task overrides for a rollout or evaluation. The Task field
 * appears only for language-conditioned policies (smolvla, pi0) and is
 * prefilled with the task the checkpoint was trained on when the server
 * reports it.
 */
export default function PolicyRunFields({
  model,
  fields,
  onChange,
  validation,
  disabled,
  taskMode = "edit",
}: PolicyRunFieldsProps) {
  const language = modelNeedsTask(model);
  const fpsError = !validation.ok && validation.field === "fps" ? validation.error : null;
  const taskError = !validation.ok && validation.field === "task" ? validation.error : null;
  const fpsHint =
    typeof model?.fps === "number"
      ? `Trained at ${model.fps} fps. Blank lets the server choose: the training rate where it supports that, else 30.`
      : "Blank lets the server choose: the checkpoint's training rate where it supports that, else 30.";
  const taskHint = model?.task
    ? "Prefilled with the training task. Edit it to give a different instruction; blank uses the training task."
    : "Language instruction for the policy. Blank uses the checkpoint's training task.";
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <Field label="FPS (optional)" hint={fpsHint} error={fpsError}>
        {(control) => (
          <input
            {...control}
            type="number"
            inputMode="numeric"
            min={MIN_POLICY_FPS}
            max={MAX_POLICY_FPS}
            step={1}
            placeholder="server default"
            value={fields.fps}
            disabled={disabled}
            onChange={(event) => onChange({ ...fields, fps: event.target.value })}
            className={INPUT_CLASS}
          />
        )}
      </Field>
      {language && taskMode === "checkpoint" && (
        <div className="rounded-lg border border-slate-700/60 bg-slate-900/40 px-3 py-2 text-xs leading-relaxed text-slate-400 sm:col-span-2">
          <span className="font-semibold text-slate-300">Task:</span>{" "}
          {model?.task ? (
            <>
              this run uses the task the checkpoint was trained on, &ldquo;
              <span className="text-slate-200">{model.task}</span>&rdquo;.
            </>
          ) : (
            "this run uses the task the checkpoint was trained on."
          )}{" "}
          This run does not offer a task override.
        </div>
      )}
      {language && taskMode === "edit" && (
        <Field label="Task (optional)" hint={taskHint} error={taskError} className="sm:col-span-2">
          {(control) => (
            <textarea
              {...control}
              rows={2}
              maxLength={MAX_TASK_LENGTH}
              placeholder="e.g. Pick up the blue cube and place it in the bowl"
              value={fields.task}
              disabled={disabled}
              onChange={(event) => onChange({ ...fields, task: event.target.value })}
              className={`${INPUT_CLASS} resize-y`}
            />
          )}
        </Field>
      )}
    </div>
  );
}
