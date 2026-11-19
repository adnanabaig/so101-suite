import { useId, type ReactNode } from "react";

/** Shared look for text, number and select controls inside a Field. */
export const INPUT_CLASS =
  "w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-100 outline-hidden transition " +
  "placeholder:text-muted focus-visible:border-cyan-400 focus-visible:ring-2 focus-visible:ring-cyan-400/40 " +
  "disabled:cursor-not-allowed disabled:opacity-50 aria-[invalid=true]:border-rose-400/70";

/** Props a Field hands to its control so the label, hint and error are announced. */
export interface FieldControlProps {
  id: string;
  "aria-describedby"?: string;
  "aria-invalid"?: true;
}

export interface FieldProps {
  label: ReactNode;
  /** Persistent help text under the control. */
  hint?: ReactNode;
  /** Validation message; marks the control invalid. */
  error?: string | null;
  className?: string;
  /** Render the control with the wiring props spread onto it. */
  children: (control: FieldControlProps) => ReactNode;
}

/**
 * Label + control + hint + error, wired for assistive technology:
 *
 *   <Field label="FPS" hint="Blank uses the default" error={fpsError}>
 *     {(control) => <input {...control} className={INPUT_CLASS} />}
 *   </Field>
 */
export function Field({ label, hint, error, className = "", children }: FieldProps) {
  const id = useId();
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const describedBy = [hint ? hintId : null, error ? errorId : null].filter(Boolean).join(" ");
  return (
    <div className={`flex min-w-0 flex-col gap-1 ${className}`}>
      <label htmlFor={id} className="text-xs font-medium text-slate-400">
        {label}
      </label>
      {children({
        id,
        ...(describedBy ? { "aria-describedby": describedBy } : {}),
        ...(error ? { "aria-invalid": true as const } : {}),
      })}
      {hint && (
        <p id={hintId} className="text-[11px] leading-relaxed text-muted">
          {hint}
        </p>
      )}
      {error && (
        <p id={errorId} className="text-[11px] font-medium leading-relaxed text-rose-300">
          {error}
        </p>
      )}
    </div>
  );
}
