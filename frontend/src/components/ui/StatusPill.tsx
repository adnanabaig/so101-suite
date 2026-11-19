import type { ReactNode } from "react";

export type StatusTone = "neutral" | "ok" | "warn" | "danger" | "info";

/** Pill colours per tone. Every pairing keeps at least 4.5:1 text contrast. */
export const STATUS_TONE_CLASSES: Record<StatusTone, string> = {
  neutral: "bg-slate-800 text-slate-300 border-slate-700",
  ok: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  warn: "bg-amber-500/15 text-amber-300 border-amber-500/30",
  danger: "bg-red-500/15 text-red-300 border-red-500/30",
  info: "bg-cyan-500/15 text-cyan-300 border-cyan-500/30",
};

export interface StatusPillProps {
  label: ReactNode;
  value?: ReactNode;
  tone?: StatusTone;
  className?: string;
}

function hasContent(node: ReactNode): boolean {
  return node !== "" && node !== null && node !== undefined && node !== false;
}

/**
 * A short "label value" status, such as "lease owned". The tone is never the
 * only signal: the value is always words. Screen readers hear "lease: owned".
 */
export function StatusPill({ label, value, tone = "neutral", className = "" }: StatusPillProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${STATUS_TONE_CLASSES[tone]} ${className}`}
    >
      {value === undefined ? (
        // A lone word ("done") is the whole status: full strength.
        <span className="font-semibold">{label}</span>
      ) : (
        <>
          {hasContent(label) && (
            // 80% keeps the dimmed label at 4.9:1 or better on slate-800 to 950.
            <span data-pill-label="" className="opacity-80">
              {label}
              <span className="sr-only">:</span>
            </span>
          )}
          <span className="font-semibold">{value}</span>
        </>
      )}
    </span>
  );
}

/** A yes/no state as a toned pill; `invert` when "yes" is the bad answer. */
export function BoolPill({
  label,
  value,
  invert = false,
}: {
  label: string;
  value: boolean;
  invert?: boolean;
}) {
  const good = invert ? !value : value;
  return <StatusPill label={label} value={value ? "yes" : "no"} tone={good ? "ok" : "danger"} />;
}
