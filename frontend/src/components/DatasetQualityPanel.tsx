import {
  AlertTriangle,
  CheckCircle2,
  CircleSlash2,
  XCircle,
} from "lucide-react";
import type { DatasetQualityCheck, DatasetQualityReport } from "../api";
import Button from "./Button";
import { StatusPill } from "./ui/StatusPill";

const styles: Record<
  DatasetQualityCheck["status"],
  { border: string; text: string; Icon: typeof CheckCircle2 }
> = {
  pass: {
    border: "border-emerald-500/20 bg-emerald-500/[0.04]",
    text: "text-emerald-300",
    Icon: CheckCircle2,
  },
  warn: {
    border: "border-amber-500/20 bg-amber-500/[0.04]",
    text: "text-amber-300",
    Icon: AlertTriangle,
  },
  fail: {
    border: "border-red-500/25 bg-red-500/[0.05]",
    text: "text-red-300",
    Icon: XCircle,
  },
  skip: {
    border: "border-slate-700 bg-slate-800/30",
    text: "text-slate-400",
    Icon: CircleSlash2,
  },
};

export default function DatasetQualityPanel({
  report,
  loading,
  onRun,
}: {
  report: DatasetQualityReport | null;
  loading: boolean;
  onRun: () => void;
}) {
  if (!report) {
    return (
      <div className="flex min-h-40 flex-col items-center justify-center gap-3 text-center">
        <p className="max-w-xl text-xs leading-relaxed text-slate-400">
          Scan frame schemas, NaN/Inf values, timestamp continuity, SO-101 action
          limits, camera chunk alignment, and interrupted captures.
        </p>
        <Button onClick={onRun} disabled={loading}>
          {loading ? "Running checks…" : "Run quality control"}
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <StatusPill
            label="overall"
            value={report.overall}
            tone={
              report.overall === "pass"
                ? "ok"
                : report.overall === "fail"
                  ? "danger"
                  : "warn"
            }
          />
          <StatusPill label="frames checked" value={report.checked_frames} />
        </div>
        <Button size="sm" variant="secondary" onClick={onRun} disabled={loading}>
          {loading ? "Checking…" : "Run again"}
        </Button>
      </div>

      <div className="grid gap-2 lg:grid-cols-2">
        {report.checks.map((check) => {
          const style = styles[check.status];
          return (
            <details
              key={check.id}
              className={`group rounded-xl border px-3.5 py-3 ${style.border}`}
              open={check.status === "fail"}
            >
              <summary className="flex cursor-pointer list-none items-start gap-2.5">
                <style.Icon className={`mt-0.5 shrink-0 ${style.text}`} size={16} />
                <span className="min-w-0 flex-1">
                  <span className="flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold text-slate-200">{check.label}</span>
                    <span className={`text-[10px] font-bold uppercase tracking-wider ${style.text}`}>
                      {check.status}
                    </span>
                  </span>
                  <span className="mt-1 block text-[11px] leading-relaxed text-slate-400">
                    {check.detail}
                  </span>
                </span>
              </summary>
              {(check.examples.length > 0 || check.remediation) && (
                <div className="ml-6 mt-3 border-t border-slate-700/50 pt-2.5">
                  {check.examples.length > 0 && (
                    <ul className="space-y-1 font-mono text-[10px] text-slate-400">
                      {check.examples.map((example, index) => (
                        <li key={`${check.id}-${index}`} className="break-all">
                          {example}
                        </li>
                      ))}
                    </ul>
                  )}
                  {check.remediation && (
                    <p className="mt-2 text-[10px] leading-relaxed text-muted">
                      Next: {check.remediation}
                    </p>
                  )}
                </div>
              )}
            </details>
          );
        })}
      </div>
    </div>
  );
}
