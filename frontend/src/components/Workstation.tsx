import type { ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Info, XCircle, type LucideIcon } from "lucide-react";

export function PageHeader({
  eyebrow,
  title,
  description,
  icon: Icon,
  actions,
  headingLevel = 1,
}: {
  eyebrow?: string;
  title: string;
  description?: string;
  icon?: LucideIcon;
  actions?: ReactNode;
  headingLevel?: 1 | 2;
}) {
  const Heading = headingLevel === 1 ? "h1" : "h2";
  return (
    <div className="flex flex-col justify-between gap-4 md:flex-row md:items-end">
      <div className="flex min-w-0 items-start gap-3.5">
        {Icon && (
          <div className="mt-0.5 flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl border border-cyan-300/15 bg-cyan-400/10 text-cyan-300 shadow-lg shadow-cyan-950/20">
            <Icon aria-hidden="true" className="h-5 w-5" />
          </div>
        )}
        <div>
          {eyebrow && (
            <div className="mb-1 text-[10px] font-bold uppercase tracking-[0.22em] text-cyan-500/80">
              {eyebrow}
            </div>
          )}
          <Heading className="text-2xl font-black tracking-tight text-white sm:text-3xl">
            {title}
          </Heading>
          {description && (
            <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-400">
              {description}
            </p>
          )}
        </div>
      </div>
      {actions && <div className="flex shrink-0 flex-wrap gap-2">{actions}</div>}
    </div>
  );
}

export function MetricTile({
  label,
  value,
  detail,
  tone = "neutral",
  icon: Icon,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  tone?: "neutral" | "cyan" | "green" | "amber" | "red";
  icon?: LucideIcon;
}) {
  const tones = {
    neutral: "text-slate-300 bg-slate-400/5 border-white/5",
    cyan: "text-cyan-300 bg-cyan-400/[0.06] border-cyan-300/10",
    green: "text-emerald-300 bg-emerald-400/[0.06] border-emerald-300/10",
    amber: "text-amber-300 bg-amber-400/[0.06] border-amber-300/10",
    red: "text-rose-300 bg-rose-400/[0.06] border-rose-300/10",
  };
  return (
    <div className={`rounded-2xl border p-4 ${tones[tone]}`}>
      <div className="flex items-center justify-between gap-3">
        <span className="text-[10px] font-bold uppercase tracking-[0.16em] text-muted">
          {label}
        </span>
        {Icon && <Icon aria-hidden="true" className="h-4 w-4 opacity-70" />}
      </div>
      <div className="mt-2 text-2xl font-black tracking-tight text-current">{value}</div>
      {detail && <div className="mt-1 text-xs text-muted">{detail}</div>}
    </div>
  );
}

export function Notice({
  tone,
  title,
  children,
}: {
  tone: "info" | "success" | "warn" | "danger";
  title: string;
  children?: ReactNode;
}) {
  const styles = {
    info: "border-cyan-400/20 bg-cyan-400/[0.06] text-cyan-200",
    success: "border-emerald-400/20 bg-emerald-400/[0.06] text-emerald-200",
    warn: "border-amber-400/20 bg-amber-400/[0.06] text-amber-200",
    danger: "border-rose-400/20 bg-rose-400/[0.06] text-rose-200",
  };
  const Icon =
    tone === "success"
      ? CheckCircle2
      : tone === "danger"
        ? XCircle
        : tone === "warn"
          ? AlertTriangle
          : Info;
  return (
    <div className={`flex gap-3 rounded-xl border px-4 py-3 ${styles[tone]}`}>
      <Icon aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0" />
      <div>
        <div className="text-xs font-bold">{title}</div>
        {children && <div className="mt-1 text-xs leading-relaxed text-slate-400">{children}</div>}
      </div>
    </div>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="text-[10px] font-bold uppercase tracking-[0.2em] text-muted">
      {children}
    </div>
  );
}

export function EmptyState({
  title,
  detail,
  icon: Icon = Info,
  action,
}: {
  title: string;
  detail: string;
  icon?: LucideIcon;
  action?: ReactNode;
}) {
  return (
    <div className="flex min-h-40 flex-col items-center justify-center rounded-xl border border-dashed border-slate-700/80 bg-slate-950/30 px-6 py-8 text-center">
      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-slate-800/80 text-slate-400">
        <Icon aria-hidden="true" className="h-5 w-5" />
      </div>
      <div className="mt-3 text-sm font-semibold text-slate-200">{title}</div>
      <div className="mt-1 max-w-md text-xs leading-relaxed text-muted">{detail}</div>
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}
