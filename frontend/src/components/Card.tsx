import type { ReactNode } from "react";

interface CardProps {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode;
  className?: string;
  bodyClassName?: string;
}

export default function Card({
  title,
  subtitle,
  actions,
  children,
  className = "",
  bodyClassName = "",
}: CardProps) {
  return (
    <div
      className={`rounded-2xl border border-slate-800 bg-slate-900/60 shadow-lg shadow-black/20 ${className}`}
    >
      {(title || actions || subtitle) && (
        <div className="flex items-start justify-between gap-3 border-b border-slate-800 px-5 py-3">
          <div>
            {/* Pages own the h1; a card is a section under it. */}
            {title && (
              <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-200">
                {title}
              </h2>
            )}
            {subtitle && (
              <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p>
            )}
          </div>
          {actions && <div className="flex items-center gap-2">{actions}</div>}
        </div>
      )}
      <div className={`p-5 ${bodyClassName}`}>{children}</div>
    </div>
  );
}
