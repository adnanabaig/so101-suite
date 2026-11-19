interface BlockerListProps {
  reasons: string[];
  /** Lead-in for the first reason. */
  title?: string;
  className?: string;
}

/**
 * Every reason an action is unavailable, first one emphasised. Disabled
 * buttons alone do not tell an operator what to fix.
 */
export default function BlockerList({ reasons, title = "Start is blocked", className = "" }: BlockerListProps) {
  if (reasons.length === 0) return null;
  return (
    <div className={`text-xs leading-relaxed text-slate-400 ${className}`}>
      <span className="font-semibold text-amber-200">{title}: </span>
      {reasons[0]}
      {reasons.length > 1 && (
        <ul className="mt-1 list-disc space-y-0.5 pl-5 text-muted">
          {reasons.slice(1).map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
