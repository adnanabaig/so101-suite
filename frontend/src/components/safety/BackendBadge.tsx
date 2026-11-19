import { BACKEND_BADGE, backendDescription, backendInfo } from "../../lib/backend";
import { useSafety } from "./SafetyProvider";

const STYLES = {
  // The normal style: commands move the physical arm.
  real: "border-slate-600 bg-slate-900 text-slate-200",
  // Clearly not the real arm: dashed and tinted.
  sim: "border-dashed border-violet-300/70 bg-violet-500/15 text-violet-100",
  mock: "border-dashed border-sky-300/70 bg-sky-500/15 text-sky-100",
} as const;

/** REAL, SIM or MOCK, always in the header, from the shared status poll. */
export default function BackendBadge() {
  const { status } = useSafety();
  const info = backendInfo(status);
  if (!info) return null;
  const description = backendDescription(info);
  return (
    <div
      data-testid="backend-badge"
      data-backend={info.kind}
      title={description}
      className={`inline-flex h-7 shrink-0 items-center rounded-md border px-2 font-mono text-[10px] font-black tracking-wider ${STYLES[info.kind]}`}
    >
      <span aria-hidden="true">{BACKEND_BADGE[info.kind]}</span>
      <span className="sr-only">{`Backend: ${description}.`}</span>
    </div>
  );
}
