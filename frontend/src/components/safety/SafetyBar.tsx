import { useEffect, useId, useRef, useState, type PointerEvent } from "react";
import { NavLink, useLocation } from "react-router-dom";
import { ArrowDownToLine, ChevronDown, Hand, OctagonX } from "lucide-react";
import { useNow } from "../../hooks/useNow";
import type { SafetyTone } from "../../lib/safetyView";
import { useSafety } from "./SafetyProvider";

/** A click this soon after a pointerdown that already fired is the same press. */
const CLICK_DEDUPE_MS = 1000;

const CHIP_TONES: Record<SafetyTone, string> = {
  danger: "border-rose-400/50 bg-rose-500/15 text-rose-100",
  warn: "border-amber-400/50 bg-amber-400/10 text-amber-100",
  neutral: "border-slate-700 bg-slate-900 text-slate-300",
  unknown: "border-dashed border-slate-600 bg-slate-900 text-slate-400",
};

const STOP_NAMES = { estop: "E-stop", halt: "Halt", collapse: "Collapse" } as const;

function latchPersistenceLabel(value: string | undefined, stale: boolean): string {
  if (stale) return "unknown — status stale";
  if (value === "saved") return "durable store";
  if (value === "marker") return "fallback marker";
  if (value === "unsaved") return "memory only — do not restart";
  return "unknown";
}

/**
 * Fires on pointerdown (no wait for release) and on click, so keyboard and
 * assistive-tech activation work too, without double-firing one press.
 */
function useImmediatePress(action: () => void) {
  const lastPointerFire = useRef(-Infinity);
  return {
    onPointerDown: (event: PointerEvent<HTMLButtonElement>) => {
      lastPointerFire.current = event.timeStamp;
      action();
    },
    onClick: (event: { timeStamp: number }) => {
      if (event.timeStamp - lastPointerFire.current < CLICK_DEDUPE_MS) return;
      action();
    },
  };
}

/**
 * E-stop. Always enabled, including while its own request is in flight (F6),
 * and alone in its slot so nothing else can ever occupy its position (F2).
 */
export function EstopButton() {
  const { fireEstop, estopPending } = useSafety();
  const press = useImmediatePress(() => fireEstop("button"));
  return (
    <div className="shrink-0" data-slot="estop">
      <button
        type="button"
        aria-label="E-stop"
        aria-keyshortcuts="Escape"
        title="E-stop: kill all motor power now (Escape)"
        className="relative inline-flex h-11 w-[6.75rem] items-center justify-center gap-1.5 rounded-xl border border-rose-300/50 bg-rose-600 text-xs font-black uppercase tracking-[0.14em] text-white shadow-lg shadow-rose-950/40 transition hover:bg-rose-700 focus-visible:outline-hidden focus-visible:ring-4 focus-visible:ring-rose-300/70 active:bg-rose-800"
        onContextMenu={(event) => event.preventDefault()}
        {...press}
      >
        <OctagonX aria-hidden="true" className="h-4 w-4" />
        E-stop
        {estopPending && (
          <span
            aria-hidden="true"
            data-testid="estop-pending"
            className="absolute right-1.5 top-1.5 h-1.5 w-1.5 motion-safe:animate-ping rounded-full bg-white"
          />
        )}
      </button>
    </div>
  );
}

/** Halt: freeze in place with motors holding. Amber; H key; every width. */
export function HaltButton({ label = "Halt" }: { label?: string }) {
  const { fireHalt, view } = useSafety();
  const press = useImmediatePress(() => fireHalt("button"));
  return (
    <button
      type="button"
      aria-label={label}
      aria-keyshortcuts="h"
      title={
        view.stopState === "collapsing"
          ? "Halt stops the descent and holds position (H)"
          : "Halt: freeze in place; motors stay on and hold position (H)"
      }
      className="inline-flex h-11 shrink-0 items-center gap-1.5 rounded-xl border border-amber-300/60 bg-amber-400 px-3 text-xs font-black uppercase tracking-[0.12em] text-slate-950 shadow-lg shadow-amber-950/30 transition hover:bg-amber-300 focus-visible:outline-hidden focus-visible:ring-4 focus-visible:ring-amber-200/70 active:bg-amber-500"
      {...press}
    >
      <Hand aria-hidden="true" className="h-4 w-4" />
      {label}
      <kbd
        aria-hidden="true"
        className="hidden rounded-sm border border-slate-950/30 px-1 font-mono text-[10px] sm:inline"
      >
        H
      </kbd>
    </button>
  );
}

/**
 * Collapse: lower to rest at a capped speed, then motors off. Deliberately
 * neutral and click-only so it is never mistaken for E-stop.
 */
export function CollapseButton({ className = "", onDone }: { className?: string; onDone?: () => void }) {
  const { fireCollapse } = useSafety();
  return (
    <button
      type="button"
      title="Collapse: lower the arm to rest at a capped speed, then turn motors off"
      className={`h-10 shrink-0 items-center gap-1.5 rounded-lg border border-slate-600 bg-slate-800 px-3 text-xs font-semibold text-slate-200 transition hover:border-slate-500 hover:bg-slate-700 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-slate-300/60 ${className}`}
      onClick={() => {
        fireCollapse("button");
        onDone?.();
      }}
    >
      <ArrowDownToLine aria-hidden="true" className="h-4 w-4" />
      Collapse
    </button>
  );
}

/** Legacy POST /api/robot/stop, shown only on servers without Halt. */
function SoftStopButton({ className = "", onDone }: { className?: string; onDone?: () => void }) {
  const { softStop } = useSafety();
  return (
    <button
      type="button"
      title="Return the robot to idle (servers without Halt)"
      className={`h-10 shrink-0 items-center rounded-lg border border-slate-700 bg-slate-900 px-3 text-xs font-semibold text-slate-300 transition hover:border-slate-600 hover:bg-slate-800 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-slate-300/60 ${className}`}
      onClick={() => {
        softStop();
        onDone?.();
      }}
    >
      Soft stop
    </button>
  );
}

function SafetyChip() {
  const { view, status, statusStale, statusUpdatedAt } = useSafety();
  const [open, setOpen] = useState(false);
  const popoverId = useId();
  const wrapperRef = useRef<HTMLDivElement>(null);
  const location = useLocation();
  const now = useNow(1000);

  // Navigating closes the popover.
  const [pathname, setPathname] = useState(location.pathname);
  if (pathname !== location.pathname) {
    setPathname(location.pathname);
    setOpen(false);
  }

  useEffect(() => {
    if (!open) return;
    const onPointerDown = (event: Event) => {
      if (!wrapperRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", onPointerDown);
    return () => document.removeEventListener("pointerdown", onPointerDown);
  }, [open]);

  const statusAge =
    statusUpdatedAt === null ? null : Math.max(0, Math.round((now - statusUpdatedAt) / 1000));
  const sourceText =
    view.source === "telemetry"
      ? "Live telemetry"
      : !view.known
        ? statusAge === null
          ? "No response from the robot server yet"
          : `No fresh source (status ${statusAge}s old)`
        : `Status poll, ${statusAge ?? 0}s ago (telemetry link down)`;

  return (
    <div ref={wrapperRef} className={`min-w-0 max-w-full ${open ? "basis-full" : ""}`}>
      <button
        type="button"
        data-testid="safety-chip"
        aria-expanded={open}
        aria-controls={popoverId}
        onClick={() => setOpen((value) => !value)}
        className={`inline-flex h-10 max-w-[9.5rem] items-center gap-1 rounded-lg border px-2.5 text-[10px] font-bold uppercase tracking-wider sm:max-w-[14rem] ${CHIP_TONES[view.tone]}`}
      >
        <span className="sr-only">Safety state: </span>
        <span className="truncate">{view.chipLabel}</span>
        <ChevronDown aria-hidden="true" className={`h-3 w-3 shrink-0 transition ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div
          id={popoverId}
          className="mt-2 w-full max-w-72 break-words rounded-xl border border-slate-700 bg-slate-950 p-3 text-xs text-slate-300"
        >
          <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-1.5">
            <dt className="text-muted">State</dt>
            <dd className="font-semibold text-slate-100">{view.chipLabel}</dd>
            <dt className="text-muted">Control</dt>
            <dd>{view.safetyState.replace(/_/g, " ")}</dd>
            <dt className="text-muted">Torque</dt>
            <dd>{view.torqueState}</dd>
            <dt className="text-muted">Latch storage</dt>
            <dd className={status?.latch_persistence === "unsaved" ? "font-semibold text-rose-200" : undefined}>
              {latchPersistenceLabel(status?.latch_persistence, statusStale)}
            </dd>
            <dt className="text-muted">Source</dt>
            <dd>{sourceText}</dd>
            <dt className="text-muted">Stops</dt>
            <dd>{view.stopKinds.map((kind) => STOP_NAMES[kind]).join(", ")}</dd>
          </dl>
          <div className="mt-3 flex flex-wrap gap-2">
            {view.canCollapse && (
              <CollapseButton className="inline-flex sm:hidden" onDone={() => setOpen(false)} />
            )}
            {!view.canHalt && (
              <SoftStopButton className="inline-flex sm:hidden" onDone={() => setOpen(false)} />
            )}
            <NavLink
              to="/diagnostics"
              className="inline-flex h-10 items-center rounded-lg border border-slate-700 px-3 font-semibold text-cyan-200 hover:bg-slate-800"
            >
              Open diagnostics
            </NavLink>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * The persistent safety bar: state chip, then the stops ordered from least to
 * most drastic, with E-stop always last and always in the same place.
 */
export function SafetyDetails() {
  const { view } = useSafety();
  return <div aria-label="Robot state and rest controls" role="group" className="flex flex-wrap items-center gap-2">
    <SafetyChip />
    {view.canCollapse && <CollapseButton className="inline-flex" />}
    {!view.canHalt && <SoftStopButton className="inline-flex" />}
  </div>;
}

export default function SafetyBar({ compact = false }: { compact?: boolean }) {
  const { view } = useSafety();
  return (
    <div role="group" aria-label="Robot stops" className="flex shrink-0 items-center gap-1.5 sm:gap-2">
      {!compact && <SafetyDetails />}
      {view.canHalt && <HaltButton />}
      <EstopButton />
    </div>
  );
}
