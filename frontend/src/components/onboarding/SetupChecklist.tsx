import { Fragment, useRef, type ReactNode, type Ref } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, Keyboard } from "lucide-react";
import { getCalibrationStatus, getLatestPreflight, type CapabilitiesResponse } from "../../api";
import { usePolling } from "../../hooks/usePolling";
import { backendInfo } from "../../lib/backend";
import { CHECKLIST_DISMISSED_KEY, STOPS_ACKNOWLEDGED_KEY, usePersistentFlag } from "../../lib/onboarding";
import {
  isComplete,
  isRehearsalBackend,
  setupItems,
  type Fetched,
  type SetupItem,
  type SetupItemState,
} from "../../lib/setupChecklist";
import Button from "../Button";
import Card from "../Card";
import { useSafety } from "../safety/SafetyProvider";
import { useShortcuts } from "../shortcuts/ShortcutsProvider";
import { StatusPill, type StatusTone } from "../ui/StatusPill";
import { useToast } from "../ui/Toast";

/** Calibration and the preflight report change rarely; a slow refresh keeps them honest. */
export const SETUP_POLL_MS = 15_000;

export const CHECKLIST_HIDDEN_TOAST = "Setup checklist hidden. Show it again from Settings.";

const getFollowerCalibration = () => getCalibrationStatus("follower");
const getLeaderCalibration = () => getCalibrationStatus("leader");

const STATE_PILLS: Record<SetupItemState, { label: string; tone: StatusTone }> = {
  done: { label: "done", tone: "ok" },
  rehearsal: { label: "rehearsal", tone: "info" },
  todo: { label: "to do", tone: "warn" },
  problem: { label: "problem", tone: "danger" },
  optional: { label: "optional", tone: "neutral" },
  running: { label: "in progress", tone: "info" },
  checking: { label: "checking", tone: "neutral" },
  unknown: { label: "unknown", tone: "neutral" },
  unavailable: { label: "not available", tone: "neutral" },
};

/** Our summaries mark commands with backticks; show them as code. */
function withCode(text: string): ReactNode {
  return text.split("`").map((part, index) =>
    index % 2 === 1 ? (
      <code key={index} className="rounded-sm bg-slate-800 px-1 py-px font-mono text-[11px] text-slate-100">
        {part}
      </code>
    ) : (
      <Fragment key={index}>{part}</Fragment>
    ),
  );
}

function formatWhen(iso: string): string {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

function ItemRow({
  item,
  extra,
  headingRef,
}: {
  item: SetupItem;
  extra?: ReactNode;
  headingRef?: Ref<HTMLHeadingElement>;
}) {
  const pill = STATE_PILLS[item.state];
  return (
    <li className="py-3 first:pt-0 last:pb-0" data-testid={`setup-${item.id}`} data-state={item.state}>
      <div className="flex flex-wrap items-center gap-2">
        <h3 ref={headingRef} tabIndex={headingRef ? -1 : undefined} className="text-sm font-semibold text-slate-100 focus:outline-hidden">
          {item.title}
        </h3>
        <StatusPill label={pill.label} tone={pill.tone} />
        {!item.required && item.state !== "optional" && <span className="text-[11px] text-slate-400">Optional</span>}
      </div>
      <p className="mt-1 text-xs leading-relaxed text-slate-300">{withCode(item.summary)}</p>
      {item.when && (
        <p className="mt-0.5 text-[11px] text-slate-400">
          {item.when.label} <time dateTime={item.when.iso}>{formatWhen(item.when.iso)}</time>
        </p>
      )}
      {item.details.length > 0 && (
        <ul className="mt-1.5 space-y-0.5 break-words font-mono text-[11px] leading-relaxed text-slate-400">
          {item.details.map((line, index) => (
            <li key={index}>{line}</li>
          ))}
        </ul>
      )}
      {(item.link || extra) && (
        <div className="mt-2 flex flex-wrap items-center gap-2">
          {item.link && (
            <Link
              to={item.link.to}
              className="inline-flex items-center gap-1 rounded-sm text-xs font-semibold text-cyan-300 hover:text-cyan-200"
            >
              {item.link.label} <ArrowRight aria-hidden="true" className="h-3.5 w-3.5" />
            </Link>
          )}
          {extra}
        </div>
      )}
    </li>
  );
}

function Checklist({
  capabilities,
  preflightAction,
  onHide,
}: {
  capabilities: Fetched<CapabilitiesResponse>;
  preflightAction?: ReactNode;
  onHide: () => void;
}) {
  const { view, status } = useSafety();
  const shortcuts = useShortcuts();
  const follower = usePolling(getFollowerCalibration, SETUP_POLL_MS);
  const leader = usePolling(getLeaderCalibration, SETUP_POLL_MS);
  const preflight = usePolling(getLatestPreflight, SETUP_POLL_MS);
  const [stopsAcknowledged, setStopsAcknowledged] = usePersistentFlag(STOPS_ACKNOWLEDGED_KEY);
  const stopsHeading = useRef<HTMLHeadingElement>(null);

  const backend = backendInfo(capabilities.data) ?? backendInfo(status);
  const items = setupItems({
    capabilities,
    backend,
    follower,
    leader,
    preflight,
    stopsAcknowledged,
    canHalt: view.canHalt,
    canCollapse: view.canCollapse,
  });
  const required = items.filter((item) => item.required);
  // A rehearsal counts only on a mock or sim server (isComplete).
  const done = required.filter((item) => isComplete(item, backend)).length;
  const complete = done === required.length;
  const rehearsal = isRehearsalBackend(backend);

  const extras: Partial<Record<SetupItem["id"], ReactNode>> = {
    preflight: preflightAction ?? (
      <span className="text-[11px] text-slate-400">Running preflight from this page comes with the preflight wizard.</span>
    ),
    stops: (
      <>
        {!stopsAcknowledged && (
          <Button
            size="sm"
            variant="secondary"
            onClick={() => {
              setStopsAcknowledged(true);
              // The button goes away; keep focus on this item.
              stopsHeading.current?.focus();
            }}
          >
            I know the stops
          </Button>
        )}
        <Button size="sm" variant="ghost" onClick={shortcuts.open} aria-haspopup="dialog">
          <Keyboard aria-hidden="true" className="h-3.5 w-3.5" /> All keyboard shortcuts
        </Button>
      </>
    ),
  };

  return (
    <Card
      title={complete ? (rehearsal ? "Rehearsal setup complete" : "Setup complete") : "Set up this workstation"}
      subtitle={
        complete
          ? rehearsal
            ? `Everything a first run needs is in place on the ${backend?.kind} backend. None of it is evidence about a real arm.`
            : "Everything a first run needs is in place."
          : `${done} of ${required.length} required steps done. Read from the server; nothing here runs on the robot.`
      }
      actions={
        <Button size="sm" variant="ghost" onClick={onHide}>
          Hide checklist
        </Button>
      }
    >
      <ol className="divide-y divide-white/5">
        {items.map((item) => (
          <ItemRow
            key={item.id}
            item={item}
            extra={extras[item.id]}
            headingRef={item.id === "stops" ? stopsHeading : undefined}
          />
        ))}
      </ol>
    </Card>
  );
}

/**
 * First-run setup checklist for Overview. Every item comes from the server
 * (capabilities, saved calibration, the last preflight report) except the
 * stops acknowledgement. Hiding it persists in this browser; Settings shows
 * it again. `preflightAction` is the slot for the wave-2 preflight wizard.
 */
export default function SetupChecklist({
  capabilities,
  preflightAction,
}: {
  capabilities: Fetched<CapabilitiesResponse>;
  preflightAction?: ReactNode;
}) {
  const [hidden, setHidden] = usePersistentFlag(CHECKLIST_DISMISSED_KEY);
  const toast = useToast();
  if (hidden) return null;
  return (
    <Checklist
      capabilities={capabilities}
      preflightAction={preflightAction}
      onHide={() => {
        // The card and its button go away; move focus to the page.
        document.getElementById("main-content")?.focus();
        setHidden(true);
        toast.show(CHECKLIST_HIDDEN_TOAST);
      }}
    />
  );
}
