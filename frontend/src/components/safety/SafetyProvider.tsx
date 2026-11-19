import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  ApiError,
  collapseRobot,
  estop,
  getControlState,
  getRobotStatus,
  haltRobot,
  resetEstop,
  stopRobot,
  type ControlState,
  type RobotStatus,
} from "../../api";
import { useNow } from "../../hooks/useNow";
import { usePolling } from "../../hooks/usePolling";
import { useTelemetry } from "../../hooks/useTelemetry";
import { errorMessage, joinSentences } from "../../lib/errors";
import { isEditableTarget } from "../../lib/keyboard";
import { deriveSafetyView, type SafetyView } from "../../lib/safetyView";

/** After any E-stop, Reset stays locked this long so a double tap cannot clear it. */
export const RESET_LOCKOUT_MS = 3000;
export const STATUS_POLL_MS = 1500;
export const CONTROL_POLL_MS = 2000;

/**
 * The E-stop route sets the latch before it answers, so a 503 means "latched,
 * but torque-off is not confirmed (yet)". That is shown as latched plus cut
 * power, never as a failed E-stop.
 */
export const ESTOP_UNCONFIRMED_TITLE = "E-stop latched, torque-off not confirmed: cut physical power";

export interface SafetyAlert {
  id: string;
  severity: "critical" | "warning";
  title: string;
  message: string;
}

export type StopOrigin = "button" | "keyboard" | "banner";

export interface SafetyContextValue {
  /** False for the inert default used outside the provider (isolated tests). */
  available: boolean;
  view: SafetyView;
  status: RobotStatus | null;
  control: ControlState | null;
  statusUpdatedAt: number | null;
  /** The retained HTTP status has failed its freshness threshold. */
  statusStale: boolean;
  controlUpdatedAt: number | null;
  /** Why the latest status poll failed; null once one succeeds. */
  statusError: Error | null;
  /** An E-stop request is in flight. The button stays enabled regardless (F6). */
  estopPending: boolean;
  /** Reset is refused before this client time. */
  lockoutUntil: number | null;
  /** Failed safety requests, shown until acknowledged. */
  alerts: SafetyAlert[];
  fireEstop: (origin: StopOrigin) => void;
  fireHalt: (origin: StopOrigin, reason?: string) => void;
  fireCollapse: (origin: StopOrigin) => void;
  /** Legacy POST /api/robot/stop, for servers without C11 Halt. */
  softStop: () => void;
  resetEstop: () => Promise<boolean>;
  dismissAlert: (id: string) => void;
  /** Ask polite assistive technology to read a message. */
  announce: (message: string) => void;
  refresh: () => void;
}

const noop = () => undefined;

const DEFAULT_VIEW = deriveSafetyView({
  telemetry: null,
  transportStale: true,
  telemetryAt: null,
  status: null,
  statusAt: null,
  statusStale: false,
  control: null,
  controlStale: false,
  now: 0,
});

const DEFAULT_CONTEXT: SafetyContextValue = {
  available: false,
  view: DEFAULT_VIEW,
  status: null,
  control: null,
  statusUpdatedAt: null,
  statusStale: true,
  controlUpdatedAt: null,
  statusError: null,
  estopPending: false,
  lockoutUntil: null,
  alerts: [],
  fireEstop: noop,
  fireHalt: noop,
  fireCollapse: noop,
  softStop: noop,
  resetEstop: () => Promise.resolve(false),
  dismissAlert: noop,
  announce: noop,
  refresh: noop,
};

const SafetyContext = createContext<SafetyContextValue>(DEFAULT_CONTEXT);

function upsert(alerts: SafetyAlert[], alert: SafetyAlert): SafetyAlert[] {
  return [...alerts.filter((entry) => entry.id !== alert.id), alert];
}

/**
 * F28: the derived view keeps its identity while its content is unchanged, so
 * a 20 Hz telemetry stream does not re-render every safety consumer (the bar,
 * banners, jog panel, camera tiles and pages) on every frame.
 */
function useStableView(next: SafetyView): SafetyView {
  const key = JSON.stringify(next);
  const [stable, setStable] = useState({ key, view: next });
  if (stable.key !== key) {
    setStable({ key, view: next });
    return next;
  }
  return stable.view;
}

function without(alerts: SafetyAlert[], ...ids: string[]): SafetyAlert[] {
  const next = alerts.filter((entry) => !ids.includes(entry.id));
  return next.length === alerts.length ? alerts : next;
}

/**
 * Owns the shell's view of robot safety: one status poll, one control poll,
 * the derived safety view, the stop actions, global Escape/H keys and the
 * polite live region. Mounted once by Layout.
 */
export function SafetyProvider({ children }: { children: ReactNode }) {
  const { telemetry, transportStale, lastMessageAt } = useTelemetry();
  const statusPoll = usePolling(getRobotStatus, STATUS_POLL_MS);
  const controlPoll = usePolling(getControlState, CONTROL_POLL_MS);
  const refreshStatus = statusPoll.refresh;
  const refreshControl = controlPoll.refresh;
  const now = useNow(1000);

  const derivedView = useMemo(
    () =>
      deriveSafetyView({
        telemetry,
        transportStale,
        telemetryAt: lastMessageAt,
        status: statusPoll.data,
        statusAt: statusPoll.lastSuccessAt,
        statusStale: statusPoll.stale,
        control: controlPoll.data,
        controlStale: controlPoll.stale,
        now,
      }),
    [
      telemetry,
      transportStale,
      lastMessageAt,
      statusPoll.data,
      statusPoll.lastSuccessAt,
      statusPoll.stale,
      controlPoll.data,
      controlPoll.stale,
      now,
    ],
  );
  const view = useStableView(derivedView);

  const [estopInFlight, setEstopInFlight] = useState(0);
  const [lockoutUntil, setLockoutUntil] = useState<number | null>(null);
  const [alerts, setAlerts] = useState<SafetyAlert[]>([]);
  const [notice, setNotice] = useState("");

  // A latch observed from any source (another station, a fault) also starts
  // the reset lockout, measured from the newest evidence we have.
  const [seenTripped, setSeenTripped] = useState(view.tripped);
  if (seenTripped !== view.tripped) {
    setSeenTripped(view.tripped);
    if (view.tripped) {
      const evidenceAt = Math.max(now, lastMessageAt ?? 0, statusPoll.lastSuccessAt ?? 0);
      setLockoutUntil((current) => Math.max(current ?? 0, evidenceAt + RESET_LOCKOUT_MS));
    }
  }

  const refresh = useCallback(() => {
    refreshStatus();
    refreshControl();
  }, [refreshStatus, refreshControl]);

  const fireEstop = useCallback(
    (origin: StopOrigin) => {
      setEstopInFlight((count) => count + 1);
      setLockoutUntil(Date.now() + RESET_LOCKOUT_MS);
      setNotice(origin === "keyboard" ? "E-stop sent from the Escape key." : "E-stop sent.");
      void (async () => {
        try {
          await estop();
          setAlerts((current) => without(current, "estop-failed", "estop-unconfirmed"));
          setNotice("E-stop latched. Motors off.");
        } catch (caught) {
          if (caught instanceof ApiError && caught.status === 503) {
            setAlerts((current) =>
              upsert(without(current, "estop-failed"), {
                id: "estop-unconfirmed",
                severity: "critical",
                title: ESTOP_UNCONFIRMED_TITLE,
                message: errorMessage(caught),
              }),
            );
            setNotice("E-stop latched. Torque-off not confirmed: cut physical power if anything moves.");
            return;
          }
          setAlerts((current) =>
            upsert(current, {
              id: "estop-failed",
              severity: "critical",
              title: "E-stop request failed",
              message: joinSentences(
                errorMessage(caught),
                "Cut physical power if the arm is moving or you cannot confirm the motors are off.",
              ),
            }),
          );
        } finally {
          setEstopInFlight((count) => Math.max(0, count - 1));
          refresh();
        }
      })();
    },
    [refresh],
  );

  const fireHalt = useCallback(
    (origin: StopOrigin, reason?: string) => {
      setNotice("Halt sent.");
      void (async () => {
        try {
          await haltRobot(
            reason ?? (origin === "keyboard" ? "operator pressed H" : "operator pressed Halt"),
          );
          setAlerts((current) => without(current, "halt-failed"));
          setNotice("Halted. Holding position.");
        } catch (caught) {
          setAlerts((current) =>
            upsert(current, {
              id: "halt-failed",
              severity: "critical",
              title: "Halt request failed",
              message: joinSentences(errorMessage(caught), "Use E-stop if the arm must stop now."),
            }),
          );
        } finally {
          refresh();
        }
      })();
    },
    [refresh],
  );

  const collapseInFlight = useRef(false);
  const fireCollapse = useCallback(
    (origin: StopOrigin) => {
      if (collapseInFlight.current) return;
      collapseInFlight.current = true;
      setNotice("Collapse requested. Lowering to rest.");
      void (async () => {
        try {
          await collapseRobot(
            origin === "banner" ? "operator chose Collapse now" : "operator pressed Collapse",
          );
          setAlerts((current) => without(current, "collapse-failed"));
          setNotice("Collapsing. Lowering to rest, then motors off.");
        } catch (caught) {
          setAlerts((current) =>
            upsert(current, {
              id: "collapse-failed",
              severity: "warning",
              title: "Collapse request failed",
              message: joinSentences(
                errorMessage(caught),
                "The arm was not lowered. Halt holds it in place; E-stop cuts motor power.",
              ),
            }),
          );
        } finally {
          collapseInFlight.current = false;
          refresh();
        }
      })();
    },
    [refresh],
  );

  const softStop = useCallback(() => {
    setNotice("Soft stop sent.");
    void (async () => {
      try {
        await stopRobot();
        setAlerts((current) => without(current, "stop-failed"));
        setNotice("Soft stop complete. The robot is idle.");
      } catch (caught) {
        setAlerts((current) =>
          upsert(current, {
            id: "stop-failed",
            severity: "critical",
            title: "Soft stop failed",
            message: joinSentences(errorMessage(caught), "Use E-stop."),
          }),
        );
      } finally {
        refresh();
      }
    })();
  }, [refresh]);

  const doResetEstop = useCallback(async () => {
    try {
      await resetEstop();
      setAlerts((current) => without(current, "estop-failed", "estop-unconfirmed", "reset-failed"));
      setNotice("E-stop reset. The robot is idle and disarmed.");
      return true;
    } catch (caught) {
      setAlerts((current) =>
        upsert(current, {
          id: "reset-failed",
          severity: "warning",
          title: "E-stop reset refused",
          message: errorMessage(caught),
        }),
      );
      return false;
    } finally {
      refresh();
    }
  }, [refresh]);

  const dismissAlert = useCallback((id: string) => {
    setAlerts((current) => without(current, id));
  }, []);

  const announce = useCallback((message: string) => setNotice(message), []);

  // Global keys. Escape is E-stop everywhere, including text fields; H halts
  // only outside text entry and only when the server implements Halt.
  const fireEstopRef = useRef(fireEstop);
  const fireHaltRef = useRef(fireHalt);
  const canHaltRef = useRef(view.canHalt);
  useEffect(() => {
    fireEstopRef.current = fireEstop;
    fireHaltRef.current = fireHalt;
    canHaltRef.current = view.canHalt;
  });
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        if (!event.repeat) fireEstopRef.current("keyboard");
        return;
      }
      if (
        (event.key === "h" || event.key === "H") &&
        !event.repeat &&
        !event.ctrlKey &&
        !event.metaKey &&
        !event.altKey &&
        canHaltRef.current &&
        !isEditableTarget(event.target)
      ) {
        event.preventDefault();
        fireHaltRef.current("keyboard");
      }
    };
    window.addEventListener("keydown", onKeyDown, true);
    return () => window.removeEventListener("keydown", onKeyDown, true);
  }, []);

  const value = useMemo<SafetyContextValue>(
    () => ({
      available: true,
      view,
      status: statusPoll.data,
      control: controlPoll.data,
      statusUpdatedAt: statusPoll.lastSuccessAt,
      statusStale: statusPoll.stale,
      controlUpdatedAt: controlPoll.lastSuccessAt,
      statusError: statusPoll.error,
      estopPending: estopInFlight > 0,
      lockoutUntil,
      alerts,
      fireEstop,
      fireHalt,
      fireCollapse,
      softStop,
      resetEstop: doResetEstop,
      dismissAlert,
      announce,
      refresh,
    }),
    [
      view,
      statusPoll.data,
      controlPoll.data,
      statusPoll.lastSuccessAt,
      statusPoll.stale,
      controlPoll.lastSuccessAt,
      statusPoll.error,
      estopInFlight,
      lockoutUntil,
      alerts,
      fireEstop,
      fireHalt,
      fireCollapse,
      softStop,
      doResetEstop,
      dismissAlert,
      announce,
      refresh,
    ],
  );

  return (
    <SafetyContext.Provider value={value}>
      {children}
      <div className="sr-only" aria-live="polite" aria-atomic="true" data-testid="safety-notice">
        {notice}
      </div>
      <div className="sr-only" aria-live="polite" aria-atomic="true">
        {view.known ? `Safety state: ${view.chipLabel}` : ""}
      </div>
    </SafetyContext.Provider>
  );
}

/** Safety state and stop actions. Outside the provider, an inert default. */
export function useSafety(): SafetyContextValue {
  return useContext(SafetyContext);
}
