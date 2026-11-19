import { useState } from "react";
import { NavLink } from "react-router-dom";
import { AlertOctagon, Hand, PowerOff, WifiOff } from "lucide-react";
import { ApiError } from "../../api";
import { useNow } from "../../hooks/useNow";
import { useTelemetry, type TelemetryLinkIssue } from "../../hooks/useTelemetry";
import { collapsePercent, haltRemainingMs, unsupervisedRemainingMs, unsupervisedText } from "../../lib/safetyView";
import HoldToConfirmButton from "./HoldToConfirmButton";
import { useSafety, type SafetyAlert } from "./SafetyProvider";

/** `m:ss`, rounded-sm up so the display reaches 0:00 only when time is up. */
export function formatCountdown(ms: number): string {
  const total = Math.max(0, Math.ceil(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

/** Move focus somewhere stable after the control that had it disappears. */
export function focusMain() {
  document.getElementById("main-content")?.focus();
}

const BANNER = "flex flex-wrap items-start gap-x-4 gap-y-2 rounded-xl border px-4 py-3 text-sm shadow-lg";

function AlertBanner({ alert }: { alert: SafetyAlert }) {
  const { dismissAlert } = useSafety();
  const critical = alert.severity === "critical";
  return (
    <div
      role={critical ? "alert" : undefined}
      className={`${BANNER} ${
        critical
          ? "border-rose-400/60 bg-rose-950/95 text-rose-50 shadow-rose-950/40"
          : "border-amber-400/50 bg-amber-950/95 text-amber-50 shadow-amber-950/30"
      }`}
    >
      <AlertOctagon aria-hidden="true" className="mt-0.5 h-5 w-5 shrink-0" />
      <div className="min-w-0 flex-1">
        <div className="font-bold">{alert.title}</div>
        <p className="mt-0.5 text-xs leading-relaxed opacity-90">{alert.message}</p>
      </div>
      <button
        type="button"
        aria-label={`Dismiss: ${alert.title}`}
        onClick={() => dismissAlert(alert.id)}
        className="h-9 shrink-0 rounded-lg border border-white/30 px-3 text-xs font-semibold hover:bg-white/10"
      >
        Dismiss
      </button>
    </div>
  );
}

function TorqueUnconfirmedBanner() {
  return (
    <div role="alert" className={`${BANNER} border-rose-300 bg-rose-700 text-white shadow-rose-950/50`}>
      <PowerOff aria-hidden="true" className="mt-0.5 h-5 w-5 shrink-0" />
      <div className="min-w-0 flex-1">
        <div className="font-black uppercase tracking-wide">Fault: cut physical power if anything moves</div>
        <p className="mt-0.5 text-xs leading-relaxed">
          The server could not confirm that the motors are off, so treat the arm as live. Clearing the
          fault in Diagnostics retries the torque-off first.
        </p>
      </div>
      <NavLink
        to="/diagnostics"
        className="inline-flex h-9 shrink-0 items-center rounded-lg border border-white/50 px-3 text-xs font-bold hover:bg-white/10"
      >
        Diagnostics
      </NavLink>
    </div>
  );
}

function EstopLatchedBanner() {
  const { lockoutUntil, resetEstop } = useSafety();
  const now = useNow(250);
  const [resetting, setResetting] = useState(false);
  const remainingMs = lockoutUntil === null ? 0 : Math.max(0, lockoutUntil - now);
  const locked = remainingMs > 0;
  const lockSeconds = Math.ceil(remainingMs / 1000);

  const confirm = () => {
    setResetting(true);
    void resetEstop()
      .then((ok) => {
        if (ok) focusMain();
      })
      .finally(() => setResetting(false));
  };

  return (
    <div
      data-testid="estop-latched"
      className={`${BANNER} items-center border-rose-400/60 bg-rose-950/95 text-rose-50 shadow-rose-950/40`}
    >
      <AlertOctagon aria-hidden="true" className="h-5 w-5 shrink-0" />
      <div className="min-w-0 flex-1">
        <div className="font-bold">E-stop latched. All motors are off.</div>
        <p className="mt-0.5 text-xs leading-relaxed opacity-90">
          Before you reset, make sure the workspace is clear. Reset returns the robot to idle and
          disarmed; nothing moves until someone arms it again.
        </p>
      </div>
      <HoldToConfirmButton
        label="Reset E-stop"
        onConfirm={confirm}
        locked={locked}
        busy={resetting}
        lockedReason={locked ? `Reset unlocks in ${lockSeconds} seconds.` : undefined}
        className="h-10 shrink-0 rounded-xl border border-amber-300/60 bg-amber-400 px-4 text-xs font-black uppercase tracking-[0.1em] text-slate-950"
      >
        {resetting ? "Resetting" : locked ? `Reset in ${lockSeconds}s` : "Hold to reset E-stop"}
      </HoldToConfirmButton>
    </div>
  );
}

function HaltedBanner() {
  const { view, fireHalt, fireCollapse } = useSafety();
  const now = useNow(500);
  const remaining = view.halt ? haltRemainingMs(view.halt, now) : null;
  const unsupervised = view.halt ? unsupervisedRemainingMs(view.halt, view.supervision, now) : null;
  const reason = view.halt?.info.reason;

  // Announce the switch, not every tick of the countdown.
  const isUnsupervised = unsupervised !== null;
  const [announced, setAnnounced] = useState(isUnsupervised);
  const [announcement, setAnnouncement] = useState(isUnsupervised ? `${unsupervisedText(unsupervised)}.` : "");
  if (announced !== isUnsupervised) {
    setAnnounced(isUnsupervised);
    setAnnouncement(
      isUnsupervised ? `${unsupervisedText(unsupervised)}.` : "Supervised again: back to the hold timer.",
    );
  }
  return (
    <div
      data-testid="halted-banner"
      className={`${BANNER} items-center border-amber-300/60 bg-amber-950/95 text-amber-50 shadow-amber-950/30`}
    >
      <Hand aria-hidden="true" className="h-5 w-5 shrink-0" />
      <div className="min-w-0 flex-1">
        <div className="font-bold">
          Halted — holding position
          {unsupervised !== null ? (
            <span className="ml-3 font-mono font-black text-rose-200" data-testid="unsupervised-countdown">
              {unsupervisedText(unsupervised)}
            </span>
          ) : (
            remaining !== null && (
              <span className="ml-3 font-mono font-black" data-testid="halt-countdown">
                {formatCountdown(remaining)}
              </span>
            )
          )}
        </div>
        <p className="mt-0.5 text-xs leading-relaxed opacity-90">
          {unsupervised !== null
            ? "No operator page is sending a supervision heartbeat, so the hold lasts only the grace period. When it ends, the arm will lower to rest and turn off."
            : remaining === null
              ? "The hold timer is unknown. When it runs out, the arm will lower to rest and turn off."
              : `When the countdown reaches zero, the arm will lower to rest and turn off.`}
          {reason ? ` Reason: ${reason}.` : ""}
        </p>
        <p className="sr-only" aria-live="polite" data-testid="supervision-notice">
          {announcement}
        </p>
      </div>
      <div className="flex shrink-0 flex-wrap gap-2">
        <button
          type="button"
          onClick={() => fireHalt("banner", "operator kept holding")}
          className="h-10 rounded-lg border border-amber-300/60 bg-amber-400 px-3 text-xs font-bold text-slate-950 hover:bg-amber-300"
        >
          Keep holding
        </button>
        {view.canCollapse && (
          <button
            type="button"
            onClick={() => fireCollapse("banner")}
            className="h-10 rounded-lg border border-slate-500 bg-slate-800 px-3 text-xs font-semibold text-slate-100 hover:bg-slate-700"
          >
            Collapse now
          </button>
        )}
      </div>
    </div>
  );
}

function CollapsingBanner() {
  const { view, fireHalt } = useSafety();
  const percent = view.collapse ? collapsePercent(view.collapse) : null;
  const eta = view.collapse?.eta_s;
  return (
    <div
      data-testid="collapsing-banner"
      className={`${BANNER} items-center border-slate-500 bg-slate-900/95 text-slate-100 shadow-black/40`}
    >
      <div className="min-w-0 flex-1">
        <div className="font-bold">Collapsing — lowering to rest, then motors off</div>
        <div className="mt-2 flex items-center gap-3">
          <div
            role="progressbar"
            aria-label="Collapse progress"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={percent === null ? undefined : Math.round(percent)}
            className="h-2 w-40 overflow-hidden rounded-full bg-slate-700"
          >
            <div className="h-full bg-slate-200" style={{ width: `${percent ?? 0}%` }} />
          </div>
          <span className="font-mono text-xs">
            {percent === null ? "progress unknown" : `${Math.round(percent)}%`}
            {typeof eta === "number" && Number.isFinite(eta) ? ` · about ${Math.ceil(eta)} s left` : ""}
          </span>
        </div>
        <p className="mt-1 text-xs opacity-80">
          {view.canHalt ? "Halt stops the descent and holds. " : ""}E-stop cuts motor power immediately.
        </p>
      </div>
      {view.canHalt && (
        <button
          type="button"
          onClick={() => fireHalt("banner", "operator halted the descent")}
          className="h-10 shrink-0 rounded-lg border border-amber-300/60 bg-amber-400 px-3 text-xs font-bold text-slate-950 hover:bg-amber-300"
        >
          Halt the descent
        </button>
      )}
    </div>
  );
}

function ConnectionLostBanner() {
  return (
    <div className={`${BANNER} items-center border-amber-400/40 bg-slate-900/95 text-amber-100`}>
      <WifiOff aria-hidden="true" className="h-5 w-5 shrink-0" />
      <p className="min-w-0 flex-1 text-xs leading-relaxed">
        <span className="font-bold">Robot state unknown.</span> The server has stopped answering, so
        the state shown may be out of date. The stop buttons still send; confirm the arm visually.
      </p>
    </div>
  );
}

/** How to let this page through the C1 guard. */
export const LINK_ALLOWLIST_HINT =
  "Open the UI from the robot server's own address, or add this page's hostname to SO101_ALLOWED_HOSTS on the server. " +
  "If this page comes from a different port or through a proxy, also add its origin (scheme, host and port) to " +
  "SO101_CORS_ORIGINS. Restart the server after either change.";

function LinkRefusedBanner({ issue }: { issue: TelemetryLinkIssue }) {
  return (
    <div role="status" className={`${BANNER} items-center border-amber-400/40 bg-slate-900/95 text-amber-100`}>
      <WifiOff aria-hidden="true" className="h-5 w-5 shrink-0" />
      <p className="min-w-0 flex-1 text-xs leading-relaxed">
        {issue.kind === "refused" ? (
          <>
            <span className="font-bold">The server refused the live telemetry link</span>
            {issue.detail ? (
              <>
                : <span className="font-mono">{issue.detail}</span>.
              </>
            ) : (
              " (policy violation, code 1008)."
            )}{" "}
          </>
        ) : (
          <>
            <span className="font-bold">The live telemetry link has failed {issue.attempts} times without connecting</span>{" "}
            while the server still answers requests. It is probably refusing this page&apos;s address.{" "}
          </>
        )}
        {LINK_ALLOWLIST_HINT} The stop buttons still send.
      </p>
    </div>
  );
}

/**
 * C1 Host guard: the server answers every request from this address with 400
 * and a detail that names SO101_ALLOWED_HOSTS. The detail is shown as sent.
 */
function HostRefusedBanner({ detail }: { detail: string }) {
  return (
    <div role="alert" className={`${BANNER} items-center border-rose-400/50 bg-slate-900/95 text-rose-100`}>
      <WifiOff aria-hidden="true" className="h-5 w-5 shrink-0" />
      <p className="min-w-0 flex-1 text-xs leading-relaxed">
        <span className="font-bold">The robot server refused this page&apos;s address:</span>{" "}
        <span className="font-mono">{detail}</span>. Robot state is unknown here. {LINK_ALLOWLIST_HINT} E-stop, Halt
        and Soft stop still send: the server&apos;s guard always lets them through.
      </p>
    </div>
  );
}

/** Safety notices pinned under the header. Renders nothing when all is quiet. */
export default function SafetyBanners() {
  const { view, alerts, statusUpdatedAt, statusError } = useSafety();
  const { linkIssue } = useTelemetry();
  const hostRefused = statusError instanceof ApiError && statusError.status === 400 ? statusError.message : null;
  const halted = !view.tripped && view.stopState === "halted";
  const collapsing = !view.tripped && view.stopState === "collapsing";
  const lost = !view.known && statusUpdatedAt !== null;
  // A server that answers nothing gets the "state unknown" banner instead:
  // a dead server also fails every socket.
  const refused = linkIssue !== null && (linkIssue.kind === "refused" || (!lost && statusUpdatedAt !== null));
  if (
    alerts.length === 0 &&
    !view.torqueUnconfirmed &&
    !view.tripped &&
    !halted &&
    !collapsing &&
    !lost &&
    !refused &&
    !hostRefused
  ) {
    return null;
  }
  return (
    <section aria-label="Safety notices" className="sticky top-16 z-30 mt-16 shrink-0 space-y-2 px-3 pt-2 sm:px-6">
      {alerts.map((alert) => (
        <AlertBanner key={alert.id} alert={alert} />
      ))}
      {view.torqueUnconfirmed && <TorqueUnconfirmedBanner />}
      {view.tripped && <EstopLatchedBanner />}
      {halted && <HaltedBanner />}
      {collapsing && <CollapsingBanner />}
      {hostRefused !== null && <HostRefusedBanner detail={hostRefused} />}
      {lost && hostRefused === null && <ConnectionLostBanner />}
      {refused && linkIssue && hostRefused === null && <LinkRefusedBanner issue={linkIssue} />}
    </section>
  );
}
