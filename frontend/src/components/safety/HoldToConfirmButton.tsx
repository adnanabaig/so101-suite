import { useEffect, useId, useRef, useState, type ReactNode } from "react";

export const DEFAULT_HOLD_MS = 1500;

export interface HoldToConfirmButtonProps {
  /** Accessible name. Stays constant so assistive tech never hears it change mid-hold. */
  label: string;
  /** Visible content (defaults to the label). */
  children?: ReactNode;
  onConfirm: () => void;
  holdMs?: number;
  /** Refuse to start a hold (rendered aria-disabled so it stays focusable). */
  locked?: boolean;
  /** Why it is locked, read as part of the description. */
  lockedReason?: string;
  busy?: boolean;
  className?: string;
}

const CONFIRM_KEYS = new Set([" ", "Enter"]);

/**
 * A button that fires only after a continuous press (pointer, Space or Enter)
 * of `holdMs`. Releasing, cancelling, or losing focus aborts the hold, and a
 * plain click does nothing, so a stray tap can never confirm.
 */
export default function HoldToConfirmButton({
  label,
  children,
  onConfirm,
  holdMs = DEFAULT_HOLD_MS,
  locked = false,
  lockedReason,
  busy = false,
  className = "",
}: HoldToConfirmButtonProps) {
  const descriptionId = useId();
  const [holding, setHolding] = useState(false);
  const [releasedEarly, setReleasedEarly] = useState(false);
  const timerRef = useRef<number | null>(null);
  const blocked = locked || busy;

  // Latest values for the timer callback.
  const onConfirmRef = useRef(onConfirm);
  const blockedRef = useRef(blocked);
  useEffect(() => {
    onConfirmRef.current = onConfirm;
    blockedRef.current = blocked;
  });

  useEffect(
    () => () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    },
    [],
  );

  // A lock or busy state that arrives mid-hold aborts the hold.
  if (holding && blocked) setHolding(false);

  const start = () => {
    if (blocked) return;
    // A timer left over from a hold that a lock aborted is discarded.
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    setHolding(true);
    setReleasedEarly(false);
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null;
      setHolding(false);
      if (!blockedRef.current) onConfirmRef.current();
    }, holdMs);
  };

  const cancel = () => {
    if (timerRef.current === null) return;
    window.clearTimeout(timerRef.current);
    timerRef.current = null;
    setHolding(false);
    setReleasedEarly(true);
  };

  const seconds = (holdMs / 1000).toFixed(1).replace(/\.0$/, "");
  const duration = `${seconds} ${seconds === "1" ? "second" : "seconds"}`;
  const description = blocked
    ? (lockedReason ?? (busy ? "Working." : "Unavailable."))
    : releasedEarly
      ? `Released too early. Press and hold for ${duration}.`
      : `Press and hold for ${duration} to confirm.`;

  return (
    <>
      <button
        type="button"
        aria-label={label}
        aria-describedby={descriptionId}
        aria-disabled={blocked || undefined}
        data-holding={holding || undefined}
        className={`relative overflow-hidden touch-none select-none ${
          blocked ? "cursor-not-allowed opacity-60" : ""
        } ${className}`}
        onPointerDown={(event) => {
          if (event.pointerType === "mouse" && event.button !== 0) return;
          start();
          try {
            event.currentTarget.setPointerCapture?.(event.pointerId);
          } catch {
            // Capture is best-effort; release still cancels via pointerup.
          }
        }}
        onPointerUp={cancel}
        onPointerCancel={cancel}
        onLostPointerCapture={cancel}
        onKeyDown={(event) => {
          if (!CONFIRM_KEYS.has(event.key)) return;
          event.preventDefault();
          if (!event.repeat) start();
        }}
        onKeyUp={(event) => {
          if (CONFIRM_KEYS.has(event.key)) cancel();
        }}
        onBlur={cancel}
        onContextMenu={(event) => event.preventDefault()}
      >
        {/* The fill is how long is left to hold, so reduced motion keeps it
            (index.css exempts data-essential-motion): without it the bar
            would jump to full the moment the press starts. */}
        <span
          aria-hidden="true"
          data-essential-motion=""
          className="pointer-events-none absolute inset-0 origin-left bg-white/25"
          style={{
            transform: holding ? "scaleX(1)" : "scaleX(0)",
            transition: holding ? `transform ${holdMs}ms linear` : "none",
          }}
        />
        <span className="relative">{children ?? label}</span>
      </button>
      <span id={descriptionId} className="sr-only">
        {description}
      </span>
    </>
  );
}
