import { useCallback, useEffect, useRef, type ReactNode } from "react";

export const DEADMAN_KEYS = new Set([" ", "Enter"]);

export interface DeadmanButtonProps {
  /** Pressed: start the held action. */
  onHoldStart: () => void;
  /**
   * The hold ended: released, cancelled, focus or window lost, tab hidden,
   * disabled or unmounted while held. Called at most once per press.
   */
  onHoldEnd: (reason: string) => void;
  /** The held action is live (aria-pressed and the visual state). */
  active: boolean;
  disabled?: boolean;
  children: ReactNode;
  className?: string;
  "aria-describedby"?: string;
}

/**
 * A press-and-hold control with the same release semantics as the teleop
 * deadman: pointer capture, release on pointerup/pointercancel/lost capture
 * anywhere in the window, Space or Enter held as the keyboard alternative,
 * and a release whenever focus, the window or the tab goes away.
 */
export default function DeadmanButton({
  onHoldStart,
  onHoldEnd,
  active,
  disabled = false,
  children,
  className = "",
  "aria-describedby": describedBy,
}: DeadmanButtonProps) {
  const holding = useRef(false);
  const endRef = useRef(onHoldEnd);
  useEffect(() => {
    endRef.current = onHoldEnd;
  });

  const end = useCallback((reason: string) => {
    if (!holding.current) return;
    holding.current = false;
    endRef.current(reason);
  }, []);

  const start = () => {
    if (holding.current || disabled) return;
    holding.current = true;
    onHoldStart();
  };

  // A release anywhere ends a hold that pointer capture missed; releases
  // with no hold in progress do nothing.
  useEffect(() => {
    const release = () => end("deadman released");
    const onBlur = () => end("window lost focus");
    const onVisibility = () => {
      if (document.visibilityState !== "visible") end("tab hidden");
    };
    window.addEventListener("pointerup", release);
    window.addEventListener("pointercancel", release);
    window.addEventListener("blur", onBlur);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      window.removeEventListener("pointerup", release);
      window.removeEventListener("pointercancel", release);
      window.removeEventListener("blur", onBlur);
      document.removeEventListener("visibilitychange", onVisibility);
      end("control closed");
    };
  }, [end]);

  useEffect(() => {
    if (disabled) end("control disabled");
  }, [disabled, end]);

  return (
    <button
      type="button"
      aria-pressed={active}
      aria-describedby={describedBy}
      disabled={disabled}
      onPointerDown={(event) => {
        if (!event.isPrimary || event.button !== 0) return;
        try {
          event.currentTarget.setPointerCapture?.(event.pointerId);
        } catch {
          // Capture is best-effort; the window listener still sees the release.
        }
        start();
      }}
      onPointerUp={() => end("deadman released")}
      onPointerCancel={() => end("deadman released")}
      onLostPointerCapture={() => end("deadman released")}
      onKeyDown={(event) => {
        if (!DEADMAN_KEYS.has(event.key)) return;
        event.preventDefault();
        if (!event.repeat) start();
      }}
      onKeyUp={(event) => {
        if (DEADMAN_KEYS.has(event.key)) end("deadman released");
      }}
      onBlur={() => end("deadman lost focus")}
      onContextMenu={(event) => event.preventDefault()}
      className={`touch-none select-none ${className}`}
    >
      {children}
    </button>
  );
}
