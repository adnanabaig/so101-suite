import {
  useCallback,
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
  type RefObject,
} from "react";
import { createPortal } from "react-dom";
import { AlertTriangle } from "lucide-react";
import Button from "../Button";

/*
 * In-app dialogs replace window.confirm(), which blocks the page's JavaScript
 * and swallows Escape: while a native confirm is open, Escape cannot E-stop.
 *
 * Rules (operator-ui.md):
 * - A dialog never closes on Escape. Escape always reaches the global E-stop
 *   handler, and the dialog stays open so the operator sees what was pending.
 * - The overlay sits below the sticky header (z-50), so E-stop and Halt stay
 *   visible and clickable while a dialog is open.
 * - Focus moves into the dialog (Cancel by default), Tab cycles inside it, and
 *   focus returns to the control that opened it.
 * - Clicking the backdrop does nothing; only the dialog's own buttons decide.
 */

export type DialogTone = "neutral" | "warn" | "danger";

export interface DialogProps {
  open: boolean;
  title: string;
  /** Short explanation, announced as the dialog's description. */
  description?: ReactNode;
  children?: ReactNode;
  /** Footer buttons. */
  actions?: ReactNode;
  /** Element to focus when the dialog opens; defaults to the first control. */
  initialFocusRef?: RefObject<HTMLElement | null>;
  tone?: DialogTone;
  /** "alertdialog" for confirmations that interrupt the operator. */
  role?: "dialog" | "alertdialog";
  /** "wide" for reference content such as the shortcut list. */
  size?: "normal" | "wide" | "workspace";
}

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

function focusableIn(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
    (element) => !element.hasAttribute("inert") && element.getAttribute("aria-hidden") !== "true",
  );
}

const TONE_BORDER: Record<DialogTone, string> = {
  neutral: "border-white/10",
  warn: "border-amber-400/40",
  danger: "border-rose-400/50",
};

function DialogSurface({
  title,
  description,
  children,
  actions,
  initialFocusRef,
  tone = "neutral",
  role = "dialog",
  size = "normal",
}: Omit<DialogProps, "open">) {
  const titleId = useId();
  const descriptionId = useId();
  const panelRef = useRef<HTMLDivElement>(null);

  // Focus in on open, back to the opener on close.
  useEffect(() => {
    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const panel = panelRef.current;
    const target = initialFocusRef?.current ?? (panel ? focusableIn(panel)[0] : null) ?? panel;
    target?.focus();
    return () => {
      if (opener && opener.isConnected) opener.focus();
    };
    // Focus only on open; a changing ref target must not steal focus later.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    // Escape is deliberately not handled here: it propagates to the E-stop.
    if (event.key !== "Tab") return;
    const panel = panelRef.current;
    if (!panel) return;
    const focusable = focusableIn(panel);
    if (focusable.length === 0) {
      event.preventDefault();
      panel.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    if (event.shiftKey && (active === first || active === panel)) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && active === last) {
      event.preventDefault();
      first.focus();
    }
  };

  return (
    <div className="fixed inset-0 z-40 flex items-start justify-center overflow-y-auto bg-black/70 px-4 pb-6 pt-24 backdrop-blur-xs">
      <div
        ref={panelRef}
        role={role}
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={description ? descriptionId : undefined}
        tabIndex={-1}
        onKeyDown={onKeyDown}
        className={`w-full ${size === "workspace" ? "max-h-[calc(100dvh-7rem)] max-w-[1600px] overflow-y-auto" : size === "wide" ? "max-w-2xl" : "max-w-lg"} rounded-2xl border bg-slate-900 p-5 text-slate-100 shadow-2xl focus:outline-hidden ${TONE_BORDER[tone]}`}
      >
        <div className="flex items-start gap-3">
          {tone !== "neutral" && (
            <AlertTriangle
              aria-hidden="true"
              className={`mt-0.5 h-5 w-5 shrink-0 ${tone === "danger" ? "text-rose-300" : "text-amber-300"}`}
            />
          )}
          <div className="min-w-0 flex-1">
            <h2 id={titleId} className="text-base font-bold text-white">
              {title}
            </h2>
            {description && (
              <div id={descriptionId} className="mt-1.5 text-sm leading-relaxed text-slate-300">
                {description}
              </div>
            )}
          </div>
        </div>
        {children && <div className="mt-4">{children}</div>}
        {actions && <div className="mt-5 flex flex-wrap justify-end gap-2">{actions}</div>}
        <p className="mt-4 text-[11px] text-muted">Escape is E-stop. It does not close this dialog.</p>
      </div>
    </div>
  );
}

/** Modal dialog rendered into document.body, below the safety header. */
export function Dialog({ open, ...rest }: DialogProps) {
  if (!open) return null;
  return createPortal(<DialogSurface {...rest} />, document.body);
}

export interface ConfirmOptions {
  title: string;
  body?: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  tone?: DialogTone;
  /** When set, a checkbox with this label must be ticked before confirming. */
  acknowledgement?: string;
}

interface ConfirmDialogProps extends ConfirmOptions {
  open: boolean;
  onResolve: (confirmed: boolean) => void;
}

export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  tone = "warn",
  acknowledgement,
  onResolve,
}: ConfirmDialogProps) {
  const [acknowledged, setAcknowledged] = useState(false);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const ackId = useId();
  // A reopened dialog starts unacknowledged.
  const [wasOpen, setWasOpen] = useState(open);
  if (wasOpen !== open) {
    setWasOpen(open);
    if (open) setAcknowledged(false);
  }
  const canConfirm = !acknowledgement || acknowledged;
  return (
    <Dialog
      open={open}
      role="alertdialog"
      title={title}
      description={body}
      tone={tone}
      initialFocusRef={cancelRef}
      actions={
        <>
          <Button ref={cancelRef} variant="secondary" onClick={() => onResolve(false)}>
            {cancelLabel}
          </Button>
          <Button
            variant={tone === "danger" ? "danger" : "primary"}
            disabled={!canConfirm}
            onClick={() => {
              if (canConfirm) onResolve(true);
            }}
          >
            {confirmLabel}
          </Button>
        </>
      }
    >
      {acknowledgement && (
        <label
          htmlFor={ackId}
          className="flex items-start gap-3 rounded-xl border border-amber-400/30 bg-amber-400/[0.06] p-3 text-sm text-amber-50"
        >
          <input
            id={ackId}
            type="checkbox"
            checked={acknowledged}
            onChange={(event) => setAcknowledged(event.target.checked)}
            className="mt-0.5 h-4 w-4 accent-amber-400"
          />
          <span>{acknowledgement}</span>
        </label>
      )}
    </Dialog>
  );
}

interface PendingConfirm {
  id: number;
  options: ConfirmOptions;
}

/**
 * Promise-based replacement for window.confirm():
 *
 *   const [confirm, confirmDialog] = useConfirm();
 *   if (!(await confirm({ title: "Start rollout?" }))) return;
 *   ...
 *   return <>{confirmDialog}...</>;
 *
 * Resolves false when cancelled, when another confirmation replaces it, or
 * when the component unmounts.
 */
export function useConfirm(): [(options: ConfirmOptions) => Promise<boolean>, ReactNode] {
  const [pending, setPending] = useState<PendingConfirm | null>(null);
  const resolverRef = useRef<{ id: number; resolve: (confirmed: boolean) => void } | null>(null);
  const nextIdRef = useRef(0);

  const confirm = useCallback(
    (options: ConfirmOptions) =>
      new Promise<boolean>((resolve) => {
        resolverRef.current?.resolve(false);
        const id = ++nextIdRef.current;
        resolverRef.current = { id, resolve };
        setPending({ id, options });
      }),
    [],
  );

  const resolve = useCallback((id: number, confirmed: boolean) => {
    if (resolverRef.current?.id !== id) return;
    const { resolve: settle } = resolverRef.current;
    resolverRef.current = null;
    setPending(null);
    settle(confirmed);
  }, []);

  useEffect(
    () => () => {
      resolverRef.current?.resolve(false);
      resolverRef.current = null;
    },
    [],
  );

  const element = pending ? (
    <ConfirmDialog
      key={pending.id}
      open
      {...pending.options}
      onResolve={(confirmed) => resolve(pending.id, confirmed)}
    />
  ) : null;
  return [confirm, element];
}
