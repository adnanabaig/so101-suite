import type { ReactNode } from "react";
import { JOG_STEP, JOG_STEP_LARGE } from "../../lib/jog";
import Button from "../Button";
import { SHORTCUT_ITEMS } from "../navigation";
import { Dialog } from "../ui/Dialog";

/** A key cap. `name` is what a screen reader says when the cap is a symbol. */
export function Key({ children, name }: { children: ReactNode; name?: string }) {
  return (
    <kbd className="inline-flex min-w-[1.75rem] items-center justify-center rounded-md border border-slate-600 bg-slate-800 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-slate-100 shadow-[inset_0_-1px_0_rgba(0,0,0,0.4)]">
      {name ? (
        <>
          <span aria-hidden="true">{children}</span>
          <span className="sr-only">{name}</span>
        </>
      ) : (
        children
      )}
    </kbd>
  );
}

function Row({ keys, children }: { keys: ReactNode; children: ReactNode }) {
  return (
    <div className="grid grid-cols-[minmax(7.5rem,auto)_1fr] items-baseline gap-3 py-1.5">
      <dt className="flex flex-wrap items-center gap-1 text-xs text-slate-300">{keys}</dt>
      <dd className="text-sm leading-relaxed text-slate-200">{children}</dd>
    </div>
  );
}

function Section({ title, note, children }: { title: string; note?: string; children: ReactNode }) {
  return (
    <div className="border-t border-white/5 pt-3 first:border-t-0 first:pt-0">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-300">{title}</h3>
      {note && <p className="mt-1 text-xs leading-relaxed text-slate-400">{note}</p>}
      <dl className="mt-1 divide-y divide-white/5">{children}</dl>
    </div>
  );
}

/**
 * The `?` list. Like every dialog here it does not close on Escape, which
 * stays E-stop; `?` or the Close button closes it.
 */
export default function ShortcutsDialog({
  open,
  canHalt,
  onClose,
}: {
  open: boolean;
  canHalt: boolean;
  onClose: () => void;
}) {
  return (
    <Dialog
      open={open}
      size="wide"
      title="Keyboard shortcuts"
      description="Letter keys are ignored while you type in a field. Escape works everywhere."
      actions={
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
      }
    >
      <div className="max-h-[60vh] space-y-4 overflow-y-auto pr-1">
        <Section title="Safety">
          <Row keys={<Key>Esc</Key>}>
            E-stop: cut motor power now. Works on every page, in text fields and in dialogs.
          </Row>
          <Row keys={<Key>H</Key>}>
            {canHalt
              ? "Halt: freeze in place with the motors holding position."
              : "Halt. This server does not offer Halt, so H does nothing here."}
          </Row>
          <Row keys={<span className="text-slate-400">Button only</span>}>
            Collapse has no key. It lowers the arm to rest, then turns the motors off.
          </Row>
        </Section>

        <Section
          title="Deadman and jog"
          note="Data collection, Manual joint control. Jog keys do nothing unless the deadman is held and motion is armed."
        >
          <Row
            keys={
              <>
                <Key>Space</Key>
                <span className="text-slate-400">or</span>
                <Key>Enter</Key>
              </>
            }
          >
            Hold on the deadman button to stream commands. Letting go stops motion.
          </Row>
          <Row
            keys={
              <>
                <Key name="Left arrow">←</Key>
                <Key name="Right arrow">→</Key>
              </>
            }
          >
            {`Jog the selected joint's target by ${JOG_STEP}, or ${JOG_STEP_LARGE} with Shift, inside its limits.`}
          </Row>
          <Row
            keys={
              <>
                <Key name="Up arrow">↑</Key>
                <Key name="Down arrow">↓</Key>
              </>
            }
          >
            Choose the joint to jog.
          </Row>
        </Section>

        <Section title="Go to a page" note="Press G, let go, then the letter.">
          {SHORTCUT_ITEMS.map((item) => (
            <Row
              key={item.to}
              keys={
                <>
                  <Key>G</Key>
                  <span className="text-slate-400">then</span>
                  <Key>{item.shortcut.toUpperCase()}</Key>
                </>
              }
            >
              {item.label}
            </Row>
          ))}
        </Section>

        <Section title="Help">
          <Row keys={<Key>?</Key>}>Show or hide this list.</Row>
        </Section>
      </div>
    </Dialog>
  );
}
