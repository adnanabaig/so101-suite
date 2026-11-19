import { useId, useRef, useState, type KeyboardEvent, type ReactNode } from "react";

export interface TabItem {
  id: string;
  label: ReactNode;
  panel: ReactNode;
}

export interface TabsProps {
  /** Names the tab list for assistive technology. */
  label: string;
  tabs: readonly TabItem[];
  /** Controlled selection; leave unset to let the tabs keep their own. */
  selected?: string;
  /** First tab shown when uncontrolled (defaults to the first tab). */
  defaultSelected?: string;
  onSelect?: (id: string) => void;
  className?: string;
  /** Classes for the active panel, e.g. spacing below the tab list. */
  panelClassName?: string;
}

/**
 * WAI-ARIA tabs with automatic activation: Left and Right move between tabs
 * (wrapping), Home and End jump to the ends, and only the selected tab is in
 * the Tab order. Only the selected panel is rendered, so a heavy panel costs
 * nothing until it is opened; state that must survive a tab switch belongs to
 * the parent.
 */
export function Tabs({
  label,
  tabs,
  selected,
  defaultSelected,
  onSelect,
  className = "",
  panelClassName = "pt-4",
}: TabsProps) {
  const baseId = useId();
  const [own, setOwn] = useState(defaultSelected ?? tabs[0]?.id ?? "");
  const wanted = selected ?? own;
  const active = tabs.find((tab) => tab.id === wanted) ?? tabs[0];
  const buttons = useRef(new Map<string, HTMLButtonElement>());

  const tabId = (id: string) => `${baseId}-tab-${id}`;
  const panelId = (id: string) => `${baseId}-panel-${id}`;

  const choose = (id: string) => {
    if (selected === undefined) setOwn(id);
    onSelect?.(id);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!active || tabs.length === 0) return;
    const index = tabs.findIndex((tab) => tab.id === active.id);
    let next: number;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    else if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    else return;
    event.preventDefault();
    const target = tabs[next];
    choose(target.id);
    buttons.current.get(target.id)?.focus();
  };

  if (!active) return null;
  return (
    <div className={className}>
      <div
        role="tablist"
        aria-label={label}
        onKeyDown={onKeyDown}
        className="flex gap-1 overflow-x-auto border-b border-slate-800"
      >
        {tabs.map((tab) => {
          const isActive = tab.id === active.id;
          return (
            <button
              key={tab.id}
              ref={(element) => {
                if (element) buttons.current.set(tab.id, element);
                else buttons.current.delete(tab.id);
              }}
              type="button"
              role="tab"
              id={tabId(tab.id)}
              aria-selected={isActive}
              aria-controls={isActive ? panelId(tab.id) : undefined}
              tabIndex={isActive ? 0 : -1}
              onClick={() => choose(tab.id)}
              className={`-mb-px inline-flex shrink-0 items-center gap-2 rounded-t-lg border-b-2 px-3.5 py-2 text-sm font-semibold transition-colors focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-cyan-300 ${
                isActive
                  ? "border-cyan-400 text-cyan-100"
                  : "border-transparent text-slate-400 hover:border-slate-600 hover:text-slate-200"
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
      <div role="tabpanel" id={panelId(active.id)} aria-labelledby={tabId(active.id)} className={panelClassName}>
        {active.panel}
      </div>
    </div>
  );
}
