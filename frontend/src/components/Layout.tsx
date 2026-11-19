import {
  useEffect,
  useId,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from "react";
import { NavLink, useLocation } from "react-router-dom";
import { ChevronRight, Menu, X } from "lucide-react";
import { useNow } from "../hooks/useNow";
import { useTelemetry } from "../hooks/useTelemetry";
import { ErrorBoundary, UpdateBanner, useUpdateAvailable } from "./ErrorBoundary";
import SafetyBanners from "./safety/SafetyBanners";
import SafetyBar from "./safety/SafetyBar";
import { NAV_ITEMS, routeLabel } from "./navigation";
import { SafetyProvider } from "./safety/SafetyProvider";
import { ShortcutsProvider } from "./shortcuts/ShortcutsProvider";
import { ToastProvider } from "./ui/Toast";

/** "0.4s", "12s", "3m": compact age for the link indicator. */
function formatAge(ms: number): string {
  if (ms < 10_000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 120_000) return `${Math.round(ms / 1000)}s`;
  return `${Math.round(ms / 60_000)}m`;
}

/**
 * Telemetry link and joint-stream state. Transport and sample freshness are
 * separate (C6): an idle robot with a healthy link is neutral, not a warning
 * (F8), and the time since the last frame is always visible (F26).
 */
export function ConnectionIndicator() {
  const { telemetry, transportStale, sampleState, lastMessageAt, linkIssue } = useTelemetry();
  const now = useNow(1000);
  const ageMs = lastMessageAt === null ? null : Math.max(0, now - lastMessageAt);

  let tone: string;
  let label: string;
  if (transportStale) {
    tone = "bg-rose-500";
    label =
      linkIssue?.kind === "refused"
        ? "Live link refused by the server"
        : lastMessageAt === null
          ? "Connecting to robot"
          : "Telemetry link down · reconnecting";
  } else if (sampleState === "idle") {
    tone = "bg-slate-400";
    label = "Idle · no live joint stream";
  } else if (sampleState === "stale") {
    tone = "bg-amber-400";
    label = "Joint samples stale";
  } else {
    tone = "bg-emerald-400";
    label = "Joint stream live";
  }
  const detail =
    ageMs === null
      ? "no frames yet"
      : `last msg ${formatAge(ageMs)} ago${
          !transportStale && telemetry && sampleState === "live" ? ` · ${telemetry.fps.toFixed(0)} fps` : ""
        }`;
  const live = !transportStale && sampleState === "live";

  return (
    <div
      data-testid="connection-indicator"
      className="flex min-w-0 items-center gap-2.5 rounded-xl border border-white/5 bg-slate-950/60 px-2.5 py-2 shadow-inner shadow-black/20 sm:px-3"
      title={`${label} (${detail})`}
    >
      <span aria-hidden="true" className={`relative h-2.5 w-2.5 shrink-0 rounded-full ${tone}`}>
        {live && <span className="absolute inset-0 motion-safe:animate-ping rounded-full bg-emerald-400 opacity-50" />}
      </span>
      <div className="sr-only sm:not-sr-only sm:min-w-0 sm:leading-none">
        <div className="truncate text-[11px] font-semibold text-slate-200">{label}</div>
        <div className="mt-1 font-mono text-[9px] text-muted">{detail}</div>
      </div>
    </div>
  );
}

/** The sidebar and brand share one navigation landmark. */
function Sidebar({
  onNavigate,
  navRef,
  id,
  compact = false,
}: {
  onNavigate?: () => void;
  navRef?: RefObject<HTMLElement>;
  id?: string;
  compact?: boolean;
}) {
  return (
    <nav id={id} ref={navRef} aria-label="Main" className="flex h-full flex-col bg-slate-950/95">
      <div className={`flex h-16 shrink-0 items-center border-b border-white/5 ${compact ? "justify-center" : "px-5"}`}>
        <span className="text-sm font-bold tracking-wide text-white">SO101</span>
      </div>

      <div className={`scroll-thin flex-1 space-y-1 overflow-y-auto py-5 ${compact ? "px-2" : "px-3"}`}>
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.end}
                    onClick={onNavigate}
                    title={item.label}
                    className={({ isActive }) =>
                      `group flex items-center rounded-xl text-sm font-medium transition-all focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-cyan-400/60 ${compact ? "mx-auto h-12 w-12 justify-center" : "gap-3 px-3 py-2.5"} ${
                        isActive
                          ? "bg-cyan-400/10 text-cyan-200 shadow-inner shadow-cyan-900/10"
                          : "text-slate-400 hover:bg-white/[0.04] hover:text-slate-100"
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        <Icon
                          aria-hidden="true"
                          className={`${compact ? "h-5 w-5" : "h-[17px] w-[17px]"} ${isActive ? "text-cyan-300" : "text-muted group-hover:text-slate-300"}`}
                        />
                        <span className={compact ? "sr-only" : "min-w-0 flex-1 truncate"}>{item.label}</span>
                        {!compact && item.badge && (
                          <span
                            aria-hidden="true"
                            className="rounded-md border border-white/5 bg-white/5 px-1.5 py-0.5 font-mono text-[8px] text-muted"
                          >
                            {item.badge}
                          </span>
                        )}
                        {!compact && isActive && <ChevronRight aria-hidden="true" className="h-3.5 w-3.5 text-cyan-500" />}
                      </>
                    )}
                  </NavLink>
                );
              })}
      </div>
    </nav>
  );
}

function setInert(element: HTMLElement | null, inert: boolean) {
  if (!element) return;
  if (inert) element.setAttribute("inert", "");
  else element.removeAttribute("inert");
}

function Shell({ children }: { children: ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  const label = routeLabel(location.pathname, location.search);
  const updateAvailable = useUpdateAvailable();
  const drawerNavId = useId();

  const mainRef = useRef<HTMLElement>(null);
  const sidebarRef = useRef<HTMLDivElement>(null);
  const drawerNavRef = useRef<HTMLElement>(null);
  const menuButtonRef = useRef<HTMLButtonElement>(null);
  const wasOpenRef = useRef(false);
  const currentPath = location.pathname + location.search;
  const operate = location.pathname === "/operate";
  const previousPathRef = useRef(currentPath);

  // Navigating closes the drawer.
  const [drawerPath, setDrawerPath] = useState(currentPath);
  if (drawerPath !== currentPath) {
    setDrawerPath(currentPath);
    setMobileOpen(false);
  }

  // The floating global stops stay interactive while the navigation is open.
  useEffect(() => {
    setInert(mainRef.current, mobileOpen);
    setInert(sidebarRef.current, mobileOpen);
    if (mobileOpen) {
      drawerNavRef.current?.querySelector<HTMLElement>("a")?.focus();
    } else if (wasOpenRef.current && previousPathRef.current === currentPath) {
      menuButtonRef.current?.focus();
    }
    wasOpenRef.current = mobileOpen;
  }, [mobileOpen, currentPath]);

  // Route change: announce via title and move focus to the new page (F27).
  useEffect(() => {
    document.title = `${label} · SO-101`;
    if (previousPathRef.current !== currentPath) {
      previousPathRef.current = currentPath;
      mainRef.current?.focus();
    }
  }, [label, currentPath]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <a
        href="#main-content"
        onClick={(event) => {
          event.preventDefault();
          mainRef.current?.focus();
        }}
        className="sr-only z-[60] rounded-lg bg-cyan-500 px-4 py-2 text-sm font-bold text-slate-950 focus:not-sr-only focus:fixed focus:left-3 focus:top-3"
      >
        Skip to content
      </a>

      <div ref={sidebarRef} className="fixed inset-y-0 left-0 z-30 hidden w-20 border-r border-white/5 lg:block">
        <Sidebar id="sidebar-nav" compact />
      </div>

      {mobileOpen && (
        <div data-testid="nav-drawer" className="fixed inset-x-0 bottom-0 top-20 z-40 lg:hidden">
          <button
            type="button"
            tabIndex={-1}
            aria-hidden="true"
            className="absolute inset-0 bg-black/70 backdrop-blur-xs"
            onClick={() => setMobileOpen(false)}
          />
          <div className="absolute inset-y-0 left-0 w-[min(19rem,88vw)] border-r border-white/10 shadow-2xl">
            <Sidebar id={drawerNavId} navRef={drawerNavRef} onNavigate={() => setMobileOpen(false)} />
          </div>
        </div>
      )}

      <div className={`lg:pl-20 ${operate ? "lg:flex lg:h-dvh lg:flex-col" : ""}`}>
        <header aria-label="Global robot stops" className="fixed right-3 top-3 z-50 rounded-xl bg-slate-950/90 p-1 shadow-lg backdrop-blur-sm">
          <SafetyBar compact />
        </header>
        <button
          ref={menuButtonRef}
          type="button"
          onClick={() => setMobileOpen((open) => !open)}
          className="fixed left-3 top-3 z-50 rounded-xl border border-white/10 bg-slate-950 p-2.5 text-slate-300 hover:bg-slate-900 lg:hidden"
          aria-label={mobileOpen ? "Close navigation" : "Open navigation"}
          aria-expanded={mobileOpen}
          aria-controls={mobileOpen ? drawerNavId : undefined}
        >
          {mobileOpen ? <X aria-hidden="true" className="h-5 w-5" /> : <Menu aria-hidden="true" className="h-5 w-5" />}
        </button>

        <SafetyBanners />
        {updateAvailable && (
          <div className="sticky top-16 z-30 px-3 pt-2 sm:px-6">
            <UpdateBanner />
          </div>
        )}

        <main
          id="main-content"
          ref={mainRef}
          tabIndex={-1}
          aria-label={label}
          className={`workstation-grid focus:outline-hidden ${operate ? "p-3 pt-20 lg:min-h-0 lg:flex-1 lg:overflow-hidden lg:pt-3" : "min-h-screen p-4 pt-20 sm:p-6 sm:pt-20 xl:p-8 xl:pt-20"}`}
        >
          <div className={`mx-auto w-full ${operate ? "lg:h-full lg:min-h-0" : "max-w-[1600px]"}`}>
            <ErrorBoundary resetKey={currentPath}>{children}</ErrorBoundary>
          </div>
        </main>
      </div>
    </div>
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <SafetyProvider>
      <ToastProvider>
        <ShortcutsProvider>
          <Shell>{children}</Shell>
        </ShortcutsProvider>
      </ToastProvider>
    </SafetyProvider>
  );
}
