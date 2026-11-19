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
import { useLocation, useNavigate } from "react-router-dom";
import { isEditableTarget } from "../../lib/keyboard";
import { navItemForShortcut } from "../navigation";
import { useSafety } from "../safety/SafetyProvider";
import ShortcutsDialog from "./ShortcutsDialog";

/** After G, the page letter must follow within this many milliseconds. */
export const GO_SEQUENCE_MS = 1500;

export interface ShortcutsApi {
  isOpen: boolean;
  open: () => void;
  close: () => void;
}

const INERT_API: ShortcutsApi = { isOpen: false, open: () => undefined, close: () => undefined };
const ShortcutsContext = createContext<ShortcutsApi>(INERT_API);

function modalOpen(): boolean {
  return document.querySelector('[aria-modal="true"]') !== null;
}

/**
 * Navigation and help keys. Safety keys are not here: SafetyProvider takes
 * Escape (E-stop) and H (Halt) in the capture phase and marks them handled,
 * so this bubble-phase listener never sees them as navigation (G then H
 * halts; it does not navigate). Jog keys belong to Manual joint control,
 * which listens only while its deadman is held.
 *
 * Navigation is safe at any time: leaving the teleop page unmounts the
 * manual panel, which stops motion and releases its lease.
 */
export function ShortcutsProvider({ children }: { children: ReactNode }) {
  const { view } = useSafety();
  const navigate = useNavigate();
  const location = useLocation();
  const [isOpen, setIsOpen] = useState(false);

  const open = useCallback(() => setIsOpen(true), []);
  const close = useCallback(() => setIsOpen(false), []);

  const stateRef = useRef({ isOpen, navigate, pathname: location.pathname });
  useEffect(() => {
    stateRef.current = { isOpen, navigate, pathname: location.pathname };
  });

  useEffect(() => {
    let goTimer: number | null = null;
    const clearGo = () => {
      if (goTimer !== null) window.clearTimeout(goTimer);
      goTimer = null;
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (
        event.defaultPrevented ||
        event.isComposing ||
        event.ctrlKey ||
        event.metaKey ||
        event.altKey ||
        isEditableTarget(event.target)
      ) {
        clearGo();
        return;
      }
      if (event.repeat) return;
      const state = stateRef.current;
      if (event.key === "?") {
        if (state.isOpen) setIsOpen(false);
        else if (!modalOpen()) setIsOpen(true);
        else return;
        clearGo();
        event.preventDefault();
        return;
      }
      // Nothing navigates out from under an open dialog.
      if (state.isOpen || modalOpen()) {
        clearGo();
        return;
      }
      if (goTimer !== null) {
        clearGo();
        const item = navItemForShortcut(event.key);
        if (!item) return;
        event.preventDefault();
        if (item.to !== state.pathname) void state.navigate(item.to);
        return;
      }
      if (event.key === "g" || event.key === "G") {
        goTimer = window.setTimeout(() => {
          goTimer = null;
        }, GO_SEQUENCE_MS);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      clearGo();
    };
  }, []);

  const api = useMemo<ShortcutsApi>(() => ({ isOpen, open, close }), [isOpen, open, close]);

  return (
    <ShortcutsContext.Provider value={api}>
      {children}
      <ShortcutsDialog open={isOpen} canHalt={view.canHalt} onClose={close} />
    </ShortcutsContext.Provider>
  );
}

/** Open or close the shortcut list. Outside the provider it does nothing. */
export function useShortcuts(): ShortcutsApi {
  return useContext(ShortcutsContext);
}
