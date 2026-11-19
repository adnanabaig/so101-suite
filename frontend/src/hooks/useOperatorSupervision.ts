import { useEffect, useRef, useState } from "react";
import { heartbeatOperator } from "../api";

/**
 * Proves that the owning operation page is visible and responsive. Navigating
 * away, backgrounding the tab, losing the network, or unmounting stops pulses;
 * the backend then stops its loop within two seconds.
 */
export function useOperatorSupervision() {
  const inFlight = useRef(false);
  const [healthy, setHealthy] = useState(false);

  useEffect(() => {
    let disposed = false;
    const pulse = async () => {
      if (disposed || document.visibilityState !== "visible" || inFlight.current) return;
      inFlight.current = true;
      try {
        await heartbeatOperator();
        if (!disposed) setHealthy(true);
      } catch {
        if (!disposed) setHealthy(false);
      } finally {
        inFlight.current = false;
      }
    };
    const onVisibility = () => {
      if (document.visibilityState !== "visible") setHealthy(false);
      else void pulse();
    };
    void pulse();
    const interval = window.setInterval(() => void pulse(), 500);
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      disposed = true;
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  return healthy;
}
