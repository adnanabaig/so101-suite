import { useState } from "react";
import { getSimulationCapabilities } from "../../api";
import { usePolling } from "../../hooks/usePolling";
import { renderProbing } from "./engine";

/** How often capabilities are read once the server's render probe has answered. */
export const CAPABILITIES_POLL_MS = 10_000;
/**
 * While the server is still probing GL (render_available null), ask again this
 * often. The server counts a probe as hung after 10 s and then reports it as
 * unavailable, so the fast poll ends by itself.
 */
export const RENDER_PROBE_POLL_MS = 1000;

/** GET /api/simulation/capabilities, polled faster while the render probe runs. */
export function useSimulationCapabilities() {
  const [probing, setProbing] = useState(false);
  const poll = usePolling(getSimulationCapabilities, probing ? RENDER_PROBE_POLL_MS : CAPABILITIES_POLL_MS);
  const nowProbing = renderProbing(poll.data);
  if (nowProbing !== probing) setProbing(nowProbing);
  return poll;
}
