import { useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { ListChecks, Keyboard, Settings as SettingsIcon } from "lucide-react";
import Button from "../components/Button";
import Card from "../components/Card";
import CameraSetup from "../components/CameraSetup";
import { useShortcuts } from "../components/shortcuts/ShortcutsProvider";
import { BoolPill, StatusPill } from "../components/ui/StatusPill";
import { useToast } from "../components/ui/Toast";
import { Notice, PageHeader } from "../components/Workstation";
import { getHealth } from "../api";
import { usePolling } from "../hooks/usePolling";
import { backendInfo } from "../lib/backend";
import { cameraLabel } from "../lib/cameras";
import { CHECKLIST_DISMISSED_KEY, usePersistentFlag } from "../lib/onboarding";

export const CHECKLIST_SHOWN_TOAST = "The setup checklist is back on Overview.";

export default function Settings() {
  const { data: health, error, refresh } = usePolling(getHealth, 5000);
  const healthBackend = backendInfo(health);
  const shortcuts = useShortcuts();
  const toast = useToast();
  const [checklistHidden, setChecklistHidden] = usePersistentFlag(CHECKLIST_DISMISSED_KEY);
  // "Show setup checklist" goes away when pressed; focus moves to the link
  // that replaces it instead of falling back to the page.
  const overviewLink = useRef<HTMLAnchorElement>(null);
  const focusOverviewLink = useRef(false);
  useEffect(() => {
    if (!checklistHidden && focusOverviewLink.current) {
      focusOverviewLink.current = false;
      overviewLink.current?.focus();
    }
  }, [checklistHidden]);

  return (
    <div className="space-y-6">
      <PageHeader
        icon={SettingsIcon}
        title="Settings"
        description="Backend health, configuration and operator help."
      />

      {error && (
        <Notice tone="danger" title="Backend health could not be read">
          {error.message}
        </Notice>
      )}

      <Card
        title="Operator help"
        subtitle="Reference for the keys this workstation answers to"
      >
        <div className="flex flex-wrap items-center gap-3">
          <Button variant="secondary" onClick={shortcuts.open} aria-haspopup="dialog">
            <Keyboard aria-hidden="true" className="h-4 w-4" /> Keyboard shortcuts
          </Button>
          <p className="text-xs text-slate-400">
            Press ? on any page for the same list. Escape is E-stop everywhere.
          </p>
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-3 border-t border-white/5 pt-4">
          {checklistHidden ? (
            <>
              <Button
                variant="secondary"
                onClick={() => {
                  focusOverviewLink.current = true;
                  setChecklistHidden(false);
                  toast.show(CHECKLIST_SHOWN_TOAST);
                }}
              >
                <ListChecks aria-hidden="true" className="h-4 w-4" /> Show setup checklist
              </Button>
              <p className="text-xs text-slate-400">It goes back on Overview, above the readiness summary.</p>
            </>
          ) : (
            <p className="text-xs text-slate-400">
              The setup checklist is showing on{" "}
              <Link ref={overviewLink} to="/" className="font-semibold text-cyan-300 hover:text-cyan-200">
                Overview
              </Link>
              . Hide it there when you are done with it.
            </p>
          )}
        </div>
      </Card>

      <Card
        title="Backend health"
        actions={
          <Button size="sm" variant="secondary" onClick={refresh}>
            Refresh
          </Button>
        }
      >
        {!health ? (
          <p className="text-sm text-muted">Loading…</p>
        ) : (
          <div className="space-y-4">
            <div className="flex flex-wrap gap-2">
              <BoolPill label="ok" value={health.ok} />
              <StatusPill label="version" value={health.version} tone="info" />
              <StatusPill label="mode" value={health.mode} tone="neutral" />
              <StatusPill
                label="backend"
                value={healthBackend ? healthBackend.kind : "unknown"}
                tone={healthBackend?.kind === "real" ? "neutral" : "warn"}
              />
              <StatusPill label="fps" value={health.fps.toFixed(0)} tone="neutral" />
            </div>
            <div className="flex flex-wrap gap-2">
              <BoolPill
                label={healthBackend?.kind === "sim" ? "follower (simulated)" : "follower"}
                value={health.follower_connected}
              />
              <BoolPill label="leader" value={health.leader_connected} />
              <StatusPill
                label="e-stop"
                value={health.estopped ? "TRIPPED" : "clear"}
                tone={health.estopped ? "danger" : "ok"}
              />
            </div>
            <div>
              <div className="mb-1 text-xs uppercase tracking-wide text-muted">
                Cameras
              </div>
              {health.cameras.length === 0 ? (
                <p className="text-sm text-muted">None reported.</p>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {health.cameras.map((c) => (
                    <StatusPill key={c} label="" value={cameraLabel(c)} tone="info" />
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </Card>

      <CameraSetup />

      <Card title="Configuration">
        <div className="space-y-3 text-sm text-slate-300">
          <p>
            The backend is configured via{" "}
            <code className="rounded-sm bg-slate-800 px-1.5 py-0.5 font-mono text-cyan-300">
              SO101_*
            </code>{" "}
            environment variables (ports, device paths, mock mode, camera config,
            etc.). These settings are read-only from this UI.
          </p>
          <p>
            For the full interactive API reference, see{" "}
            <a
              href="/docs"
              target="_blank"
              rel="noreferrer"
              className="text-cyan-400 underline-offset-2 hover:underline"
            >
              /docs
            </a>
            .
          </p>
        </div>
      </Card>
    </div>
  );
}
