import { useState } from "react";
import { getCameraPreviewStatus, startCameraPreview, stopCameraPreview } from "../api";
import { usePolling } from "../hooks/usePolling";
import { cameraLabel } from "../lib/cameras";
import { useSafety } from "./safety/SafetyProvider";
import Button from "./Button";
import Card from "./Card";
import CameraView from "./CameraView";

export default function CameraSetup() {
  const preview = usePolling(getCameraPreviewStatus, 1000);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const { status } = useSafety();
  async function toggle() {
    setBusy(true);
    setActionError(null);
    try {
      preview.mutate(await (preview.data?.active ? stopCameraPreview() : startCameraPreview()));
    } catch (error) {
      setActionError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }
  return (
    <Card title="Camera setup" subtitle="Preview the currently configured cameras. No motor connection or motion."
      actions={<Button size="sm" variant="secondary" disabled={busy || !preview.data || !!preview.error || !preview.data.cameras.length} onClick={() => void toggle()}>
        {busy ? "Working…" : preview.data?.active ? "Stop camera preview" : "Start camera preview"}
      </Button>}>
      <p className="mb-3 text-sm text-muted">Stop preview before calibration, teleop or recording. Camera names remain generic until you identify their physical views.</p>
      {(actionError || preview.error || preview.data?.error) && <p role="alert" className="mb-3 text-sm text-red-300">{actionError ?? preview.error?.message ?? preview.data?.error}</p>}
      {!preview.data && !preview.error && <p className="text-sm text-muted">Loading camera configuration…</p>}
      {preview.data?.cameras.length === 0 && <p className="text-sm text-muted">No cameras configured.</p>}
      <div className="grid gap-4 xl:grid-cols-2">
        {preview.data?.cameras.map((cam) => {
          const mode = preview.data?.devices[cam]?.device.streams?.color;
          return <div key={cam} className="min-w-0 space-y-2">
            <CameraView cam={cam} active={(status?.live_cameras ?? []).includes(cam)} />
            {mode && <p className="text-xs text-muted">{cameraLabel(cam)}: negotiated {mode.width} × {mode.height} · {mode.fps} fps{mode.fourcc ? ` · ${mode.fourcc}` : ""}</p>}
          </div>;
        })}
      </div>
    </Card>
  );
}
