import { useState } from "react";
import { getMotorVision, startMotorVision, stopMotorVision } from "../api";
import { usePolling } from "../hooks/usePolling";
import { errorMessage } from "../lib/errors";
import Button from "./Button";

export default function MotorVisionPanel() {
  const vision = usePolling(getMotorVision, 1000);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const toggle = async () => {
    setBusy(true); setError(null);
    try { vision.mutate(await (vision.data?.running ? stopMotorVision() : startMotorVision())); }
    catch (caught) { setError(errorMessage(caught)); }
    finally { setBusy(false); }
  };
  return <div className="rounded-xl border border-emerald-400/20 p-3">
    <div className="mb-2 flex flex-wrap items-center gap-3">
      <h3 className="font-semibold text-emerald-300">Computer vision · green motor tags</h3>
      <Button variant="secondary" disabled={busy} onClick={() => void toggle()}>{vision.data?.running ? "Stop vision" : "Start vision"}</Button>
      <a className="text-xs text-cyan-300 underline" href="/api/mcap/vision/tags.png" target="_blank" rel="noreferrer">Printable motor tags</a>
    </div>
    <p className="mb-3 text-xs text-slate-400">Independent image detections use ArUco 4×4 tags 1–6 attached beside the matching motors. 1 pan · 2 lift · 3 elbow · 4 wrist flex · 5 wrist roll · 6 gripper. Keep tags and adhesive clear of joints, cables and travel.</p>
    {vision.data?.live && !vision.stale ? <img className="aspect-video w-full object-contain" src="/api/mcap/vision/stream.mjpg" alt="OBSBOT with green detected motor tag centers and connecting lines" /> : <p className="p-4 text-sm text-slate-400">Waiting for a fresh vision frame.</p>}
    <p className="my-2 text-xs text-emerald-300">Detected: {vision.data?.detected.map((point) => point.id).join(", ") || "none"} / 6</p>
    <p className="text-xs text-amber-200">Tag centers are visual landmarks. Camera projection and tag-to-pivot offsets are not calibrated; use this comparison for observation, not motion or automatic calibration.</p>
    {(error || vision.data?.error) && <p role="alert" className="mt-2 text-sm text-rose-300">{error || vision.data?.error}</p>}
  </div>;
}
