import { useState } from "react";
import { api } from "../api";
import { usePolling } from "../hooks/usePolling";
import { useSafety } from "./safety/SafetyProvider";
import Button from "./Button";

interface Control { value: number; minimum: number; maximum: number; step: number; writable: boolean; inactive: boolean; label: string; choices?: Record<string, string> }
interface State { live: boolean; actual_capture_fps: number; camera_profile: { identity: string; controls: Record<string, Control>; saved_values: Record<string, number>; profile_error: string | null; exposure_unit: string; profile_sha256: string }; image_quality: { mean_luma?: number; dark_fraction?: number; clipped_fraction?: number } }

export default function CameraTuning() {
  const [camera, setCamera] = useState("camera_1");
  const [draft, setDraft] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const state = usePolling(() => api.get<State>(`/api/mcap/cameras/${camera}/controls`), 1500, { key: camera });
  const { status, statusStale } = useSafety();
  const editable = status?.mode === "idle" && !statusStale && state.data?.live;
  async function apply() {
    setBusy(true); setError(null);
    try {
      await api.post(`/api/mcap/cameras/${camera}/controls`, { values: draft });
      setDraft({}); state.refresh();
    } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)); }
    finally { setBusy(false); }
  }
  const profile = state.data?.camera_profile;
  return <section className="space-y-3">
    <p className="text-sm text-slate-300">Match each view under the same lighting with a neutral white/gray reference. Different sensors need different exposure and white balance values. Save settings while teleop and recording are stopped.</p>
    <label className="block text-sm">Camera <select aria-label="Camera settings device" value={camera} onChange={(event) => { setCamera(event.target.value); setDraft({}); setError(null); }} className="ml-2 rounded bg-slate-800 p-2">{["obsbot", "camera_1", "camera_2", "realsense"].map(name => <option key={name}>{name}</option>)}</select></label>
    <img alt={`Tuning preview ${camera}`} src={`/api/mcap/cameras/${camera}/stream.mjpg`} className="max-h-56 w-full object-contain" />
    <p className="text-xs text-slate-400">Measured capture {state.data?.actual_capture_fps?.toFixed(1) ?? "—"} fps · exposure units: {profile?.exposure_unit ?? "unavailable"}. The device reporting 30 fps is not proof of delivered 30 fps.</p>
    <p className="text-xs">Luma {state.data?.image_quality?.mean_luma?.toFixed(1) ?? "—"}/255 · dark {((state.data?.image_quality?.dark_fraction ?? 0) * 100).toFixed(1)}% · clipped {((state.data?.image_quality?.clipped_fraction ?? 0) * 100).toFixed(1)}%</p>
    {(error || state.error || profile?.profile_error) && <p role="alert" className="text-amber-200">{error ?? state.error?.message ?? profile?.profile_error}</p>}
    {!editable && <p className="text-amber-200 text-xs">Controls require an online camera and an idle workstation.</p>}
    <div className="grid gap-3 md:grid-cols-2">{Object.entries(profile?.controls ?? {}).map(([name, control]) => <label key={name} className="text-xs">
      <span className="block mb-1">{control.label} · readback {control.value}{control.inactive ? " · automatic mode controls this setting" : ""}</span>
      {control.choices ? <select aria-label={name} value={draft[name] ?? control.value} disabled={!editable || busy || !control.writable} className="w-full rounded bg-slate-800 p-2" onChange={event => setDraft(values => ({ ...values, [name]: Number(event.target.value) }))}>{Object.entries(control.choices).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select> : <input aria-label={name} type="number" min={control.minimum} max={control.maximum} step={control.step || 1} value={draft[name] ?? control.value} disabled={!editable || busy || !control.writable} className="w-full rounded bg-slate-800 p-2" onChange={event => setDraft(values => ({ ...values, [name]: Number(event.target.value) }))} />}
    </label>)}</div>
    <Button disabled={!editable || busy || !Object.keys(draft).length} onClick={() => void apply()}>{busy ? "Verifying readback…" : "Apply and save camera settings"}</Button>
    <p className="text-xs text-slate-400">Disable automatic exposure/white balance and apply first to unlock manual values. Settings are saved per device, restored after reconnect and included in MCAP camera status. Unsupported controls are omitted.</p>
    <details><summary className="text-xs text-cyan-300">Device identity and saved profile</summary><pre className="max-h-52 overflow-auto text-xs">{JSON.stringify({ identity: profile?.identity, saved: profile?.saved_values, sha256: profile?.profile_sha256 }, null, 2)}</pre></details>
  </section>;
}
