import type { CalibrationLiveStatus } from "../api";
import Card from "./Card";

const JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"];

function reading(value: number | undefined | null) {
  return value == null ? "—" : String(value);
}

export default function CalibrationLive({
  live, error,
}: {
  live?: CalibrationLiveStatus | null;
  error?: string | null;
}) {
  const stale = Boolean(error || (live && !live.frozen && live.age_seconds > 2));
  return (
    <Card title="Live calibration readings" subtitle="Raw encoder ticks · expected limits 0–4095 · measured span at least 100 ticks">
      {error && <p role="alert" className="mb-3 text-sm text-amber-200">Status unavailable: {error}. Readings may be stale.</p>}
      {!live ? (
        <p className="text-sm text-muted">No live encoder samples for this attempt. Readings appear during a real calibration; mock mode has no encoder measurements.</p>
      ) : (
        <>
          <p className="mb-3 text-xs text-muted">
            {live.joint ? `Joint baseline: ${live.joint} · ` : ""}Phase: {live.phase} · {live.sample_count} samples · {live.elapsed_seconds.toFixed(1)} s captured · update #{live.sequence}
            {" · "}{live.frozen ? "Frozen final readings" : stale ? "Stale readings — waiting for worker" : "Live worker readings"}
            {" · "}{live.age_seconds.toFixed(1)} s since update
          </p>
          {live.error && <p className="mb-3 text-sm text-red-300">{live.error}</p>}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <caption className="sr-only">Per-motor encoder values and calibration checks</caption>
              <thead className="text-muted">
                <tr>{["Motor", "Check", "Current", "Start", "Min", "Max", "Span", "Homing requested / read", "Reason"].map((title) => (
                  <th key={title} scope="col" className="whitespace-nowrap p-2">{title}</th>
                ))}</tr>
              </thead>
              <tbody>
                {JOINTS.filter((motor) => !live.joint || live.joint === motor).map((motor) => {
                  const fixed = motor === "wrist_roll" && !live.joint;
                  const min = live.range_mins?.[motor];
                  const max = live.range_maxes?.[motor];
                  const span = min != null && max != null ? max - min : null;
                  const requested = live.homing_offsets_requested?.[motor];
                  const readback = live.homing_readback?.[motor];
                  const homingBad = Boolean(readback && requested != null && (
                    readback.homing_offset !== requested || readback.range_min !== 0 || readback.range_max !== 4095
                  ));
                  const boundsBad = !fixed && ((min != null && min < 0) || (max != null && max > 4095));
                  const short = !fixed && span != null && span < 100;
                  const ended = live.phase === "failed" || live.phase === "complete";
                  const namedFailure = live.affected_motors?.includes(motor);
                  const failed = homingBad || boundsBad || namedFailure || (ended && short);
                  const ready = Boolean(readback && !homingBad && (fixed || (span != null && span >= 100)));
                  const label = failed ? "FAIL" : live.phase === "complete" && ready ? "PASS" : ready ? "Valid so far" : "Pending";
                  const reason = homingBad ? "Homing read-back mismatch"
                    : boundsBad ? "Outside 0–4095"
                    : namedFailure ? "Reported motor failure"
                    : short ? ended ? "Span below 100 ticks" : "Sweep needed: span below 100 ticks"
                    : fixed ? "Fixed 0–4095; no sweep measurement"
                    : ready ? "Range checks met; full safe sweep still required"
                    : "Waiting for midpoint / encoder data";
                  return (
                    <tr key={motor} className="border-t border-white/5">
                      <th scope="row" className="whitespace-nowrap p-2 font-mono">{motor}</th>
                      <td className={`whitespace-nowrap p-2 font-semibold ${failed ? "text-red-300" : ready ? "text-emerald-300" : "text-amber-200"}`}>
                        {stale && !live.frozen ? "Stale · " : ""}{label}
                      </td>
                      <td className="p-2 font-mono">{reading(live.positions?.[motor])}</td>
                      <td className="p-2 font-mono">{reading(live.starting_positions?.[motor])}</td>
                      <td className="p-2 font-mono">{fixed ? "0 (fixed)" : reading(min)}</td>
                      <td className="p-2 font-mono">{fixed ? "4095 (fixed)" : reading(max)}</td>
                      <td className="p-2 font-mono">{fixed ? "—" : reading(span)}</td>
                      <td className="whitespace-nowrap p-2 font-mono">{reading(requested)} / {reading(readback?.homing_offset)}</td>
                      <td className="min-w-48 p-2 text-muted">{reason}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-xs text-muted">
            “Valid so far” checks the observed values; it does not prove that both physical end stops were reached.
            {live.baseline_only ? "Joint baselines restore prior calibration registers and do not qualify the full arm." : "Failed attempts restore the prior calibration."} Frozen values describe the attempted sweep, not the restored registers.
          </p>
        </>
      )}
    </Card>
  );
}
