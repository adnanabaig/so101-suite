// `physics_state` from the MuJoCo engine: contacts, servo load and the
// catch-up counter. The kinematic engine sends null and this is not shown.
import { Fragment } from "react";
import type { SimulationPhysicsState } from "../../api";
import { SectionLabel } from "../Workstation";

const JOINT_LABELS: Record<string, string> = {
  shoulder_pan: "Shoulder pan",
  shoulder_lift: "Shoulder lift",
  elbow_flex: "Elbow flex",
  wrist_flex: "Wrist flex",
  wrist_roll: "Wrist roll",
  gripper: "Gripper",
};

function formatEffort(value: number | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(1)} %` : "—";
}

function count(value: number | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? String(value) : "—";
}

export default function PhysicsReadout({
  physics,
  jointOrder,
}: {
  physics: SimulationPhysicsState;
  jointOrder: readonly string[];
}) {
  // physics_state is untyped on the server, so read it defensively.
  const contacts: Partial<SimulationPhysicsState["contacts"]> = physics.contacts ?? {};
  const effort: Partial<Record<string, number>> = physics.effort_pct ?? {};
  const joints = jointOrder.length ? jointOrder : Object.keys(effort);
  const penetration = contacts.max_penetration_m;
  return (
    <section aria-label="Physics state" className="space-y-3">
      <SectionLabel>Physics</SectionLabel>
      <dl className="grid grid-cols-3 gap-2 text-xs">
        {[
          ["Contacts", count(contacts.total)],
          ["Arm on table", count(contacts.arm_table)],
          ["Arm on itself", count(contacts.arm_self)],
        ].map(([term, value]) => (
          <div key={term} className="rounded-lg bg-slate-950/50 p-2.5">
            <dt className="text-slate-400">{term}</dt>
            <dd className="mt-1 font-mono font-semibold text-slate-200">{value}</dd>
          </div>
        ))}
      </dl>
      <div>
        <div className="text-[10px] uppercase tracking-wide text-slate-400">
          Servo load (% of force limit)
        </div>
        <dl className="mt-1 grid grid-cols-[1fr_auto] gap-x-3 gap-y-0.5 font-mono text-[11px]">
          {joints.map((joint) => (
            <Fragment key={joint}>
              <dt className="text-slate-400">{JOINT_LABELS[joint] ?? joint}</dt>
              <dd className="text-right text-slate-200">
                {formatEffort(effort[joint])}
              </dd>
            </Fragment>
          ))}
        </dl>
      </div>
      <p className="text-[11px] leading-relaxed text-slate-400">
        Servo {physics.servo?.mode ?? "unknown"} mode · torque{" "}
        {physics.torque_enabled ? "on" : "off"}
        {typeof penetration === "number" && Number.isFinite(penetration)
          ? ` · deepest penetration ${(penetration * 1000).toFixed(2)} mm`
          : ""}
      </p>
      {physics.dropped_ticks > 0 && (
        <p role="note" className="rounded-lg border border-amber-400/20 bg-amber-400/[0.06] px-3 py-2 text-xs leading-relaxed text-amber-200">
          {physics.dropped_ticks} tick{physics.dropped_ticks === 1 ? " was" : "s were"} skipped
          because nothing polled the simulator for more than 2 s during play. The clock counts
          only simulated ticks.
        </p>
      )}
    </section>
  );
}
