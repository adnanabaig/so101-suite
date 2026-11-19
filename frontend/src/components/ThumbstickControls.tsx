import { useEffect, useMemo, useRef, useState } from "react";
import { isEditableTarget } from "../lib/keyboard";
import { CENTERED_STICKS, keyboardAxes, STICK_KEYS, type StickAxes } from "../lib/thumbsticks";

export default function ThumbstickControls({ enabled, onAxesChange }: {
  enabled: boolean;
  onAxesChange: (axes: StickAxes) => void;
}) {
  const [keys, setKeys] = useState<Set<string>>(new Set());
  const [pointer, setPointer] = useState({ left: null as { x: number; y: number } | null, right: null as { x: number; y: number } | null });
  const pointers = useRef<Record<string, number>>({});
  const axes = useMemo(() => {
    if (!enabled) return CENTERED_STICKS;
    const result = keyboardAxes(keys);
    if (pointer.left) { result.leftX = pointer.left.x; result.leftY = pointer.left.y; }
    if (pointer.right) { result.rightX = pointer.right.x; result.rightY = pointer.right.y; }
    return result;
  }, [enabled, keys, pointer]);

  useEffect(() => { onAxesChange(axes); }, [axes, onAxesChange]);
  useEffect(() => {
    if (!enabled) return;
    const clear = () => { setKeys(new Set()); setPointer({ left: null, right: null }); pointers.current = {}; };
    const down = (event: KeyboardEvent) => {
      const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
      if (!STICK_KEYS.has(key) || event.ctrlKey || event.altKey || event.metaKey || isEditableTarget(event.target)) return;
      event.preventDefault();
      setKeys((previous) => new Set([...previous, key]));
    };
    const up = (event: KeyboardEvent) => {
      const key = event.key.length === 1 ? event.key.toLowerCase() : event.key;
      if (!STICK_KEYS.has(key)) return;
      setKeys((previous) => { const next = new Set(previous); next.delete(key); return next; });
    };
    window.addEventListener("keydown", down); window.addEventListener("keyup", up);
    window.addEventListener("blur", clear); document.addEventListener("visibilitychange", clear);
    return () => {
      window.removeEventListener("keydown", down); window.removeEventListener("keyup", up);
      window.removeEventListener("blur", clear); document.removeEventListener("visibilitychange", clear);
      onAxesChange(CENTERED_STICKS);
    };
  }, [enabled, onAxesChange]);

  return <div className="mb-5 rounded-xl border border-cyan-400/20 bg-slate-950/30 p-4">
    <div className="grid grid-cols-2 gap-6">
      {(["left", "right"] as const).map((side) => {
        const x = side === "left" ? axes.leftX : axes.rightX;
        const y = side === "left" ? axes.leftY : axes.rightY;
        const update = (event: React.PointerEvent<HTMLDivElement>) => {
          if (!enabled || pointers.current[side] !== event.pointerId) return;
          const box = event.currentTarget.getBoundingClientRect();
          const radius = box.width * 0.35;
          const horizontal = (event.clientX - box.left - box.width / 2) / radius;
          const vertical = (box.top + box.height / 2 - event.clientY) / radius;
          const magnitude = Math.max(1, Math.hypot(horizontal, vertical));
          setPointer((previous) => ({ ...previous, [side]: { x: horizontal / magnitude, y: vertical / magnitude } }));
        };
        const release = () => { delete pointers.current[side]; setPointer((previous) => ({ ...previous, [side]: null })); };
        return <div key={side} className="text-center">
          <div role="group" aria-label={`${side} thumbstick`} aria-disabled={!enabled}
            className={`relative mx-auto aspect-square w-36 touch-none rounded-full border border-white/20 bg-slate-900 ${enabled ? "cursor-grab" : "opacity-40"}`}
            onPointerDown={(event) => {
              if (!enabled || (event.pointerType !== "touch" && event.button !== 0)) return;
              event.preventDefault(); pointers.current[side] = event.pointerId;
              event.currentTarget.setPointerCapture(event.pointerId); update(event);
            }} onPointerMove={update} onPointerUp={release} onPointerCancel={release} onLostPointerCapture={release}>
            <div className="absolute left-1/2 top-1/2 h-12 w-12 rounded-full border border-cyan-300 bg-cyan-500/40"
              style={{ transform: `translate(calc(-50% + ${x * 48}px), calc(-50% - ${y * 48}px))` }} />
          </div>
          <p className="mt-2 text-sm text-white">{side === "left" ? "WASD" : "Arrow keys"}</p>
          <p className="text-xs text-slate-400">{side === "left" ? "Shoulder pan / lift" : "Elbow / wrist flex"}</p>
        </div>;
      })}
    </div>
    <p className="mt-3 text-xs text-slate-300">Hold “Hold to move” first. Q/E: wrist roll · Z/X: gripper. Maximum 4 degrees or gripper units per second.</p>
    <p className="mt-1 text-xs text-slate-400">Releasing the motion hold stops control. Key release centers its input. Confirm each joint’s physical direction during bring-up.</p>
  </div>;
}
