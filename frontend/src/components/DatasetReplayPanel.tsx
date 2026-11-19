import { useEffect, useMemo, useState } from "react";
import { Pause, Play, RotateCcw, ShieldCheck } from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EpisodeTimeline } from "../api";
import Button from "./Button";
import { StatusPill } from "./ui/StatusPill";

const COLORS = ["#22d3ee", "#818cf8", "#f59e0b", "#34d399", "#f472b6", "#a3e635"];

function formatJoint(name: string) {
  return name.replace(/_/g, " ");
}

export default function DatasetReplayPanel({
  timeline,
  loading,
}: {
  timeline: EpisodeTimeline | null;
  loading: boolean;
}) {
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);
  // Rewind when a different episode arrives. Adjusting state during render
  // (instead of in an effect) avoids painting one frame at the old cursor.
  const timelineKey = timeline ? `${timeline.repo_id}#${timeline.episode}` : null;
  const [cursorKey, setCursorKey] = useState(timelineKey);
  if (cursorKey !== timelineKey) {
    setCursorKey(timelineKey);
    setCursor(0);
    setPlaying(false);
  }

  useEffect(() => {
    if (!playing || !timeline?.points.length) return;
    const delay = Math.max(
      30,
      Math.round((1000 * timeline.stride) / Math.max(timeline.fps, 1)),
    );
    const timer = window.setInterval(() => {
      setCursor((current) => {
        if (current >= timeline.points.length - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, delay);
    return () => window.clearInterval(timer);
  }, [playing, timeline]);

  const chartData = useMemo(
    () =>
      (timeline?.points ?? []).map((point) => ({
        frame: point.frame_index,
        seconds: point.relative_time_seconds,
        ...Object.fromEntries(
          point.action.map((value, index) => [`joint_${index}`, value]),
        ),
      })),
    [timeline],
  );
  const point = timeline?.points[cursor];

  if (loading) {
    return (
      <div className="flex min-h-72 items-center justify-center text-sm text-muted">
        <span className="mr-3 h-5 w-5 motion-safe:animate-spin rounded-full border-2 border-slate-700 border-t-cyan-400" />
        Loading read-only replay preview…
      </div>
    );
  }

  if (!timeline) {
    return (
      <div className="flex min-h-72 items-center justify-center text-sm text-muted">
        Select an episode to inspect its action trace.
      </div>
    );
  }

  if (!timeline.capability.available) {
    return (
      <div className="flex min-h-72 flex-col items-center justify-center px-6 text-center">
        <p className="text-sm font-semibold text-amber-200">Preview capability unavailable</p>
        <p className="mt-2 max-w-xl text-xs leading-relaxed text-slate-400">
          {timeline.capability.detail}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <StatusPill label="episode" value={timeline.episode} tone="info" />
          <StatusPill
            label="sampled"
            value={`${timeline.returned_frames}/${timeline.total_frames ?? "?"}`}
          />
          <StatusPill label="stride" value={timeline.stride} />
          <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/25 bg-emerald-500/10 px-2.5 py-1 text-[11px] font-semibold text-emerald-300">
            <ShieldCheck aria-hidden="true" size={13} /> Preview only · no hardware
          </span>
        </div>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="secondary"
            onClick={() => setCursor(0)}
            disabled={!timeline.points.length}
          >
            <RotateCcw aria-hidden="true" size={13} /> Reset
          </Button>
          <Button
            size="sm"
            onClick={() => {
              if (cursor >= timeline.points.length - 1) setCursor(0);
              setPlaying((value) => !value);
            }}
            disabled={!timeline.points.length}
          >
            {playing ? <Pause aria-hidden="true" size={13} /> : <Play aria-hidden="true" size={13} />}
            {playing ? "Pause" : "Play trace"}
          </Button>
        </div>
      </div>

      {chartData.length > 0 ? (
        <div className="h-64 rounded-xl border border-slate-800 bg-slate-950/60 p-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 8, right: 12, left: -14, bottom: 2 }}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="seconds"
                type="number"
                domain={["dataMin", "dataMax"]}
                tickFormatter={(value: number) => `${value.toFixed(1)}s`}
                tick={{ fill: "#64748b", fontSize: 10 }}
                axisLine={{ stroke: "#334155" }}
              />
              <YAxis
                tick={{ fill: "#64748b", fontSize: 10 }}
                axisLine={{ stroke: "#334155" }}
                tickFormatter={(value: number) => `${Math.round(value)}°`}
              />
              <Tooltip
                contentStyle={{
                  background: "#020617",
                  border: "1px solid #334155",
                  borderRadius: 10,
                  fontSize: 11,
                }}
                labelFormatter={(value) => `${Number(value).toFixed(3)} seconds`}
              />
              {timeline.joint_names.map((joint, index) => (
                <Line
                  key={joint}
                  type="monotone"
                  dataKey={`joint_${index}`}
                  name={formatJoint(joint)}
                  stroke={COLORS[index % COLORS.length]}
                  dot={false}
                  strokeWidth={1.6}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-8 text-center text-sm text-muted">
          This episode contains no previewable action frames.
        </div>
      )}

      {timeline.points.length > 0 && point && (
        <>
          <input
            aria-label="Replay preview frame"
            type="range"
            min={0}
            max={timeline.points.length - 1}
            value={cursor}
            onChange={(event) => {
              setPlaying(false);
              setCursor(Number(event.target.value));
            }}
            className="w-full accent-cyan-400"
          />
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {timeline.joint_names.map((joint, index) => (
              <div
                key={joint}
                className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2"
              >
                <div className="flex items-center justify-between gap-2 text-xs">
                  <span className="capitalize text-slate-400">{formatJoint(joint)}</span>
                  <span className="font-mono font-semibold text-slate-100">
                    {point.action[index]?.toFixed(2) ?? "—"}°
                  </span>
                </div>
                <div className="mt-1 text-[10px] text-muted">
                  desired {point.desired_action[index]?.toFixed(2) ?? "—"}°
                </div>
              </div>
            ))}
          </div>
          <p className="text-right font-mono text-[10px] text-muted">
            frame {point.frame_index} · {point.relative_time_seconds.toFixed(3)}s
          </p>
        </>
      )}
    </div>
  );
}
