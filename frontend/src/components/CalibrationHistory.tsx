import { getCalibrationHistory, type CalibrationAttempt } from "../api";
import { usePolling } from "../hooks/usePolling";
import Button from "./Button";
import Card from "./Card";
import { StatusPill, type StatusTone } from "./ui/StatusPill";

const RESULTS: Record<CalibrationAttempt["result"], { label: string; tone: StatusTone }> = {
  pass: { label: "PASS", tone: "ok" },
  fail: { label: "FAIL", tone: "danger" },
  running: { label: "RUNNING", tone: "info" },
  incomplete: { label: "INCOMPLETE", tone: "warn" },
};

function attemptLog(attempt: CalibrationAttempt): string {
  const lines = attempt.log.map((entry) => `${entry.timestamp} [${entry.level}] ${entry.message}`);
  if (Object.keys(attempt.diagnostics).length > 0) {
    lines.push("", "Encoder and homing diagnostics", JSON.stringify(attempt.diagnostics, null, 2));
  }
  return lines.join("\n");
}

export default function CalibrationHistory() {
  const history = usePolling(getCalibrationHistory, 2000);
  const attempts = history.data?.attempts ?? [];
  return (
    <Card
      title="Calibration history"
      subtitle="Latest 100 attempts, newest first. Saved on this workstation across refreshes and restarts."
      actions={<Button variant="secondary" onClick={history.refresh}>Refresh history</Button>}
    >
      {history.error && (
        <p role="alert" className="mb-4 text-sm text-red-300">
          Could not refresh calibration history: {history.error.message}
          {attempts.length > 0 ? " Showing the last received records." : ""}
        </p>
      )}
      {history.loading && <p className="text-sm text-slate-400">Loading calibration history…</p>}
      {!history.loading && !history.error && attempts.length === 0 && (
        <p className="text-sm text-slate-400">No calibration attempts recorded yet.</p>
      )}
      {attempts.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <caption className="sr-only">Calibration attempts and their diagnostic logs</caption>
            <thead className="border-b border-slate-700 text-xs uppercase text-slate-400">
              <tr>
                {["Time", "Arm", "Pass / fail", "Affected motors", "Error log"].map((heading) => (
                  <th key={heading} scope="col" className="px-3 py-3 font-medium">{heading}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {attempts.map((attempt) => {
                const result = RESULTS[attempt.result];
                const timestamp = attempt.started_at ?? attempt.finished_at;
                return (
                  <tr key={attempt.id} className="align-top">
                    <td className="min-w-40 px-3 py-4 text-xs text-slate-400">
                      {timestamp ? <time dateTime={timestamp} title={timestamp}>{new Date(timestamp).toLocaleString()}</time> : "Not recorded"}
                      <div className="mt-1">{attempt.sample_seconds === null ? "Duration not recorded" : `${attempt.sample_seconds}s requested`}</div>
                    </td>
                    <td className="px-3 py-4">
                      <div className="font-medium text-slate-200">{attempt.arm}</div>
                      <div className="mt-1 text-xs text-cyan-300">{attempt.joint ? `Joint baseline · ${attempt.joint}` : "Full arm"}</div>
                      <div className="mt-1 text-xs text-slate-400">{attempt.backend ?? "Backend not recorded"}</div>
                      {attempt.device_id && <div className="mt-1 break-all text-xs text-muted">{attempt.device_id}</div>}
                    </td>
                    <td className="px-3 py-4">
                      <StatusPill label={result.label} tone={result.tone} />
                      {attempt.result === "fail" && attempt.saved === true && (
                        <p className="mt-2 text-xs text-amber-300">File saved; attempt still failed.</p>
                      )}
                      {attempt.result === "incomplete" && <p className="mt-2 text-xs text-slate-400">No completion recorded.</p>}
                    </td>
                    <td className="min-w-36 px-3 py-4 font-mono text-xs text-slate-300">
                      {attempt.affected_motors.length > 0
                        ? attempt.affected_motors.map((motor) => <div key={motor}>{motor}</div>)
                        : attempt.result === "fail" ? "Not identified" : "—"}
                    </td>
                    <td className="min-w-72 max-w-xl px-3 py-4 text-xs">
                      <p className="whitespace-pre-wrap break-words text-slate-300">
                        {attempt.error ?? (attempt.result === "pass" ? attempt.joint ? "Joint baseline saved; arm calibration unchanged." : "Calibration saved successfully." : "Awaiting completion.")}
                      </p>
                      <details className="mt-3">
                        <summary className="cursor-pointer text-cyan-300">View full log</summary>
                        <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-slate-950 p-3 font-mono text-slate-300">{attemptLog(attempt)}</pre>
                      </details>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}
