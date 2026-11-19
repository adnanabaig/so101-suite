import { useCallback, useEffect, useRef, useState } from "react";
import {
  armControl,
  createControlLease,
  disarmControl,
  haltRobot,
  heartbeatControlLease,
  releaseControlLease,
} from "../api";
import { useSafety } from "../components/safety/SafetyProvider";
import { errorMessage, joinSentences } from "../lib/errors";

/**
 * Round trips slower than this do not update the clock offset (the midpoint
 * estimate is too uncertain) and block arming, because motion commands carry
 * a 300 ms TTL measured on the server clock (F12).
 */
export const MAX_SYNC_ROUND_TRIP_MS = 250;
export const LEASE_HEARTBEAT_MS = 2500;

/**
 * Exclusive lease plus a press-and-hold arm window for manual/tracking input.
 *
 * Stopping (deadman release, blur, a failed command) asks the server to Halt
 * when it advertises Halt (C11: deadman release is a Halt), otherwise it
 * disarms with the lease (C5; on v0.1 servers that turns torque off). It only
 * sends anything when this hook may have armed the server.
 */
export function useMotionLease(source: string) {
  const { view } = useSafety();
  const [leaseId, setLeaseId] = useState<string | null>(null);
  const [motionActive, setMotionActive] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [clockSynced, setClockSynced] = useState(false);
  const leaseRef = useRef<string | null>(null);
  const motionGeneration = useRef(0);
  const motionAllowed = useRef(false);
  const sequence = useRef(0);
  const clockOffsetMs = useRef(0);
  const clockSyncedRef = useRef(false);
  const lastRoundTripMs = useRef<number | null>(null);
  // True from the moment an arm request may have been sent until a stop succeeds.
  const serverMayBeArmed = useRef(false);
  const canHaltRef = useRef(view.canHalt);
  useEffect(() => {
    canHaltRef.current = view.canHalt;
  });

  /** Record a round trip; only a fast one updates the clock offset. */
  const updateClock = useCallback((serverTime: string, startedAt: number, endedAt: number) => {
    const roundTripMs = Math.max(0, endedAt - startedAt);
    lastRoundTripMs.current = roundTripMs;
    setLatencyMs(roundTripMs);
    const serverMs = Date.parse(serverTime);
    if (roundTripMs > MAX_SYNC_ROUND_TRIP_MS || Number.isNaN(serverMs)) return false;
    clockOffsetMs.current = serverMs - (startedAt + endedAt) / 2;
    clockSyncedRef.current = true;
    setClockSynced(true);
    return true;
  }, []);

  /** Halt when available, else disarm with the lease. Throws on failure. */
  const sendStop = useCallback(async (lease: string | null, reason: string) => {
    if (canHaltRef.current) {
      await haltRobot(reason);
    } else if (lease) {
      await disarmControl(lease);
    }
  }, []);

  const stopMotion = useCallback(
    async (reason = "deadman released") => {
      motionGeneration.current += 1;
      motionAllowed.current = false;
      setMotionActive(false);
      if (!serverMayBeArmed.current) return;
      serverMayBeArmed.current = false;
      try {
        await sendStop(leaseRef.current, reason);
      } catch (caught) {
        serverMayBeArmed.current = true;
        setError(`Stopping motion failed: ${joinSentences(errorMessage(caught), "Use E-stop if the arm is moving.")}`);
      }
    },
    [sendStop],
  );

  /**
   * Explicit "done, power down": POST /api/control/disarm with the lease (C5).
   * On servers that advertise Collapse this lowers the arm to rest, then turns
   * the motors off; on v0.1 servers it turns torque off where the arm stands.
   */
  const disarm = useCallback(async () => {
    motionGeneration.current += 1;
    motionAllowed.current = false;
    setMotionActive(false);
    const lease = leaseRef.current;
    if (!lease) return false;
    try {
      await disarmControl(lease);
      serverMayBeArmed.current = false;
      return true;
    } catch (caught) {
      setError(`Disarm failed: ${errorMessage(caught)}`);
      return false;
    }
  }, []);

  const release = useCallback(async () => {
    await stopMotion("lease released");
    const current = leaseRef.current;
    leaseRef.current = null;
    setLeaseId(null);
    clockSyncedRef.current = false;
    setClockSynced(false);
    if (current) await releaseControlLease(current).catch(() => undefined);
  }, [stopMotion]);

  const acquire = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const startedAt = Date.now();
      const lease = await createControlLease(source, 10);
      updateClock(lease.server_time, startedAt, Date.now());
      leaseRef.current = lease.id;
      sequence.current = 0;
      setLeaseId(lease.id);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  }, [source, updateClock]);

  const startMotion = useCallback(
    async (firstCommand?: () => Promise<unknown>): Promise<boolean> => {
      const current = leaseRef.current;
      if (!current || document.visibilityState !== "visible") return false;
      const roundTrip = lastRoundTripMs.current;
      if (!clockSyncedRef.current || roundTrip === null || roundTrip > MAX_SYNC_ROUND_TRIP_MS) {
        setError(
          roundTrip === null
            ? "No round trip to the robot server measured yet; cannot arm."
            : `Control latency is ${roundTrip.toFixed(0)} ms. Arming needs under ${MAX_SYNC_ROUND_TRIP_MS} ms because commands expire after 300 ms.`,
        );
        return false;
      }
      const generation = ++motionGeneration.current;
      setError(null);
      serverMayBeArmed.current = true;
      try {
        if (firstCommand) {
          const accepted = await firstCommand();
          if (accepted === false) throw new Error("First motion command was not accepted.");
        } else await armControl(current);
        if (generation !== motionGeneration.current || document.visibilityState !== "visible") {
          // Released while arming: the stop may have raced ahead of the arm,
          // so stop again unconditionally.
          serverMayBeArmed.current = true;
          await stopMotion("released while arming");
          return false;
        }
        motionAllowed.current = true;
        setMotionActive(true);
        return true;
      } catch (caught) {
        setError(errorMessage(caught));
        return false;
      }
    },
    [stopMotion],
  );

  const nextSequence = useCallback(() => {
    const value = sequence.current;
    sequence.current += 1;
    return value;
  }, []);
  const currentLeaseId = useCallback(() => leaseRef.current, []);
  const canSend = useCallback(
    () => motionAllowed.current && leaseRef.current !== null && document.visibilityState === "visible",
    [],
  );
  const timestampFor = useCallback(
    (clientTimeMs = Date.now()) => new Date(clientTimeMs + clockOffsetMs.current).toISOString(),
    [],
  );

  useEffect(() => {
    if (!leaseId) return;
    let disposed = false;
    const pulse = async () => {
      if (disposed || document.visibilityState !== "visible") return;
      const startedAt = Date.now();
      try {
        const renewed = await heartbeatControlLease(leaseId);
        if (!disposed) updateClock(renewed.server_time, startedAt, Date.now());
      } catch {
        if (!disposed) {
          setError("Motion lease heartbeat failed; control was released.");
          await release();
        }
      }
    };
    const onVisibility = () => {
      if (document.visibilityState !== "visible") void release();
    };
    const onBlur = () => void stopMotion("window lost focus");
    const interval = window.setInterval(() => void pulse(), LEASE_HEARTBEAT_MS);
    document.addEventListener("visibilitychange", onVisibility);
    window.addEventListener("blur", onBlur);
    return () => {
      disposed = true;
      window.clearInterval(interval);
      document.removeEventListener("visibilitychange", onVisibility);
      window.removeEventListener("blur", onBlur);
    };
  }, [leaseId, release, stopMotion, updateClock]);

  useEffect(
    () => () => {
      motionGeneration.current += 1;
      motionAllowed.current = false;
      const current = leaseRef.current;
      const mayBeArmed = serverMayBeArmed.current;
      serverMayBeArmed.current = false;
      leaseRef.current = null;
      if (!current) return;
      void (async () => {
        if (mayBeArmed) await sendStop(current, "control closed").catch(() => undefined);
        await releaseControlLease(current).catch(() => undefined);
      })();
    },
    [sendStop],
  );

  return {
    leaseId,
    motionActive,
    busy,
    error,
    /** Latest lease round trip in ms (reported, not a reason to drop control). */
    latencyMs,
    clockSynced,
    acquire,
    release,
    disarm,
    startMotion,
    stopMotion,
    nextSequence,
    currentLeaseId,
    canSend,
    timestampFor,
  };
}
