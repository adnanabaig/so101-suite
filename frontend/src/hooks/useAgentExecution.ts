import { useCallback, useEffect, useRef, useState } from "react";
import {
  approveAgentProposal,
  cancelAgentProposal,
  executeAgentProposal,
  getAgentProposal,
  getAgentProposals,
  holdAgentProposal,
  isMissingRoute,
  type AgentProposal,
} from "../api";
import {
  isExecutableStatus,
  isTerminalStatus,
  proposalStatus,
  requiresDeadman,
} from "../lib/agentGates";
import { checkTargetDigest } from "../lib/canonicalJson";
import { errorMessage, joinSentences } from "../lib/errors";

/** C8 asks for a hold at least every 300 ms; ping well inside that. */
export const AGENT_HOLD_INTERVAL_MS = 100;
/**
 * Holds may overlap, so one slow answer does not hold back the next ping and
 * open a 300 ms gap on the server. Bounded, so a stalled server is not flooded.
 */
export const MAX_HOLDS_IN_FLIGHT = 3;
export const AGENT_POLL_INTERVAL_MS = 250;
/** C8: approval tokens live 30 s on the server. */
export const APPROVAL_MAX_MS = 30_000;
/**
 * C8: the server approves and executes a proposal only within 120 s of its
 * creation (its clock); after that it marks the proposal "expired".
 */
export const PROPOSAL_MAX_AGE_MS = 120_000;
/** Stop offering an approval this long before the server would refuse it. */
export const APPROVAL_MARGIN_MS = 1_000;
/** Keep polling this long past the proposal's own duration limit. */
const POLL_GRACE_MS = 30_000;
const DEFAULT_MAX_DURATION_S = 30;

export const PRE_C8_MESSAGE =
  "This server predates two-step approval (C8); agent motion needs the v0.2 backend.";
export const DIGEST_MISMATCH_MESSAGE =
  "Target digest mismatch: the targets shown are not the ones the server would execute. Refresh and review again; do not approve.";
export const PROPOSAL_TOO_OLD_MESSAGE =
  "This proposal is too old to run: the server approves and executes a proposal only within 2 minutes of its creation. Create a fresh proposal.";

/** A single-use C8 approval held by this tab. */
export interface AgentApprovalGrant {
  proposalId: string;
  token: string;
  /** The digest that was approved (of the targets on screen). */
  digest: string;
  leaseId: string;
  /** Client time after which the token is not offered for execution. */
  deadline: number;
}

export interface AgentRun {
  proposalId: string;
  leaseId: string;
  requireDeadman: boolean;
  /**
   * "starting" until the execute call returns 202, "cancelling" once a cancel
   * was sent, "ended" at a terminal status.
   */
  phase: "starting" | "running" | "cancelling" | "ended";
}

/**
 * Client time at which the server stops approving or executing a proposal:
 * 120 s after its `created_at`, on the server clock (via the lease clock
 * offset). Null when the proposal has no readable creation time.
 */
export function proposalAgeDeadline(
  createdAt: string | undefined,
  now: number,
  timestampFor: (clientTimeMs?: number) => string,
): number | null {
  if (!createdAt) return null;
  const created = Date.parse(createdAt);
  const serverNow = Date.parse(timestampFor(now));
  if (Number.isNaN(created) || Number.isNaN(serverNow)) return null;
  return now + (created + PROPOSAL_MAX_AGE_MS - serverNow);
}

/**
 * Client deadline for an approval: the server's `expires_at` measured on the
 * server clock (via the lease clock offset), never later than 30 s after the
 * request was sent nor past the proposal's own 120 s age limit, minus a
 * safety margin.
 */
export function approvalDeadline(
  expiresAt: string,
  sentAt: number,
  receivedAt: number,
  timestampFor: (clientTimeMs?: number) => string,
  ageDeadline: number | null = null,
): number {
  const cap = sentAt + APPROVAL_MAX_MS;
  const expires = Date.parse(expiresAt);
  const serverNow = Date.parse(timestampFor(receivedAt));
  const byServer =
    Number.isNaN(expires) || Number.isNaN(serverNow) ? cap : receivedAt + (expires - serverNow);
  return Math.min(cap, byServer, ageDeadline ?? Infinity) - APPROVAL_MARGIN_MS;
}

/**
 * The approval may be used for this proposal right now: same proposal and
 * lease, unexpired, still executable, and the server's digest (when it sends
 * one) is still the approved one.
 */
export function isApprovalUsable(
  grant: AgentApprovalGrant | null,
  proposal: AgentProposal,
  leaseId: string | null,
  now: number,
): grant is AgentApprovalGrant {
  if (!grant || grant.proposalId !== proposal.id || grant.leaseId !== leaseId) return false;
  if (now >= grant.deadline || !isExecutableStatus(proposalStatus(proposal))) return false;
  const serverDigest = proposal.target_digest;
  return typeof serverDigest !== "string" || serverDigest.toLowerCase() === grant.digest;
}

export interface UseAgentExecutionOptions {
  leaseId: string | null;
  /** Server-clock ISO time for a client time (from the lease clock sync). */
  timestampFor: (clientTimeMs?: number) => string;
  /** E-stop latched: the server aborts the run; holds stop, polling goes on. */
  tripped: boolean;
  /** Proposals and safety state should be refetched. */
  onChanged: () => void;
}

/**
 * C8 approve → execute → (hold) → poll → result for Agent Lab.
 *
 * - Approval is computed from the targets on screen and refused on a digest
 *   mismatch. The token is single-use and expires client-side early.
 * - Execution needs a usable approval; the digest is checked again first.
 * - `require_deadman` runs are held. The server starts its 300 ms hold window
 *   when it answers 202, so the first hold goes out before anything else
 *   (polling, refreshes), then every 100 ms with up to three holds in flight.
 *   Releasing (or blur, keyup, a hidden tab) stops holding and cancels; a
 *   release before the 202 cancels when it arrives. A failed hold cancels
 *   unless the run had already finished.
 * - A latched E-stop stops the holds but not the polling: the server aborts
 *   and reports the end state.
 * - A hidden tab, a lost lease or leaving the page cancels an active run.
 */
export function useAgentExecution({
  leaseId,
  timestampFor,
  tripped,
  onChanged,
}: UseAgentExecutionOptions) {
  const [approval, setApproval] = useState<AgentApprovalGrant | null>(null);
  const [run, setRunState] = useState<AgentRun | null>(null);
  const [latest, setLatest] = useState<AgentProposal | null>(null);
  const [holding, setHolding] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const approvalRef = useRef<AgentApprovalGrant | null>(null);
  const runRef = useRef<AgentRun | null>(null);
  const leaseRef = useRef(leaseId);
  const generation = useRef(0);
  const holdingRef = useRef(false);
  const holdTimer = useRef<number | null>(null);
  const pollTimer = useRef<number | null>(null);
  const pollInFlight = useRef(false);
  const pollDeadline = useRef(0);
  /** Cancel asked for while the execute call was in flight, and why. */
  const cancelRequested = useRef<string | null>(null);
  const cancelSent = useRef(false);
  const singleGetMissing = useRef(false);
  const onChangedRef = useRef(onChanged);
  const timestampRef = useRef(timestampFor);
  const trippedRef = useRef(tripped);
  useEffect(() => {
    onChangedRef.current = onChanged;
    timestampRef.current = timestampFor;
    trippedRef.current = tripped;
  });

  const setGrant = useCallback((grant: AgentApprovalGrant | null) => {
    approvalRef.current = grant;
    setApproval(grant);
  }, []);

  const setRun = useCallback((next: AgentRun | null) => {
    runRef.current = next;
    setRunState(next);
  }, []);

  const stopHolds = useCallback(() => {
    if (holdTimer.current !== null) window.clearInterval(holdTimer.current);
    holdTimer.current = null;
  }, []);

  const stopPolling = useCallback(() => {
    if (pollTimer.current !== null) window.clearInterval(pollTimer.current);
    pollTimer.current = null;
  }, []);

  /** GET /{id}, or the list on servers without the single-proposal route. */
  const fetchProposal = useCallback(async (id: string): Promise<AgentProposal | null> => {
    if (!singleGetMissing.current) {
      try {
        return await getAgentProposal(id);
      } catch (caught) {
        if (!isMissingRoute(caught)) throw caught;
        singleGetMissing.current = true;
      }
    }
    const response = await getAgentProposals();
    const list = Array.isArray(response) ? response : response.proposals;
    return list.find((proposal) => proposal.id === id) ?? null;
  }, []);

  const finish = useCallback(
    (proposal: AgentProposal | null) => {
      stopHolds();
      stopPolling();
      const current = runRef.current;
      if (current && current.phase !== "ended") setRun({ ...current, phase: "ended" });
      if (proposal) setLatest(proposal);
      onChangedRef.current();
    },
    [setRun, stopHolds, stopPolling],
  );

  const cancelRun = useCallback(
    async (reason: string) => {
      const current = runRef.current;
      if (!current || current.phase === "ended" || cancelSent.current) return;
      stopHolds();
      if (current.phase === "starting") {
        // Not running yet: cancel the moment the execute call answers.
        cancelRequested.current ??= reason;
        return;
      }
      cancelSent.current = true;
      setRun({ ...current, phase: "cancelling" });
      const gen = generation.current;
      try {
        const cancelled = await cancelAgentProposal(current.proposalId);
        if (gen !== generation.current) return;
        setNotice(`Run cancelled (${reason}). The result records what moved.`);
        if (cancelled && typeof cancelled === "object" && cancelled.id === current.proposalId) {
          setLatest(cancelled);
          if (isTerminalStatus(proposalStatus(cancelled))) finish(cancelled);
        }
      } catch (caught) {
        if (gen !== generation.current) return;
        // Refused because the run already ended (reached, timed out, aborted)?
        const after = await fetchProposal(current.proposalId).catch(() => null);
        if (gen !== generation.current) return;
        if (after && isTerminalStatus(proposalStatus(after))) {
          finish(after);
          return;
        }
        setError(
          `Cancel failed: ${joinSentences(errorMessage(caught), "Use Halt or E-stop if the arm is moving.")}`,
        );
      }
    },
    [fetchProposal, finish, setRun, stopHolds],
  );

  const startPolling = useCallback(
    (id: string, gen: number, maxDurationS: number) => {
      stopPolling();
      pollDeadline.current = Date.now() + maxDurationS * 1000 + POLL_GRACE_MS;
      const tick = async () => {
        if (gen !== generation.current || pollInFlight.current) return;
        if (Date.now() > pollDeadline.current) {
          stopPolling();
          setError(
            "The server has not reported a terminal status for this run. Check the arm, then refresh.",
          );
          finish(null);
          return;
        }
        pollInFlight.current = true;
        try {
          const proposal = await fetchProposal(id);
          if (gen !== generation.current || !proposal) return;
          setLatest(proposal);
          if (isTerminalStatus(proposalStatus(proposal))) finish(proposal);
        } catch {
          // Transient: keep polling until the deadline.
        } finally {
          pollInFlight.current = false;
        }
      };
      pollTimer.current = window.setInterval(() => void tick(), AGENT_POLL_INTERVAL_MS);
    },
    [fetchProposal, finish, stopPolling],
  );

  const holdFailed = useCallback(
    async (id: string, gen: number, caught: unknown) => {
      stopHolds();
      // A run that just reached its target refuses further holds; that is not
      // a failure, so look before cancelling.
      const current = await fetchProposal(id).catch(() => null);
      if (gen !== generation.current) return;
      if (current && isTerminalStatus(proposalStatus(current))) {
        finish(current);
        return;
      }
      setError(
        `The hold-to-run ping failed: ${joinSentences(errorMessage(caught), "The run is being cancelled.")}`,
      );
      await cancelRun("hold failed");
    },
    [cancelRun, fetchProposal, finish, stopHolds],
  );

  const startHolds = useCallback(
    (id: string, lease: string, gen: number) => {
      stopHolds();
      // Per run, so a previous run's stragglers never count against this one.
      let inFlight = 0;
      const pulse = async () => {
        if (gen !== generation.current || !holdingRef.current || inFlight >= MAX_HOLDS_IN_FLIGHT) {
          return;
        }
        inFlight += 1;
        try {
          await holdAgentProposal(id, lease);
        } catch (caught) {
          // holdFailed clears the timer first, so only the first failure acts.
          if (gen === generation.current && holdingRef.current && holdTimer.current !== null) {
            await holdFailed(id, gen, caught);
          }
        } finally {
          inFlight -= 1;
        }
      };
      holdTimer.current = window.setInterval(() => void pulse(), AGENT_HOLD_INTERVAL_MS);
      void pulse();
    },
    [holdFailed, stopHolds],
  );

  const approve = useCallback(
    async (proposal: AgentProposal) => {
      const lease = leaseRef.current;
      setError(null);
      setNotice(null);
      if (!lease) {
        setError("Acquire the Agent Lab lease first.");
        return false;
      }
      setBusy(`approve:${proposal.id}`);
      try {
        const check = checkTargetDigest(proposal.targets ?? {}, proposal.target_digest);
        if (check.status === "mismatch") {
          setError(DIGEST_MISMATCH_MESSAGE);
          return false;
        }
        const sentAt = Date.now();
        const grant = await approveAgentProposal(proposal.id, lease, check.digest);
        const receivedAt = Date.now();
        if (!grant || typeof grant.approval_token !== "string" || !grant.approval_token) {
          throw new Error("The server returned no approval token.");
        }
        const approvedDigest = grant.proposal?.target_digest;
        if (typeof approvedDigest === "string" && approvedDigest.toLowerCase() !== check.digest) {
          setError(
            "The server approved a different target set than the one shown. Do not execute; refresh and review again.",
          );
          onChangedRef.current();
          return false;
        }
        if (leaseRef.current !== lease) {
          setError("The lease changed while approving. Approve again.");
          return false;
        }
        const ageDeadline = proposalAgeDeadline(
          grant.proposal?.created_at ?? proposal.created_at,
          receivedAt,
          timestampRef.current,
        );
        const deadline = approvalDeadline(
          grant.expires_at,
          sentAt,
          receivedAt,
          timestampRef.current,
          ageDeadline,
        );
        if (deadline <= Date.now()) {
          setError(
            ageDeadline !== null && ageDeadline - APPROVAL_MARGIN_MS <= Date.now()
              ? PROPOSAL_TOO_OLD_MESSAGE
              : "The approval expired before it arrived. Check the connection and approve again.",
          );
          return false;
        }
        setGrant({
          proposalId: proposal.id,
          token: grant.approval_token,
          digest: check.digest,
          leaseId: lease,
          deadline,
        });
        onChangedRef.current();
        return true;
      } catch (caught) {
        setError(isMissingRoute(caught) ? PRE_C8_MESSAGE : `Approval was refused: ${errorMessage(caught)}`);
        // A refusal can change the proposal (an expired one is marked so).
        onChangedRef.current();
        return false;
      } finally {
        setBusy(null);
      }
    },
    [setGrant],
  );

  const execute = useCallback(
    async (proposal: AgentProposal) => {
      const active = runRef.current;
      if (active && active.phase !== "ended") return false;
      const grant = approvalRef.current;
      const lease = leaseRef.current;
      if (!isApprovalUsable(grant, proposal, lease, Date.now())) {
        setError("Approve this proposal first. Approvals are single-use and last 30 seconds.");
        return false;
      }
      setError(null);
      setNotice(null);
      const check = checkTargetDigest(proposal.targets ?? {}, proposal.target_digest);
      if (check.status === "mismatch" || check.digest !== grant.digest) {
        setGrant(null);
        setError(DIGEST_MISMATCH_MESSAGE);
        return false;
      }
      // Single use: the token is spent whatever the answer.
      setGrant(null);
      const gen = ++generation.current;
      const deadman = requiresDeadman(proposal);
      cancelRequested.current = null;
      cancelSent.current = false;
      setRun({ proposalId: proposal.id, leaseId: grant.leaseId, requireDeadman: deadman, phase: "starting" });
      setLatest(null);
      setBusy(`execute:${proposal.id}`);
      try {
        const started = await executeAgentProposal(proposal.id, grant.leaseId, grant.token);
        if (gen !== generation.current) return false;
        setRun({ proposalId: proposal.id, leaseId: grant.leaseId, requireDeadman: deadman, phase: "running" });
        if (started && typeof started === "object" && started.id === proposal.id) setLatest(started);
        const maxDuration = proposal.constraints?.max_duration_seconds ?? DEFAULT_MAX_DURATION_S;
        const pending =
          cancelRequested.current ??
          (deadman && !holdingRef.current ? "released before the run started" : null);
        if (pending) {
          // The cancel goes out first; polling then reports how the run ended.
          const cancelling = cancelRun(pending);
          startPolling(proposal.id, gen, maxDuration);
          onChangedRef.current();
          await cancelling;
          return false;
        }
        // The server's 300 ms hold window is already running: the first hold
        // goes on the wire before the poll and the refreshes.
        if (deadman) startHolds(proposal.id, grant.leaseId, gen);
        startPolling(proposal.id, gen, maxDuration);
        onChangedRef.current();
        return true;
      } catch (caught) {
        if (gen !== generation.current) return false;
        setRun(null);
        setError(isMissingRoute(caught) ? PRE_C8_MESSAGE : `Execution was refused: ${errorMessage(caught)}`);
        onChangedRef.current();
        return false;
      } finally {
        setBusy(null);
      }
    },
    [cancelRun, setGrant, setRun, startHolds, startPolling],
  );

  /**
   * Deadman pressed on a `require_deadman` proposal: execute and hold. A run
   * that was released cannot be resumed by pressing again; it needs a new
   * approval once it ends.
   */
  const pressHold = useCallback(
    (proposal: AgentProposal) => {
      const active = runRef.current;
      if (active && active.phase !== "ended") return;
      holdingRef.current = true;
      setHolding(true);
      void execute(proposal);
    },
    [execute],
  );

  /** Deadman released (or lost): stop holding and cancel the held run. */
  const releaseHold = useCallback(
    (reason: string) => {
      if (!holdingRef.current) return;
      holdingRef.current = false;
      setHolding(false);
      stopHolds();
      const current = runRef.current;
      if (!current || current.phase === "ended" || !current.requireDeadman) return;
      // After an E-stop the server has already aborted; polling reports it.
      if (trippedRef.current && current.phase !== "starting") return;
      void cancelRun(current.phase === "starting" ? "released before the run started" : reason);
    },
    [cancelRun, stopHolds],
  );

  /** Cancel a proposal that is not running here (proposed, or another tab's run). */
  const cancelProposal = useCallback(
    async (proposal: AgentProposal) => {
      const current = runRef.current;
      if (current && current.phase !== "ended" && current.proposalId === proposal.id) {
        await cancelRun("cancelled by the operator");
        return;
      }
      setBusy(`cancel:${proposal.id}`);
      setError(null);
      setNotice(null);
      try {
        await cancelAgentProposal(proposal.id);
        if (approvalRef.current?.proposalId === proposal.id) setGrant(null);
        setNotice("Proposal cancelled.");
      } catch (caught) {
        setError(`Cancel failed: ${errorMessage(caught)}`);
      } finally {
        setBusy(null);
        onChangedRef.current();
      }
    },
    [cancelRun, setGrant],
  );

  const dismiss = useCallback(() => {
    setError(null);
    setNotice(null);
  }, []);

  // A lost or changed lease cancels an active run. (The approval is bound to
  // the lease: isApprovalUsable refuses it under any other lease.)
  useEffect(() => {
    leaseRef.current = leaseId;
    const current = runRef.current;
    if (current && current.phase !== "ended" && current.leaseId !== leaseId) {
      void cancelRun("the lease was lost");
    }
  }, [leaseId, cancelRun]);

  // E-stop aborts on the server; holding on would only produce errors.
  useEffect(() => {
    if (tripped) stopHolds();
  }, [tripped, stopHolds]);

  // Expire the approval a little before the server would.
  useEffect(() => {
    if (!approval) return;
    const timer = window.setTimeout(
      () => {
        if (approvalRef.current === approval) setGrant(null);
      },
      Math.max(0, approval.deadline - Date.now()),
    );
    return () => window.clearTimeout(timer);
  }, [approval, setGrant]);

  useEffect(() => {
    const onVisibility = () => {
      if (document.visibilityState !== "visible") void cancelRun("the tab was hidden");
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, [cancelRun]);

  // Leaving the page cancels an active run (fire and forget).
  useEffect(
    () => () => {
      generation.current += 1;
      stopHolds();
      stopPolling();
      const current = runRef.current;
      if (current && current.phase !== "ended" && !cancelSent.current) {
        cancelSent.current = true;
        void cancelAgentProposal(current.proposalId).catch(() => undefined);
      }
    },
    [stopHolds, stopPolling],
  );

  return {
    approval,
    run,
    /** Freshest copy of the proposal being (or last) run here. */
    latest,
    holding,
    busy,
    error,
    notice,
    approve,
    execute,
    pressHold,
    releaseHold,
    cancelProposal,
    dismiss,
  };
}
