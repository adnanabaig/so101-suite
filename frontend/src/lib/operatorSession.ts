let cachedOperatorId: string | null = null;

/** Stable for one browser tab; never shared through localStorage. */
export function getOperatorId(): string {
  if (cachedOperatorId) return cachedOperatorId;
  const generated =
    typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `operator-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  cachedOperatorId = generated;
  return generated;
}
