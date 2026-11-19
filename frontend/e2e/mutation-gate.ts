import type { APIRequestContext, Page } from "@playwright/test";
import { requireInstalledE2EIdentity, type E2EApi } from "./contract";
import { E2EAggregateError } from "./errors";

const MUTATING_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);

export async function dispatchInstalledMutation<T>(
  api: E2EApi,
  dispatch: () => Promise<T>,
): Promise<T> {
  // The server checks the token again on the dispatched request. This identity
  // probe prevents even attempting a write after the URL is replaced by a
  // real or token-unconfigured server.
  await requireInstalledE2EIdentity(api);
  return await dispatch();
}

export interface BrowserMutationGate {
  assertNoBlockedMutation: () => void;
}

export async function installBrowserMutationGate(
  page: Page,
  api: APIRequestContext,
): Promise<BrowserMutationGate> {
  const blocked: Error[] = [];
  await page.route("**/*", async (route) => {
    if (!MUTATING_METHODS.has(route.request().method())) {
      await route.continue();
      return;
    }
    try {
      await dispatchInstalledMutation(api, () => route.continue());
    } catch (reason) {
      blocked.push(
        reason instanceof Error
          ? reason
          : new Error("browser mutation was blocked because installed E2E identity was lost"),
      );
      await route.abort("blockedbyclient");
    }
  });
  return {
    assertNoBlockedMutation() {
      if (blocked.length) {
        throw new E2EAggregateError(
          "one or more browser mutations lost installed E2E authorization",
          blocked,
        );
      }
    },
  };
}
