import process from "node:process";
import type { FullConfig, FullResult, Reporter } from "@playwright/test/reporter";
import { redactTokenArtifacts } from "./artifact-audit";
import { E2E_TOKEN_ENV, requiredE2EToken } from "./config";

export default class TokenAuditReporter implements Reporter {
  private outputDirs: string[] = [];

  onBegin(config: FullConfig) {
    this.outputDirs = Array.from(new Set(config.projects.map((project) => project.outputDir)));
  }

  async onEnd(_result: FullResult): Promise<{ status?: FullResult["status"] } | undefined> {
    try {
      const token = requiredE2EToken(process.env[E2E_TOKEN_ENV]);
      const redacted = this.outputDirs.flatMap((directory) => redactTokenArtifacts(directory, token));
      if (redacted.length) {
        process.stderr.write(
          `installed UI E2E: redacted the ephemeral token from ${redacted.length} artifact(s); failing the run\n`,
        );
        return { status: "failed" };
      }
      return undefined;
    } catch (reason) {
      const detail = reason instanceof Error ? reason.message : "artifact audit failed";
      process.stderr.write(`installed UI E2E: ${detail}; failing the run\n`);
      return { status: "failed" };
    }
  }

  printsToStdio() {
    return false;
  }
}
