export function fixedChildEnvironment(
  source: Record<string, string | undefined>,
): Record<string, string>;
export function browserMetadata(source: Record<string, string | undefined>): Promise<{
  schema_version: number;
  kind: string;
  source: string;
  executable_path: string;
  version: string;
}>;
export function main(
  args?: string[],
  source?: Record<string, string | undefined>,
): Promise<number>;
