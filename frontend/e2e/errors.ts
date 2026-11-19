export class E2EAggregateError extends Error {
  readonly errors: readonly Error[];

  constructor(message: string, errors: readonly Error[]) {
    super(`${message}: ${errors.map((error) => error.message).join("; ")}`);
    this.name = "E2EAggregateError";
    this.errors = errors;
  }
}
