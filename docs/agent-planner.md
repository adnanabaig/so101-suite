# Optional intelligent proposal planner

Agent Contract Lab defaults to an offline deterministic skill router. An
optional OpenAI Responses API provider can select one strict, named proposal
tool. It never receives a joint-command, lease, arm, E-stop-reset, or execution
tool. This follows the official OpenAI
[function-calling flow](https://developers.openai.com/api/docs/guides/function-calling):
the model emits structured function arguments, while the application owns and
validates all actions.

## Configure

```bash
SO101_AGENT_PROVIDER=openai_responses
SO101_AGENT_MODEL=<model available to your project>
SO101_OPENAI_API_KEY=<secret>
```

The API key remains server-side. The planner receives the operator objective
and a reduced snapshot containing mode, joint state, safety state, armed state,
and lease owner—not camera pixels. Set `SO101_OPENAI_BASE_URL` only for an
explicit compatible gateway.

The model can select only:

- `inspect_scene`
- `home`
- `open_gripper`
- `close_gripper`
- `wrist_wave`

Its output becomes a durable **proposal**, never motion. The existing human
review, exact target display, short lease, operator approval bound to the
reviewed target digest (see [control-plane-security.md](control-plane-security.md)),
hold-to-run execution, calibration limits, measured-relative clamp, watchdog,
audit, and E-stop remain mandatory. When a skill completes, is cancelled or
times out, the arm Halts: it holds the final pose with torque on, and the hold
timer then lowers it with a Collapse ([control-plane-safety.md](control-plane-safety.md)).
Provider failure is surfaced; it never silently changes to another model or
executes a fallback motion.
