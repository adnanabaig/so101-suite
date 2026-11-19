# v0.1 audit (2026-10-01)

Before the v0.2 work began, five read-only audits examined the v0.1 baseline
(commit `1d3a14a`; `7b59eda` has identical code). v0.2 commit messages cite the
audits' bug IDs, and this folder is where those IDs resolve.

| Report | Scope | IDs |
| --- | --- | --- |
| [safety.md](safety.md) | Control plane: RobotManager, leases, E-stop, watchdog, SafetyGuard, cross-origin handling | B1–B12 |
| [sim.md](sim.md) | Kinematic sim, hand tracking and WebXR, the LLM agent planner | B1–B21 |
| [frontend.md](frontend.md) | React UI | F1–F28 |
| [data.md](data.md) | Data collection, training, policy serving, evaluation | B1–B28 |
| [devex.md](devex.md) | Developer setup, tests, packaging, deployment, docs | B1–B18 |

Every report numbers its own bugs from B1, so cite them with the report name:
`safety-B4`, `sim-B19`, `data-B3`, `devex-B12`. Frontend IDs (`F1`) are
already unique. Early commits sometimes say just "audit B17". In those, the
commit prefix tells you the report: `data:` means data.md, `quality:` means
devex.md, and `sim:` means sim.md.

## Reading these reports

- Line numbers refer to the v0.1 code, not the current tree.
- Paths under `/tmp/so101_audit/` refer to a scratch directory that was not
  kept. That includes the repro scripts and the auditors' copies of the repo.
  v0.2 turned repros into regression tests in `tests/` and
  `frontend/src/test/`.
- Each report has a "what is good" section: behavior v0.2 had to preserve.
