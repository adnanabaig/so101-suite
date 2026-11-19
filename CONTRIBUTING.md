# Contributing

## Setup

You need Python 3.12 or 3.13 and Node 24 LTS ([Prerequisites](README.md#prerequisites)).

```bash
make bootstrap              # .venv with the package (editable) + dev tools, and frontend deps
source .venv/bin/activate   # puts so101 on PATH
```

## The dev loop

```bash
SO101_MOCK=1 make dev       # API with auto-reload + Vite (mock hardware only)
make test                   # tests (mock hardware; no device needed)
make lint                   # Ruff check, Ruff format check, mypy
make fmt                    # Ruff format + autofix
make coverage               # tests with branch coverage, then the coverage floors
make ci                     # every CI job, before you push
```

`make help` lists every target. No test, CI job or `make ci` step moves
hardware. The stop targets (`make estop`, `make halt`, `make collapse`) send a
stop to a running server and are never run by CI; `tests/test_makefile.py`
checks this.

The first Ubuntu G15 run uses the opt-in, pre-bootstrap
[`make g15-ci-evidence`](docs/g15-ci.md) target. It refuses the wrong host,
source, reusable build state, non-private evidence, or inherited `SO101_*`
configuration; then it runs `make bootstrap` once and `make ci` once and seals
their exact result. It is deliberately not part of `make ci` and never runs a
hardware diagnostic or HIL step.

## `make ci` is the CI workflow

Each job in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs one make
target, so `make ci` runs the same steps locally. It runs the jobs in order and
stops at the first failure.

| CI job | Local target | What it runs |
| --- | --- | --- |
| backend | `make ci-backend` | Hosted CI repeats this target on Python 3.12 and 3.13: `make lint`, `make coverage` (skip budget and floors), then `make ci-thor-worker` (stdlib-only qualification-pack contracts; no Thor/CUDA/Docker) |
| frontend | `make ci-frontend` | checks that `package.json` defines `lint`, `typecheck`, `test` and `build`; `npm ci`; `npm run lint`, `typecheck`, `test`, `build`; `npm audit --audit-level=high` |
| smoke | `make ci-smoke` | On Python 3.12 and 3.13, `scripts/run_smoke.py`: a mock API with a throwaway data dir, `scripts/smoke_suite.py` against it, then shutdown |
| wheel | `make ci-wheel` | On Python 3.12 and 3.13, builds the UI, runs the backend path-containment test over every built file, builds the sdist and wheel (`SO101_REQUIRE_UI=1`), checks their contents (`scripts/check_dist.py`), installs the wheel into a clean venv and runs `so101 --help` and `so101 info` (`scripts/wheel_smoke.py`) |

- A missing frontend script fails `ci-frontend` before anything is installed.
  `make ci` never skips a step.
- `make ci-smoke` uses a free port. `SMOKE_PORT=8814` pins one; a busy port fails
  with a message rather than testing whatever is listening there.
- `scripts/smoke_suite.py` (also `make smoke`) E-stops and resets, arms and
  drives every joint to 0, so before anything else it reads
  `/api/robot/status` and refuses with exit 3 unless `backend` is `mock` and
  nothing is latched (no E-stop, fault or unconfirmed torque-off). It only
  resets the E-stop it set itself. Never point it at a server with an arm.
- `PY=/path/to/python` runs the Python steps with another interpreter. By
  default make uses `.venv/bin/python`, or `python3` when there is no venv (as
  in CI).
- `tests/test_ci_workflow.py` fails if the workflow and the `ci` targets drift
  apart.

### Git worktrees that share `node_modules`

`npm ci` deletes `node_modules` before installing. If `frontend/node_modules` is
a symlink to another checkout's install, every target that would run `npm ci`
refuses. `FRONTEND_INSTALL=0` reuses the existing install instead, but then
`npm run build` writes TypeScript and Vite caches into the shared install. To
keep the other checkout untouched, run the frontend and wheel jobs in a copy:

```bash
rsync -a --exclude .git --exclude .venv --exclude node_modules --exclude dist ./ /tmp/so101-ci/
make -C /tmp/so101-ci ci-frontend
make -C /tmp/so101-ci ci-wheel FRONTEND_INSTALL=0 PY="$PWD/.venv/bin/python"
```

## Tests

- Tests use mock hardware and a hermetic config: `tests/conftest.py` hides
  your `.env` and any `SO101_*` variables, so a customized setup cannot change
  the results.
- No reruns or retries. A flaky test is a bug in the test or the code: fix it,
  or mark it skipped with the reason and an issue. Do not add
  pytest-rerunfailures, retry loops or longer sleeps to get it green.
- Timing-sensitive code should take its clock or timeout as a parameter, so a
  test can control time instead of waiting for it.

## Coverage floors

`make coverage` writes `coverage.json`, and `scripts/check_coverage.py` compares
it with [`coverage-floors.toml`](coverage-floors.toml): per-file floors for the
safety and control paths, and per-function floors for the lease and critical
stop paths. Measured floors are set just below their observed value; newly
inventoried critical stop, reset and recovery helpers use conservative
minimums pending the first supported G15 coverage report and may only be raised.

When your change raises coverage on one of these paths, raise its floor in the
same change. Do not lower a floor to land a change; add tests. A renamed file
or function must move its floor, because a target missing from the report
fails the check.

## Skip budget

A skipped test is lost coverage, so a full run may skip only what
[`tests/skip-budget.toml`](tests/skip-budget.toml) allows. `make test` and
`make coverage` (and so the CI backend job and `make ci`) pass
`--skip-budget tests/skip-budget.toml`; the hook is `tests/skip_budget.py`,
loaded by `tests/conftest.py`. The run fails (exit 1, with each offending test
and reason listed under the summary) when:

- a test skips for a reason no `[[allowed]]` entry matches,
- more tests skip for one entry than its `max`, or
- more tests skip in total than `max_skipped`.

Each entry has `reason` (a regular expression searched in the skip message),
`max` and `why`. Today's 16 skips need LeRobot, the LeRobot sdist, Pillow or
OpenCV, pyarrow, pyrealsense2 or a built `frontend/dist`; with the UI built
the count is 15. One more entry covers the rest-pose example check, which
skips only without the sim extra (MuJoCo). To add a skip, first try to make the test run; if the skip is
right, add or raise an entry with its reason and raise `max_skipped` to the
sum of the caps, in the same change. Running a few tests by hand
(`pytest tests/test_x.py`) checks nothing; add `--skip-budget
tests/skip-budget.toml` to a full run to check it. The CI backend checkout
fetches the full history (`fetch-depth: 0`), because the HIL quote tests read
the v0.2 design at an older commit and would skip in a shallow clone.

## Lint policy

Ruff targets Python 3.12. Five pyupgrade rules (UP017, UP040, UP042, UP046,
UP047) are ignored in `pyproject.toml` until one codemod applies them after the
v0.2 streams merge. Do not apply them by hand in unrelated changes; that would
conflict with the other streams.

mypy checks `src/so101`, `scripts` and `hatch_build.py`.

## The wheel and the UI

`hatch_build.py` puts the built UI (`frontend/dist`) into the wheel as
`so101/web`. `make wheel` builds the UI first, then the sdist and wheel, then
runs `scripts/check_dist.py` on them.

- Without `frontend/dist` the build warns and produces a wheel without a UI.
  That is fine for backend work.
- Release builds must set `SO101_REQUIRE_UI=1`, which turns a missing UI into a
  build error. `make wheel`, `make ci-wheel` and the CI wheel job do this.
- Every `npm run build` ends by writing `frontend/dist/so101-ui-stamp.json`
  (`frontend/build-stamp.ts`): a digest of the UI's sources (`frontend/src`,
  `frontend/public`, `index.html`, `package.json`, the lockfile, the Vite and
  TypeScript configs), the git commit, and the build's manifest (every file in
  `dist` with its SHA-256). The build refuses a `frontend/dist` without a
  stamp, one whose sources digest is not the current sources' (a stale UI), or
  one with a file the manifest doesn't list or with another hash. Run
  `make build-ui`, or delete `frontend/dist` to build without the UI.
- In a git checkout the build also refuses untracked files under `src/so101`
  (hatchling packs every file there that isn't ignored), and warns about
  uncommitted changes. Commit, delete or ignore the file. A build hook can't
  take files out of what hatchling packs, so refusing is how a local hack stays
  out; `check_dist.py` then checks the archives again for files git doesn't
  track.
- Check a build with:

  ```bash
  make build-ui
  rm -rf dist && SO101_REQUIRE_UI=1 python -m build --outdir dist .
  python scripts/check_dist.py dist --ui required
  python scripts/wheel_smoke.py dist --ui required
  ```

  `wheel_smoke.py` installs the wheel into a clean venv, runs the CLI, then
  starts the installed app (mock hardware, temp data folders) and checks that
  `GET /` is the built `index.html` from the package and that its assets load.

### Installed-artifact end-to-end rehearsal

`make installed-e2e` is the opt-in M3 product-flow gate. It builds the strict
UI-bearing wheel, installs that exact file with the simulation extra into one
clean venv, and exercises isolated mock and MuJoCo servers from the installed
package: backend guard, rehearsal preflight, dataset creation, inspection,
QC, human labels, the numpy baseline, model registry/load, bounded rollout,
operator-scored evaluation, and E-stop/Halt/Collapse semantics. It never uses
real ports or cameras; every mutating request stays locked until the server
has proved that it is the requested `mock` or `sim` backend. Mock UI requests
carry an ephemeral `SO101_E2E_TOKEN`; the server revalidates its
`X-SO101-E2E-Token` header and mock identity before every browser mutation.

The selected wheel is first copied without overwrite into the private evidence
tree, and that exact copy is installed. Logs, the full dataset, action-motion
proof, report and journal, checkpoint, shutdown events and recursive hashes go
to a new timestamped mode-0700 `.artifacts/installed-e2e-*` directory (files
are 0600); the venv and temporary data are then removed. Override the output
or run one backend while debugging with:

```bash
make installed-e2e INSTALLED_E2E_EVIDENCE=/outside/repo/evidence/run-1
make installed-e2e INSTALLED_E2E_BACKEND=mock
make installed-e2e INSTALLED_E2E_UI=1
```

The optional UI gate runs only the fixed frontend `npm run test:e2e` contract
against the still-fresh installed mock server; callers cannot append commands
or Playwright flags. It runs with `SO101_E2E_BASE_URL` set to its loopback
URL. Its output joins the same evidence ledger, and
`PLAYWRIGHT_ARTIFACTS_DIR` points at its
`mock/playwright/` folder for failure screenshots and token-audited
diagnostics. The backend is then re-checked for idle, unlatched state and no
active lease, training job or evaluation before the Python workflow continues.
Use a separate dataset id
(the UI suite reserves `e2e/installed_ui`; the harness uses
`installed_e2e/mock_reach`). The launcher records the exact system override or
pinned Playwright browser path and version in hashed `browser.json` evidence.

This target is deliberately not part of `make ci` yet. Promote it only after
it passes on the supported dependency host and its runtime and stability are
measured; `make ci` keeps its existing backend, frontend, smoke and wheel
ordering in the meantime. The backend job also runs the required stdlib-only
Thor qualification-pack host contracts; those need no Thor, CUDA or Docker.

## Release gate

1. `make ci` passes: lint, type check, tests with the coverage floors, the
   frontend checks, the smoke run and the wheel checks.
2. On a real arm, run the hardware preflight in
   [docs/hardware-setup.md](docs/hardware-setup.md) and
   [docs/SUITE_V0.md](docs/SUITE_V0.md) before calling that arm supported.

`make verify` remains a faster development check (tests, lint and a strict UI
build). It is not the release gate because it omits the coverage floors, skip
budget enforcement as configured by `make coverage`, frontend audit, live
smoke test, package-content checks and installed-wheel smoke.

## Validating CI and Docker files

The workflow, the Dockerfiles and the compose file are checked by linters that
are not Python packages. Install them from their release pages (and check the
published checksums) or with a package manager, then run:

```bash
actionlint -shellcheck="$(command -v shellcheck)" .github/workflows/ci.yml
hadolint docker/Dockerfile.api docker/Dockerfile.suite docker/Dockerfile.ui
docker compose config --quiet   # needs Docker
```

Without Docker, validate `docker-compose.yml` against the
[compose-spec JSON schema](https://github.com/compose-spec/compose-spec/blob/main/schema/compose-spec.json)
with any JSON-schema validator. `tests/test_ci_workflow.py` and
`tests/test_docker.py` cover what the linters cannot see, such as SHA-pinned
actions, read-only permissions, non-root users and health checks.

The images were not built when the non-root users and health checks were
added. To check them: `make docker`, then `docker compose ps` shows the api
service as healthy.

### Updating a pinned action

Actions are pinned to full commit SHAs, with the release in a comment. To move
to a new release, look up the tag's commit:

```bash
git ls-remote --tags https://github.com/actions/checkout 'v7.0.1*'
```

For an annotated tag, use the commit on the `^{}` line, not the tag object.
Replace the SHA and the `# vX.Y.Z` comment together.
