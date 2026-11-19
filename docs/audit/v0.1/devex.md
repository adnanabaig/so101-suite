# SO-101 Suite v0.1: DevEx, tests, packaging, deployment and docs audit

## Summary

Every finding below was reproduced, and the report includes the commands and output. B18 (Docker) is the exception: it is PLAUSIBLE only, because Docker is not installed.

Top findings, most severe first:

1. **HIGH: the SPA fallback lets anyone read any file** (`app.py:89-92`). Requests like `/..%2F..%2FX` and `//etc/hosts` return HTTP 200 with the file's contents. That exposes `.env` API keys, the audit DB and `~/.ssh`.
2. **HIGH: `make api` runs with `--reload`** (Makefile:28), and that is the documented hardware run path (README:63). Touching any `.py` file restarts the server and clears a latched E-stop.
3. **HIGH: the wheel ships no UI** (`app.py:81`, `pyproject.toml:55-56`). An installed `so101 serve` returns a JSON placeholder.
4. **MED-HIGH: control-plane timing tests flake under CPU load** (`test_control_plane.py:116-129`, `:302-323`). 8 of 224 runs failed.
5. **MED: a developer's `.env` leaks into the tests** (`config.py:15`, `conftest.py:12-17`), causing 2 failures.
6. **MED: `[robot]` installs opencv-python 5.0.0.93 next to LeRobot's opencv-python-headless<4.14** (`pyproject.toml:29`). Uninstalling either one breaks `cv2`.
7. **MED: `so101 estop` ignores Make's PORT** (`cli.py:265-273`, Makefile:4-5). It fails with ConnectError while the robot keeps running.
8. **MED: `.env` and state are resolved against the cwd.** Every CLI call, even `--help`, creates `data/`, `datasets/` and `outputs/` there (`config.py:15,33-35,78-81`; `app.py:110`).
9. **MED: `make bootstrap` accepts Python 3.9** (Makefile:18), and `install-robot` installs globally when `.venv` is missing (Makefile:3,24).
10. **LOW-MED: `doctor` exits 0 on FAIL and Rich strips `[robot]` from its advice** (`cli.py:242-262`, `operations.py:102`). Also, README:63 `make api && make ui` never starts the UI.

**Release claim:** the 105 tests, Ruff, mypy, the strict frontend build, 0 npm vulnerabilities and smoke (10/10) all hold. But mypy only checks `src`, the tests flake under load, and the wheel has no UI. Branch coverage is 71%; `cli.py` has 0%.


- **Date:** 2026-10-01
- **Repo:** `/Users/baiadnan/Documents/so101`
  - HEAD is `7b59eda` on branch `v0.2`.
  - The code is identical to `1d3a14a`. `7b59eda` adds only `docs/plans/2026-10-01-suite-v02-design.md`.
- **Environment:**
  - macOS (Darwin 25.6), 10 CPUs.
  - Python 3.12.14 is first on PATH. Apple's `/usr/bin/python3` is 3.9.6.
  - node v24.18.0, npm 11.16.0.
  - Docker is not installed, so Docker findings come from reading the files. Headless Chrome was not used.
- **Method:** everything ran under `/tmp/so101_audit/devex/`:
  - `repo_copy/`: a file copy with its own `.venv`, `node_modules` and `dist`.
  - `venv_wheel/`: a fresh venv installed from the built wheel.
  - `venv_resolve/`: a scratch `[robot]` install.
  - All `SO101_*_DIR` variables pointed at scratch dirs. Nothing under the repo was created, modified or deleted, and git state was not touched.

### Git baseline and test-suite side effects

- The count was 111 at the start, then 0 after the outside commits `1d3a14a` (14:08, which committed the v0.1 tree) and `7b59eda` (14:42, the v0.2 design doc).
- The branch is now `v0.2`, with five worktrees under `/Users/baiadnan/Documents/so101-wt/`.
- The count stayed 0 after every pytest run, and the final check was also 0.
- **The suite never modifies tracked files.** It does create gitignored files:
  - `.pytest_cache/` (unless run with `-p no:cacheprovider`)
  - `__pycache__/`
  - empty `data/`, `datasets/` and `outputs/` in the cwd. These appear at collection time, when test modules import `so101.services.app` (B8).

### Release-claim verification

| Claim | Verdict | Evidence |
| --- | --- | --- |
| 105 tests pass | True under normal load, flaky under CPU contention | `105 passed` in 5 default-order runs (5.7-7.6 s), 4 random seeds, `--reverse`, and xdist `-n 4`/`-n 8`. Under load, 1 test fails in 2 of 3 runs (B4). |
| Ruff | True | `All checks passed!` and `73 files already formatted`. But `target-version="py310"` (`pyproject.toml:60`) hides 28 findings: `ruff check --target-version py312` reports `26 UP017`, `2 UP042`, `Found 28 errors.` |
| mypy | True for `src/so101` only | `Success: no issues found in 58 source files`. Running `mypy tests scripts` gives `Found 10 errors in 2 files`. |
| Strict frontend build | True | `npm run build` (`tsc && vite build`) passes in the scratch copy. |
| Zero npm vulnerabilities | True | `npm audit` reports `found 0 vulnerabilities`. |
| Live smoke | True | `scripts/smoke_suite.py --base ...` against a fresh mock suite: 10 PASS and `Suite smoke passed.`, twice in a row (idempotent). |
| Wheel validation | Partly true | The wheel builds and installs, and `so101 --help`, `SO101_MOCK=1 so101 info` and `so101 doctor` work. But there is no UI (B3) and no automated contents check. |
| `make verify` is the gate (Makefile:66) | Incomplete | It runs only `test lint build-ui`. Smoke, the wheel and `npm audit` are manual. It passes from a fresh copy in about 18 s. |

---

## 1. Bugs (most severe first)

### B1. HIGH: the SPA fallback serves any file on disk (path traversal). CONFIRMED

- **Where:** `src/so101/services/app.py:89-92`. The route is registered whenever `frontend/dist` exists (`app.py:82`).
- **Defect:** `spa()` returns `FileResponse(dist / path)` for any path that is a file. There is no containment check, so three kinds of path escape `dist`:
  - `../` segments
  - `%2F`-encoded slashes
  - absolute paths, because `Path(dist) / "/etc/hosts"` evaluates to `/etc/hosts`
- **Failure scenario:** anyone who can reach the port can read:
  - `.env` at `dist/../../.env`, which holds `SO101_OPENAI_API_KEY` and `SO101_HF_TOKEN`
  - `data/control_plane.sqlite3`
  - calibration files
  - `~/.ssh/*`

  This is reachable in two documented setups:
  - the documented Quest/remote setup, an HTTPS reverse proxy (README.md:33-34)
  - the docker `suite` image

  PLAUSIBLE: there is no Host-header allow-list (`curl -H 'Host: evil.example:8797' .../api/health` returns 200), so a DNS-rebinding page in the operator's browser could reach a server that only listens on loopback.
- **Evidence** (mock suite from the scratch copy on :8796, with a decoy file at the repo root):
  ```
  $ curl -s --path-as-is "$B/../../AUDIT_DECOY.txt"
  decoy-secret-from-repo-root
   [HTTP 200 text/plain; charset=utf-8]
  $ curl -s "$B/..%2F..%2FAUDIT_DECOY.txt"
  decoy-secret-from-repo-root
   [HTTP 200 text/plain; charset=utf-8]
  $ curl -s "$B/..%2F..%2F..%2F..%2F..%2F..%2F..%2F..%2F..%2F..%2Fetc%2Fpasswd"
  nobody:*:-2:-2:Unprivileged User:/var/empty:/usr/bin/false
  root:*:0:0:System Administrator:/var/root:/bin/sh
  [HTTP 200]
  $ curl -s "$B//etc/hosts"            # plain curl, no --path-as-is needed
  127.0.0.1	localhost
  $ curl -s --path-as-is "$B/../../pyproject.toml"
  [project]
  name = "so101"
  server log: "GET /../../../../../../../../../../etc/passwd HTTP/1.1" 200 OK
  ```
- **Fix:**
  1. `root = dist.resolve(); t = (root / path).resolve()`. Serve `t` only if `t.is_relative_to(root) and t.is_file()`; otherwise serve `index.html`.
  2. Return 404 for `api/` and `ws/` prefixes (B12).
  3. Add a regression test for `..%2F`, `//etc/hosts` and `/api/nope`.

### B2. HIGH: the documented run command auto-reloads, and each reload silently clears the E-stop latch. CONFIRMED

- **Where:** `Makefile:26-28` (`uvicorn ... --reload`). The docs present this as:
  - the real-hardware run step (README.md:63)
  - the two-process mode (README.md:32, getting-started.md:25)

  The latch exists only in process memory.
- **Defect:** `make api` watches the repo root and restarts on any `.py` change. The new process starts un-latched.
- **Failure scenario:**
  - An operator latches E-stop.
  - Then a file is saved in the editor, or someone runs `make fmt`, `git pull` or a branch switch.
  - The API restarts un-latched with no operator action.

  This contradicts getting-started.md:72-74 ("Reset ... never happens as a side effect of reconnecting"). It also kills an in-flight teleop or recording. `shutdown()` (`robot_manager.py:1307-1320`) does disarm, so motion stops during the reload. The defect is the lost latch.
- **Evidence:**
  ```
  $ SO101_MOCK=1 make api PORT=8797                    # scratch copy
  $ curl -s -X POST $B/api/robot/estop
  {"ok":true,"detail":"E-STOP engaged; all motion stopped"}
  estopped = True
  $ touch src/so101/__init__.py
  estopped = False
  log: INFO:     Will watch for changes in these directories: ['/private/tmp/so101_audit/devex/repo_copy']
       WARNING:  WatchFiles detected changes in 'src/so101/__init__.py'. Reloading...
       INFO:     Shutting down
       INFO:     Started server process [85240]
  ```
- **Fix:**
  1. Remove `--reload` from `make api` and add an explicit `make api-dev`.
  2. Point the hardware docs at `so101 serve` or `make suite`.
  3. Optionally, persist the latch in the SQLite store so a restart comes back latched.

### B3. HIGH (packaging): the wheel has no UI, so an installed `so101 serve` returns a JSON placeholder. CONFIRMED

- **Where:**
  - `app.py:81`: `Path(__file__).resolve().parents[3] / "frontend" / "dist"`
  - `pyproject.toml:55-56`: the wheel packages only `src/so101`
  - `app.py:101-102`: the remediation text
- **Defect:** the code looks for the UI relative to a source checkout, and the build never packages `frontend/dist`. Any non-editable install therefore has no UI, and the suggested fix (`make ui`) does not exist outside a checkout.
- **Failure scenario:**
  - `pip install <wheel> && so101 serve` shows a JSON blob, and `/datasets` returns 404.
  - `docker/Dockerfile.suite:16` works only because it uses `pip install -e .`. Switching to `pip install .` would silently drop the UI.
- **Evidence:**
  ```
  $ unzip -Z1 dist/so101-0.1.0-py3-none-any.whl | grep -c '\.py$'      -> 58
  non-.py entries: so101/simulation/scenarios.yaml + dist-info (METADATA, WHEEL, entry_points.txt, licenses/LICENSE, RECORD)
  index.html / *.js / *.css entries: 0
  $ so101 serve --port 8798            # fresh venv from the wheel
  GET /  -> {"service":"so101","version":"0.1.0","ui":"frontend not built — run `make ui` (dev) or `cd frontend && npm run build`","docs":"/docs"} [HTTP 200]
  GET /datasets -> HTTP 404
  UI lookup path: /private/tmp/so101_audit/devex/venv_wheel/lib/python3.12/frontend/dist
  ```
- **Fix:**
  1. Build the UI into `src/so101/web/` and ship it with hatch `artifacts`, `force-include` or a build hook.
  2. Locate it with `importlib.resources.files("so101") / "web"`.
  3. Add a CI check that the wheel contains `so101/web/index.html`.

### B4. MED-HIGH: control-plane tests depend on wall-clock timing and fail under CPU contention. CONFIRMED

- **Where:** `tests/test_control_plane.py`. 12 of the suite's 15 `time.sleep` calls are in this file.

  | Lines | What depends on timing |
  | --- | --- |
  | `:116-129` | lease TTLs of 0.15 s and 0.1 s |
  | `:132-157` | deadman / stale window |
  | `:92-113` | `operator_timeout_ms=100` plus `sleep(0.25)` |
  | `:302-323` | first frame against the 500 ms watchdog |
  | `:226-244` | `sleep(0.65)` against the 500 ms watchdog |

- **Defect:** the tests assume consecutive TestClient requests arrive within 100-500 ms of each other.
- **Failure scenario:** shared 2-vCPU CI runners, or a busy laptop, produce intermittent red builds. They hit the file that guards leases, deadman and watchdog.
- **Evidence under normal load:** always `105 passed`, including random seeds 1/42/1234/98765, reverse order and xdist. So the suite is order-independent and parallel-safe.
- **Evidence under load** (20 busy loops on 10 CPUs):
  ```
  $ for i in $(seq 1 20); do (yes > /dev/null &); done
  $ SO101_MOCK=1 python -m pytest -p no:cacheprovider -p no:randomly -o addopts="" -q
  run 1: 1 failed, 104 passed in 31.56s
         test_tracking_atomic_first_frame_reports_actual_limited_command
         tests/test_control_plane.py:318  {"detail":"control is disarmed"}
  run 2: 105 passed in 23.15s
  run 3: 1 failed, 104 passed in 28.20s
         test_lease_conflict_heartbeat_release_and_expiry (:119 assert conflict.status_code == 409)
  $ ... pytest --count=8 tests/test_control_plane.py tests/test_api.py      (same load)
  8 failed, 216 passed in 182.79s
    lease_conflict x5, command_rejects_deadman_stale x2, legacy_loop x1
  ```
- **Possible product impact (PLAUSIBLE):**
  - The 500 ms first-command deadman clock starts at `robot_manager.py:680`, before `follower.enable_torque()` at `:698`.
  - So slow bus writes eat into the operator's first-command window, which is exactly how the `:318` failure arises.
  - This fails safe, but could make arming flaky on real hardware. It falls under the control-plane audit.
- **Fix:**
  1. Inject a clock, or make the TTL, watchdog and operator timeouts settings that tests can scale.
  2. Poll against deadlines instead of using fixed sleeps.
  3. Start the deadman clock only after torque-enable succeeds.

### B5. MED: tests read the developer's `.env` and fail when it is customized. CONFIRMED

- **Where:**
  - `config.py:15`: `env_file=".env"`
  - `tests/conftest.py:12-17`: overrides only the dirs and cameras
  - `tests/test_agent_planner.py:89-99`: `delenv` cannot remove a value that came from `.env`
  - `tests/test_hardware.py:406`
- **Defect:** pydantic-settings loads `./.env` during tests. Any `SO101_*` value that the fixture does not override leaks in.
- **Failure scenario:** README.md:59 tells users to `cp .env.example .env` and edit it. A developer who sets a device ID or the optional planner gets a red `make test`. The "fails closed without credentials" test also stops testing anything.
- **Evidence** (scratch copy; the stock `.env.example` alone passes all 105):
  ```
  $ cp .env.example .env && printf 'SO101_FOLLOWER_ID=lab-arm-7\nSO101_AGENT_PROVIDER=openai_responses\nSO101_AGENT_MODEL=gpt-test\nSO101_OPENAI_API_KEY=sk-test-not-real\n' >> .env
  $ make test
  E       Failed: DID NOT RAISE PlannerError
  E       AssertionError: assert 'lab-arm-7' == 'so101-follower'
  FAILED tests/test_agent_planner.py::test_configured_planner_fails_closed_without_credentials
  FAILED tests/test_hardware.py::test_real_calibration_delegates_to_lerobot_format
  2 failed, 103 passed in 5.88s
  make: *** [test] Error 1
  ```
- **Fix:**
  1. In conftest, disable dotenv, e.g. `monkeypatch.setitem(Settings.model_config, "env_file", None)` or an `SO101_ENV_FILE` switch.
  2. `delenv` every inherited `SO101_*` variable before setting the test values.

### B6. MED: the `[robot]` extra installs two OpenCV distributions into one `cv2` package. CONFIRMED

- **Where:** `pyproject.toml:29` adds `opencv-python>=4.9`. LeRobot 0.6.1 already requires `opencv-python-headless<4.14.0,>=4.9.0`.
- **Defect:** both wheels own the same `cv2/` directory.
  - The resolver installs opencv-python 5.0.0.93 next to opencv-python-headless 4.13.0.92.
  - `cv2` ends up one major version above LeRobot's tested range.
  - Removing or upgrading either wheel breaks `cv2`.
- **Failure scenario:** after `make install-robot` on the robot host, `cv2` is 5.0, which LeRobot was not tested against. Any later uninstall or upgrade of one wheel leaves `cv2` empty, and `hardware/cameras.py:58` (`cv2.VideoCapture`) raises.
- **Evidence:**
  ```
  $ pip install --dry-run --report resolve_robot.json ".[robot]"
  opencv-python-headless 4.13.0.92
  opencv-python 5.0.0.93
  $ unzip -p lerobot-0.6.1-py3-none-any.whl '*/METADATA' | grep -i opencv
  Requires-Dist: opencv-python-headless<4.14.0,>=4.9.0
  scratch venv after install: cv2 5.0.0; then `pip uninstall -y opencv-python`:
  $ python -c "import cv2; print(cv2.__file__); cv2.VideoCapture"
  None
  AttributeError: module 'cv2' has no attribute 'VideoCapture'
  ```
- **Fix:** drop `opencv-python` from `[robot]`, or pin `opencv-python-headless>=4.9,<4.14` explicitly.

### B7. MED: `so101 estop` and `make smoke` ignore the port the suite was started on. CONFIRMED

- **Where:**
  - `cli.py:265-273`: `estop` uses `SO101_HOST:SO101_PORT`. It has no `--url`/`--port`, no error handling, and doesn't check the status code.
  - `Makefile:4-5, 28, 44`: Make's `HOST`/`PORT` ignore `SO101_HOST`/`SO101_PORT`.
  - `Makefile:68-70`: `smoke` never passes `--base`.
- **Defect:** the server's port comes from Make's `PORT`, while the E-stop CLI uses `SO101_PORT`. Nothing keeps them in sync.
- **Failure scenarios:**
  - `PORT=8792 make suite` (because 8000 is busy), then `so101 estop` gets ConnectError and exits 1, while the arm keeps moving.
  - PLAUSIBLE: if another instance is on 8000 (a second checkout, or the compose `api` service), that instance gets E-stopped and the CLI prints `{'ok': True}`.
  - `SO101_PORT=8793 make api` still serves on 8000.
- **Evidence:**
  ```
  $ so101 serve --port 8799 &              # equivalent to PORT=8799 make suite
  $ so101 estop
  ConnectError: [Errno 61] Connection refused        exit: 1
  robot on 8799 estopped = False
  $ SO101_PORT=8799 so101 estop
  {'ok': True, 'detail': 'E-STOP engaged; all motion stopped'}
  $ SO101_HOST=0.0.0.0 SO101_PORT=8793 make -n api suite
  .venv/bin/python -m uvicorn so101.services.app:app --reload --host 127.0.0.1 --port 8000
  cd frontend && npm run build
  .venv/bin/python -m uvicorn so101.services.app:app --host 127.0.0.1 --port 8000
  $ make -n smoke PORT=8792
  .venv/bin/python scripts/smoke_suite.py          # still :8000
  ```
- **Fix:**
  1. Use `PORT ?= $(or $(SO101_PORT),8000)` and the same for HOST, and pass `--base http://$(HOST):$(PORT)` to smoke.
  2. Add `estop --url`. On failure print "E-STOP NOT DELIVERED to <url>; use the power switch" and exit non-zero.

### B8. MED: config and state follow the cwd, and every CLI call creates `data/ datasets/ outputs/` there. CONFIRMED

- **Where:**
  - `config.py:15`: `.env` is read from the cwd
  - `config.py:33-35`: `./data`, `./datasets`, `./outputs`
  - `config.py:78-81`: `get_settings()` calls `ensure_dirs()`
  - `app.py:110`: `app = create_app()` runs at import
  - `cli.py:32-34`: the root callback calls `get_settings()` even for `--help`
- **Defect:** the `.env` (MOCK, ports, IDs), the calibration files and the audit DB all resolve against the cwd. Importing the app, or running any subcommand, creates three directories there.
- **Failure scenarios:**
  - `so101 calibrate follower` run from `frontend/` or `~` writes calibration the API never sees, so motion stays rejected as uncalibrated.
  - The audit trail splits across several `control_plane.sqlite3` files.
  - Running from a subdirectory drops `.env`, so `SO101_MOCK=1` becomes real mode with the default ports.
  - `so101 --help` in `~` litters `~/data`, `~/datasets` and `~/outputs`.
- **Evidence:**
  ```
  proj/.env = SO101_MOCK=1, SO101_FOLLOWER_ID=from-dotenv
  proj     -> mock = True  | follower_id = from-dotenv    | data_dir = .../cwd_demo2/proj/data
  proj/sub -> mock = False | follower_id = so101-follower | data_dir = .../cwd_demo2/proj/sub/data
  $ cd empty  && so101 datasets --help >/dev/null && ls        -> data  datasets  outputs
  $ cd empty2 && python -c "import so101.services.app" && ls   -> data  datasets  outputs
  ```
- **Fix:**
  1. Use a lazy app factory (`uvicorn --factory ...:create_app`) and create dirs on first write.
  2. Use a stable data root: `SO101_HOME`, else platformdirs, else the repo root for editable installs.
  3. Read `.env` from `SO101_ENV_FILE` or that root.

### B9. MED: `make bootstrap` uses whatever `python3` comes first on PATH, which is 3.9 on stock macOS. CONFIRMED

- **Where:**
  - `Makefile:16-19`: `python3 -m venv .venv`
  - `Makefile:3`: `PY` then prefers that venv
  - No doc states the Python 3.12+ or Node requirement. README.md:7 says only "FastAPI/Python core".
- **Defect:** there is no interpreter check. pip fails with a misleading setuptools message, and the broken 3.9 `.venv` stays behind for every target.
- **Evidence** (fresh copy):
  ```
  $ env -i HOME=$HOME PATH=/usr/bin:/bin:/usr/sbin:/sbin make bootstrap
  python3 on stock PATH: /usr/bin/python3 Python 3.9.6
  python3 -m venv .venv
  .venv/bin/python -m pip install -e ".[dev]"
  ERROR: File "setup.py" or "setup.cfg" not found. Directory cannot be installed in editable mode: /private/tmp/so101_audit/devex/bootstrap39
  (A "pyproject.toml" file was found, but editable mode currently requires a setuptools-based build.)
  make: *** [bootstrap] Error 1
  $ .venv/bin/python --version  -> Python 3.9.6
  $ make -n test                -> SO101_MOCK=1 .venv/bin/python -m pytest
  ```
- **Fix:**
  1. Use `PYTHON ?= $(shell command -v python3.12 || command -v python3)` plus a guard: `$(PYTHON) -c 'import sys; sys.exit(sys.version_info < (3,12))' || (echo "Python >= 3.12 required"; exit 1)`.
  2. Add a Prerequisites section to the docs.

### B10. MED: `make install-robot` installs into the global interpreter when `.venv` is missing. CONFIRMED (dry run)

- **Where:** `Makefile:3, 22-24`. getting-started.md:10-12 says "or on the robot host: make install-robot", and README.md:58 lists it too.
- **Defect:** with no `.venv`, `PY` falls back to bare `python3`.
- **Failure scenario:** torch and LeRobot land in the system Python. PLAUSIBLE: on Debian/Ubuntu 23.04+ it fails with PEP 668 "externally-managed-environment".
- **Evidence:**
  ```
  $ make -n install-robot      # no .venv
  python3 -m pip install -e ".[robot,train,dev]"
  $ python3 -c 'import sys; print(sys.executable, sys.prefix, "in venv:", sys.prefix != sys.base_prefix)'
  /usr/local/fbcode/platform010/Python3.12.framework/Versions/3.12/bin/python3.12 /usr/local/fbcode/platform010/Python3.12.framework/Versions/3.12 in venv: False
  ```
- **Fix:** make the install targets depend on a `.venv` target built with the checked interpreter (B9).

### B11. LOW-MED: the README's `make api && make ui` never starts the UI. CONFIRMED

- **Where:** README.md:63.
- **Defect:** `make api` stays in the foreground, so `make ui` runs only after the API exits.
- **Evidence:** after `SO101_MOCK=1 make api PORT=8791 && make ui`:
  - After 10 s there was no vite/npm process, and `curl :5173` returned `000`.
  - Vite started only after the API was killed.
- **Fix:** write "two terminals: `make api` / `make ui`", or use `make dev` or `make suite`.

### B12. LOW-MED: unknown `/api/*` GETs return `200 text/html` once the UI is built. CONFIRMED

- **Where:** `app.py:89-92`.
- **Defect:** the catch-all route answers unmatched `/api/...` and `/ws/...` GETs with `index.html` and status 200.
- **Failure scenario:** typos and removed endpoints look like success, and clients fail later on JSON parsing. Without a built UI the same request returns a JSON 404, so dev and suite mode differ.
- **Evidence:**
  ```
  GET  /api/does-not-exist -> HTTP 200 text/html; charset=utf-8
  POST /api/does-not-exist -> HTTP 405
  ```
- **Fix:** return 404 for the `api/` and `ws/` prefixes in `spa()`, as part of the B1 fix.

### B13. LOW-MED: `so101 doctor` exits 0 on FAIL, and Rich markup strips `[robot]` from the remediation. CONFIRMED

- **Where:**
  - `cli.py:242-262`: no exit code, and detail/remediation are passed to Rich unescaped
  - `operations.py:102`
- **Failure scenario:**
  - `so101 doctor && so101 teleop` proceeds past a FAIL.
  - The operator is told "Install `.`" instead of `.[robot]`.
- **Evidence:**
  ```
  $ SO101_MOCK=0 so101 doctor; echo EXIT=$?
                         SO-101 diagnostics · FAIL
  │ FAIL   │ Robot backend │ Real mode is selected but LeRobot is not installed. │ Install `.` or explicitly set SO101_MOCK=1. │
  EXIT=0
  ```
- **Fix:** use `rich.markup.escape(...)`, and `raise typer.Exit(1)` when `result.overall == "fail"`.

### B14. LOW: `so101` is not on PATH after `make bootstrap`, and no doc says to activate the venv. CONFIRMED

- **Where:**
  - README.md:45-52 and :62
  - getting-started.md:49, 57, 66-67
  - hardware-setup.md:58-59 and 72
- **Evidence:** `SO101_MOCK=1 so101 info` gives `zsh: command not found: so101`, while `.venv/bin/so101` exists.
- **Fix:** add `source .venv/bin/activate` after `make bootstrap` in the docs.

### B15. LOW: `make setup-motors` claims to be interactive and to assign IDs, but it only prints, and only the follower command. CONFIRMED

- **Where:**
  - `Makefile:77`: "Interactive Feetech motor ID / baudrate setup"
  - README.md:61: "assign Feetech ids 1..6 per arm"
  - hardware-setup.md:88: "redo `make setup-motors`"
  - `scripts/setup_motors.py:2-6`: the docstring says it "wraps LeRobot"
  - `scripts/setup_motors.py:31-35`: prints only `--robot.type=so101_follower`
  - `scripts/setup_motors.py:13`: cwd-relative `sys.path.insert(0, "src")`
- **Defect:** the script never runs LeRobot's tool and omits the leader command. Only hardware-setup.md:40 describes it accurately.
- **Failure scenario:** a user believes the IDs are assigned. The troubleshooting row then sends them back to the same command.
- **Evidence:** `make setup-motors` prints the ports, the `id 1: shoulder_pan ... id 6: gripper` table, and "LeRobot not installed. Install with: pip install -e '.[robot]'". With LeRobot present it prints only the follower command.
- **Fix:** either run both commands (follower `--robot.*`, leader `--teleop.*`), or retitle it as "print motor-setup instructions" and print both.

### B16. LOW: `docs/SUITE_V0.md:64` hardcodes the author's path. CONFIRMED

- **Evidence:** `sed -n 64p docs/SUITE_V0.md` returns `cd /Users/baiadnan/Documents/so101`. Line 65 also uses bare `python3 -m venv`, which runs into B9.
- **Fix:** say "from the repo root" and use `python3.12`.

### B17. LOW: mock mode accepts nonexistent models and datasets and reports success, and a test locks this in. CONFIRMED

- **Where:**
  - `cli.py:161-188` (`train`) and `cli.py:201-211` (`run`)
  - `tests/test_api.py:58-72` trains on `"test/x"` and asserts `succeeded`
- **Evidence:**
  ```
  $ SO101_MOCK=1 so101 run /does/not/exist --duration 0.5; echo EXIT=$?
  rollout done: 15 steps in 0.5s          EXIT=0
  $ SO101_MOCK=1 so101 train no/such-dataset --policy act --steps 10; echo EXIT=$?
  explicit mock mode: running deterministic trainer for bc253720120b          EXIT=0
  ```
- **Fix:**
  1. Validate that the dataset and model path exist before starting, in both modes.
  2. Have the test record a dataset first.

### B18. LOW: the Docker real-hardware hint cannot work, images run as root, and there are no health checks. PLAUSIBLE (no Docker)

- **Where:**
  - `docker-compose.yml:1-3, 34-37`: hint using the macOS path `/dev/tty.usbmodem58760431541`
  - `docker-compose.yml:47-48`: `depends_on` without a condition
  - `docker/Dockerfile.api` and `Dockerfile.suite`: no `USER`, no `HEALTHCHECK`, and apt installs `libgl1 libglib2.0-0` with no OpenCV in the image
- **Defect:**
  - The image has no `[robot]` extra, and `SO101_FOLLOWER_PORT` is never set to `/dev/ttyUSB0`, so real mode fails closed even on Linux.
  - Docker Desktop on macOS cannot pass USB serial devices through.
  - Root containers create root-owned `./data`, `./datasets` and `./outputs` on Linux hosts.
- **Fix:**
  1. Document that the host runs real hardware, or add a `robot` image target.
  2. Add `USER 1000`, a urllib `HEALTHCHECK` on `/api/health`, and `depends_on: {api: {condition: service_healthy}}`.

---

## 2. Coverage

**Command:** `COVERAGE_FILE=/tmp/so101_audit/devex/.coverage SO101_MOCK=1 python -m pytest --cov=so101 --cov-branch -p no:cacheprovider -p no:randomly`

**Result:** 105 passed. TOTAL: 5783 statements, 1440 missed, 1416 branches, 355 partial, **71%**. The numbers vary by about 1 line between runs because of timing.

| Module | Stmts | Miss | Branch | BrPart | Cover |
| --- | ---: | ---: | ---: | ---: | ---: |
| hardware/safety.py | 110 | 15 | 52 | 12 | 82% |
| services/robot_manager.py | 860 | 240 | 260 | 61 | 68% |
| services/store.py | 99 | 7 | 6 | 1 | 92% |
| services/app.py | 44 | 6 | 4 | 1 | 85% |
| config.py | 48 | 0 | 2 | 0 | 100% |
| routers/control.py | 71 | 10 | 0 | 0 | 86% |
| routers/tracking.py | 95 | 12 | 40 | 5 | 84% |
| routers/eval.py | 223 | 29 | 46 | 16 | 83% |
| routers/agent.py | 103 | 19 | 36 | 10 | 79% |
| routers/operations.py | 67 | 12 | 12 | 5 | 78% |
| routers/robot.py | 38 | 10 | 2 | 1 | 72% |
| routers/training.py | 37 | 10 | 8 | 2 | 64% |
| routers/models.py | 21 | 9 | 2 | 0 | 52% |
| routers/datasets.py | 102 | 50 | 2 | 1 | 51% |
| routers/recording.py | 34 | 20 | 2 | 0 | 39% |
| routers/telemetry.py | 37 | 22 | 4 | 1 | 39% |
| routers/calibration.py | 54 | 41 | 4 | 0 | 22% |
| simulation/engine.py | 219 | 11 | 56 | 11 | 92% |
| data/recorder.py | 336 | 55 | 100 | 30 | 79% |
| data/dataset.py | 273 | 64 | 94 | 30 | 74% |
| hardware/cameras.py | 79 | 17 | 14 | 2 | 75% |
| agents/planner.py | 88 | 22 | 24 | 6 | 68% |
| hardware/calibration.py | 90 | 28 | 30 | 5 | 59% |
| hardware/motors.py | 294 | 112 | 92 | 22 | 58% |
| eval/evaluator.py | 50 | 18 | 6 | 0 | 57% |
| policy/runner.py | 71 | 28 | 12 | 0 | 54% |
| training/trainer.py | 104 | 46 | 44 | 7 | 49% |
| **cli.py** | **173** | **173** | 24 | 0 | **0%** |

### Gaps

- **`cli.py` has 0% coverage.** That includes the E-stop command. There are no `CliRunner` tests, which is why B7 and B13 went unnoticed.
- **`robot_manager.py`:** the untested lines are mostly safety failure paths:
  - `clear_fault` (148-158)
  - calibration claim, finish and cancel (166-227)
  - heartbeat fault (528-543)
  - arm and torque failure (653-660, 699-707)
  - `send_goal` failure (841-852)
  - I/O-timeout de-energize (929-941, 958-971)
  - recording through the manager (1143-1206)
- **`safety.py`:** lines 56, 59, 66, 68-74, 78, 97, 119, 122, 125, 147-148 and 156 are missed. Branch 170->176 (the first command is not rate-limited) is never taken.
- **Routers:** calibration 22%, recording 39%, telemetry 39%, datasets and models about 51%.
- **Frontend:** no tests and no linter. The only npm scripts are `dev`, `build` and `preview`.

### Test-quality notes

- `test_api.py:46-55` asserts `mode == "teleop"` twice (lines 52 and 54) and never checks that the follower tracks the leader.
- `:58-72` covers B17.
- The tests at `:28`, `:34` and `:75` are smoke-level only.
- **Strengths:** order-independent, xdist-safe, about 6 s, never touches tracked files, and has config-drift tests.
- **Weaknesses:** timing (B4), `.env` leakage (B5), and directory creation in the cwd at collection time (B8).

---

## 3. Top improvements (max 8)

1. **Contain static serving (S).** Resolve the path and check `is_relative_to`, return 404 for `api/` and `ws/`, and add a traversal test.
   - Why: closes B1, the only HIGH remote file read, and B12.
2. **Package the UI in the wheel (M).** Build into `src/so101/web`, load it via `importlib.resources`, add a wheel-contents test, and make the suite image non-editable.
   - Why: makes "wheel validation" real (B3).
3. **Make the control-plane clock injectable (M).** Use a fake or scaled clock fixture, poll against deadlines, and start the deadman after torque-enable.
   - Why: the safety-critical tests become deterministic on 2-vCPU CI (B4). Any CI gate needs this first.
4. **Hermetic tests plus CLI tests (S).** In conftest, disable dotenv and clear `SO101_*`. Add `CliRunner` tests for `info`, `doctor` (exit code) and `estop` (`--url`, failure path).
   - Why: fixes B5, takes `cli.py` off 0%, and locks in the B7 and B13 fixes.
5. **One machine-checked release gate (M).** Add `make ci`:
   - lint, plus mypy on tests and scripts
   - pytest with per-module coverage floors
   - `build-ui` and `npm audit`
   - wheel build and contents check
   - install from the wheel into a fresh venv
   - mock smoke

   Then add `.github/workflows/ci.yml` mirroring it (v0.2 plan §3).
   - Why: half of the release claim is manual today.
6. **Separate the developer and operator run paths (S; persisting the latch is M).**
   - `make api` without reload, plus an `api-dev` target.
   - Make's `HOST`/`PORT` default from `SO101_*`.
   - `smoke --base`.
   - `estop --url` with a loud, non-zero failure.
   - A `doctor` exit code.
   - Optionally, persist the E-stop latch.
   - Why: the run and stop paths should not depend on which tool started the server (B2, B7, B13).
7. **Stable state location (M).** A lazy app factory, no directory creation in `get_settings()`, an `SO101_HOME`/platformdirs root, and `.env` from `SO101_ENV_FILE` or that root.
   - Why: B8.
8. **Install and dependency hygiene (S).**
   - A Python 3.12 check in bootstrap.
   - A Prerequisites section (Python 3.12, Node 24, plus the proxy CA-bundle note).
   - Install targets that depend on `.venv`.
   - Drop `opencv-python`.
   - Set Ruff to `py312`, which surfaces 28 findings.
   - Remove the unused `python-multipart`, `pytest-asyncio` and `SO101_HF_USER`.
   - `.env.example` parity.
   - Use `npm ci` in `make ui`.
   - Add the venv activation step to the docs.
   - Honest wording for `setup-motors`.
   - Why: B6, B9, B10, B11, B14, B15, B16.

---

## 4. What is good and must be preserved

- **Explicit mock flag, and real mode fails closed.** With `SO101_MOCK=0` and no LeRobot, `doctor` shows FAIL instead of substituting mock data. Every check comes with remediation, in both `doctor` and `/api/diagnostics`.
- **A smoke script that is safe to rerun.** It is idempotent, refuses non-mock hardware, and passed 10/10 twice.
- **Config-drift guards:** `test_config_assets.py` and the sync test for the packaged `scenarios.yaml`.
- **Safe network defaults.** The server binds to loopback by default, compose publishes only on 127.0.0.1, and the docs warn against exposing the API to a LAN.
- **Clean repo and package hygiene.**
  - `.gitignore` covers data, `.env` and build artifacts.
  - The sdist excludes `dist`, `node_modules`, `.env` and data.
  - No secrets or SQLite files are tracked.
  - Both `.dockerignore` files are correct.
  - The suite image is multi-stage and uses `npm ci`.
- **Pinned dependencies.** LeRobot is pinned to `==0.6.1`, the lockfile is committed, and npm audit reports 0 vulnerabilities.
- **Static checks are clean.** The strict TypeScript build passes, and Ruff, format and mypy are clean on `src`.
- **A fast, isolated test suite.**
  - It is quick, order-independent and parallel-safe under normal load.
  - It never touches tracked files.
  - The store and simulation engine are at 92% or higher.
  - The WAL audit trail is tested across a store reopen.
- **Consistent versioning.** Version 0.1.0 matches across pyproject, the API and the UI, and the LICENSE ships in the wheel.
- **Accurate docs.** Most mock-mode CLI examples work exactly as written. The `lerobot-setup-motors` flags (hardware-setup.md:43-44) match LeRobot 0.6.1, and the troubleshooting endpoints exist.

---

## Appendix A: documented commands in mock mode

These ran from the wheel venv or the scratch copy, with dirs under `/tmp/so101_audit/devex/docs_run`.

| Doc:line | Command | Result |
| --- | --- | --- |
| README:27, GS:10 | `make bootstrap` | OK with Python 3.12 first on PATH. Fails with stock macOS `python3` (B9). `so101` is not on PATH afterwards (B14). |
| README:28, GS:18 | `SO101_MOCK=1 make suite` | OK. Smoke 10/10. |
| README:32, GS:25 | `make api` + `make ui` in two terminals | OK, but it auto-reloads (B2). |
| README:39 | `docker compose --profile standalone up --build suite` | Not run (no Docker). |
| README:45-46 | `so101 info`, `so101 doctor` | OK. `doctor` exits 0 on FAIL (B13). |
| README:47 | `teleop --duration 5` | OK, 5.4 s. |
| README:48 | `record demo/pick --episodes 2 --episode-seconds 3` | OK, 2 x 90 frames. |
| README:49 | `datasets ls` | OK. |
| README:50 | `train demo/pick --policy act --steps 500` | OK. The progress line has no newlines when piped (cosmetic). |
| README:51 | `models ls` | OK. The path is truncated at 80 columns (cosmetic). |
| README:52, GS:66 | `run <model_path> --duration 5` | OK, 150 steps. Also succeeds on a nonexistent path (B17). |
| GS:67 | `eval <model_path> --episodes 2` (doc says 10) | OK, about 30 s per episode, 100% success. |
| GS:49, 57 | `record` 30x20 s, `train` 100k steps | Same code paths, run scaled down. OK. |
| README:58, GS:12 | `make install-robot` | Installs globally when `.venv` is missing (B10). Pulls in two OpenCVs (B6). |
| README:59 | `cp .env.example .env` | OK as shipped. Once customized, it breaks the tests (B5). |
| README:60, HW:50 | `make find-cameras` | Without OpenCV it prints "opencv-python not installed" and exits 0 (`scripts/find_cameras.py:11-12`). |
| README:61, HW:40 | `make setup-motors` | Prints only, follower command only (B15). |
| README:62, HW:58-59 | `calibrate follower` / `calibrate leader` | OK in mock. Writes `data/calibration_*.json` under the cwd (B8). |
| README:63 | `make api && make ui` | The UI never starts (B11). |
| HW:43-44 | `lerobot-setup-motors ...` | Flags match the 0.6.1 wheel. Not executed (needs hardware). |
| HW:72 | `teleop --duration 10` | OK. |
| TS:70 | `GET /api/diagnostics`, `GET /api/events` | Both return 200 JSON. |
| CLI | `replay demo/pick --episode 0` | OK, 90 frames. A missing episode prints a raw `FileNotFoundError` traceback. |
| Makefile | `make cli`, `make fmt`, `make verify` (fresh copy, about 18 s) | OK. |
| SUITE_V0:64 | `cd /Users/baiadnan/...` | Author-specific path (B16). |

## Appendix B: `.env.example` vs `config.py`

| Key | `.env.example` | `config.py` default | Note |
| --- | --- | --- | --- |
| `SO101_FOLLOWER_PORT` / `SO101_LEADER_PORT` | `/dev/tty.usbmodem58760431541` / `...551` (compose:37 uses the same) | `/dev/tty.usbmodem585A0077531` / `...541` | The placeholders disagree. Use one obvious placeholder. |
| `SO101_CAMERAS` | `0,1` | `"0"` | On a one-camera host this configures a missing camera 1. |
| `SO101_HF_USER` | present | field exists, unused anywhere in `src/` | Remove it or use it. |
| `SO101_AGENT_TIMEOUT_SECONDS` | missing | `20.0`, used by the planner | Add it. |
| All other keys | match | match | |

## Appendix C: packaging and Docker

- **Wheel:** 117,481 B. It contains 58 `.py` files, `scenarios.yaml` and the LICENSE. The CLI works from a fresh venv. The API started from another cwd runs, but has no UI (B3), and its state lands in that cwd (B8).
- **Editable install:** finds the UI from any cwd, but `.env` and data still follow the cwd.
- **Sdist:** 240,519 B. It has the frontend sources and no `dist`, `node_modules`, `.env` or data.
- **`Dockerfile.api`:** `COPY src` comes before `pip install -e .`, so every source edit re-installs the dependencies. It has no UI, which is fine for the dev stack.
- **`Dockerfile.suite`:** depends on the editable install so that `parents[3]` is `/app`. It exposes B1, including the bind-mounted `/app/data`.

## Appendix D: CI readiness

There is no `.github/` directory. The v0.2 plan (§3) proposes `ci.yml` and `make ci`. The minimum pipeline:

- **backend** (Python 3.12):
  - `pip install -e ".[dev]"`
  - ruff check and format, and `mypy src/so101` (add tests and scripts after their 10 errors are fixed)
  - `SO101_MOCK=1 pytest -p no:cacheprovider --cov --cov-branch`, with an overall floor of 70% and per-module floors
- **frontend** (Node 24): `npm ci`, `npm run build`, `npm audit --audit-level=high`.
- **package:**
  - `python -m build`
  - assert the wheel contains the UI (this fails today, B3)
  - install into a fresh venv and run `so101 --help` and `SO101_MOCK=1 so101 doctor`
- **smoke:** start `SO101_MOCK=1 so101 serve`, wait for `/api/health`, then run `scripts/smoke_suite.py`.
- **optional:** `docker build` of both images.

Blockers: B4 (flaky on 2-4 vCPU runners) and B3. B5 does not affect CI, since there is no `.env` there.

## Environment note (not a project bug)

On this Mac, pip and npm needed `PIP_CERT` and `NODE_EXTRA_CA_CERTS` pointed at a system CA bundle (`/etc/ssl/cert.pem`) to get through the TLS-intercepting proxy. This is environmental. A docs suggestion: the Prerequisites section could add one line saying that behind such a proxy, users should point those two variables at their CA bundle.

**Scratch artifacts** are in `/tmp/so101_audit/devex/`: `repo_copy`, `venv_wheel`, `venv_resolve`, `dist/` (wheel and sdist), `.coverage`, and the server logs under `*_run*/`.
