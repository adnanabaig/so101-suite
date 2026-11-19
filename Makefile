# SO-101 platform: common tasks. `make help` lists them by section.
.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory

# --- Python ----------------------------------------------------------------------
VENV ?= .venv
# Creates $(VENV): python3.12 when it is on PATH, else python3.13, else python3.
# The venv target refuses anything outside 3.12-3.13 (pyproject.toml
# requires-python >=3.12,<3.14).
PYTHON ?= $(shell command -v python3.12 2>/dev/null || command -v python3.13 2>/dev/null || command -v python3 2>/dev/null)
# Runs everything else: the venv's interpreter when there is one. CI installs
# into the runner's Python and has no venv.
PY ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
# The so101 CLI.
SO101 ?= $(PY) -m so101.cli
PY_VERSION_OK = import sys; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, 13))
PY_VERSION = import platform; print(platform.python_version())

# --- Server address ----------------------------------------------------------------
# Defaults follow SO101_HOST / SO101_PORT, so `make api`, `make suite`, `make
# smoke`, the stop targets and the so101 CLI agree on one server. make does not
# read .env: export SO101_HOST / SO101_PORT, or pass HOST=... PORT=..., to move it.
HOST ?= $(or $(SO101_HOST),127.0.0.1)
PORT ?= $(or $(SO101_PORT),8000)
# What uvicorn serves: the app factory, so importing the module has no side effects.
UVICORN_APP = --factory so101.services.app:create_app
# Clients cannot connect to a wildcard bind address; they use loopback instead.
CLIENT_HOST = $(if $(filter 0.0.0.0 ::,$(HOST)),127.0.0.1,$(HOST))
CLIENT_URL = http://$(if $(findstring :,$(CLIENT_HOST)),[$(CLIENT_HOST)],$(CLIENT_HOST)):$(PORT)

# --- Frontend ----------------------------------------------------------------------
# npm ci deletes node_modules before installing. When frontend/node_modules is a
# symlink (a git worktree sharing another checkout's install), that would wipe
# the other checkout's install, so every npm ci here refuses to run through one.
NODE_MODULES_GUARD = if [ -L frontend/node_modules ]; then echo "error: frontend/node_modules is a symlink; npm ci would delete the install it points to. Remove the link to install here, or pass FRONTEND_INSTALL=0 to reuse it." >&2; exit 1; fi
# The CI jobs install with npm ci. FRONTEND_INSTALL=0 reuses the existing
# frontend/node_modules instead.
FRONTEND_INSTALL ?= 1
# The CI frontend job runs these package.json scripts; a missing one fails.
FRONTEND_CI_SCRIPTS = lint typecheck test build

LINT_PATHS = src tests scripts hatch_build.py
TYPED_PATHS = src/so101 scripts hatch_build.py
# Port for the throwaway smoke server; 0 picks a free one.
SMOKE_PORT ?= 0
# Opt-in M3 rehearsal evidence. A timestamp makes every invocation a new ledger
# entry; the harness refuses a non-empty directory rather than mixing runs.
INSTALLED_E2E_EVIDENCE ?= $(CURDIR)/.artifacts/installed-e2e-$(shell date -u +%Y%m%dT%H%M%SZ)
INSTALLED_E2E_BACKEND ?= both
INSTALLED_E2E_UI ?= 0
INSTALLED_E2E_UI_CWD ?= $(CURDIR)/frontend
G15_CI_EXPECTED_COMMIT ?=
G15_CI_EVIDENCE ?=
HANDOFF_APPROVED_COMMIT ?=

.PHONY: help
help: ## Show this help
	@awk 'BEGIN { FS = ":.*## " } /^##@ / { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } /^[a-zA-Z0-9_-]+:.*## / { printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

##@ Setup

$(VENV)/bin/python:
	@test -n "$(PYTHON)" || { echo "error: no python3.12, python3.13 or python3 on PATH. Install Python 3.12 (README: Prerequisites)." >&2; exit 1; }
	@"$(PYTHON)" -c '$(PY_VERSION_OK)' || { echo "error: $(PYTHON) is Python $$("$(PYTHON)" -c '$(PY_VERSION)'); the suite needs Python 3.12 or 3.13. Install python3.12, or pass PYTHON=/path/to/python3.12." >&2; exit 1; }
	"$(PYTHON)" -m venv $(VENV)

.PHONY: venv
venv: $(VENV)/bin/python ## Create .venv with Python 3.12 or 3.13 (or check the existing one)
	@$(VENV)/bin/python -c '$(PY_VERSION_OK)' || { echo "error: $(VENV) is Python $$($(VENV)/bin/python -c '$(PY_VERSION)'); the suite needs 3.12 or 3.13. Remove it (rm -rf $(VENV)) and run make bootstrap again." >&2; exit 1; }

.PHONY: install
install: venv ## Install core + dev deps into .venv (editable)
	$(VENV)/bin/python -m pip install -e ".[dev]"

.PHONY: install-robot
install-robot: venv ## Install hardware + training deps into .venv (on the robot host)
	$(VENV)/bin/python -m pip install -e ".[robot,train,dev]"

.PHONY: frontend-deps
frontend-deps: frontend/node_modules/.package-lock.json ## Install the locked frontend deps (npm ci) if the lockfile changed

frontend/node_modules/.package-lock.json: frontend/package-lock.json
	@$(NODE_MODULES_GUARD)
	cd frontend && npm ci
	@touch $@

.PHONY: bootstrap
bootstrap: install frontend-deps ## Create .venv and install backend + frontend dependencies
	@echo "Next: source $(VENV)/bin/activate   # puts so101 on PATH"

##@ Run

.PHONY: api
api: ## Run the backend API (no auto-reload; the one to use with hardware)
	$(PY) -m uvicorn $(UVICORN_APP) --host $(HOST) --port $(PORT)

# DEV ONLY. --reload restarts the server whenever a source file changes, and
# each restart is a service shutdown: the arm Collapses to its rest pose with
# the motors off, and any running motion session ends. The E-stop latch and
# faults persist and come back after the restart (RobotManager._restore_latches),
# but a real arm would still collapse on every file save, so never run it with an
# arm connected: use `make api`, `make suite` or `so101 serve`. The guard refuses
# unless SO101_MOCK=1 (environment or .env).
# `make dev` runs it first too, so that Vite does not start on its own.
MOCK_ONLY_GUARD = $(PY) -c 'import sys; from so101.config import Settings; sys.exit(not Settings().mock)' || { echo "error: $@ auto-reloads, and every reload shuts the server down, which Collapses the arm. It runs only with mock hardware (SO101_MOCK=1). With an arm connected use make api, make suite or so101 serve." >&2; exit 1; }

.PHONY: api-dev
api-dev: ## DEV ONLY: backend with auto-reload; refuses unless SO101_MOCK=1
	@$(MOCK_ONLY_GUARD)
	$(PY) -m uvicorn $(UVICORN_APP) --reload --host $(HOST) --port $(PORT)

.PHONY: ui
ui: frontend-deps ## Run the Vite dev server (installs locked deps first)
	cd frontend && npm run dev

.PHONY: build-ui
build-ui: ## Strict TypeScript check + production frontend build
	cd frontend && npm run build

.PHONY: dev
dev: ## DEV ONLY: api-dev + ui together; refuses unless SO101_MOCK=1 (Ctrl-C stops both)
	@$(MOCK_ONLY_GUARD)
	$(MAKE) -j2 api-dev ui

.PHONY: suite
suite: build-ui ## Build the UI and serve the whole suite from one process
	$(PY) -m uvicorn $(UVICORN_APP) --host $(HOST) --port $(PORT)

.PHONY: cli
cli: ## Show the so101 CLI help
	$(SO101) --help

##@ Quality

# A full run fails when tests skip beyond tests/skip-budget.toml (CONTRIBUTING.md).
SKIP_BUDGET = --skip-budget tests/skip-budget.toml

.PHONY: test
test: ## Run the test suite (mock hardware, no device needed) within the skip budget
	SO101_MOCK=1 $(PY) -m pytest $(SKIP_BUDGET)

.PHONY: coverage
coverage: ## Tests with branch coverage and the skip budget, then enforce coverage-floors.toml
	SO101_MOCK=1 $(PY) -m pytest --cov --cov-report=term --cov-report=json:coverage.json $(SKIP_BUDGET)
	$(PY) scripts/check_coverage.py coverage.json

.PHONY: lint
lint: ## Ruff lint + format check, mypy
	$(PY) -m ruff check $(LINT_PATHS)
	$(PY) -m ruff format --check $(LINT_PATHS)
	$(PY) -m mypy $(TYPED_PATHS)

.PHONY: fmt
fmt: ## Auto-format
	$(PY) -m ruff format $(LINT_PATHS)
	$(PY) -m ruff check --fix $(LINT_PATHS)

.PHONY: verify
verify: test lint build-ui ## Fast local tests, lint and UI build (the release gate is make ci)

.PHONY: smoke
smoke: ## Smoke-test a running mock suite at http://HOST:PORT
	$(PY) scripts/smoke_suite.py --base $(CLIENT_URL)

# The UI is built first, so the packages never carry a stale frontend/dist:
# hatch_build.py refuses a dist whose build stamp does not match the frontend
# sources, and check_dist.py checks what was packed (CONTRIBUTING.md).
.PHONY: wheel
wheel: frontend-deps ## Build the UI, then the sdist + wheel into dist/, and check their contents
	$(MAKE) build-ui
	rm -rf dist
	SO101_REQUIRE_UI=1 $(PY) -m build --outdir dist .
	$(PY) scripts/check_dist.py dist --ui required

.PHONY: installed-e2e
installed-e2e: wheel ## Install the UI wheel and run isolated mock + MuJoCo product workflows
	$(PY) scripts/installed_e2e.py dist --backend $(INSTALLED_E2E_BACKEND) \
		--evidence-dir "$(INSTALLED_E2E_EVIDENCE)" \
		$(if $(filter 1 true yes,$(INSTALLED_E2E_UI)),--ui-e2e --ui-cwd "$(INSTALLED_E2E_UI_CWD)",)

# This is deliberately opt-in and is not a dependency of `ci`: it enforces
# the exact Ubuntu G15 host/source/fresh-checkout contract before bootstrap.
.PHONY: g15-ci-evidence
g15-ci-evidence: ## One-shot Ubuntu 24.04 G15 bootstrap + CI evidence (no hardware)
	@test -n "$(G15_CI_EXPECTED_COMMIT)" || { echo "error: pass G15_CI_EXPECTED_COMMIT=<full candidate commit>" >&2; exit 2; }
	@test -n "$(G15_CI_EVIDENCE)" || { echo "error: pass G15_CI_EVIDENCE=<new external evidence directory>" >&2; exit 2; }
	/usr/bin/python3.12 -I scripts/run_g15_ci.py \
		--expected-commit "$(G15_CI_EXPECTED_COMMIT)" \
		--evidence-dir "$(G15_CI_EVIDENCE)"

.PHONY: handoff
handoff: ## Build the Zephyrus handoff folder from a clean commit: make handoff OUT=<dir>
	@test -n "$(OUT)" || { echo "error: pass OUT=<dir>, an empty or new folder outside the repo (make handoff OUT=/media/usb/so101-handoff)" >&2; exit 1; }
	$(PY) scripts/make_handoff.py --out "$(OUT)"

.PHONY: handoff-archive
handoff-archive: ## Build one exact-commit handoff .tar.gz plus its SHA-256 sidecar
	@test -n "$(OUT)" || { echo "error: pass OUT=<new .tar.gz path outside the repo>" >&2; exit 1; }
	@test -n "$(HANDOFF_APPROVED_COMMIT)" || { echo "error: pass HANDOFF_APPROVED_COMMIT=<full approved commit>" >&2; exit 1; }
	$(PY) scripts/make_handoff.py --archive "$(OUT)" \
		--expected-commit "$(HANDOFF_APPROVED_COMMIT)"

##@ CI (make ci mirrors .github/workflows/ci.yml; never touches hardware)

.PHONY: ci
ci: ## Run every CI job in order: backend, frontend, smoke, wheel
	$(MAKE) ci-backend
	$(MAKE) ci-frontend
	$(MAKE) ci-smoke
	$(MAKE) ci-wheel FRONTEND_INSTALL=0
	@echo "make ci: backend, frontend, smoke and wheel jobs passed"

.PHONY: ci-backend
ci-backend: lint coverage ci-thor-worker ## CI backend: lint, tests/floors, Thor host contracts

.PHONY: ci-thor-worker
ci-thor-worker: ## Stdlib-only optional Thor worker contract/security tests (no Thor/Docker/CUDA)
	$(MAKE) -C thor-worker test PYTHON="$$($(PY) -c 'import sys; print(sys.executable)')"

.PHONY: ci-frontend
ci-frontend: frontend-scripts ## CI frontend job: npm ci, lint, typecheck, test, build, audit
	$(MAKE) frontend-ci-install
	cd frontend && npm run lint
	cd frontend && npm run typecheck
	cd frontend && npm run test
	cd frontend && npm run build
	cd frontend && npm audit --audit-level=high

.PHONY: ci-smoke
ci-smoke: ## CI smoke job: throwaway mock API (temp data) + scripts/smoke_suite.py
	$(PY) scripts/run_smoke.py --port $(SMOKE_PORT)

.PHONY: ci-wheel
ci-wheel: ## CI wheel job: UI, sdist + wheel, contents check, clean-venv install smoke
	$(MAKE) frontend-ci-install
	$(MAKE) build-ui
	SO101_MOCK=1 BUILT_UI_PATHS_REQUIRED=1 $(PY) -m pytest \
		tests/test_gate_guard.py::test_c1_every_file_in_the_built_ui_passes_the_path_check
	rm -rf dist
	SO101_REQUIRE_UI=1 $(PY) -m build --outdir dist .
	$(PY) scripts/check_dist.py dist --ui required
	$(PY) scripts/wheel_smoke.py dist --ui required

# Fails, before anything is installed, unless package.json defines every script
# the CI frontend job runs. make ci never skips a missing step.
.PHONY: frontend-scripts
frontend-scripts:
	@cd frontend && node -e 'const s = require("./package.json").scripts || {}; const want = process.argv.slice(1); const missing = want.filter((n) => typeof s[n] !== "string"); if (missing.length) { console.error("error: frontend/package.json has no " + missing.join(", ") + " script. The CI frontend job runs npm run " + want.join(", ") + ", and make ci does not skip a missing one."); process.exit(1); }' $(FRONTEND_CI_SCRIPTS)

.PHONY: frontend-ci-install
ifeq ($(FRONTEND_INSTALL),1)
frontend-ci-install:
	@$(NODE_MODULES_GUARD)
	cd frontend && npm ci
else
frontend-ci-install:
	@echo "FRONTEND_INSTALL=$(FRONTEND_INSTALL): reusing the existing frontend/node_modules (no npm ci)"
endif

##@ Hardware

.PHONY: find-cameras
find-cameras: ## List connected physical cameras (real backend only)
	$(PY) scripts/find_cameras.py

.PHONY: setup-motors
setup-motors: ## Print the LeRobot motor id/baudrate commands for both arms (prints only)
	$(PY) scripts/setup_motors.py

##@ Safety (sent to the server at http://HOST:PORT; the power switch is the last resort)

# Never called by CI or by any target CI runs (tests/test_makefile.py checks).
.PHONY: estop
estop: ## E-stop: cut torque on every motor now; latched until a deliberate reset
	@echo "E-stop -> $(CLIENT_URL)"
	@$(SO101) estop --url $(CLIENT_URL) || { echo "E-STOP NOT CONFIRMED by $(CLIENT_URL). Cut power at the switch." >&2; exit 1; }

.PHONY: halt
halt: ## Halt: freeze in place; motors stay on and hold position (then collapse on timeout)
	@echo "Halt -> $(CLIENT_URL)"
	@$(SO101) halt --url $(CLIENT_URL) || { echo "HALT NOT CONFIRMED by $(CLIENT_URL). If the arm moves, run make estop or cut power." >&2; exit 1; }

.PHONY: collapse
collapse: ## Collapse: lower slowly to the rest pose, then turn the motors off
	@echo "Collapse -> $(CLIENT_URL)"
	@$(SO101) collapse --url $(CLIENT_URL) || { echo "COLLAPSE NOT CONFIRMED by $(CLIENT_URL). If the arm moves, run make estop or cut power." >&2; exit 1; }

##@ Docker

# The containers run as uid 1000. Creating the bind-mounted dirs first keeps
# Docker from creating them as root.
.PHONY: docker
docker: ## Build + run the mock stack in Docker (real hardware runs on the host)
	mkdir -p data datasets outputs
	docker compose up --build
