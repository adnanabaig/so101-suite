# Development preview — November 18, 2026

This is an experimental source release for reviewing the SO-101 workstation's
engineering. It is not a qualified hardware release or a demonstration of
autonomous task performance. Package version remains `0.2.2rc5`.

## Publication checks

Local checks used Python 3.12.13 and Node 22.23.3 on the G15's Ubuntu 22.04
environment. CI targets Node 24 and Python 3.12/3.13; local results do not
substitute for that separate matrix.

- Repository-wide Ruff lint and formatting pass; mypy passes for 160 source files.
- Frontend: 765 tests across 71 files, lint, TypeScript and production build pass.
- Compatible dependency updates clear the frontend audit: zero reported vulnerabilities.
- Isolated mock HTTP workflow smoke passes; no physical arms are connected by it.
- Source-archive HIL quote verification: 24 focused cases pass. Without the old
  Git history, the existing archived design is accepted only when its SHA-256
  matches the document at the cited original commit; no skip budget is raised.
- The complete backend run finished: **3,753 passed, 51 failed, 9 skipped**,
  with 23 subtests passed. `make ci` stopped at that failed backend gate;
  frontend and HTTP smoke were then checked independently. The full software
  release gate has **not passed**.
- Committed source snapshot `993d0e6`: wheel and source distribution build,
  package contents checks and clean-venv install smoke pass. The installed
  application serves the bundled UI and assets. The final publication changes
  after that build affect documentation only. The laptop's untracked source
  files were included in the reviewed snapshot before running these checks;
  the packaging guard was preserved.

Full CI, coverage floors and physical qualification remain separate gates.
Known regression failures are published rather than hidden by disabling jobs,
lowering thresholds or removing tests.

The failed backend tests include bus ownership/adapter contracts, preflight/HIL
instructions, persistence and dataset handling, simulator episode/rendering
paths, subprocess lifecycle handling and documentation assertions. These are
unresolved regression signals, including checks on safety-related paths; a
passing mock workflow does not clear them for physical operation. Some failures
may depend on the laptop's installed dependencies or process environment; they
have not all been isolated to a root cause. This preview is suitable for source
review and development, not a claim that every supported workflow works.

## Bench evidence and remaining work

Bench observations date from October 23–November 1, 2026. Publication does not
requalify the physical rig. See [G15 status](g15.md) for the dated evidence.

- Installed follower calibration has ten accepted sweeps per joint; the leader
  has an installed six-sweep baseline, with wrist-roll interpretation/clipping
  and physical frame alignment still open.
- Teleop tuning episodes contain four RGB feeds, control ticks, encoder and
  health data. They are not accepted as ten clean task demonstrations.
- OBSBOT and D435i RGB exercised near 30 fps; icSpring feeds remain below target.
- An isolated D435i RGB-D test delivered approximately 30 fps. Live depth/IMU
  recording and D405 depth delivery are unfinished.
- Encoder-driven geometry is exercised; camera-to-table and robot-base alignment
  remain unqualified. Phone target captures are candidates, not motion authority.
- The old shoulder/rest-pose preflight blocker was resolved at the October 27
  checkpoint. Physical collision clearance, reliable all-pose rest return and
  on-kit validation of the foreground stop sequence remain operator work.
- No real trained-policy manipulation performance, safety certification or
  end-to-end hardware qualification is claimed.

## Source snapshot

The initial public repository contains the current application, tests, service
templates, example configuration and documentation. It starts a new Git history
instead of uploading the laptop's backup history. Recordings, private machine
settings, credentials, local session notes, caches, build products and old
archives are excluded. Hardware service examples require per-rig configuration;
start with the mock quick start in the README.
