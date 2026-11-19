# G15 one-shot software CI evidence

This is the M0 software gate for the ASUS Zephyrus G15. It runs before any
installation or hardware bring-up and deliberately does **not** run `doctor`,
preflight, HIL, camera discovery, calibration, or an installed-artifact
rehearsal. Leave the arms and cameras disconnected and unpowered.

The runner is standard-library-only and runs before bootstrap. It accepts the
full commit printed in the candidate manifest; do not derive the expected value
from the checkout being tested, because that would make the identity check
meaningless.

From a fresh candidate clone, as the normal user (never with `sudo`):

```bash
/usr/bin/make g15-ci-evidence \
  G15_CI_EXPECTED_COMMIT=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  G15_CI_EVIDENCE="$PWD/../evidence/g15-ci-aaaaaaaa"
```

Replace the example `a...a` value with the candidate's exact 40-character
commit. The evidence leaf may be absent or empty, but an existing leaf must be
mode `0700`. It must be outside the checkout. The runner creates a new leaf as
`0700` and every evidence file as `0600`.

## Refusal is intentional

Before either make command starts, the runner requires all of the following:

- DMI identity for an ASUS Zephyrus G15/GA503, plus Linux x86-64, Ubuntu
  24.04, Python 3.12, Node 24, and a non-root user;
- the exact expected Git commit and no tracked, staged, or untracked changes;
- no prior `.venv`, frontend dependencies/build, distribution, coverage, or
  other reusable CI output, and no root/frontend `.env` file;
- no inherited environment variable whose name starts with `SO101_`, in any
  letter case; and
- a new or empty private evidence directory outside the repository.

Python, Node, npm, Git, and GNU Make must resolve from the fixed system path to
root-owned executable files whose resolved parents are not group/other
writable. Their absolute paths, modes, owners, and SHA-256 digests are recorded,
and the two gates use the resolved absolute GNU Make path. A shell-local shim,
alias, virtual environment, or user-writable `PATH` entry cannot substitute a
passing command. The Make target starts the runner with Python's isolated
(`-I`) mode; the runner refuses a direct non-isolated invocation.
The kernel DMI `sys_vendor` and `product_name` values are included in
`environment.json`; missing or non-G15 values refuse the run before any probe
or make command. This binds a passing G15 evidence set to the target model,
not merely to an arbitrary Ubuntu 24.04 x86-64 host.

Only variable names are reported; values are never copied into evidence. Start
from a fresh clone when reusable state is present. Do not delete or overwrite a
failed evidence folder and do not rerun a failed test to make it green.

Other process-control overrides such as `MAKEFLAGS`, `PYTEST_ADDOPTS`,
`PYTHONPATH`, `NODE_OPTIONS`, and `FRONTEND_INSTALL` are removed from the child
environment, and child `PATH` is replaced with the fixed system path. Both make
commands receive a private SO-101 home, mock backend,
empty camera configuration, and deliberately nonexistent serial-port paths as
defence in depth. Registry/proxy configuration remains available for
bootstrap; credential-like environment values are redacted if a tool prints
them. `HOME`, XDG folders, Python user-site behavior, npm user configuration
and script shell, pip configuration, and temporary directories are replaced
with private controlled locations; only explicit registry/proxy/CA variables
from a narrow allowlist survive. Raw command output is held in that private
scratch tree under the fixed system `/tmp`, never under a caller-supplied
`TMPDIR`, and is only atomically published to evidence after redaction.

## Exactly two gates

The fixed order is:

1. `make bootstrap`, once;
2. `make ci`, once, and only when bootstrap returned zero.

An operator `SIGINT`, spawn error, or nonzero exit stops the run and remains the
primary failure. The full merged stdout/stderr is retained in `bootstrap.log` or
`ci.log`; there is no retry. On success, a final read-only Git check proves the
source stayed clean at the same commit.

The evidence directory contains:

```text
environment.json
bootstrap.log
ci.log                 # only when bootstrap passed
SHA256SUMS
result.json            # atomically published last
```

`SHA256SUMS` covers every other evidence file, including the exact final
`result.json` bytes; as usual, a checksum manifest excludes itself. Verify it
from inside the evidence directory:

```bash
sha256sum --check SHA256SUMS
```

`result.json` records the schema, expected and actual commits, fixed argv,
durations, exact exit codes, log hashes, first failure, and source postflight.
Even a passed software run always records `release_ready: false` and
`hardware_touched: false`. A pass authorizes building the next verified
handoff; it does not authorize connecting or moving an arm.
