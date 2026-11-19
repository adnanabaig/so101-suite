# Repository integration

The root CI hook below is implemented. The optional top-level handoff copy is
still deferred because every handoff's exact-commit `so101/` clone already
contains this qualification pack.

## Root CI hook

The root target preserves the gate's selected interpreter:

```make
.PHONY: ci-thor-worker
ci-thor-worker:
	$(MAKE) -C thor-worker test \
		PYTHON="$$($(PY) -c 'import sys; print(sys.executable)')"

ci-backend: lint coverage ci-thor-worker
```

The hosted backend job invokes the same target explicitly after coverage, and
the Make/workflow contract tests keep the local and hosted gate aligned. The
Thor tests are not optional and do not skip when CUDA, Docker or Thor is absent;
the host suite deliberately requires none of those.

ShellCheck is an additional maintainer check:

```bash
shellcheck thor-worker/collect-host-facts.sh thor-worker/scripts/*.sh
```

The built-in test target always runs `bash -n`, so CI remains reproducible when
ShellCheck is not installed.

## Optional handoff convenience hook

Every handoff's `so101/` clone already contains this folder at the exact handoff
commit. No packaging change is required for correctness. If the release layout
needs a convenient top-level optional `thor-worker/` sibling, add an
`--include-thor-worker` flag to `scripts/make_handoff.py` and, only after the
staged clone's HEAD is verified, copy:

```python
shutil.copytree(staged / "so101" / "thor-worker", staged / "thor-worker")
```

Do this before `SHA256SUMS` is generated; the existing `checksums(staged)` then
covers every copied worker file. Wire a Make variable through `make handoff`,
and add handoff tests proving the copy comes from the staged commit, contains no
symlinks or generated evidence, and is covered by `SHA256SUMS`.

This hook must package only the qualification pack. A real Thor wheelhouse,
compatibility record, or benchmark result is run evidence and stays outside the
source handoff until its canonical manifest is independently verified.
