# Optional Jetson Thor batch worker

> **Current status: qualification-only.** The checked-in container does not
> install a Thor wheelhouse and cannot execute promotion benchmarks. It may
> collect compatibility facts and test binary dependency resolution. Benchmark
> execution fails closed until an actual Thor wheelhouse has a verified
> canonical manifest, the runtime reports LeRobot 0.6.1 plus complete CUDA
> truth, and that manifest digest is recorded as `locked`.

This folder is the software-only qualification pack for the optional Jetson
Thor in the SO-101 v1 plan. It measures whether a particular Thor software
stack is useful for offline training, dataset processing, synthetic inference
benchmarks, or simulation. The ASUS G15 remains the only robot host and the
only source of hardware-in-the-loop evidence.

The pack is deliberately independent from the `so101` application package. It
uses the Python standard library for its probes, validation, artifact handling,
and tests. Optional benchmark cases report `unavailable` when PyTorch or
MuJoCo is not installed; they never install a dependency at runtime.

## Hard boundary

- No follower or leader arm is connected or passed into the container.
- No USB or TTY device is mounted.
- No robot, Feetech, serial, RealSense, API-server, or control-plane extra is
  installed.
- No SO-101 credential, lease, command endpoint, or motor-bus code is present.
- Remote inference that can propose real-arm actions is out of scope and
  disabled. See [SCOPE.md](SCOPE.md) for the later gate.
- A failed or absent Thor changes nothing about G15 operation.

Read [SECURITY.md](SECURITY.md) before moving an artifact between machines.

## T0: inventory the actual Thor

Run these directly on the Thor, from a checkout of the exact source revision:

```bash
./collect-host-facts.sh > host-facts.txt
python3 scripts/runtime_probe.py \
  --source-commit "$(git rev-parse HEAD)" \
  --container-image 'HOST-NATIVE' \
  --output compatibility.json
python3 scripts/dependency_probe.py --print-command
```

`collect-host-facts.sh` does not enumerate `/dev`, USB, serial ports, network
configuration, environment variables, or credentials. `runtime_probe.py`
requires an exact 40-character source commit. Inside a container it also
requires an image reference pinned with `@sha256:`; `HOST-NATIVE` is reserved
for the initial host inventory.

The dependency command is a dry-run. Add `--run --output dependencies.json`
only when network access is intentionally available. It asks pip to resolve
binary distributions without installing anything and fails when a required
wheel is unavailable. Run it natively on Thor or inside the selected NVIDIA
image so pip can account for the PyTorch build already supplied by JetPack. A
cross-platform approximation is also available with `--platform`,
`--python-version`, and `--abi`.

## T1: build the qualification container

The Dockerfile has a deliberately unusable all-zero digest placeholder. Pick
the NVIDIA image that matches the measured Thor JetPack generation, verify its
published digest, and provide the complete reference. The build helper requires
a committed `thor-worker/` tree and bakes that exact source revision into the
image:

```bash
BASE_IMAGE='nvcr.io/nvidia/<jetpack-image>:<tag>@sha256:<64 lowercase hex>'
./scripts/build-container.sh "$BASE_IMAGE" so101-thor-worker:local
```

The image defaults to uid/gid 10001, contains no suite or robot dependencies,
and has no entry point that can control a robot. The wrappers run with the
invoking non-root uid/gid and reject extra arguments so a device mount cannot
be smuggled into the documented command:

```bash
IMAGE_ID="$(docker image inspect --format '{{.Id}}' so101-thor-worker:local)"
# Name the locally measured image with its immutable image id. A pushed image
# should use its registry repo digest instead.
IDENTITY="so101-thor-worker:local@sha256:${IMAGE_ID#sha256:}"
mkdir -p "$PWD/thor-evidence"
./scripts/probe-container.sh so101-thor-worker:local \
  "$(git rev-parse HEAD)" "$IDENTITY" "$PWD/thor-evidence"
```

That probe intentionally records `qualification.mode=qualification-only`.
`run-container.sh` rejects it. Locking requires a separate reviewed step on the
real Thor: resolve and install an immutable ARM64 wheelhouse, create and verify
its canonical artifact manifest, then rerun the probe with
the verified digest as the wrapper's fifth argument (forwarded as
`--wheelhouse-manifest-sha256`). The current build helper has no option that
pretends this evidence exists. The reviewed derived image must also bake
`THOR_WORKER_QUALIFICATION_ONLY=0` and the identical digest in
`THOR_WORKER_WHEELHOUSE_MANIFEST_SHA256`; the probe and runner reject an
argument that is not bound to those immutable image settings.

Once a derived image and compatibility document are genuinely locked, use new,
dedicated input and evidence directories for each run:

```bash
mkdir -p "$PWD/incoming" "$PWD/thor-results"
cp configs/smoke.json "$PWD/incoming/benchmark-config.json"
cp /verified/locked/compatibility.json "$PWD/incoming/compatibility.json"
./scripts/run-container.sh so101-thor-worker:locked \
  "$PWD/incoming" "$PWD/thor-results"
```

Do not place credentials or unrelated files in either directory. Both wrappers
refuse root, overlapping input/output trees, mutable image tags that do not
match the recorded digest, and replacement of existing evidence.

The benchmark wrapper mounts only `incoming` read-only and the evidence folder
read-write, drops Linux capabilities, applies `no-new-privileges`, uses a
read-only root filesystem, and disables networking. It never passes `--device`,
`--privileged`, a Docker socket, `/dev`, a TTY, or a USB path. NVIDIA's Jetson
container runtime may provide GPU compute devices according to the host JetPack
installation; this pack does not grant general device access.

## Immutable artifact round trip

Create a manifest before transfer:

```bash
python3 scripts/artifact_manifest.py create dataset-bundle \
  --kind dataset --name reach-v1 --source-commit "$(git rev-parse HEAD)"
python3 scripts/artifact_manifest.py verify dataset-bundle
sha256sum dataset-bundle/artifact-manifest.json
```

Transfer into a new staging directory. Verify there, then atomically rename
the directory into its final name on the same filesystem. Never add files to a
published bundle. A verification failure means the bundle is discarded, not
repaired in place. The verifier rejects undeclared files, changed sizes or
hashes, non-canonical manifests, symlinks, special files, traversal/absolute
paths, and lock/incomplete/partial/temp markers.

## Benchmarks

The checked-in smoke configuration is intentionally small:

```bash
python3 scripts/run_benchmarks.py \
  --config configs/smoke.json \
  --compatibility /verified/locked/compatibility.json \
  --expected-source-commit "$(git rev-parse HEAD)" \
  --expected-container-image '<verified-name>@sha256:<digest>' \
  --output benchmark-result.json
python3 scripts/validate_result.py benchmark-result.json
```

This direct command is also locked-runtime-only; the checked-in qualification
image will refuse it. The runner records the complete compatibility document, source commit,
container digest, config digest, seed, policy, precision, batch size, workload
parameters, deterministic/TF32 policy, durations, throughput, correctness
signatures, and explicit numeric comparison tolerances. The dataset case is
synthetic in the smoke config; a real case must name a verified artifact root
and its manifest SHA-256.

`benchmark_contract.compare_benchmark_results(reference, candidate)` validates
both documents, requires the same config and source commit, applies exact
dataset hashes plus the configured numeric tolerances, and reports the
configured throughput-ratio verdict for every enabled workload.

Compare the locked Thor result with a G15 result from the same commit, config,
and immutable dataset. Thor is qualified for a workload only when correctness
is within the recorded tolerances and it sustains at least 1.5x representative
throughput without thermal or memory instability. The v1 plan additionally
requires a 60-minute soak; this smoke runner does not claim that evidence.

Capacity-only promotion is not implemented. A G15 `unavailable`, OOM, or failed
result is not enough to establish equivalent correctness for a larger Thor-only
shape. That later gate needs reproducible G15 OOM evidence plus correctness on a
shared reduced shape and the larger Thor result; until then the comparator
returns unqualified when either enabled workload did not pass.

The worker runner intentionally accepts only `target=thor`; it cannot mint its
own G15 reference. The promotion comparator requires a separately produced,
complete, dependency-locked `target=g15` result. That G15 benchmark adapter is
a deferred integration hook, so the current qualification pack cannot declare
Thor promoted by itself.

## Host-only tests

No Thor, GPU, Docker daemon, network, or third-party Python package is needed:

```bash
make -C thor-worker test
```

The tests cover artifact-path and symlink attacks, incomplete bundles,
canonical manifests, configuration/result validation, container boundary
checks, dependency command construction, shell syntax, and a tiny stdlib-only
dataset benchmark.

The root backend CI now runs these host-only contracts through
`make ci-thor-worker`. See [INTEGRATION.md](INTEGRATION.md) for the active hook
and the still-optional top-level handoff convenience copy.
