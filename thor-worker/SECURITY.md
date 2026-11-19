# Security and artifact trust

Treat Thor as a batch worker, not as part of the robot trust boundary.

## Isolation

- Do not attach the SO-101 arms or RealSense to Thor for this workflow.
- Do not run the container privileged, with host networking/PID/IPC, with a
  Docker socket, or with a USB/TTY/device mount.
- Do not copy `.env`, API credentials, operator leases, SSH private keys, cloud
  tokens, or an entire SO-101 data home into an artifact.
- Use an NVIDIA base image appropriate for the measured JetPack version and
  pin it by a verified SHA-256 digest. Tags alone are mutable.
- The wrappers inspect the local Docker image, require its image id or repo
  digest to match the recorded identity, and run that immutable image id. A
  tag changing between qualification and execution fails closed.
- The build helper refuses an uncommitted worker tree and bakes its Git revision
  into the image. The runtime probe rejects a different claimed source commit.
- The checked-in image is labeled qualification-only. Promotion execution
  requires a non-placeholder wheelhouse-manifest digest, a complete Thor tuple,
  Python 3.12/3.13, LeRobot 0.6.1, and consistent CUDA device/build/capability
  evidence. The runner verifies these before importing a workload dependency.
- Run the worker through the wrapper's invoking non-root uid and keep input
  mounts read-only. The image's standalone default is uid/gid 10001.
- Output must be a new, empty, non-overlapping evidence directory. JSON
  publication uses atomic create-if-absent semantics; it never replaces a file
  that appeared after the wrapper's precheck.
- A benchmark needing network access should fetch dependencies before the
  measured run; the documented runtime has networking disabled.

The runtime wrapper has a fixed argument list and fixed security options. If a
workload needs broader access, it is outside this pack and requires review.

## Artifact rules

`scripts/artifact_manifest.py` is the canonical producer and verifier.

- Paths are NFC-normalized, relative POSIX paths. Empty components, `.`, `..`,
  backslashes, absolute paths, control characters, and duplicate paths fail.
- The bundle may contain only ordinary files and directories. Symlinks and
  special files fail whether or not the manifest names them.
- Lock, incomplete, partial, and temporary marker names fail. A bundle is not
  evidence while a producer could still be writing it.
- Every non-manifest file must be declared exactly once with exact byte size
  and a lowercase SHA-256 digest. Undeclared and missing files fail.
- The JSON bytes themselves must use the producer's canonical serialization;
  reformatting a manifest invalidates it.
- Verification opens files without following symlinks where the platform
  supports it and rechecks file identity after hashing to detect replacement.

Hash verification detects corruption and substitution relative to the supplied
manifest; it does not authenticate who supplied that manifest. Transfer the
manifest digest over an authenticated channel or sign it with an independently
managed release key. This pack intentionally does not invent a key-management
scheme.

## Publication

Receive into a new staging directory on the destination filesystem. Verify the
canonical manifest and its separately communicated digest. `fsync` the files
and staging directory when durability matters, then rename the verified staging
directory to an unused final name on the same filesystem. Never overwrite an
existing published dataset or checkpoint. A consumer opens only final names,
never staging names.

Results returning from Thor follow the same rule. Verify them on the G15 before
registration or evaluation. A checkpoint hash mismatch, incomplete marker, or
unavailable compatibility field is a failed transfer.
