# Scope and promotion gates

## T0/T1 implemented here

This pack may:

- record the Thor SKU, JetPack/L4T, kernel, architecture, Python, CUDA, cuDNN,
  TensorRT, PyTorch, MuJoCo, NumPy, and LeRobot versions;
- test dependency resolution without installing packages;
- validate and hash immutable offline datasets, checkpoints, and benchmark
  results;
- run offline synthetic PyTorch, dataset-I/O, inference, and MuJoCo benchmarks;
- write result files in a documented JSON format; and
- run in an unprivileged, device-free qualification container.

The checked-in Dockerfile is qualification-only. Running the benchmark stubs is
supported only from a later derived image whose exact Thor wheelhouse manifest
is recorded in a complete locked compatibility document. A command-line flag
or image tag is not evidence of that lock.

The implemented promotion comparison covers shared workloads that pass on both
machines. It does not infer a capacity win from a G15 OOM or unavailable result;
Thor-only model-size qualification remains a later evidence design.

The G15 workflow must install and work when this entire folder and the Thor are
absent. Thor results are supplementary evidence and cannot replace G15 doctor,
preflight, stop, or HIL evidence.

## Explicitly prohibited

The Thor worker must not:

- open, discover, own, proxy, or receive an SO-101 motor bus;
- connect to a follower, leader, serial adapter, RealSense, or other robot USB
  device;
- hold an operator lease or a control-plane credential;
- call an arm command, stop, reset, arming, preflight, or HIL endpoint;
- run the suite API/server or import its robot manager;
- turn a benchmark output into a real-arm action; or
- become required for a stop, shutdown, data capture, or local G15 rollout.

## Future remote-inference gate (not implemented)

Remote inference stays disabled for v1 hardware acceptance. A separate design,
review, fault-injection suite, and operator-approved HIL gate are required
before even bounded action proposals can leave Thor. At minimum that future
gate must prove:

1. authenticated observation, model, dataset, configuration, and command-epoch
   identity;
2. G15-local monotonic issue and expiry deadlines independent of Thor clocks;
3. finite, schema-valid, joint-limited proposals with no direct bus access;
4. at-most-once handling and zero execution of stale, duplicate, reordered, or
   previous-epoch proposals;
5. a G15-local Halt on late, invalid, disconnected, or unhealthy inference;
6. stop latency and behavior that are independent of network and Thor health;
7. bounded queue depth, cancellation, reconnect, overload, and reboot behavior;
8. an operator-visible mode boundary and explicit opt-in; and
9. real-arm Halt/disconnect measurements with the physical cutoff attended.

Until all of those pass, the protocol does not exist in this repository. The
benchmark named `inference` is an in-process synthetic compute workload only.
