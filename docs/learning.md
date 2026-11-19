# Learning: data, training and evaluation

This page covers the learning half of the suite: which episodes a policy
learns from, the trainers, the numpy behaviour-cloning baseline and its
checkpoint format, and how a trained policy runs and is evaluated. Recording
is covered in [getting started](getting-started.md) and the physics simulator
in [simulation](simulation.md).

## Trainers

| Preset | Trainer | Needs | Notes |
| --- | --- | --- | --- |
| `act` (default), `diffusion`, `smolvla`, `pi0` | `lerobot-train` subprocess (`python -m lerobot.scripts.lerobot_train` with the suite's Python, so PATH doesn't matter) | the `[train]` extra (torch, accelerate) | one job at a time |
| `bc_baseline` | numpy, in-process | the core install | CPU, seconds; a baseline, not a manipulation policy |

`GET /api/training/presets` lists every preset with its default steps and
batch size, its `trainer` and whether it `requires_torch`.

Without `lerobot-train`, the LeRobot presets behave differently by mode. In
mock mode (`SO101_MOCK=1`) they run a deterministic simulated loss curve, so
the UI works end to end, and the result is a mock checkpoint that is not a
usable policy. With `SO101_MOCK=0` they refuse to start. `bc_baseline` trains
a real policy in every mode.

Every run writes `so101_run.json` (job, preset, dataset, episode selection,
task, fps, trainer) into its output directory. The model registry, `so101 run`
and `so101 eval` read the task and frame rate from it, so a policy runs with
the instruction and control rate it was trained on unless you override them. The file is flushed to
disk before it replaces an older one. If it exists but can't be read (for
example, cut short by a power failure), `so101 run`, `so101 eval` and the
rollout API refuse the checkpoint rather than guess the task and rate, and the
model registry lists it as unreadable. Restore the file, or delete it and pass
`--task` and `--fps` yourself.

`GET /api/models` also reports `needs_task`. It comes from both the known VLA
policy types and the checkpoint's saved tokenizer preprocessor, so the UI can
offer a task for custom language policies without relying on its older-server
fallback type list.

### Job records and restarts

The server keeps one record per training job in `<outputs>/.jobs/<id>.json`,
so the job list survives a restart. Records are flushed to disk (file and
folder) before they replace the previous version, so a power cut leaves either
the old record or the new one. A record that still can't be read is listed
with status `corrupt`: its outcome is unknown, and the entry keeps the output
directory (and the dataset, from the run's `so101_run.json`) when they can be
recovered. The damaged file stays in place; delete it to remove the entry.

A job that was still queued or running when the server stopped is listed as
`interrupted` after the restart, and is never restarted automatically.
`lerobot-train` runs in its own session, so it can outlive a server that was
killed or crashed. The job record carries the trainer's pid, process group and
start time from the moment it starts. When the server starts again, it checks
that this pid is still that trainer: same process group and start time, and a
command line that runs the training module for that job's output folder. If
it is, the server stops the process group (SIGTERM, then SIGKILL after 10 s).
The job shows as running until the trainer is gone, and a new LeRobot job
waits (`queued`) until then, so two trainers never share the GPU. A pid that
now belongs to another process is never signalled; the job's error says so.
On Windows the server can't check, so it leaves any process alone.

Closing the terminal the suite runs in (the Linux app-menu launcher opens
one) stops the server the same way Ctrl+C does: training jobs are cancelled
and the arm collapses to its rest pose before the process exits.

## Which episodes a policy learns from

Each recorded episode carries an outcome label: `unlabeled` (new
recordings), `success`, `failure`, `partial` or `discarded`. Label episodes in
the Datasets page or with `POST /api/datasets/{repo_id}/episodes/{n}/outcome`,
in both dataset formats. A human label always wins over an automatic one (the
simulator labels its own episodes from the task predicate), and each label
keeps its source and the labels it replaced.

Where the label is stored depends on the format:

- SO-101 fallback datasets keep it in each episode's `episode.json`.
- LeRobot datasets (real recordings, and `so101 sim demos` when LeRobot is
  installed) keep it in `so101_outcomes.json` in the dataset folder, because
  LeRobot's own metadata has no outcome field. The file is written atomically.
  If it cannot be read, every episode reads as unlabeled, so `kept` and
  `success` are refused until it is repaired or restored, and no label is
  written over it. It travels with the folder (trash, restore, Hub push), but
  LeRobot's own tools do not know it: re-indexing episodes with them leaves
  the labels on the old indexes.

Every preset selects episodes with `so101 train --episodes` (the API's
`episodes`):

| Selection | Learns from | Baseline quality checks |
| --- | --- | --- |
| `kept` (default) | `success` and `unlabeled` | skips episodes with out-of-limit commands or non-increasing timestamps |
| `success` | `success` only | same as `kept` |
| `all` | every outcome, failures included | skips only unusable data |

The baseline also runs the quality checks as it reads the data. It skips
unreadable or non-finite state, action and environment-state frames in every
mode. With `all`, missing or non-finite timestamps are reconstructed from the
dataset fps, while finite duplicate or backwards timestamps are kept; strict
selections skip those episodes. Its training log and checkpoint manifest list
each skipped episode with its reason.

The LeRobot presets select by label alone. The suite passes the selected
episodes to `lerobot-train` as `--dataset.episodes=[0,2,5]`, after any
override, so an override can't change them. The flag is left out when every
episode is kept, as with `all`. `lerobot-train` reads the frames itself, so the
baseline's quality checks don't apply. If the labels can't be read, or the
selection keeps no episode, the job is refused before `lerobot-train` starts.
The API answers 422 at submit, and a queued job whose labels changed fails
with the same message. A Hugging Face Hub dataset (`--hub-dataset`) has no
labels here, so it trains only with `--episodes all`. The run's
`so101_run.json` records the choice under `episode_selection`: the selection,
the episodes used, and each skipped episode with its outcome. Mock training
selects the same way.

The baseline reads SO-101 fallback datasets (`meta.json` plus one
`frames.jsonl` per episode) with the core install. It reads LeRobot datasets
through the `lerobot` package (the `[robot]` or `[train]` extra); without it
the job is refused before it starts. Images are not loaded: the baseline uses
joint state and, for simulator data, `observation.environment_state`.

## The numpy baseline

`bc_baseline` is a small multilayer perceptron trained by behaviour cloning.
Its input is the follower's joint state (`observation.state`, six values) and,
when the dataset has it, the simulator's object state
(`observation.environment_state`). Its output is the next absolute joint
target. By default it learns the change from the current state (`target
delta`), which is easier to fit than the absolute target.

Training is plain and reproducible:

- Episodes, not frames, are split into training and held-out sets (20% held
  out, at least one episode when there are two or more). Normalization
  statistics come from the training episodes only.
- AdamW with cosine learning-rate decay to 10% of `lr`, on mean squared error.
- Every `eval_every` steps the held-out loss is measured. The best weights are
  kept, and training stops after `patience` evaluations without improvement.
- Every random draw comes from the seed, so the same data, settings and seed
  give byte-identical weights on the same machine and numpy build.

### Privileged and state-only baselines

`observation.environment_state` holds true object positions that only the
simulator knows. A baseline trained with it is a **privileged-state sim
baseline**, and that label follows it into the training log, the model
registry, rollouts and eval reports. It is refused, with a message naming
the missing inputs, on every backend except `sim`:

- `so101 run` and `so101 eval` refuse it before the arm moves;
- `POST /api/models/rollout/start` answers 422;
- loading it through the policy server on a mock or real backend fails;
- if it is somehow run without object state, it fails at its first step.

Train with `--state-only` to ignore the object state. The result, a
**state-only BC baseline**, can drive the mock or real arm. It sees only the
joints, so it has no way to know where an object is. Treat it as a smoke test
of the record, train and run loop, not as a manipulation policy.

### Training it

```bash
so101 train you/reach --policy bc_baseline                  # privileged if the data has object state
so101 train you/reach --policy bc_baseline --state-only     # joints only; can drive the arm
so101 train you/reach --policy bc_baseline --episodes all --seed 3 --steps 5000
so101 train you/reach --policy bc_baseline --arg hidden=256,256 --arg history=2
```

Steps and batch size default to the preset's (3000 and 256). Other settings
are `--arg KEY=VALUE` overrides, also accepted as the API's `extra_args`:

| Key | Default | Meaning |
| --- | --- | --- |
| `seed` | 0 | random seed (`--seed`) |
| `episodes` | `kept` | `kept`, `success` or `all` (`--episodes`) |
| `state_only` | false | ignore `observation.environment_state` (`--state-only`) |
| `hidden` | 128,128 | hidden layer sizes, 1 to 4 layers of up to 1024 |
| `activation` | relu | `relu` or `tanh` |
| `lr` | 1e-3 | learning rate |
| `weight_decay` | 1e-4 | decoupled weight decay on weight matrices |
| `val_fraction` | 0.2 | share of episodes held out, 0 to 0.5 |
| `patience` | 10 | held-out evaluations without improvement before stopping |
| `eval_every` | 100 | steps between held-out evaluations |
| `history` | 1 | states the policy sees; 2 or more adds frame-to-frame differences |
| `time_feature` | none | `time` adds seconds since the episode started |
| `target` | delta | `delta` (action minus state) or `absolute` |

An unknown key, or a flag that conflicts with an `--arg`, is refused before
anything is written.

The API trains the same way. `POST /api/training/jobs` with

```json
{"repo_id": "you/reach", "policy": "bc_baseline", "extra_args": {"state_only": "true"}}
```

starts a job; an omitted `steps` or `batch_size` takes the preset's default,
and an omitted `episodes` means `kept`.
The job reports progress and loss like a LeRobot job, and its log carries the
`[bc]` summary lines: the data used, the model, each held-out evaluation, the
held-out action error next to two reference predictors (the mean action and
holding position), and the checkpoint path.

The output goes to `outputs/bc_baseline_<time>_<job>/` (or `--output-dir`),
and the checkpoint to `checkpoints/last/pretrained_model/` inside it. The
registry lists it as `<run>@last` with its label (`so101 models ls`,
`GET /api/models`).

To try it without recording, write the toy reach demonstrations as a dataset:

```bash
python scripts/bc_baseline_measure.py --write-dataset you/toy_reach
so101 train you/toy_reach --policy bc_baseline
```

### Checkpoint format

`checkpoints/last/pretrained_model/` holds three files:

`policy_manifest.json`
: `type` `numpy_bc_mlp` and format `version` 1; `label` and
  `privileged_state`; `features` (input names in order, environment-state
  names, history, time feature, target, fps, units); `normalization` (input
  and output mean and standard deviation); `network` (layer sizes,
  activation); `weights` (file name, SHA-256, dtype); `action_limits`; and
  metadata: `task`, `dataset` (repo id, format, fps, the episodes used and
  held out, skipped episodes with reasons), `training` (the full
  configuration, numpy and so101 versions) and `metrics` (held-out loss and
  action error per joint, the reference predictors, best step, wall time).

`weights.npz`
: float32 matrices `W0..Wn` and biases `b0..bn`, written without pickles
  and with fixed archive timestamps, so equal weights give equal bytes.

`train_log.jsonl`
: one line per held-out evaluation: step, learning rate, training and
  held-out loss, elapsed seconds.

Loading verifies everything before a policy is returned: the manifest type
and version (a newer version asks you to upgrade), the weights hash, the
array names, shapes and dtypes, and that `privileged_state` agrees with the
inputs. A damaged checkpoint fails closed, in mock mode too.

### Running it

The baseline follows the LeRobot-style policy protocol:

```python
from so101.policy.baseline import is_baseline_checkpoint, load_baseline_policy

policy = load_baseline_policy(path, backend="sim")  # refuses "mock" and "real" if privileged
policy.reset()                                      # at every episode start
action = policy.select_action({
    "observation.state": state,                     # float32 (6,), JOINT_NAMES order
    "observation.environment_state": env,           # privileged baselines only
})                                                  # float32 (6,), absolute targets
```

Actions are degrees for the arm joints and 0 (closed) to 100 (open) for the
gripper, clipped to the joint limits. `backend=None` skips the backend check;
it is meant for the simulator's in-process evaluator, where a privileged policy
still needs `observation.environment_state` at every step. The policy counts
steps since `reset()`, so run it at the dataset's frame rate (the CLI and the
rollout API do so by default).

`so101.policy.server.load_policy` serves baselines to `so101 run`,
`so101 eval` and the rollout API. It passes the active backend and wraps the
policy in `JointDictPolicy`, which turns the action vector into the
`{joint: value}` form the control loops use and rejects a wrong-sized or
non-finite action. The observation includes `observation.environment_state`
whenever the follower's class provides an `environment_state()` method. The
real arm has none, so the real backend never sees object state.

For the simulator's batch evaluator, the baseline registers through
`so101.simulation.policy.register_policy_loader(predicate, loader)`:

```python
from functools import partial

register_policy_loader(is_baseline_checkpoint, partial(load_baseline_policy, backend="sim"))
```

Point it at the `pretrained_model` directory. Policy preflight compares the
checkpoint's `spec.env_state_names` with the active simulation task's
`env_state_names` and refuses a mismatch before loading the policy, so a reach
baseline is never scored on a pick task.

## Measured on the toy reach task

`so101.training.toy_reach` is a numpy-only kinematic stand-in for the
simulator's reach task, used by the tests and for these measurements. A
two-link arm turned by the pan joint follows each command with a lag. A
scripted expert reaches a random target 16 to 26 cm from the base, and
success means the gripper point stays within 2 cm of the target for 0.3 s.
It is not a physics simulation, and these numbers are not simulator or
real-arm success rates. They show that the trainer, the checkpoint and the
policy work together.

Setup: 60 demonstrations, the default 3000 steps with a 128x128 network,
training seeds 0, 1 and 2, and 100 closed-loop trials per trained policy on
targets the training data never saw (2026-10-02, Apple silicon laptop CPU).

| Variant | Episodes | Train s | Held-out MAE | Successes per seed | Pooled rate | Wilson 95% |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| scripted expert (upper bound) | | | | 100 | 100% | 96.3 to 100% |
| hold position (lower bound) | | | | 0 | 0% | 0 to 3.7% |
| privileged (default) | 60 | 0.58 | 0.376 | 95, 85, 83 | 87.7% | 83.5 to 90.9% |
| privileged, absolute target | 60 | 0.37 | 0.695 | 78, 85, 85 | 82.7% | 78.0 to 86.5% |
| privileged, history 3 | 60 | 0.41 | 0.325 | 62, 40, 51 | 51.0% | 45.4 to 56.6% |
| privileged, time feature | 60 | 0.52 | 0.364 | 89, 93, 86 | 89.3% | 85.3 to 92.3% |
| state-only | 60 | 0.62 | 0.536 | 2, 1, 2 | 1.7% | 0.7 to 3.8% |
| privileged, 30% distracted data, `kept` | 44 | 0.48 | 0.439 | 90, 78, 90 | 86.0% | 81.6 to 89.5% |
| privileged, 30% distracted data, `all` | 60 | 0.38 | 0.640 | 26, 37, 30 | 31.0% | 26.0 to 36.4% |

MAE is the mean absolute action error on held-out episodes in degrees
(gripper 0 to 100), averaged over seeds. On the default's held-out episodes,
predicting the mean action scores 8.092 and holding position 0.853. The Wilson
interval pools the three seeds' trials as if they were independent, so it
understates seed-to-seed variation: the default's seeds alone range from 83
to 95 successes. Training times vary by about 10% between runs.

What the table shows:

- **Outcome labels matter most.** In the distracted rows, 16 of 60 expert
  demonstrations reach for the wrong spot and are labeled `failure`. The
  default `kept` selection drops them and keeps 86% success; learning from
  all of them drops it to 31%.
- **Open-loop error does not predict closed-loop success.** Three frames of
  history give the lowest held-out MAE and the worst privileged success. The
  likely cause: the frame-to-frame differences let the network continue the
  last motion, which scores well offline and drifts once its own actions feed
  back.
- **Predicting the change from the current state** halves the held-out error
  of predicting the absolute target, with a smaller closed-loop gain (87.7%
  against 82.7%).
- **The time feature does not help significantly** (89.3% against 87.7%,
  overlapping intervals) and ties the policy to the episode clock, so it is off
  by default.
- **Joints alone cannot find the target.** The state-only baseline is close to
  the hold-position bound. Its job is to exercise the arm path, not to
  succeed.

Reproduce the table (about 18 s) or run a 1 s smoke version:

```bash
python scripts/bc_baseline_measure.py --seeds 0,1,2
python scripts/bc_baseline_measure.py --quick
```

## Evaluation

`so101 eval <model> --episodes N` runs mock/simulation rollouts; the Evaluation
page/API is also the supported real-hardware path. They ask you to score each
rollout. Before every rollout after the first they require
confirmation that the workspace is reset. Each scored trial is appended to a
journal as it happens, and an interrupted evaluation still saves a partial
report. The API accepts only a checkpoint listed by `GET /api/models`; the CLI
also accepts an explicit local checkpoint path. Both default to the
checkpoint's task and control rate, load the policy once, and write the same
report model. The report records the success rate with its Wilson 95% interval,
the backend it ran on (`real`, `sim` or `mock`), the policy implementation's
label (for example "privileged-state sim baseline"), its task and its control
rate. Real standalone CLI motion is refused because it cannot retain the
server's watchdog and timed stop lifecycle. An explicit or checkpoint-derived
control rate must be a whole 1-240 Hz;
invalid metadata is refused rather than truncated or replaced with 30 Hz.
Results from `sim` or `mock` are reported as such, never as real-robot success.
In mock mode the report label names the deterministic `MockPolicy` stand-in,
not the checkpoint whose weights were deliberately not loaded. An API eval
also refuses before motion if its initial journal or audit event cannot be
made durable; loss of a later trial journal ends the eval before another
trial and keeps that trial in the terminal JSON report with the error. A CLI
report-publication failure is itself written to the journal as `phase=failed`,
never as a successful end with a null report.

## Limitations

- The baseline is a single-step MLP: no images, no action chunking, no
  language conditioning. It is a sanity check and a simulation baseline.
- Privileged baselines run only in simulation. State-only baselines can move
  the arm but have no way to perceive objects.
- Inference counts control steps, so a policy run at a different rate than
  its data sees time and velocities scaled.
- Byte-identical weights hold for one machine and numpy build; a different
  BLAS may change the last bits.
