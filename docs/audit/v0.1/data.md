# SO-101 Suite v0.1 audit: data collection, training, policy serving, evaluation

## Scope and method

- **Scope.** data/{recorder,dataset,inspection,replay}.py, training/{trainer,jobs,configs}.py, policy/{runner,server}.py, eval/evaluator.py, models.py, services/routers/{datasets,recording,training,models,eval}.py, services/jobs.py, cli.py, configs/training/*.yaml and the related tests.
- **Code version.** Audited at git HEAD 7b59eda. During the audit someone else committed the working tree as 1d3a14a, plus a docs-only commit 7b59eda. The in-scope files match what was audited. This audit changed no files and no VCS state in the repo.
- **Path conventions.**
  - Repo paths are relative to `/Users/baiadnan/Documents/so101`. The primary location of each bug is given as an absolute path.
  - `LR/` means `/tmp/so101_audit/data/lrvenv/lib/python3.12/site-packages/lerobot`, a pip-installed lerobot==0.6.1. It was used as the API reference and for the real-mode repros.
  - Repro scripts are under `/tmp/so101_audit/data/`.
- **Real-mode legs.** Run with the lrvenv: LeRobot 0.6.1, so101 `src` on PYTHONPATH, SO101_MOCK=0, every data dir under /tmp. They covered:
  - LeRobot v3 recording and resume through EpisodeRecorder;
  - real `lerobot-train` (ACT on MPS);
  - checkpoint load and `select_action` through so101's LeRobotPolicy.
- **Live API.** A mock server on 127.0.0.1:8772 was driven through record, datasets/QC, train, models and eval (score, continue, stop, SIGTERM). The server has been stopped.
- **Not run.** pytest, because it writes into the repo. No Hugging Face hub writes were made.

## 1. Bugs (most severe first)

### B1. CRITICAL: real training always fails because so101 pre-creates the output dir
- **Where:** /Users/baiadnan/Documents/so101/src/so101/training/trainer.py:76
- **Defect:** `run_training` runs `Path(job.output_dir).mkdir(parents=True, exist_ok=True)` and then launches `lerobot-train --output_dir=<same dir>`. LeRobot 0.6.1 refuses an existing output_dir unless resuming.
- **Failure:** every real job exits with code 1 seconds after launch, from both `POST /api/training/jobs` and `so101 train`. No checkpoint is ever produced. Mock mode hides this.
- **Evidence:**
  - Upstream check: LR/configs/train.py:259-263.
  - `/tmp/so101_audit/data/real_train.py service`, run with real 0.6.1, printed: `FileExistsError: Output directory /private/tmp/so101_audit/data/real/outputs/act_1790890415820556000_service already exists and resume is False` and then `RuntimeError lerobot-train exited 1`.
- **Status:** CONFIRMED
- **Fix:** move the mkdir into `_run_mock`. For real runs create only `Path(job.output_dir).parent`.

### B2. CRITICAL: real training always fails because policy.push_to_hub defaults to True with no repo_id
- **Where:** /Users/baiadnan/Documents/so101/src/so101/training/trainer.py:38-46 (base argv)
- **Defect:** the argv never passes `--policy.push_to_hub=false` or a `--policy.repo_id`. In 0.6.1 every policy config defaults to `push_to_hub=True`, and validation rejects that when no repo_id is set.
- **Failure:** with B1 fixed, every real job still dies at config validation with `'repo_id' argument missing. Please specify it to push the model to the hub.`
- **Evidence:**
  - Upstream: LR/configs/policies.py:70 and LR/configs/train.py:283-289. Reproduced with the real trainer.
  - With both problems worked around, ACT trained end to end on MPS: `/tmp/so101_audit/data/real/outputs/act_fixed`. That checkpoint loads and serves through so101.
  - tests/test_data_integration.py:105-125 only asserts that `--dataset.root` is present, so neither B1 nor B2 is caught.
- **Status:** CONFIRMED
- **Fix:** add `"--policy.push_to_hub=false"` to the base argv, and make it impossible to override (see B16).

### B3. HIGH (cross-area: app shell): the SPA fallback serves any file on disk
- **Where:** /Users/baiadnan/Documents/so101/src/so101/services/app.py:89-92
- **Defect:** `target = dist / path` uses the request path, which may be absolute or contain `..` once `%2F` is decoded. It returns `FileResponse(target)` if that is a file. There is no containment check, no auth and no Host validation.
- **Failure:**
  - Any local process can read the HF token (`~/.cache/huggingface/token`), `~/.ssh/id_*`, the repo `.env` (SO101_HF_TOKEN), and datasets.
  - So can a web page using DNS rebinding, because any Host header is accepted.
  - This becomes CRITICAL if the API is ever exposed beyond loopback without the auth proxy documented in README.md:34.
- **Evidence:** live, against 127.0.0.1:8772 with `Host: attacker.example`:
  - `GET //etc/passwd` returns /etc/passwd;
  - `GET /%2Fetc%2Fpasswd` returns /etc/passwd;
  - `GET /..%2F..%2Fpyproject.toml` returns the repo's pyproject.toml.
- **Status:** server side CONFIRMED. Browser delivery via DNS rebinding is PLAUSIBLE; it depends on the browser's local-network protections.
- **Fix:**
  - `target = (dist / path).resolve()`. Serve it only if `target.is_relative_to(dist.resolve()) and target.is_file()`; otherwise serve index.html.
  - Add `TrustedHostMiddleware(allowed_hosts=["localhost", "127.0.0.1"])`.

### B4. HIGH: web record and rollout loops run well below the requested rate, so the dataset time axis is wrong
- **Where:** /Users/baiadnan/Documents/so101/src/so101/services/robot_manager.py:1221+1242 (record/teleop) and :1282+1296 (rollout)
- **Defect:** `now` is sampled after the bus, camera or inference work. The loop then sleeps `period - (monotonic() - now)`. Each iteration therefore takes work + period instead of max(work, period).
- **Failure:**
  - Web recordings are captured at about 20-26 Hz but stamped as 30 fps. LeRobot sets timestamp = frame_index / fps, and `add_frame` forbids a caller-supplied timestamp.
  - So every demonstration is stored 13-31% faster than it happened, and a policy trained on it moves faster than it was taught.
  - Web rollouts and evals run a policy with 25 ms inference at about 13 Hz instead of 30.
- **Evidence:**
  - Upstream: LR/datasets/dataset_writer.py:209-211 and :223-227.
  - Measured with the real loop functions (`/tmp/so101_audit/data/ratecheck/rec_rate.py`, `rate2.py`), all against 30 Hz requested:

    | Loop | Extra work per iteration | Achieved |
    |---|---|---|
    | Record (web) | 0 ms bus time | 26.3 Hz |
    | Record (web) | 10 ms bus time | 20.7 Hz |
    | Rollout (web) | 0 ms inference | 24.6 Hz |
    | Rollout (web) | 25 ms inference | 13.3-13.6 Hz |
    | CLI PolicyRunner | 0 ms inference | 30.0 Hz |
    | CLI PolicyRunner | 25 ms inference | 27.0-28.5 Hz |
- **Status:** CONFIRMED
- **Fix:** use deadline scheduling as in PolicyRunner.run (runner.py:78-88): `next_t += period; time.sleep(max(0.0, next_t - time.monotonic()))`, and resync when more than one period behind.

### B5. HIGH: real mode fails open, recording into the untrainable fallback format without LeRobot's dataset extra
- **Where:** /Users/baiadnan/Documents/so101/src/so101/data/recorder.py:135-154
- **Defect:** if `lerobot.datasets` cannot be imported, `_find_lerobot_dataset` logs a warning and returns None even with SO101_MOCK=0. The recorder then writes `so101-fallback-v2`.
- **Failure:**
  - Installing `lerobot[feetech]==0.6.1` (motors only, as LeRobot's SO-101 guide does) gives a working arm and a "successful" recording session.
  - Training later refuses that data (trainer.py:48-53). The docstring promises the format is "convertible later" (recorder.py:7), but no converter exists.
  - The warning also hides the real cause. It shows the legacy-path error `No module named 'lerobot.common.datasets'` instead of "pip install 'lerobot[dataset]'".
- **Evidence:**
  - LeRobot 0.6.1 METADATA puts `datasets`, `pyarrow` and `av` only in the `dataset` extra.
  - `/tmp/so101_audit/data/failopen/partial_install.py` (lrvenv with `datasets` blocked, SO101_MOCK=0): the arm and robot stack import fine, `lerobot.datasets` raises ImportError, and the recorder saves episode 0 as `so101-fallback-v2`.
  - Also reproduced with the repo .venv (`/tmp/so101_audit/data/realfb/rec_fb.py`).
- **Status:** CONFIRMED
- **Fix:**
  - Re-raise when `not self.settings.mock`: `raise RuntimeError("LeRobot dataset support unavailable; install so101[robot]") from exc`.
  - Make capabilities.py:33 check `lerobot.datasets`, not only `find_spec("lerobot")`.

### B6. HIGH: language-conditioned policies are served with an empty task string
- **Where:** /Users/baiadnan/Documents/so101/src/so101/policy/server.py:163-169
- **Defect:** `prepare_observation(..., task=None, robot_type="so_follower")` becomes `observation["task"] = ""` (LR/policies/utils.py:135). Every training frame carried the recording task (recorder.py:359).
- **Failure:**
  - smolvla (the "fast results" preset, configs.py:37-46) and pi0 get an empty prompt at inference. That is a train/serve skew that degrades or breaks the policy. ACT and diffusion are unaffected.
  - The task cannot be supplied anywhere:
    - EvalConfig.task (evaluator.py:30) is never used;
    - RolloutRequest and StartEvalRequest (schemas.py:243-253) have no task field;
    - `so101 run` and `so101 eval` have no `--task` option.
- **Evidence:** the code path above. By contrast, LeRobot's own record loop passes the dataset's `single_task` into policy inference (LR/scripts/lerobot_record.py:483, 505).
- **Status:** CONFIRMED (code)
- **Fix:**
  - Add a `task` field to the rollout and eval requests and to the CLI. Default it from the checkpoint's training dataset (train_config.json, then the dataset's meta/tasks.parquet).
  - Pass it to `prepare_observation`.

### B7. HIGH: scored eval trials are lost on a normal server shutdown
- **Where:** /Users/baiadnan/Documents/so101/src/so101/services/routers/eval.py:157 and :177 (waits with no timeout) and :82-84 (daemon thread); src/so101/services/app.py:38-43 (lifespan)
- **Defect:**
  - The eval thread blocks forever on `_score.wait()` or `_continue.wait()`, and the scores live only in memory.
  - The lifespan stops the training runner and the robot manager but never calls the eval session's `stop()`.
  - So the daemon thread dies before `_save_report` (eval.py:191) runs.
- **Failure:** an operator nine trials into a real-robot eval restarts the server (Ctrl-C, `docker compose down`, a deploy) while the UI waits for a score. There is no report and no trace of the scored trials.
- **Evidence:** `/tmp/so101_audit/data/eval_live.py shutdown` started a 4-trial eval, scored 1 trial and sent SIGTERM to uvicorn. It printed `new eval reports after shutdown: NONE`.
- **Status:** CONFIRMED
- **Fix:**
  - Call the eval session's `stop()` in the lifespan before `get_manager().shutdown()`.
  - Journal each trial as it is scored (improvement 4).

### B8. HIGH: cross-site requests can score trials, start the next autonomous rollout, and publish a dataset
- **Where:** /Users/baiadnan/Documents/so101/src/so101/services/routers/eval.py:301-302 (`score(success: bool)`, a query param) and :310-311 (`continue_next()`, no body); src/so101/services/routers/datasets.py:165-166 (`push(repo_id, private: bool = True)`, a query param)
- **Defect:** these state-changing endpoints accept CORS "simple" requests, meaning no JSON body and no custom header. The CORS allowlist (app.py:51-52) does not block those. There is no auth, CSRF token, Origin check or operator_id.
- **Failure:** while the operator's UI is open (heartbeat alive), any page they visit can:
  - POST `/api/eval/continue`, so the arm starts the next policy rollout with no workspace-reset confirmation;
  - forge success scores;
  - POST `/api/datasets/<repo>/push?private=false`, which publishes the dataset publicly using SO101_HF_TOKEN.
- **Evidence:**
  - `/tmp/so101_audit/data/eval_live.py`, sending `Origin: http://evil.example` and `Content-Type: text/plain`, got:
    - `score(true) from evil origin, text/plain: 200`
    - `continue from evil origin, text/plain, no body: 200 -> rollout`
  - The forged success appears in `/tmp/so101_audit/data/outputs/eval_1790891620173_ef3bde3ab299.json`.
  - `push` was not exercised, to avoid hub writes.
- **Status:** server side CONFIRMED. Browser delivery is PLAUSIBLE; it depends on the browser's local-network protections.
- **Fix:**
  - Take JSON bodies with operator_id for score, continue and push, and check the operator heartbeat.
  - Add middleware that rejects non-GET requests whose `Origin` or `Sec-Fetch-Site` is not allowed.

### B9. MEDIUM: abort() after a failed LeRobot save always raises KeyError('size'), so image cleanup and finalize are skipped
- **Where:** /Users/baiadnan/Documents/so101/src/so101/data/recorder.py:513-519
- **Defect:** 0.6.1's `save_episode` pops `size` and `task` from the episode buffer before it writes parquet and video (LR/datasets/dataset_writer.py:282-283). If a later step fails, `has_pending_frames()` (LR/datasets/lerobot_dataset.py:468-472) raises KeyError. Neither `clear_episode_buffer(delete_images=True)` nor `finalize()` runs.
- **Failure:**
  - (a) The failed take's PNGs stay on disk, and the next take encodes them into its own video. A 20-frame episode got a 25-frame video whose last 5 frames come from the failed take (to_timestamp 0.833 s against length/fps 0.667 s). Training reads only the first `length` frames, so it is unaffected. Preview, replay and any duration check see the foreign frames.
  - (b) Until the next take overwrites it, an unreferenced parquet file holds the failed take's rows: 55 rows on disk against 30 in meta.
  - (c) The ParquetWriter is closed only when GC finalizes the dropped writer. A hard exit (crash, SIGKILL, power loss) in that window leaves a footerless parquet. The whole dataset, including every earlier episode, then fails to load with DatasetGenerationError, and QC returns HTTP 500 (B22).
  - (d) The operator sees only `recording.save_and_preserve_failed: 'size'` (robot_manager.py:1183-1189).
- **Evidence:** a simulated ENOSPC during video encoding.
  - `/tmp/so101_audit/data/rows/save_fail_rows.py` printed `abort raised: KeyError('size')`. `vidcheck.py` then showed `file-001.mp4 decoded frames 25` for the 20-frame episode.
  - `save_fail_gc_noB.py` followed by `rows/orphan_check.py` showed `hf rows 55 | meta frames 30`.
  - `save_fail.py` (`os._exit` before GC) left a footerless parquet; loading the dataset raised DatasetGenerationError.
- **Status:** CONFIRMED for a, b and d, and for c after a hard exit. Effect c in the long-running server is PLAUSIBLE, because the window is narrow.
- **Fix:**
  - Mirror `VideoEncodingManager.__exit__` (LR/datasets/video_utils.py:1270-1281). Wrap the pending check in `try/except KeyError`, then in a `finally` call `ds.finalize()` and `ds.writer.cleanup_interrupted_episode(ds.num_episodes)`.
  - Delete data and video files that meta does not reference.

### B10. MEDIUM: the training progress parser reads "step:1K" as step 1
- **Where:** /Users/baiadnan/Documents/so101/src/so101/training/trainer.py:31 and :143-144
- **Defect:** `_STEP_RE = r"step[:=]\s*(\d+)"`, but LeRobot logs `step:{format_big_number(steps)}` with precision 0 (LR/utils/logging_utils.py:211; LR/utils/utils.py:103-109).
- **Failure:** from step 1000 onward, `job.step` reads 1, 2, ... 100. Progress jumps from 800 back to 1 and stays at about 0.1% for a 100k-step run (jobs.py:71).
- **Evidence:** `/tmp/so101_audit/data/step_parse.py`, using LeRobot's own MetricsTracker:

  | True step | Parsed |
  |---|---|
  | 800 | 800 |
  | 1000 | 1 |
  | 1400 | 1 |
  | 20000 | 20 |
  | 100000 | 100 |
- **Status:** CONFIRMED
- **Fix:** parse the exact `N/TOTAL` from the tqdm "Training" bar on the same stream. As a fallback, scale `(\d+(?:\.\d+)?)([KMBTQ]?)`.

### B11. MEDIUM: CLI eval never resets the policy between episodes
- **Where:** /Users/baiadnan/Documents/so101/src/so101/eval/evaluator.py:66-70; src/so101/policy/runner.py:91-101
- **Defect:** `evaluate()` connects once, and that connect is the only `policy.reset()` (runner.py:54-58). `rollout()` resets only when it owns the connection, `own = not self.follower.connected`, which is false here.
- **Failure:** with ACT (n_action_steps=100), episode 2 opens by executing up to 99 stale open-loop actions planned for the previous scene. Diffusion keeps its observation queue. Scores are biased and the arm can move unexpectedly at trial start.
- **Evidence:** `/tmp/so101_audit/data/real_policy.py` with the real ACT checkpoint printed `policy queue len after 5 calls: 95 n_action_steps 100`. Nothing on this code path clears the queue before the next rollout.
- **Status:** CONFIRMED
- **Fix:** call `runner.policy.reset()` at the start of every episode in `evaluate()`, or unconditionally in `rollout()`.

### B12. MEDIUM: CLI eval has no workspace-reset gate, and all scores are lost on Ctrl-C
- **Where:** /Users/baiadnan/Documents/so101/src/so101/cli.py:222-229; src/so101/eval/evaluator.py:65-86
- **Defect:** the CLI passes only `score_fn`; `on_reset` is never used. The report is built and saved only after the last episode.
- **Failure:** as soon as the operator answers "success?", the next autonomous rollout starts with no chance to reset the scene. Ctrl-C at episode 9 of 10, or any exception, discards every scored trial.
- **Evidence:** the code path. evaluator.py:79-86 builds the report only after the loop, and cli.py:229 saves it.
- **Status:** CONFIRMED (code)
- **Fix:**
  - Pass `on_reset=lambda ep: ep and typer.confirm("Workspace reset? Start next rollout", abort=True)`.
  - Build and save a partial report in a `finally`, with an `aborted` flag.

### B13. MEDIUM: HTTP eval loads the policy inside each trial's time budget, and a load failure becomes a "0% success" report
- **Where:** /Users/baiadnan/Documents/so101/src/so101/services/robot_manager.py:1264; src/so101/services/routers/eval.py:110-115
- **Defect:** `load_policy` runs in the rollout thread on every trial. That is after `/api/eval/start` has already returned 200 and after the trial deadline has been set. The call is also outside the loop's try/finally.
- **Failure:**
  - Checkpoint load time is charged against `max_seconds` and repeated for every trial. That is seconds for ACT and tens of seconds for smolvla or pi0 on CPU.
  - A bad checkpoint becomes a latched runtime fault (robot_manager.py:1028-1030) that blocks every mode until reset.
  - The eval then saves a report with n_episodes=1 and success_rate=0.0.
- **Evidence:** the call chain:
  1. `start_rollout` runs `_start_thread`, which sets ROLLOUT and returns.
  2. `_rollout_loop` raises from `load_policy`.
  3. `_set_fault` latches the fault.
  4. The eval loop sees `mode != "rollout"` and appends `success: None` (eval.py:130-139).
  5. The report's rate is computed as 0/1.
- **Status:** CONFIRMED (code)
- **Fix:** load the policy once in `_EvalSession.start` and fail the request on error. Pass the loaded policy to `start_rollout`, and start the trial clock at the first action.

### B14. MEDIUM: eval reports are neither self-describing nor durable
- **Where:** /Users/baiadnan/Documents/so101/src/so101/services/routers/eval.py:201-211; src/so101/eval/evaluator.py:33-46 and :73
- **Defect:**
  - `n_episodes=len(per_episode)`. The requested count, terminal phase and interrupted count go only to a store event (eval.py:217-229), never into the file.
  - Interrupted trials (`success: None`) count as failures in success_rate.
  - `save()` is a plain `write_text`.
  - CLI reports number episodes from 0 and name files at one-second resolution (`eval_{int(created_at)}.json`). HTTP reports number from 1.
- **Failure:**
  - An aborted 2-of-4 run is saved exactly like a complete 2-trial eval.
  - A crash during the write leaves truncated JSON.
  - Two CLI evals that end in the same second overwrite each other.
- **Evidence:** `/tmp/so101_audit/data/eval_live.py stop` (4 trials requested, stopped after 2) produced `/tmp/so101_audit/data/outputs/eval_1790891620173_ef3bde3ab299.json`. It reads `"n_episodes": 2, "success_rate": 0.5`, with nothing showing the abort or the 4 requested.
- **Status:** CONFIRMED
- **Fix:**
  - Add `requested_episodes`, `scored`, `interrupted`, `phase` and `error` to EvalReport, and compute the rate over scored trials only.
  - Write via a temp file plus `os.replace`.
  - Use millisecond-plus-uuid names everywhere.

### B15. MEDIUM: policy serving never uses MPS
- **Where:** /Users/baiadnan/Documents/so101/src/so101/policy/server.py:92 and :133
- **Defect:** the device is chosen as `torch.device("cuda" if torch.cuda.is_available() else "cpu")`.
- **Failure:** on Apple Silicon, inference always runs on CPU even though LeRobot trains on MPS. For smolvla and pi0 that is very slow, and B4 turns the extra latency directly into a lower control rate.
- **Evidence:** the code above. The real ACT checkpoint loaded through so101 printed `loaded in 6.9s device=cpu` (`/tmp/so101_audit/data/real_policy.py`).
- **Status:** CONFIRMED
- **Fix:** choose `mps` when `torch.backends.mps.is_available()`, or reuse LeRobot's device auto-selection, and add a settings override.

### B16. MEDIUM: free-form extra_args can override safety-relevant trainer flags, and model_path is unconstrained
- **Where:** /Users/baiadnan/Documents/so101/src/so101/training/trainer.py:58-59; src/so101/services/schemas.py:209, 244, 250
- **Defect:** request `extra_args` are appended after the base argv, and draccus keeps the last duplicate. A request can therefore replace `output_dir`, `dataset.root`, `push_to_hub` or `repo_id`. Separately, `model_path` accepts any path or hub id.
- **Failure:**
  - A same-origin or local caller can write checkpoints anywhere the API user can write.
  - It can also push a trained model to an arbitrary hub repo using the machine's HF credentials.
  - A hub `model_path` makes the server download a checkpoint and instantiate whatever processor classes it names (LR/processor/pipeline.py:1073-1093).
- **Evidence:** `/tmp/so101_audit/data/dup_args.py` parsed so101's argv with draccus. With `extra_args={"output_dir": ATTACKER_DIR, "policy.push_to_hub": "true", "policy.repo_id": "attacker/exfil"}` it printed `parsed output_dir: /tmp/so101_audit/data/ATTACKER_DIR | push_to_hub: True | policy.repo_id: attacker/exfil`.
- **Status:** the override is CONFIRMED. Code instantiation from a hub `model_path` is PLAUSIBLE.
- **Fix:**
  - Allowlist `extra_args` keys. Reject `output_dir`, `job_name`, `resume`, `dataset.*`, `*push_to_hub`, `*repo_id` and `wandb.*`, and validate values.
  - Restrict `model_path` to entries returned by `models.list_models()`.

### B17. MEDIUM: training job admission and locking
- **Where:** /Users/baiadnan/Documents/so101/src/so101/services/jobs.py:62-66 (also :45-51)
- **Defect:**
  - A training job takes the dataset's exclusive record/export lock for its whole run, and reports contention as "is being recorded or exported".
  - Jobs on different datasets start immediately with no concurrency limit.
  - Job state lives only in memory.
- **Failure:**
  - A second job on the same dataset fails at once with a misleading message, even though nothing is recording.
  - Two jobs on different datasets compete for one GPU or MPS device.
  - After an API restart the job list is empty while output dirs remain.
- **Evidence:** on the live mock server, a second job on the same repo went FAILED with `dataset '...' is being recorded or exported`.
- **Status:** CONFIRMED for the message and the missing limit. OOM or slowdown from concurrent jobs is PLAUSIBLE.
- **Fix:** run real jobs through a single-worker queue with a `queued` status. Use a shared lock for training and an exclusive lock for record/export. Persist job records to disk.

### B18. MEDIUM: replay, rollout and eval ignore the dataset's fps
- **Where:** /Users/baiadnan/Documents/so101/src/so101/data/replay.py:57, 80, 84; src/so101/cli.py:153 and :202; src/so101/services/routers/eval.py:112; src/so101/services/schemas.py:245
- **Defect:**
  - Replay defaults to fps=30, and `period = 1.0 / (fps or source_fps)` falls back to the source fps only when fps is 0. It also paces with `sleep(period)` after each `send_goal`, so it runs slow (the same pattern as B4).
  - Eval hard-codes fps=30, and RolloutRequest defaults to 30.
- **Failure:** a dataset recorded at 15 fps replays at 2x speed on the real arm. A policy trained on 15 fps data executes its action chunks at 2x in rollouts and evals.
- **Evidence:** the code path. RecordingConfig accepts any fps, and LeRobot records it in info.json.
- **Status:** CONFIRMED (code)
- **Fix:** default `fps=None`. Use `source_fps` for replay, and the training dataset's fps (from the checkpoint's train_config or a run manifest) for rollout and eval.

### B19. LOW: a crash mid-episode leaves stale PNGs that leak into the next episode's video
- **Where:** /Users/baiadnan/Documents/so101/src/so101/data/recorder.py:169-181 (resume path of `_initialize_backend`)
- **Defect:** after resume the recorder never calls `writer.cleanup_interrupted_episode(num_episodes)` (LR/datasets/dataset_writer.py:669-679). LeRobot's own recorder does this on interruption (LR/datasets/video_utils.py:1279-1281).
- **Failure:**
  - After power loss, SIGKILL or a segfault mid-episode, the next episode reuses that index and encodes the dead take's extra frames into its video.
  - Training reads only the first `length` frames, but preview, replay and QC see foreign frames.
  - `so101 record` leaks the same way on Ctrl-C, because it never aborts (cli.py:113-123).
- **Evidence:** `/tmp/so101_audit/data/stale_png.py` killed a 50-frame take with `os._exit` and then recorded a 20-frame take. `stale_check.py` found 50 decoded video frames against 20 parquet rows.
- **Status:** CONFIRMED
- **Fix:** after `resume(...)`, call `self._lerobot_ds.writer.cleanup_interrupted_episode(self._lerobot_ds.num_episodes)`.

### B20. LOW: Ctrl-C on `so101 train` does not stop the trainer
- **Where:** /Users/baiadnan/Documents/so101/src/so101/training/trainer.py:115-135; src/so101/cli.py:179
- **Defect:** the trainer runs in its own session (`start_new_session=True`), so the terminal's SIGINT never reaches it. The kill watcher exists only when a `cancel_event` is passed, and the CLI passes none and has no `finally`.
- **Failure:** the CLI exits but lerobot-train keeps running through setup (dataset load, weight download) while holding the device. It dies with an uncaught BrokenPipeError at its first tqdm write, leaving no clean final checkpoint. Restarting immediately briefly runs two trainers.
- **Evidence:**
  - `/tmp/so101_audit/data/orphan_launch.py` (logging-only stand-in) printed `TRAINER STILL ALIVE after CLI exited`.
  - `/tmp/so101_audit/data/orphan2/launch.py`, with a stand-in that writes both logging and tqdm output: after Ctrl-C during setup the trainer was ALIVE 1 s later, then logged `DIED BrokenPipeError: [Errno 32] Broken pipe` once its progress bar started. tqdm's `DisableOnWriteError` swallows only errno 5.
- **Status:** CONFIRMED (with stand-ins that mimic lerobot-train's output channels)
- **Fix:** in `_run_real`, wrap the read loop in `try/finally` and, if `proc.poll() is None`, call `_terminate_process_tree(proc)`. This also covers GeneratorExit and KeyboardInterrupt.

### B21. LOW: the model registry labels every real checkpoint "unknown"
- **Where:** /Users/baiadnan/Documents/so101/src/so101/models.py:45
- **Defect:** it reads `meta.get("policy_type", "unknown")`, but LeRobot's config.json stores the kind under `"type"`. Only so101's mock checkpoint writes `policy_type` (trainer.py:177).
- **Failure:** the UI and CLI model lists cannot tell ACT, diffusion and smolvla checkpoints apart.
- **Evidence:** the real checkpoint `/tmp/so101_audit/data/real/outputs/act_fixed/checkpoints/last/pretrained_model/config.json` has `"type": "act"`, yet `list_models()` returned `'policy_type': 'unknown'`.
- **Status:** CONFIRMED
- **Fix:** `policy_type=meta.get("type") or meta.get("policy_type", "unknown")`.

### B22. LOW: QC returns HTTP 500 on a corrupt LeRobot dataset
- **Where:** /Users/baiadnan/Documents/so101/src/so101/data/inspection.py:950; src/so101/services/routers/datasets.py:136-139
- **Defect:** `DatasetGenerationError` (a subclass of `DatasetsError`, which derives from `Exception`) is neither caught by QC's `(RuntimeError, ValueError, OSError, KeyError)` nor mapped by the router.
- **Failure:** on exactly the datasets where QC matters, such as B9's footerless parquet, the endpoint returns 500 instead of a failing report.
- **Evidence:** `/tmp/so101_audit/data/qc_check.py` on audit/savefail printed `QC raised: datasets.exceptions.DatasetGenerationError`.
- **Status:** CONFIRMED
- **Fix:** catch `Exception` per episode and record it as a failed check. Map anything unexpected to 409.

### B23. LOW: training requests are not validated
- **Where:** /Users/baiadnan/Documents/so101/src/so101/services/schemas.py:204-209; src/so101/training/trainer.py:155-178
- **Defect:** `steps` and `batch_size` have no bounds, and the mock trainer never checks that the dataset exists.
- **Failure:**
  - `steps=-5` makes the job fail with `cannot access local variable 'loss' where it is not associated with a value`.
  - `batch_size=-3` is accepted and passed through.
  - A mock job on a nonexistent repo "succeeds" and registers a checkpoint.
- **Evidence:** each case reproduced against the live mock server via `POST /api/training/jobs`.
- **Status:** CONFIRMED
- **Fix:** `steps: int = Field(100_000, ge=1, le=10_000_000)` and `batch_size: int = Field(8, ge=1, le=4096)`. Reject unknown repo_ids in mock mode.

### B24. LOW: the pi0 preset trains from random weights, and the YAML presets are never loaded
- **Where:** /Users/baiadnan/Documents/so101/src/so101/training/configs.py:47-54; configs/training/{act,diffusion,smolvla}.yaml
- **Defect:** the pi0 preset says to fine-tune from a pretrained checkpoint but passes no pretrained path. 0.6.1 therefore builds PaliGemma from config with random weights (LR/policies/pi0/modeling_pi0.py:240). Nothing references the YAML files.
- **Failure:** a pi0 job spends GPU hours training a VLA of about 3B parameters from scratch on a few dozen episodes. Edits to the YAML files have no effect.
- **Evidence:** the code. LeRobot maps pi0 to `lerobot/pi0_base` (LR/policies/pretrained.py:351).
- **Status:** CONFIRMED (code)
- **Fix:** `extra_args={"policy.pretrained_path": "lerobot/pi0_base"}`, matching the smolvla preset. Either load the YAML files or delete them.

### B25. LOW: route order shadows datasets whose repo_id ends in /qc
- **Where:** /Users/baiadnan/Documents/so101/src/so101/services/routers/datasets.py:132 vs :143
- **Defect:** `/{repo_id:path}/qc` is declared before `/{repo_id:path}`, so `GET /api/datasets/lab/qc` runs QC on `lab` instead of returning dataset `lab/qc`.
- **Failure:** such datasets cannot be opened through the detail endpoint.
- **Evidence:** the declaration order; FastAPI uses the first matching route.
- **Status:** CONFIRMED (code)
- **Fix:** have the repo_id validator reject names whose final segment is a sub-resource name: `qc`, `trash`, `push` or `episodes`.

### B26. LOW: tqdm carriage-return refreshes flood the job log tail
- **Where:** /Users/baiadnan/Documents/so101/src/so101/training/trainer.py:115-120 and :139-147; src/so101/services/jobs.py:73-75
- **Defect:** `Popen(text=True)` uses universal newlines, so every tqdm `\r` refresh becomes its own line, and every line is appended to the 200-line `log_tail`.
- **Failure:** within seconds of the start of training, the tail holds only progress-bar frames, and the useful loss lines and early warnings are pushed out.
- **Evidence:** the code, plus LR/scripts/lerobot_train.py:589-596 (a tqdm bar on stderr, which is merged into stdout).
- **Status:** CONFIRMED (code)
- **Fix:** route lines that start with `Training:` into a single progress field instead of `log_tail`.

### B27. LOW: LeRobot's hub fallback can download into a local dataset directory
- **Where:** /Users/baiadnan/Documents/so101/src/so101/data/replay.py:36 (also reached from QC, inspection.py:949)
- **Defect:** when the local load fails (a missing file or an out-of-range episode), `LeRobotDataset(repo_id, root=root, episodes=[episode])` falls back to `get_safe_version` plus `snapshot_download(local_dir=root)` (LR/datasets/lerobot_dataset.py:271-279 and :643-673).
- **Failure:** for a dataset that was pushed with so101's push, a damaged local copy or a bad episode index during replay or QC pulls hub files into the local root. That can overwrite newer local metadata. For local-only names it becomes a slow network error.
- **Evidence:** the code path only; not exercised, because the audit made no hub access.
- **Status:** PLAUSIBLE
- **Fix:** check the episode index against meta first, and run local reads with `HF_HUB_OFFLINE=1`, or refuse when the root already exists.

### B28. LOW: CLI record and train rough edges
- **Where:** /Users/baiadnan/Documents/so101/src/so101/cli.py:113-124 (record) and :173 (train)
- **Defect:**
  - `record` starts the next episode immediately after saving, with no reset window. It has no try/finally, so there is no abort and `cams.close()` is skipped on errors.
  - `train` names output dirs at one-second resolution.
- **Failure:**
  - Each CLI episode begins by recording the operator's scene-reset motion.
  - Ctrl-C leaves stale PNGs behind (B19).
  - Two `so101 train` runs of the same policy started in the same second share a dir. Even with B1 fixed, the second one then hits lerobot's FileExistsError.
- **Evidence:** the code.
- **Status:** CONFIRMED (code)
- **Fix:**
  - Add `--reset-seconds` with a prompt between episodes, and wrap the loop in try/finally that calls `rec.abort()` and `cams.close()`.
  - Name output dirs with `time.time_ns()` plus an id, as jobs.py:45 does.

## 2. Improvements (ranked by value/effort)

1. **(S) Real-mode argv contract test.**
   - Parse the output of `build_train_command()` with draccus into 0.6.1's TrainPipelineConfig, then call `validate()` with no existing output dir.
   - Also assert that every preset's policy type is registered in LeRobot.
   - This needs no GPU and would have caught B1, B2 and B16 in CI.
2. **(S) One shared fixed-rate scheduler plus measured-rate QC.**
   - Replace the hand-written web record, rollout and replay loops with PolicyRunner's deadline scheduler.
   - Store the achieved rate per episode. The recorder already tracks first and last `step.t`.
   - Make QC fail episodes whose real duration differs from length/fps by more than a few percent.
   - This fixes B4 and makes any regression visible.
3. **(M) A run manifest beside each checkpoint.**
   - Write `so101_run.json` at job launch: dataset repo_id and root, fps, task strings, camera feature names, preset and git sha.
   - Read it in `load_policy`, rollout and eval.
   - It supplies the task for VLA serving (B6), the fps for rollout and replay (B18) and the registry type (B21). It also gives a camera-name contract check before the arm moves.
4. **(S) A durable eval journal.**
   - Append each trial to `eval_<session>.jsonl` with fsync as soon as it is scored.
   - Write the final report atomically with requested, scored and interrupted counts and the phase.
   - Stop the session in the lifespan.
   - This fixes B7 and B14 and makes interrupted evals resumable.
5. **(M) A persistent single-worker job queue.**
   - Keep job records on disk (status, pid, pgid, output_dir).
   - Run one real job per device and reconcile dead pids at startup.
   - Use a reader/writer dataset lock so training is not reported as "recording".
   - This fixes B17.
6. **(M) One LeRobot writer per recording session.**
   - Today the web path creates or resumes the dataset and finalizes it on every episode (recorder.py:400-402, with the recorder dropped after each save).
   - That writes one parquet file and one mp4 per episode, defeats v3 file packing, and multiplies the failure windows behind B9 and B19.
   - Keep the writer open for the session and finalize on stop, with VideoEncodingManager-style cleanup on error.
7. **(S) Metadata-level QC for LeRobot datasets.**
   - LeRobot datasets currently skip the timestamp and camera checks that fallback data gets.
   - Add cheap checks from meta/episodes:
     - `to_timestamp - from_timestamp` against length/fps for each camera;
     - contiguous `dataset_from_index`/`dataset_to_index`;
     - parquet row count against meta frames;
     - leftover files under images/.
   - Catch `DatasetsError`, and read v3 episode metadata (inspection.py:323-346 does not).
   - These checks would flag the damage from B9 and B19.
8. **(M) A golden real-mode smoke test on CPU.**
   - With the robot and train extras: record 2 synthetic episodes through EpisodeRecorder, run 10 ACT steps via `run_training`, then `load_policy` and `select_action`.
   - It takes about a minute and catches the LeRobot API drift that mock mode hides.

## 3. What is good and must be preserved

- **Real mode refuses mock artifacts.**
  - `load_policy` rejects checkpoints marked mock (server.py:208-215).
  - The trainer rejects fallback datasets (trainer.py:48-53).
  - `run_training` raises instead of fabricating a checkpoint when lerobot-train is missing (trainer.py:82-86).
- **The recorder uses the 0.6.1 API correctly** (verified against the source and by real recording and resume):
  - `create`/`resume` with robot_type and image_writer_threads;
  - validation of resumed features;
  - the frame schema: task included, timestamp and frame_index omitted as 0.6.1 requires, HWC uint8 images;
  - episode index taken as num_episodes-1;
  - finalize after each episode, so parquet footers get written;
  - `desired_action` kept out of the policy features.
- **Policy loading is correct and works on a real checkpoint.**
  - It uses `from_pretrained`, `make_pre_post_processors` and `prepare_observation` with the saved normalizer stats.
  - An ACT checkpoint trained by real 0.6.1 loads and returns named joint actions.
- **The fallback writer is durable:** atomic JSON, a staging dir plus `os.replace`, and INCOMPLETE.json markers on abort.
- **Dataset routes validate input.** The repo_id regex and the trash/restore validation rejected every path-traversal probe sent over HTTP.
- **One per-dataset operation lock** is shared by recording, export and training.
- **The training lifecycle is solid.**
  - A watcher thread kills the process group with SIGTERM, then SIGKILL, and works even when the trainer stops printing.
  - `JobRunner.shutdown` cancels and joins every job.
  - Job dirs are unique (time_ns plus id).
- **The CLI deadline schedulers** (`TeleopController.run`, `PolicyRunner.run`) are the right pattern to copy into the web loops.
- **The HTTP eval flow is sound.**
  - It has explicit score and workspace-reset gates.
  - It clears the signal before publishing a phase (eval.py:152-156).
  - Report names cannot collide (milliseconds plus session id).
- **Small things that are right:** the model registry glob does not double-count the `last` symlink, and QC is read-only.
- **The mock loop works end to end over HTTP.** Record, dataset list/detail/QC, mock train, models, then eval with score, continue and stop all ran, and the report was saved, entirely inside the SO101_* scratch dirs.
