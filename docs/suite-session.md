# Run the Suite while you use it

From the G15 terminal:

```bash
/home/adnan/so101-suite/so101
```

This starts the Suite, opens Operate, and stays in that terminal until shutdown.
It does not arm motors or reset safety latches. Only one foreground session can
own the services at a time. The existing `up`, `down`, `restart`, and `status`
commands remain available; `so101 session --open` explicitly selects this mode.
Use the full checkout path to avoid the older installation on PATH.

## First Ctrl+C: Halt, then safe shutdown

The launcher immediately requests the motor manager's existing Halt: motion
stops and powered motors hold. The manager's existing hold timer, supervision
watchdog and bounded Collapse to the saved rest pose remain in charge. The
launcher does not extend that timer, command a new pose or disable torque itself.
The terminal does not send a supervision heartbeat: the normal unsupervised
hold is about five seconds, then the existing rest descent begins. This is a
bounded hold rather than an indefinite freeze.
The motor service stays alive through this sequence. Once it reports confirmed
torque off, the launcher stops cameras, recording, simulation/visualization
publishers and the motor API. The recorder gets its normal close path.

## Second Ctrl+C: E-stop and forced worker shutdown

During shutdown, a second Ctrl+C sends E-stop through its separate existing API
path, even if the earlier Halt request is still blocked. E-stop cuts torque and
can let the arm drop. Camera and recording workers are force-stopped immediately;
the current MCAP may remain incomplete. The motor server is killed only after
fresh torque-off confirmation and an E-stop acknowledgement/latch. Killing a
server before it can send torque-off is not a hardware emergency stop.

If E-stop cannot confirm torque off, use the physical cutoff. The launcher
reports the uncertainty and leaves the motor API alive to finish the stop.
A stopped or unreachable process is never treated as proof of torque off.
Software requests depend on a functioning host, USB bus and motor controller.
SIGTERM uses the same staged stop path; SIGKILL cannot run cleanup.

An E-stop remains latched on the next start. Clear it through the normal UI
confirmation after checking the rig; the launcher never clears it automatically.
Independently launched tools, such as the phone alignment guide, have their own
terminal lifetime and are not started by this command.
