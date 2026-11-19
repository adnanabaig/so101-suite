"""Hardware bring-up kit: preflight checks, HIL runner and evidence (v0.2).

The kit makes the real-arm gate runnable and auditable; it does not pass it.
Mock and sim runs are rehearsals and never count as hardware evidence.

Modules are imported lazily by their callers so that ``import so101.bringup``
stays cheap and never touches LeRobot or a serial port.
"""
