"""SO-101 platform: teleop, data collection, training, serving and eval."""

__version__ = "0.2.2rc5"

# SO-101 arm has 6 joints (5 DoF arm + gripper).
JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
NUM_JOINTS = len(JOINT_NAMES)
