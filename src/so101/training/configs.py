"""Policy presets — the SO-101-relevant policies LeRobot ships.

These presets are the single source of truth for ``so101 train`` and
``GET /api/training/presets`` (the unused ``configs/training/*.yaml`` copies
were removed).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PolicyPreset:
    name: str
    policy_type: str  # LeRobot --policy.type value
    description: str
    default_steps: int
    default_batch_size: int
    notes: str = ""
    extra_args: dict[str, str] = field(default_factory=dict)
    # "lerobot": ``lerobot-train`` subprocess (needs the [train] extra);
    # "numpy_in_process": the torch-free baseline trained inside this process.
    trainer: str = "lerobot"
    requires_torch: bool = True


POLICY_PRESETS: dict[str, PolicyPreset] = {
    "act": PolicyPreset(
        name="act",
        policy_type="act",
        description="Action Chunking Transformer — the SO-101 default. Fast, "
        "robust for single-task manipulation from a few dozen episodes.",
        default_steps=100_000,
        default_batch_size=8,
        notes="Best starting point. ~50 episodes is plenty for a simple task.",
    ),
    "diffusion": PolicyPreset(
        name="diffusion",
        policy_type="diffusion",
        description="Diffusion Policy — strong multimodal behaviour, heavier to train and run.",
        default_steps=200_000,
        default_batch_size=64,
        notes="Needs more data + compute than ACT; smoother on multimodal tasks.",
    ),
    "smolvla": PolicyPreset(
        name="smolvla",
        policy_type="smolvla",
        description="SmolVLA — compact vision-language-action model, supports "
        "language-conditioned tasks and finetuning from a pretrained base.",
        default_steps=20_000,
        default_batch_size=64,
        notes="Finetune from lerobot/smolvla_base for fast results.",
        extra_args={"policy.pretrained_path": "lerobot/smolvla_base"},
    ),
    "pi0": PolicyPreset(
        name="pi0",
        policy_type="pi0",
        description="Pi0 — large flow-based VLA. Highest ceiling, highest cost.",
        default_steps=30_000,
        default_batch_size=32,
        notes="Requires a capable GPU; finetunes from lerobot/pi0_base.",
        # Without a pretrained path LeRobot builds PaliGemma with random weights.
        extra_args={"policy.pretrained_path": "lerobot/pi0_base"},
    ),
    "bc_baseline": PolicyPreset(
        name="bc_baseline",
        policy_type="numpy_bc_mlp",
        description="Numpy behaviour-cloning baseline: a small MLP from joint state (plus the "
        "simulator's object state, when the dataset has it) to the next joint target. Trains "
        "in-process on CPU in seconds; no torch, no GPU.",
        default_steps=3000,
        default_batch_size=256,
        notes="A sanity check and simulation baseline, not a manipulation policy. Trained on "
        "simulator object state it runs only in simulation; --state-only (extra_args "
        "state_only=true) trains one that can drive the arm.",
        trainer="numpy_in_process",
        requires_torch=False,
    ),
}


def get_preset(name: str) -> PolicyPreset:
    if name not in POLICY_PRESETS:
        raise KeyError(f"unknown policy {name!r}; options: {sorted(POLICY_PRESETS)}")
    return POLICY_PRESETS[name]
