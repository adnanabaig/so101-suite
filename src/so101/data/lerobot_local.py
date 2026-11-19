"""Open a local LeRobot dataset without any Hugging Face Hub fallback (audit B27).

``LeRobotDataset(repo_id, root=root)`` downloads from the Hub into ``root``
when its local load fails: a missing metadata file, a missing data file or an
out-of-range episode index all trigger ``snapshot_download(local_dir=root)``.
For a dataset that was pushed to the Hub (``POST /api/datasets/{repo_id}/push``,
the Datasets page) a damaged local copy
would then be overwritten with Hub files; for a local-only name it becomes a
slow network error.

Replay and QC read local datasets only, so this module checks the metadata
LeRobot would otherwise fetch, checks the episode index against it, and opens
the dataset through a subclass whose download step raises instead.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

# Not a PEP 440 version, so LeRobot skips its Hub version lookup
# (``get_safe_version``) when a local load is incomplete. Only Hub operations,
# which this module refuses, read the revision.
LOCAL_ONLY_REVISION = "so101-local-only"


class IncompleteLocalDatasetError(FileNotFoundError):
    """A local LeRobot dataset is missing files LeRobot would fetch from the Hub."""


def _count(info: dict[str, Any], key: str) -> int:
    try:
        return max(0, int(info.get(key) or 0))
    except (TypeError, ValueError):
        return 0


def check_local_lerobot_dataset(
    root: Path,
    repo_id: str,
    episodes: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return ``meta/info.json`` once the local metadata and episode ids check out.

    Raises :class:`IncompleteLocalDatasetError` for missing metadata files or an
    episode id outside ``0..total_episodes-1``, and ``ValueError`` for
    unreadable metadata.
    """

    root = Path(root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise IncompleteLocalDatasetError(
            f"{repo_id}: {info_path} is missing; the local LeRobot dataset is incomplete"
        )
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{repo_id}: unreadable {info_path}: {exc}") from exc
    if not isinstance(info, dict):
        raise ValueError(f"{repo_id}: {info_path} is not a JSON object")
    total_episodes = _count(info, "total_episodes")
    if _count(info, "total_tasks") and not (root / "meta" / "tasks.parquet").is_file():
        raise IncompleteLocalDatasetError(
            f"{repo_id}: meta/tasks.parquet is missing; the local LeRobot dataset is incomplete"
        )
    episodes_dir = root / "meta" / "episodes"
    if total_episodes and not (episodes_dir.is_dir() and any(episodes_dir.rglob("*.parquet"))):
        raise IncompleteLocalDatasetError(
            f"{repo_id}: meta/episodes/ has no episode metadata; the local LeRobot dataset "
            "is incomplete"
        )
    for episode in episodes or ():
        if not 0 <= int(episode) < total_episodes:
            available = f"episodes 0..{total_episodes - 1}" if total_episodes else "no episodes"
            raise IncompleteLocalDatasetError(
                f"episode {episode} not found in {repo_id} ({available})"
            )
    return info


@cache
def _local_only_class(base: type) -> type:
    class LocalOnlyLeRobotDataset(base):  # type: ignore[valid-type, misc]
        """A ``LeRobotDataset`` that never downloads into its local root."""

        def _download(self, *args: Any, **kwargs: Any) -> None:
            raise IncompleteLocalDatasetError(
                f"the local LeRobot dataset at {self.root} is missing data files; so101 "
                "never downloads from the Hugging Face Hub into a local dataset (repair "
                "or re-record it)"
            )

    return LocalOnlyLeRobotDataset


def open_local_lerobot_dataset(
    repo_id: str,
    root: Path,
    *,
    episodes: Sequence[int] | None = None,
    dataset_cls: type | None = None,
    **kwargs: Any,
) -> Any:
    """Open ``root`` as a LeRobot dataset, refusing any Hub download.

    ``dataset_cls`` defaults to ``lerobot.datasets.LeRobotDataset``. Raises
    ``RuntimeError`` when LeRobot's dataset support is not installed.
    """

    check_local_lerobot_dataset(root, repo_id, episodes)
    if dataset_cls is None:
        try:
            from lerobot.datasets import LeRobotDataset  # type: ignore[import-not-found]
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError("LeRobot dataset support is not installed") from exc
        dataset_cls = LeRobotDataset
    local_cls = _local_only_class(dataset_cls)
    return local_cls(
        repo_id,
        root=root,
        episodes=list(episodes) if episodes is not None else None,
        revision=LOCAL_ONLY_REVISION,
        **kwargs,
    )
