"""Read-only, episode-bounded MCAP inspection for the workstation UI."""

from __future__ import annotations

import bisect
import json
import math
import os
import re
import threading
import uuid
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from mcap.reader import make_reader

CAMERAS = {"obsbot", "camera_1", "camera_2", "realsense"}


def _public(value):
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if key != "operator_id"}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


def _sample(values, maximum):
    if len(values) <= maximum:
        return values
    indexes = {round(index * (len(values) - 1) / (maximum - 1)) for index in range(maximum)}
    return [values[index] for index in sorted(indexes)]


def compressed_jpeg(payload: bytes) -> bytes:
    offset = 0

    def varint():
        nonlocal offset
        value = 0
        for shift in range(0, 70, 7):
            if offset >= len(payload):
                raise ValueError("Truncated camera message")
            byte = payload[offset]
            offset += 1
            value |= (byte & 127) << shift
            if not byte & 128:
                return value
        raise ValueError("Invalid camera message")

    while offset < len(payload):
        tag = varint()
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            varint()
        elif wire in (1, 5):
            offset += 8 if wire == 1 else 4
        elif wire == 2:
            length = varint()
            end = offset + length
            if end > len(payload):
                raise ValueError("Truncated camera image")
            if field == 2:
                image = payload[offset:end]
                if not image.startswith(b"\xff\xd8\xff") or not image.endswith(b"\xff\xd9"):
                    raise ValueError("Recorded camera image is not JPEG")
                return image
            offset = end
        else:
            raise ValueError("Unsupported camera encoding")
        if offset > len(payload):
            raise ValueError("Truncated camera message")
    raise ValueError("Camera image data is missing")


@lru_cache(maxsize=2)
def _timeline(episode_json: str, fingerprints: tuple) -> dict:
    episode = json.loads(episode_json)
    start, end = episode["started_at_ns"], episode["ended_at_ns"]
    ticks: list[dict[str, Any]] = []
    snapshots: dict[str, list] = defaultdict(list)
    cameras: set[str] = set()
    counts: Counter[str] = Counter()
    strides: dict[str, int] = defaultdict(lambda: 1)
    latest = {}
    for filename, _size, _modified in fingerprints:
        with Path(filename).open("rb") as handle:
            reader = make_reader(handle, validate_crcs=True)
            summary = reader.get_summary()
            if summary is None:
                raise ValueError("Recording has no finalized index")
            json_topics = {
                channel.topic
                for channel in summary.channels.values()
                if channel.message_encoding == "json"
            }
            cameras.update(
                channel.topic.split("/")[2]
                for channel in summary.channels.values()
                if channel.topic.startswith("/camera/") and channel.topic.endswith("/image")
            )
            for _, channel, message in reader.iter_messages(
                topics=list(json_topics), end_time=end + 1
            ):
                prelude = (
                    channel.topic.startswith("/calibration/")
                    or channel.topic == "/recording/metadata"
                )
                if message.log_time < start and not prelude:
                    continue
                data = _public(json.loads(message.data))
                if (
                    channel.topic == "/recording/metadata"
                    and data.get("episode_id") != episode["id"]
                ):
                    continue
                counts[channel.topic] += 1
                seconds = (message.log_time - start) / 1e9
                if channel.topic == "/robot/control_tick":
                    if data.get("episode_id") not in (None, episode["id"]):
                        raise ValueError("Control sample belongs to a different episode")
                    if not start <= data["timestamp_ns"] <= end:
                        raise ValueError("Control timestamp is outside the episode")
                    if ticks and data["timestamp_ns"] <= ticks[-1]["timestamp_ns"]:
                        raise ValueError("Control timestamps overlap or do not increase")
                    ticks.append(
                        {
                            key: data.get(key)
                            for key in (
                                "seq",
                                "timestamp_ns",
                                "observation",
                                "action",
                                "desired_action",
                                "timing",
                                "control_profile",
                                "backend",
                                "episode_id",
                            )
                        }
                        | {"seconds": (data["timestamp_ns"] - start) / 1e9}
                    )
                    if len(ticks) > 100_000:
                        raise ValueError(
                            "Recording is too long for interactive review; "
                            "export a shorter interval"
                        )
                else:
                    row = {"seconds": seconds, "data": data}
                    latest[channel.topic] = row
                    if (counts[channel.topic] - 1) % strides[channel.topic] == 0:
                        snapshots[channel.topic].append(row)
                    if len(snapshots[channel.topic]) > 800:
                        snapshots[channel.topic] = snapshots[channel.topic][::2]
                        strides[channel.topic] *= 2
    if not ticks:
        raise ValueError("No recorded control samples are available")
    intervals = [
        (right["timestamp_ns"] - left["timestamp_ns"]) / 1e9
        for left, right in zip(ticks, ticks[1:], strict=False)
    ]
    span = (ticks[-1]["timestamp_ns"] - ticks[0]["timestamp_ns"]) / 1e9
    profile = ticks[-1].get("control_profile")
    for tick in ticks:
        tick.pop("control_profile", None)
    for topic, row in latest.items():
        if snapshots[topic][-1]["seconds"] != row["seconds"]:
            snapshots[topic].append(row)
    return {
        "episode": episode,
        "duration_seconds": (end - start) / 1e9,
        "sample_count": len(ticks),
        "samples_without_episode_identity": sum(tick.get("episode_id") is None for tick in ticks),
        "display_sample_count": min(len(ticks), 3000),
        "measured_control_hz": (len(ticks) - 1) / span if span > 0 else 0,
        "max_control_gap_seconds": max(intervals, default=0),
        "ticks": _sample(ticks, 3000),
        "profile": profile,
        "cameras": sorted(cameras & CAMERAS),
        "topics": dict(counts),
        "snapshots": {topic: _sample(values, 400) for topic, values in snapshots.items()},
        "note": (
            "Read-only recorded playback. Charts may be downsampled; timing statistics "
            "use every control sample. Snapshot panels use nearest preceding retained "
            "message. No robot commands are sent."
        ),
    }


class McapReview:
    def __init__(self, data_dir: Path, raw_dir: Path):
        self.data_dir = data_dir
        self.raw_dir = raw_dir.resolve()
        self._source_cache: dict[str, tuple] = {}
        self._cache_lock = threading.Lock()

    def episodes(self):
        path = self.data_dir / "mcap/episodes.json"
        if not path.exists():
            return []
        return _public(json.loads(path.read_text()).get("episodes", []))

    def episode(self, episode_id):
        if not re.fullmatch(r"[a-f0-9]{32}", episode_id):
            raise ValueError("Invalid episode identity")
        result = next((item for item in self.episodes() if item["id"] == episode_id), None)
        if not result or not result.get("ended_at_ns"):
            raise ValueError("Episode is not finalized")
        return result

    def sources(self, episode):
        starting = Path(episode.get("starting_segment", "")).name.removesuffix(".active")
        if not starting:
            raise ValueError("Episode never acknowledged a recording segment")
        stamp = self.raw_dir.stat().st_mtime_ns
        key = json.dumps(episode, sort_keys=True)
        with self._cache_lock:
            cached = self._source_cache.get(key)
        if cached and cached[0] == stamp:
            verified = []
            for filename, size, modified in cached[1]:
                path = Path(filename)
                if not path.resolve().is_relative_to(self.raw_dir):
                    raise ValueError("Recording segment escaped the configured directory")
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != (size, modified):
                    break
                verified.append(path)
            else:
                return verified
        results = []
        for path in sorted(self.raw_dir.glob("*.mcap")):
            resolved = path.resolve()
            if not resolved.is_relative_to(self.raw_dir):
                continue
            if path.name >= starting:
                with resolved.open("rb") as handle:
                    summary = make_reader(handle).get_summary()
                    if summary and summary.statistics:
                        stats = summary.statistics
                        if stats.message_start_time > episode["ended_at_ns"]:
                            continue
                        if stats.message_end_time >= episode["started_at_ns"]:
                            results.append(resolved)
        if not results or results[0].name != starting:
            raise ValueError("Finalized recording segment is unavailable")
        entry = (
            stamp,
            tuple((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in results),
        )
        with self._cache_lock:
            self._source_cache[key] = entry
            if len(self._source_cache) > 16:
                self._source_cache.pop(next(iter(self._source_cache)))
        return results

    def timeline(self, episode_id):
        episode = self.episode(episode_id)
        sources = self.sources(episode)
        fingerprints = tuple(
            (str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in sources
        )
        return _timeline(json.dumps(episode, sort_keys=True), fingerprints)

    def snapshot(self, episode_id, topic, seconds):
        timeline = self.timeline(episode_id)
        if not math.isfinite(seconds) or not 0 <= seconds <= timeline["duration_seconds"]:
            raise ValueError("Snapshot time is outside the episode")
        records = timeline["snapshots"].get(topic)
        if records is None:
            raise ValueError("Recorded JSON topic is unavailable")
        return {
            "topic": topic,
            "record": snapshot_at(records, seconds),
            "retained_messages": len(records),
        }

    def frame(self, episode_id, camera, seconds):
        if camera not in CAMERAS or not math.isfinite(seconds):
            raise ValueError("Invalid recorded frame request")
        episode = self.episode(episode_id)
        start, end = episode["started_at_ns"], episode["ended_at_ns"]
        target = start + round(seconds * 1e9)
        if not start <= target <= end:
            raise ValueError("Frame time is outside the episode")
        candidates = []
        for source in self.sources(episode):
            with source.open("rb") as handle:
                for _, channel, message in make_reader(handle, validate_crcs=True).iter_messages(
                    topics=[f"/camera/{camera}/image"],
                    start_time=max(start, target - 500_000_000),
                    end_time=min(end + 1, target + 500_000_001),
                ):
                    if channel.message_encoding == "protobuf":
                        candidates.append(
                            (abs(message.log_time - target), message.log_time, message.data)
                        )
        if not candidates:
            raise ValueError("No recorded camera frame within 500 ms")
        _, timestamp, data = min(candidates, key=lambda item: item[0])
        return compressed_jpeg(data), timestamp, target

    def frames(self, episode_id, camera, seconds):
        if camera not in CAMERAS or not math.isfinite(seconds):
            raise ValueError("Invalid recorded camera")
        episode = self.episode(episode_id)
        start = episode["started_at_ns"] + round(seconds * 1e9)
        if not episode["started_at_ns"] <= start <= episode["ended_at_ns"]:
            raise ValueError("Playback time is outside the episode")
        for source in self.sources(episode):
            with source.open("rb") as handle:
                for _, channel, message in make_reader(handle, validate_crcs=True).iter_messages(
                    topics=[f"/camera/{camera}/image"],
                    start_time=start,
                    end_time=episode["ended_at_ns"] + 1,
                ):
                    if channel.message_encoding == "protobuf":
                        yield message.log_time, compressed_jpeg(message.data)

    def reports(self):
        results = {}
        root = self.raw_dir.parent / "episode-reviews"
        for path in sorted(root.glob("*/report.json"), key=lambda item: item.stat().st_mtime):
            if not path.resolve().is_relative_to(root.resolve()):
                continue
            try:
                content = json.loads(path.read_text())
                for episode in content.get("episodes", [content]):
                    if episode.get("episode_id"):
                        results[episode["episode_id"]] = episode
            except (OSError, ValueError, TypeError):
                continue
        return results

    def export(self, episode_id):
        from mcap.writer import Writer

        timeline = self.timeline(episode_id)
        episode = timeline["episode"]
        directory = self.raw_dir.parent / "episode-reviews" / ("ui-" + uuid.uuid4().hex)
        directory.mkdir(parents=True, exist_ok=False)
        output = directory / f"{episode_id}.mcap"
        partial = output.with_suffix(".partial")
        schemas, channels = {}, {}
        with partial.open("xb") as handle:
            writer = Writer(handle, enable_data_crcs=True)
            writer.start(library="so101-ui-review")
            for source in self.sources(episode):
                with source.open("rb") as incoming:
                    for schema, channel, message in make_reader(
                        incoming, validate_crcs=True
                    ).iter_messages(end_time=episode["ended_at_ns"] + 1):
                        prelude = (
                            channel.topic.startswith("/calibration/")
                            or channel.topic == "/recording/metadata"
                        )
                        if message.log_time < episode["started_at_ns"] and not prelude:
                            continue
                        if (
                            channel.topic == "/recording/metadata"
                            and json.loads(message.data).get("episode_id") != episode_id
                        ):
                            continue
                        key = (schema.name, schema.encoding, schema.data) if schema else None
                        if key not in schemas:
                            schemas[key] = writer.register_schema(*key) if key else 0
                        channel_key = (channel.topic, channel.message_encoding, schemas[key])
                        if channel_key not in channels:
                            channels[channel_key] = writer.register_channel(
                                *channel_key, metadata=channel.metadata
                            )
                        writer.add_message(
                            channels[channel_key],
                            message.log_time,
                            message.data,
                            message.publish_time,
                            message.sequence,
                        )
            writer.finish()
            handle.flush()
            os.fsync(handle.fileno())
        with partial.open("rb") as check:
            for _ in make_reader(check, validate_crcs=True).iter_messages():
                pass
        partial.rename(output)
        from ..atomic_io import atomic_write_json

        report = {
            "episode_id": episode_id,
            "path": str(output),
            "human_review": "pending",
            "data_checks_passed": False,
            "issues": [
                "UI export prepared; formal data-quality acceptance and human review remain pending"
            ],
            "source_segments": [str(path) for path in self.sources(episode)],
            "sample_count": timeline["sample_count"],
            "measured_control_hz": timeline["measured_control_hz"],
            "max_control_gap_seconds": timeline["max_control_gap_seconds"],
        }
        atomic_write_json(directory / "report.json", {"episodes": [report]})
        return output

    def catalog(self):
        reports = self.reports()
        return [
            {
                "episode": episode,
                "review": reports.get(episode["id"]),
                "recording_available": bool(episode.get("starting_segment"))
                and (
                    self.raw_dir / Path(episode["starting_segment"]).name.removesuffix(".active")
                ).is_file(),
            }
            for episode in reversed(self.episodes())
        ]


def snapshot_at(records, seconds):
    if not records:
        return None
    index = bisect.bisect_right([row["seconds"] for row in records], seconds) - 1
    return records[index] if index >= 0 else None
