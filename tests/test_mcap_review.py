import json

import pytest
from mcap.writer import Writer

from so101.services.mcap_review import McapReview, compressed_jpeg

EPISODE = "a" * 32
JPEG = b"\xff\xd8\xfftest\xff\xd9"


@pytest.fixture
def review(tmp_path):
    raw = tmp_path / "drive/mcap"
    raw.mkdir(parents=True)
    data = tmp_path / "data/mcap"
    data.mkdir(parents=True)
    filename = "20261028T010000Z-first.mcap"
    episode = {
        "id": EPISODE,
        "task": "tuning",
        "status": "saved",
        "started_at_ns": 1_000_000_000,
        "ended_at_ns": 3_000_000_000,
        "starting_segment": str(raw / (filename + ".active")),
        "operator_id": "private",
    }
    (data / "episodes.json").write_text(json.dumps({"episodes": [episode]}))
    with (raw / filename).open("wb") as handle:
        writer = Writer(handle, enable_data_crcs=True)
        writer.start()
        control = writer.register_channel("/robot/control_tick", "json", 0)
        metadata = writer.register_channel("/recording/metadata", "json", 0)
        camera = writer.register_channel("/camera/obsbot/image", "protobuf", 0)
        writer.add_message(
            metadata,
            900_000_000,
            json.dumps(
                {"episode_id": EPISODE, "operator_id": "private", "task": "tuning"}
            ).encode(),
            900_000_000,
        )
        for index in range(61):
            stamp = 1_000_000_000 + round(index / 30 * 1e9)
            tick = {
                "seq": index + 1,
                "timestamp_ns": stamp,
                "observation": {"wrist_roll.pos": index},
                "action": {"wrist_roll.pos": index + 1},
                "desired_action": {"wrist_roll.pos": index + 2},
                "control_profile": {"max_velocity": 90},
                "episode_id": EPISODE,
                "backend": "real",
            }
            writer.add_message(control, stamp, json.dumps(tick).encode(), stamp)
            writer.add_message(camera, stamp, b"\x12" + bytes([len(JPEG)]) + JPEG, stamp)
        writer.finish()
    return McapReview(tmp_path / "data", raw)


def test_recorded_timeline_keeps_clock_identity_and_separates_intent_command_observation(review):
    result = review.timeline(EPISODE)
    assert result["sample_count"] == 61
    assert result["measured_control_hz"] == pytest.approx(30)
    assert result["ticks"][0]["seconds"] == 0
    assert result["ticks"][-1]["seconds"] == 2
    assert result["ticks"][0]["observation"] == {"wrist_roll.pos": 0}
    assert result["ticks"][0]["action"] == {"wrist_roll.pos": 1}
    assert result["ticks"][0]["desired_action"] == {"wrist_roll.pos": 2}
    assert result["profile"] == {"max_velocity": 90}
    assert "operator_id" not in json.dumps(result)
    snapshot = review.snapshot(EPISODE, "/recording/metadata", 0)
    assert snapshot["record"]["data"]["task"] == "tuning"


def test_recorded_camera_preserves_timestamp_and_streams_original_frames(review):
    frame, stamp, target = review.frame(EPISODE, "obsbot", 0.5)
    assert frame == JPEG
    assert stamp == target == 1_500_000_000
    frames = list(review.frames(EPISODE, "obsbot", 1.9))
    assert len(frames) == 4
    assert frames[-1] == (3_000_000_000, JPEG)
    with pytest.raises(ValueError, match="outside"):
        review.frame(EPISODE, "obsbot", 4)
    with pytest.raises(ValueError, match="Invalid"):
        review.frame(EPISODE, "../../private", 0)


def test_export_is_finalized_and_preserves_original_timestamps(review):
    from mcap.reader import make_reader

    path = review.export(EPISODE)
    with path.open("rb") as handle:
        rows = list(make_reader(handle, validate_crcs=True).iter_messages())
    control = [
        json.loads(message.data)
        for _, channel, message in rows
        if channel.topic == "/robot/control_tick"
    ]
    assert len(control) == 61
    assert control[0]["timestamp_ns"] == 1_000_000_000
    assert review.reports()[EPISODE]["human_review"] == "pending"
    assert review.reports()[EPISODE]["data_checks_passed"] is False


def test_repeated_snapshot_reads_reuse_indexes_but_reject_replaced_source_symlinks(
    review, monkeypatch
):
    import so101.services.mcap_review as module

    review.timeline(EPISODE)

    def unexpected(*args, **kwargs):
        pytest.fail("Finalized unchanged indexes were rescanned")

    monkeypatch.setattr(module, "make_reader", unexpected)
    assert review.snapshot(EPISODE, "/recording/metadata", 1)["record"]["data"]["task"] == "tuning"
    path = review.sources(review.episode(EPISODE))[0]
    outside = review.raw_dir.parent / "outside.mcap"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError):
        review.timeline(EPISODE)


@pytest.mark.parametrize("problem", ["identity", "overlap"])
def test_wrong_episode_or_overlapping_control_samples_are_rejected(review, problem):
    second = review.raw_dir / "20261028T010001Z-second.mcap"
    with second.open("wb") as handle:
        writer = Writer(handle)
        writer.start()
        channel = writer.register_channel("/robot/control_tick", "json", 0)
        data = {
            "episode_id": "b" * 32 if problem == "identity" else EPISODE,
            "timestamp_ns": 2_000_000_000,
        }
        writer.add_message(channel, 2_000_000_000, json.dumps(data).encode(), 2_000_000_000)
        writer.finish()
    with pytest.raises(
        ValueError, match="different episode" if problem == "identity" else "overlap"
    ):
        review.timeline(EPISODE)


def test_missing_camera_frame_is_reported_instead_of_showing_other_episode(review):
    with pytest.raises(ValueError, match="No recorded camera frame"):
        review.frame(EPISODE, "camera_1", 1)


def test_arbitrary_paths_and_unacknowledged_recordings_cannot_be_read(review):
    with pytest.raises(ValueError, match="identity"):
        review.timeline("../../private")
    path = review.data_dir / "mcap/episodes.json"
    data = json.loads(path.read_text())
    data["episodes"][0].pop("starting_segment")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="never acknowledged"):
        review.timeline(EPISODE)


@pytest.mark.parametrize("payload", [b"\x12\xff", b"\x12\x09missing", b"\x0f", b"\x12\x03abc"])
def test_camera_decoder_rejects_malformed_or_non_jpeg_data(payload):
    with pytest.raises(ValueError):
        compressed_jpeg(payload)
