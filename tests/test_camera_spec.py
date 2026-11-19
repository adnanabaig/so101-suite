"""SO101_CAMERAS entries: the grammar, stream order and LeRobot feature names."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from so101.config import Settings, get_settings
from so101.hardware.camera_spec import (
    DEPTH_SUFFIX,
    base_key,
    is_depth_key,
    parse_camera_entries,
    parse_camera_entry,
    parse_camera_setting,
    stream_feature_names,
    stream_keys,
)
from so101.hardware.cameras import camera_feature_name

# Feature names from before RealSense support existed. Datasets recorded with
# them, and policies trained on those datasets, look cameras up by these names,
# so every entry that is neither named nor RealSense keeps its exact name,
# credentials excepted.
PINNED_FEATURE_NAMES = [
    ("0", 0, "camera_00_0"),
    ("1", 1, "camera_01_1"),
    ("/dev/video0", 0, "camera_00_dev_video0"),
    ("rtsp://h/s?x=1", 0, "camera_00_rtsp_h_s_x_1"),
    # The one deliberate change: credentials no longer reach a feature name
    # (it was camera_04_rtsp_user_pw_10.0.0.5_554_stream1).
    ("rtsp://user:pw@10.0.0.5:554/stream1", 4, "camera_04_rtsp_10.0.0.5_554_stream1"),
    ("http://h/x.mjpg", 0, "camera_00_http_h_x.mjpg"),
    ("C:\\cams\\usb cam", 0, "camera_00_C_cams_usb_cam"),
    (
        "v4l2src device=/dev/video0 ! videoconvert ! appsink",
        1,
        "camera_01_v4l2src_device_dev_video0_videoconvert_appsink",
    ),
    ("front", 0, "camera_00_front"),
    ("...", 0, "camera_00_camera"),
    ("", 3, "camera_03_camera"),
    ("..a..", 12, "camera_12_a"),
    ("weird name!", 120, "camera_120_weird_name"),
    ("a" * 100, 2, "camera_02_" + "a" * 80),
]


@pytest.mark.parametrize(("key", "index", "name"), PINNED_FEATURE_NAMES)
def test_feature_names_of_plain_opencv_entries_are_unchanged(
    key: str, index: int, name: str
) -> None:
    assert camera_feature_name(key, index) == name


# --- grammar -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        ("0", dict(name=None, kind="opencv", source="0", serial=None, width=None, depth=False)),
        (" /dev/video2 ", dict(kind="opencv", source="/dev/video2")),
        ("front=1", dict(name="front", kind="opencv", source="1")),
        ("rtsp://h/s?x=1", dict(name=None, kind="opencv", source="rtsp://h/s?x=1")),
        ("v4l2src device=/dev/video0 ! appsink", dict(name=None, kind="opencv")),
        ("cam=rtsp://h/s?x=1", dict(name="cam", source="rtsp://h/s?x=1")),
        (
            "realsense",
            dict(
                kind="realsense",
                serial=None,
                width=640,
                height=480,
                fps=30,
                mode_explicit=False,
                depth=False,
            ),
        ),
        ("RealSense+DEPTH", dict(kind="realsense", serial=None, depth=True)),
        ("realsense:123622270882", dict(serial="123622270882", mode_explicit=False)),
        (
            "realsense:1280x720:123622270882+depth",
            dict(serial="123622270882", width=1280, height=720, fps=30, mode_explicit=True),
        ),
        (
            "wrist=realsense:123622270882:848x480@60",
            dict(name="wrist", serial="123622270882", width=848, height=480, fps=60, depth=False),
        ),
        ("realsense:640X480@15", dict(width=640, height=480, fps=15, mode_explicit=True)),
        ("realsense:f0245826", dict(serial="f0245826")),
    ],
)
def test_valid_entries_parse(entry: str, expected: dict[str, object]) -> None:
    spec = parse_camera_entry(entry)
    assert spec.entry == entry.strip()
    for field_name, value in expected.items():
        assert getattr(spec, field_name) == value, field_name


def test_stream_keys_of_a_spec() -> None:
    color = parse_camera_entry("realsense:1234")
    both = parse_camera_entry("w=realsense:1234+depth")
    assert (color.color_key, color.depth_key, color.keys) == (
        "realsense:1234",
        None,
        ("realsense:1234",),
    )
    assert both.depth_key == "w=realsense:1234+depth#depth"
    assert both.keys == ("w=realsense:1234+depth", "w=realsense:1234+depth#depth")
    assert is_depth_key(both.depth_key) and not is_depth_key(both.color_key)
    assert base_key(both.depth_key) == both.color_key == base_key(both.color_key)
    assert DEPTH_SUFFIX == "#depth"


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        ("", "camera entry is empty"),
        ("   ", "camera entry is empty"),
        ("front=", "nothing after front="),
        ("0+depth", "+depth needs a RealSense source"),
        ("side=/dev/video0+depth", "an OpenCV camera has no depth stream"),
        ("x#depth", "reserved for depth stream keys"),
        ("realsense:", "each ':' must be followed by a mode or a serial"),
        ("realsense::1234", "each ':' must be followed"),
        ("realsense+foo", "the only flag is +depth"),
        ("realsense+depth+depth", "the only flag is +depth"),
        ("realsense:1:2:3", "at most one mode and one serial"),
        ("realsense:640x480:1280x720", "more than one mode"),
        ("realsense:1234:5678", "more than one serial"),
        ("realsense:640x480x30", "'640x480x30' is not a mode"),
        ("realsense:640x", "'640x' is not a mode"),
        ("realsense:x480", "'x480' is not a mode"),
        ("realsense:640x480@", "'640x480@' is not a mode"),
        ("realsense:abc!", "neither a mode"),
        ("realsense:123", "neither a mode"),  # a serial has at least 4 characters
        ("realsense:0x480", "0x480 is outside"),
        ("realsense:640x5000", "640x5000 is outside"),
        ("realsense:640x480@0", "0 fps is outside"),
        ("realsense:640x480@301", "301 fps is outside"),
    ],
)
def test_invalid_entries_name_the_entry(entry: str, message: str) -> None:
    with pytest.raises(ValueError) as caught:
        parse_camera_entry(entry)
    text = str(caught.value)
    assert message in text
    if entry.strip():
        assert repr(entry.strip()) in text
    if entry.lower().startswith("realsense") and "#depth" not in entry:
        assert "expected realsense[:<serial>][:<W>x<H>[@<fps>]][+depth]" in text


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("0,0", "camera entry '0' is listed twice"),
        ("a=0,a=1", "camera name 'a' is used by both 'a=0' and 'a=1'"),
        ("Front=0,front=1", "camera name 'front' is used by both"),
        (
            "wrist=realsense+depth,wrist_depth=1",
            "camera entry 'wrist_depth=1': its name 'wrist_depth' is the depth stream name of "
            "'wrist=realsense+depth'",
        ),
        (
            "wrist_depth=1,wrist=realsense+depth",
            "camera entry 'wrist_depth=1': its name 'wrist_depth' is the depth stream name",
        ),
        (
            "realsense:123456,a=realsense:123456+depth",
            "RealSense serial 123456 is in both 'realsense:123456' and 'a=realsense:123456+depth'",
        ),
        ("realsense:AbC123,realsense:abc123", "RealSense serial abc123 is in both"),
        (
            "realsense:1234,realsense+depth",
            "camera entry 'realsense+depth': with 2 RealSense entries each needs a serial",
        ),
        ("realsense,a=realsense", "camera entry 'realsense': with 2 RealSense entries"),
        (
            "0,camera_00_0=1",
            "camera entries '0' and 'camera_00_0=1' both record as 'camera_00_0'",
        ),
    ],
)
def test_entries_that_collide_are_rejected(value: str, message: str) -> None:
    with pytest.raises(ValueError) as caught:
        parse_camera_setting(value)
    assert message in str(caught.value)


def test_setting_splits_on_commas_and_skips_blank_entries() -> None:
    specs = parse_camera_setting(" 0, ,wrist=realsense:1234+depth,, ")
    assert [spec.entry for spec in specs] == ["0", "wrist=realsense:1234+depth"]
    assert parse_camera_setting("") == [] and parse_camera_setting(" , ") == []
    # A depth stream named after its camera is fine when nothing else uses the name.
    assert len(parse_camera_setting("camera_00_0=1,0")) == 2


def test_parse_camera_entries_checks_the_list() -> None:
    assert [s.entry for s in parse_camera_entries(["front=0", "realsense"])] == [
        "front=0",
        "realsense",
    ]
    with pytest.raises(ValueError, match="listed twice"):
        parse_camera_entries(["1", " 1 "])


# --- feature names -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "index", "name"),
    [
        ("front=0", 3, "front"),
        ("wrist=realsense:1234+depth", 0, "wrist"),
        ("wrist=realsense:1234+depth#depth", 0, "wrist_depth"),
        ("realsense", 1, "camera_01_realsense"),
        ("realsense+depth#depth", 1, "camera_01_realsense_depth"),
        ("realsense:640x480:123622270882", 2, "camera_02_realsense_123622270882"),
        ("realsense:123622270882+depth#depth", 0, "camera_00_realsense_123622270882_depth"),
        ("front=0#depth", 0, "front_depth"),
        # Not a valid entry any more, but naming must never raise: the old rule applies.
        ("cam=realsense:!!", 0, "camera_00_cam_realsense"),
        ("0+depth", 0, "camera_00_0_depth"),
        ("x.#depth", 0, "camera_00_x._depth"),
        ("0#depth", 0, "camera_00_0_depth"),
    ],
)
def test_named_and_realsense_feature_names(key: str, index: int, name: str) -> None:
    assert camera_feature_name(key, index) == name


def test_feature_name_never_raises_on_odd_input() -> None:
    assert camera_feature_name(None, 0) == "camera_00_None"  # type: ignore[arg-type]
    assert camera_feature_name("=", 0) == "camera_00_camera"


def test_stream_order_is_color_then_depth_in_configured_order() -> None:
    specs = parse_camera_setting("a=realsense:1111+depth,0,b=realsense:2222,c=realsense:3333+depth")
    assert stream_keys(specs) == [
        "a=realsense:1111+depth",
        "0",
        "b=realsense:2222",
        "c=realsense:3333+depth",
        "a=realsense:1111+depth#depth",
        "c=realsense:3333+depth#depth",
    ]
    assert stream_keys([]) == []


def test_stream_feature_names_index_depth_by_its_color_key() -> None:
    keys = stream_keys(parse_camera_setting("1,realsense:2222+depth,2"))
    assert stream_feature_names(keys) == {
        "1": "camera_00_1",
        "realsense:2222+depth": "camera_01_realsense_2222",
        "2": "camera_02_2",
        "realsense:2222+depth#depth": "camera_01_realsense_2222_depth",
    }


@pytest.mark.parametrize(
    "keys",
    [
        ["0"],
        ["0", "1", "/dev/video0"],
        [key for key, _, _ in PINNED_FEATURE_NAMES],
        ["front=0", "realsense:1234", "rtsp://h/s?x=1"],
    ],
)
def test_color_only_stream_feature_names_equal_the_old_enumeration(keys: list[str]) -> None:
    expected = {key: camera_feature_name(key, index) for index, key in enumerate(keys)}
    assert stream_feature_names(keys) == expected


def test_a_depth_key_without_its_color_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="depth stream 'realsense\\+depth#depth' has no color"):
        stream_feature_names(["0", "realsense+depth#depth"])


# --- Settings ------------------------------------------------------------------------------------


def _settings(monkeypatch: pytest.MonkeyPatch, cameras: str) -> Settings:
    monkeypatch.setenv("SO101_CAMERAS", cameras)
    get_settings.cache_clear()
    try:
        return Settings()
    finally:
        get_settings.cache_clear()


def test_settings_accept_the_grammar_and_keep_camera_ids_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = " 0, wrist=realsense:123622270882:1280x720@15+depth ,realsense:999999999999"
    settings = _settings(monkeypatch, value)
    assert settings.cameras == value
    assert settings.camera_ids == [
        "0",
        "wrist=realsense:123622270882:1280x720@15+depth",
        "realsense:999999999999",
    ]
    assert [spec.kind for spec in settings.camera_specs] == ["opencv", "realsense", "realsense"]
    assert settings.camera_streams == settings.camera_ids + [
        "wrist=realsense:123622270882:1280x720@15+depth#depth"
    ]


def test_settings_default_and_empty_cameras(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _settings(monkeypatch, "0").camera_streams == ["0"]
    empty = _settings(monkeypatch, "")
    assert (empty.camera_ids, empty.camera_specs, empty.camera_streams) == ([], [], [])


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("0,realsense:12", "camera entry 'realsense:12'"),
        ("realsense,realsense:1234", "each needs a serial"),
        ("1+depth", "+depth needs a RealSense source"),
    ],
)
def test_settings_reject_malformed_cameras(
    monkeypatch: pytest.MonkeyPatch, value: str, message: str
) -> None:
    with pytest.raises(ValidationError) as caught:
        _settings(monkeypatch, value)
    assert message in str(caught.value)


def test_the_spec_module_imports_only_the_standard_library() -> None:
    # Settings validation imports it; it must not drag in a camera stack.
    import so101.hardware.camera_spec as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {
        (node.module or "").split(".")[0] if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert imported <= {"__future__", "re", "collections", "dataclasses", "typing"}
