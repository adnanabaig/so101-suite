"""Parse SO101_CAMERAS entries and name the LeRobot feature of every stream.

Pure on purpose: no cv2, no pyrealsense2, no numpy, so ``Settings`` can
validate the value and recorders can name features without a camera stack.

An entry is ``[<name>=]<source>``:

- ``<name>`` (``^[A-Za-z][A-Za-z0-9_]{0,31}$``) names the LeRobot feature
  ``observation.images.<name>``. Text before the first ``=`` is a name only if
  it matches; otherwise the whole entry is the source, so ``rtsp://h/s?x=1``
  and GStreamer pipelines such as ``v4l2src device=/dev/video0 ! ...`` keep
  working.
- An OpenCV source (an index such as ``0`` or a device path/URL) opens with
  ``cv2.VideoCapture``, exactly as before RealSense support.
- ``realsense[:<token>[:<token>]][+depth]`` opens an Intel RealSense with
  pyrealsense2. A token is a mode ``<W>x<H>[@<fps>]`` or a serial (4-32
  letters/digits), in any order, each at most once. Without a serial the entry
  means the only attached RealSense. The default mode is 640x480@30, color
  only; ``+depth`` adds depth aligned to color.

Every entry yields its color stream under the entry itself (``color_key``) and
a RealSense ``+depth`` entry also yields ``entry + "#depth"`` (``depth_key``).
Streams are ordered all color keys in configured order, then all depth keys.

Credentials never leave the parser: URL userinfo (``rtsp://user:pass@host``)
and ``user-pw=``/``password=``-style values are stripped from the entry that
becomes the stream key, so feature names, labels, logs, status and dataset
metadata never carry them, and error messages redact them. Only
``CameraSpec.source`` (excluded from ``repr``) keeps them, to open the camera.
See docs/realsense.md.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

CameraKind = Literal["opencv", "realsense"]

#: Appended to an entry to form the key of its depth stream.
DEPTH_SUFFIX = "#depth"
REALSENSE_SCHEME = "realsense"
DEPTH_FLAG = "+depth"
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 30
#: Loose sanity bounds: anything larger is a typo, not a camera mode.
MAX_DIMENSION = 4096
MAX_FPS = 300
REALSENSE_GRAMMAR = "realsense[:<serial>][:<W>x<H>[@<fps>]][+depth]"

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,31}")
# Checked before the serial pattern so that 640x480 is never taken as a serial.
_MODE = re.compile(r"(\d+)x(\d+)(?:@(\d+))?", re.IGNORECASE)
# Starts like a mode but is not one (640x480x30, 640x, 640x480@): a typo, not
# a serial. RealSense serials are digits, sometimes behind a letter prefix.
_MODE_LIKE = re.compile(r"\d+x|x\d+", re.IGNORECASE)
_SERIAL = re.compile(r"[0-9A-Za-z]{4,32}")
_FEATURE_COMPONENT = re.compile(r"[^a-zA-Z0-9_.-]+")
# scheme://userinfo@host: everything up to the last "@" before the path.
_URL_USERINFO = re.compile(r"(?P<scheme>\b[A-Za-z][A-Za-z0-9+.-]*://)[^/?#\s]*@")
# Secrets passed as key=value: GStreamer rtspsrc user-id/user-pw, URL queries.
_KEY_VALUE_SECRET = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])(?P<key>user-pw|user-id|password|passwd|pwd)=[^&\s!]*"
)


def strip_credentials(text: str) -> str:
    """``text`` without URL userinfo and password-style ``key=value`` pairs."""
    return _KEY_VALUE_SECRET.sub("", _URL_USERINFO.sub(r"\g<scheme>", text))


def redact_credentials(text: str) -> str:
    """``text`` with credentials replaced by ``***``, for error messages."""
    return _KEY_VALUE_SECRET.sub(r"\g<key>=***", _URL_USERINFO.sub(r"\g<scheme>***@", text))


@dataclass(frozen=True)
class CameraSpec:
    """One parsed SO101_CAMERAS entry."""

    entry: str  # stripped, credentials removed; also the color stream key
    name: str | None
    kind: CameraKind
    # The entry without ``<name>=``, credentials included: open the camera with
    # it, never print it.
    source: str = field(repr=False)
    serial: str | None = None
    width: int | None = None  # RealSense only; OpenCV cameras report their own
    height: int | None = None
    fps: int | None = None
    mode_explicit: bool = False
    depth: bool = False

    @property
    def color_key(self) -> str:
        return self.entry

    @property
    def depth_key(self) -> str | None:
        return self.entry + DEPTH_SUFFIX if self.depth else None

    @property
    def keys(self) -> tuple[str, ...]:
        """This camera's stream keys: color, then depth when enabled."""
        return (self.entry,) if self.depth_key is None else (self.entry, self.depth_key)


def is_depth_key(key: str) -> bool:
    return key.endswith(DEPTH_SUFFIX)


def base_key(key: str) -> str:
    """The entry (color key) a stream key belongs to."""
    return key[: -len(DEPTH_SUFFIX)] if is_depth_key(key) else key


def is_realsense_source(source: str) -> bool:
    lowered = source.lower()
    return lowered == REALSENSE_SCHEME or lowered.startswith(
        (REALSENSE_SCHEME + ":", REALSENSE_SCHEME + "+")
    )


def parse_camera_entry(entry: str) -> CameraSpec:
    """Parse one entry; raises ``ValueError`` naming the entry when malformed."""

    raw = entry.strip()
    if not raw:
        raise ValueError("camera entry is empty")
    shown = redact_credentials(raw)  # what error messages may quote
    if is_depth_key(raw):
        raise ValueError(
            f"camera entry {shown!r}: {DEPTH_SUFFIX!r} is reserved for depth stream keys"
        )
    name: str | None = None
    source = raw
    head, separator, tail = raw.partition("=")
    if separator and _NAME.fullmatch(head):
        name, source = head, tail.strip()
        if not source:
            raise ValueError(f"camera entry {shown!r}: nothing after {name}=")
    key = f"{name}={strip_credentials(source)}" if name else strip_credentials(source)
    if is_realsense_source(source):
        return _parse_realsense(key, shown, name, source)
    if source.lower().endswith(DEPTH_FLAG):
        raise ValueError(
            f"camera entry {shown!r}: +depth needs a RealSense source ({REALSENSE_GRAMMAR}); "
            "an OpenCV camera has no depth stream"
        )
    return CameraSpec(entry=key, name=name, kind="opencv", source=source)


def _parse_realsense(key: str, shown: str, name: str | None, source: str) -> CameraSpec:
    def bad(reason: str) -> ValueError:
        return ValueError(f"camera entry {shown!r}: {reason}; expected {REALSENSE_GRAMMAR}")

    body = source[len(REALSENSE_SCHEME) :]
    depth = body.lower().endswith(DEPTH_FLAG)
    if depth:
        body = body[: -len(DEPTH_FLAG)]
    if "+" in body:
        raise bad("the only flag is +depth, once, at the end")
    tokens = body[1:].split(":") if body else []
    if len(tokens) > 2:
        raise bad("at most one mode and one serial")
    serial: str | None = None
    mode: re.Match[str] | None = None
    for token in tokens:
        if not token:
            raise bad("each ':' must be followed by a mode or a serial")
        match = _MODE.fullmatch(token)
        if match:
            if mode is not None:
                raise bad("more than one mode")
            mode = match
        elif _MODE_LIKE.match(token):
            raise bad(f"{token!r} is not a mode; write <W>x<H> or <W>x<H>@<fps>, e.g. 640x480@30")
        elif _SERIAL.fullmatch(token):
            if serial is not None:
                raise bad("more than one serial")
            serial = token
        else:
            raise bad(f"{token!r} is neither a mode (<W>x<H>[@<fps>]) nor a serial")
    width, height, fps = DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS
    if mode is not None:
        width, height = int(mode[1]), int(mode[2])
        fps = int(mode[3]) if mode[3] else DEFAULT_FPS
        if not (0 < width <= MAX_DIMENSION and 0 < height <= MAX_DIMENSION):
            raise bad(f"{width}x{height} is outside 1x1..{MAX_DIMENSION}x{MAX_DIMENSION}")
        if not 0 < fps <= MAX_FPS:
            raise bad(f"{fps} fps is outside 1..{MAX_FPS}")
    return CameraSpec(
        entry=key,
        name=name,
        kind="realsense",
        source=source,
        serial=serial,
        width=width,
        height=height,
        fps=fps,
        mode_explicit=mode is not None,
        depth=depth,
    )


def parse_camera_entries(entries: Iterable[str]) -> list[CameraSpec]:
    """Parse entries and check them against each other."""

    specs = [parse_camera_entry(entry) for entry in entries]
    check_camera_specs(specs)
    return specs


def parse_camera_setting(value: str) -> list[CameraSpec]:
    """Parse SO101_CAMERAS: comma-separated entries; blank entries are skipped."""

    return parse_camera_entries(part.strip() for part in value.split(",") if part.strip())


def check_camera_specs(specs: Sequence[CameraSpec]) -> None:
    """Raise ``ValueError`` for entries that cannot be told apart once recorded."""

    entries: set[str] = set()
    names: dict[str, CameraSpec] = {}
    serials: dict[str, CameraSpec] = {}
    for spec in specs:
        if spec.entry in entries:
            raise ValueError(f"camera entry {spec.entry!r} is listed twice")
        entries.add(spec.entry)
        # Case-insensitive: feature names become directory names, and macOS
        # and Windows file systems do not tell front from Front.
        if spec.name is not None:
            other = names.setdefault(spec.name.casefold(), spec)
            if other is not spec:
                raise ValueError(
                    f"camera name {spec.name!r} is used by both {other.entry!r} and {spec.entry!r}"
                )
        if spec.serial is not None:
            other = serials.setdefault(spec.serial.casefold(), spec)
            if other is not spec:
                raise ValueError(
                    f"RealSense serial {spec.serial} is in both {other.entry!r} and {spec.entry!r}"
                )
    realsense = [spec for spec in specs if spec.kind == "realsense"]
    unnamed = [spec.entry for spec in realsense if spec.serial is None]
    if len(realsense) > 1 and unnamed:
        raise ValueError(
            f"camera entry {unnamed[0]!r}: with {len(realsense)} RealSense entries each needs "
            "a serial (realsense:<serial>; `make find-cameras` lists them)"
        )
    keys = stream_keys(specs)
    seen: dict[str, str] = {}
    for key, feature in stream_feature_names(keys).items():
        previous = seen.setdefault(feature.casefold(), key)
        if previous == key:
            continue
        if is_depth_key(previous) or is_depth_key(key):
            depth, named = (previous, key) if is_depth_key(previous) else (key, previous)
            raise ValueError(
                f"camera entry {named!r}: its name {feature!r} is the depth stream name of "
                f"{base_key(depth)!r}; pick another name"
            )
        raise ValueError(
            f"camera entries {previous!r} and {key!r} both record as {feature!r}; "
            "name one of them, e.g. side=<source>"
        )


def stream_keys(specs: Sequence[CameraSpec]) -> list[str]:
    """Every color key in configured order, then every depth key in configured order."""

    return [spec.color_key for spec in specs] + [
        spec.depth_key for spec in specs if spec.depth_key is not None
    ]


def _sanitized_feature_name(key: str, index: int) -> str:
    # The pre-RealSense naming, which recorded datasets depend on byte for
    # byte, except that credentials are stripped first.
    component = _FEATURE_COMPONENT.sub("_", strip_credentials(str(key))).strip("._")
    return f"camera_{index:02d}_{(component[:80] or 'camera')}"


def camera_feature_name(key: str, index: int) -> str:
    """The LeRobot feature component (``observation.images.<this>``) of a stream key.

    ``index`` is the camera's position among the color keys. A named entry
    records under its name, an unnamed RealSense under
    ``camera_<index>_realsense[_<serial>]``, and a depth key under its camera's
    name plus ``_depth``. Every other key keeps the original sanitized
    ``camera_<index>_<key>`` name. Never raises.
    """

    try:
        spec: CameraSpec | None = parse_camera_entry(base_key(key))
    except (ValueError, TypeError, AttributeError):
        spec = None
    if spec is None or (spec.name is None and spec.kind == "opencv"):
        return _sanitized_feature_name(key, index)
    if spec.name is not None:
        name = spec.name
    else:
        name = f"camera_{index:02d}_{REALSENSE_SCHEME}"
        if spec.serial is not None:
            name += f"_{spec.serial}"
    return f"{name}_depth" if is_depth_key(key) else name


def stream_feature_names(keys: Sequence[str]) -> dict[str, str]:
    """Feature component per stream key, as recorders and policy runners name them.

    A color key's index is its position among the color keys; a depth key
    reuses its color key's index. For color-only keys this equals
    ``{key: camera_feature_name(key, i) for i, key in enumerate(keys)}``.
    Raises ``ValueError`` for a depth key whose color key is absent.
    """

    index = {key: position for position, key in enumerate(k for k in keys if not is_depth_key(k))}
    names: dict[str, str] = {}
    for key in keys:
        color = base_key(key)
        if color not in index:
            raise ValueError(f"depth stream {key!r} has no color stream {color!r}")
        names[key] = camera_feature_name(key, index[color])
    return names
