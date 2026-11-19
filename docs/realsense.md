# Intel RealSense cameras

An Intel RealSense is a camera source like an OpenCV index. Its color stream
feeds the live view, recordings and policy inputs. Its depth stream is opt-in
(`+depth`). The code is `hardware/realsense.py` (built on pyrealsense2),
`hardware/camera_spec.py` (the `SO101_CAMERAS` grammar) and
`hardware/cameras.py` (`CameraManager`).

**Status.** The code is tested against a fake pyrealsense2 whose API surface is
checked against the real `pyrealsense2-macosx` 2.56.5 package, with no camera
attached. No RealSense has been streamed through it yet. The first platform to
verify is the ASUS ROG Zephyrus G15 on Ubuntu 24.04 LTS (x86-64, Python 3.12,
the official `pyrealsense2` 2.56.x wheel), with the camera plugged straight
into a USB 3 port; [First test on the G15](#first-test-on-the-g15) is the
checklist. The Windows and macOS sections are unverified.

## Supported models

The suite targets the USB D400 cameras: **D405, D415, D435, D435i and D455**.
Nothing model-specific is hard-coded; modes, depth units and firmware come
from the device through librealsense.

| Model | Status here | librealsense v2.56.5 release notes |
|---|---|---|
| D405, D415, D435, D435i, D455 | targets, not yet run on hardware | D400 firmware 5.17.0.10 or later with SDK v2.56.5 or later |
| L515 | not supported by the 2.56 line | validated up to v2.50.0; v2.54.2 "supports but not validated". The L500 code is gone from v2.55.1 on (`src/l500/` exists at tag v2.54.2, not at v2.55.1). |
| D555 | untested | "supported but still not fully validated" (firmware 7.56.19918.835, SDK v2.56.5 or later) |
| D457 | untested | not listed |

Source: <https://github.com/realsenseai/librealsense/releases/tag/v2.56.5>. The
repository moved from `IntelRealSense/librealsense` to
`realsenseai/librealsense`, and the old URLs redirect.

## Install

The `realsense` extra installs pyrealsense2, the official Python bindings of
librealsense. On macOS it installs the community build `pyrealsense2-macosx`,
which provides the same `pyrealsense2` module (published on PyPI by the
community maintainer cansik, github.com/cansik/pyrealsense2-macosx, not by
Intel or RealSense):

```toml
realsense = [
    "pyrealsense2>=2.56.4,<2.57.0 ; sys_platform != 'darwin'",
    "pyrealsense2-macosx>=2.54.2,<2.57.0 ; sys_platform == 'darwin'",
]
```

These are the distributions of LeRobot 0.6.1's `intelrealsense` extra, within
its version ranges. PyPI wheels (checked 2026-10):

| Distribution | Wheels |
|---|---|
| `pyrealsense2` 2.56.4, 2.56.5 | cp312 and cp313 for win_amd64 and manylinux x86-64; no Linux aarch64 wheel for Python 3.12 or 3.13, and no cp314. PyPI versions carry a build number (2.56.5 is `2.56.5.9235`), which the range above matches. |
| `pyrealsense2-macosx` 2.56.5 | macOS 15+ arm64, cp39 to cp314 |
| `pyrealsense2-macosx` 2.54.2 | macOS 12+ universal2, cp38 to cp312 |

So a Linux aarch64 computer (a Jetson, for example) gets no wheel from the
extra and needs librealsense built from source with its Python bindings.

### Linux: the ROG Zephyrus G15 (Ubuntu 24.04, primary)

1. Install the suite with RealSense support. The platform installer adds the
   extra, picks a Python that has a wheel, and offers to install the udev
   rules (it asks before using sudo):

   ```bash
   install/install.sh --realsense
   ```

   By hand, in the suite's venv: `pip install "so101[realsense]"`. `so101 doctor`
   reports the pyrealsense2 install, missing udev rules and the cameras it
   lists.

2. Install the udev rules, if the installer did not. librealsense's Linux
   guide makes them a setup step; without them, opening the camera can fail
   with permission errors. The suite ships librealsense's rules file
   (`99-realsense-libusb.rules` from tag v2.56.5); its installed path is
   `so101.hardware.realsense.udev_rules_path()`:

   ```bash
   RULES="$(python -c 'from so101.hardware.realsense import udev_rules_path; print(udev_rules_path())')"
   sudo cp "$RULES" /etc/udev/rules.d/
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```

   Then unplug and replug the camera. If Intel's apt package
   `librealsense2-udev-rules` is installed, it already provides the rules;
   copying the file as well makes RealSense Viewer warn "Multiple realsense
   udev-rules were found!".

3. Plug the camera straight into a USB 3 port with a USB 3 cable.

4. On the real backend, run `so101 find-cameras` (or `make find-cameras`;
   `--json` for a machine-readable report). It lists the cameras librealsense sees, with
   paste-ready entries, and on Linux also the RealSense devices on the USB bus
   with their negotiated link speed (from `/sys/bus/usb/devices`, readable
   without the udev rules). 5000 Mbit/s or more is USB 3; 480 Mbit/s is USB 2.
   A camera on the bus that librealsense does not list usually means missing
   udev rules; if they are installed, replug it and check `sudo dmesg` for
   uvcvideo errors.

librealsense v2.56.5's release notes list Ubuntu 24.04 LTS, with kernels
6.2, 6.5 and 6.8 among others, as supported.

**Depth on stock Ubuntu 24.04 is unverified.** Nobody has streamed depth
through the suite on Ubuntu 24.04's stock kernel 6.8 with the wheel's V4L2
backend and no DKMS patches yet. librealsense's v2.56.5 Linux guide says depth
cameras need its patched kernel modules; its v2.58.4 guide says most cameras
are supported by the stock kernel and only per-frame metadata needs the
patches (the suite does not read metadata). The G15 test below settles it.

Kernel modules: the PyPI wheel uses librealsense's V4L2 backend (the kernel's
uvcvideo driver), not its libusb backend; its Linux x86-64 wheel contains the
V4L2 backend classes and none of the libusb-only ones. librealsense's Linux
guide says depth cameras need its patched kernel modules, and its DKMS package
(`librealsense2-dkms`) covers the Ubuntu LTS HWE kernels 5.15, 5.19 and 6.5.
Whether the stock Ubuntu 24.04 kernel streams color and depth from the wheel
without them is not verified yet; check that first on the G15. (librealsense
documents that the unpatched uvcvideo driver gives no frame metadata; the suite
does not read metadata.)

### Windows (unverified)

`install\install.ps1 -RealSense`, or `pip install "so101[realsense]"`. There
are no udev rules on Windows. Each camera stream (color, depth) can be used by
only one program at a time, so close RealSense Viewer before starting the
suite.

### macOS (unverified)

`pip install "so101[realsense]"` installs `pyrealsense2-macosx`: 2.56.5 on
macOS 15 or later on Apple silicon, else 2.54.2. librealsense's v2.56.5
release notes call macOS "compilable, but not validated", and its macOS guide
(from v2.57.7) says sudo is required for USB access on macOS 12 and later. None
of this has been tried here. Without depth, the camera's color stream also works as an
ordinary OpenCV camera: put its index from `make find-cameras` in
`SO101_CAMERAS`.

## First test on the G15

With the camera on one of the G15's USB 3 ports and the udev rules installed:

1. `so101 doctor`: the RealSense check shows the pyrealsense2 version and the
   camera, and a missing udev rules file is reported with the commands.
2. `so101 find-cameras`: expect the camera under "Intel RealSense" with its
   serial, firmware and `USB 3.x`, the same camera under "USB bus (Linux
   sysfs)" at 5000 Mbit/s or more, and no `!` warning. Note the serial.
3. Color: `SO101_CAMERAS=realsense:<serial> so101 preflight`. The `cameras`
   check should pass with "opened and read 1 camera(s)".
4. Depth: `SO101_CAMERAS=realsense:<serial>+depth so101 preflight --json`. In
   the `cameras` check, `measured.probes[0].streams.depth` should show a
   `valid_fraction` well above 0.10 with the camera aimed at the workspace,
   and a `median_mm` close to the tape-measured distance to the table.
5. A five-second stream from Python, in the suite's venv:

   ```python
   import time
   import numpy as np
   from so101.hardware.cameras import CameraManager

   entry = "realsense:<serial>+depth"
   with CameraManager([entry]) as cams:
       end, reads = time.monotonic() + 5, 0
       while time.monotonic() < end:
           frames = cams.read()
           reads += 1
       color, depth = frames[entry], frames[entry + "#depth"]
       valid = depth[depth > 0]
       print(color.shape, color.dtype, depth.shape, depth.dtype)
       print(f"{reads} reads, {valid.size / depth.size:.0%} valid, median {np.median(valid):.0f} mm")
   ```

   Expect `(480, 640, 3) uint8 (480, 640, 1) uint16`. `read()` returns the
   newest frameset without waiting for a new one, so `reads` counts calls,
   not camera frames. A `CameraUnavailableError` names the fix.
6. If step 4 or 5 fails with timeouts or no depth while color works, the stock
   kernel is the first suspect (above): note the kernel (`uname -r`) and the
   error, and report it.

## SO101_CAMERAS

`SO101_CAMERAS` is a comma-separated list of entries. Each entry is
`[<name>=]<source>`.

| Entry | Meaning |
|---|---|
| `0`, `/dev/video2`, `rtsp://host/stream` | an OpenCV camera, exactly as before RealSense support |
| `realsense` | the only attached RealSense, color 640x480 at 30 fps |
| `realsense:123622270882` | the RealSense with that serial |
| `realsense:1280x720@15` | an explicit color mode (`<W>x<H>` alone means 30 fps) |
| `realsense:123622270882:848x480+depth` | serial and mode in either order, plus depth |
| `wrist=realsense:123622270882+depth` | a named entry |

Rules:

- `<name>` matches `^[A-Za-z][A-Za-z0-9_]{0,31}$` and names the LeRobot
  feature: `wrist=...` records `observation.images.wrist` (and
  `observation.images.wrist_depth` with `+depth`). Text before the first `=`
  is a name only if it matches, so `rtsp://h/s?x=1` and GStreamer pipelines
  such as `v4l2src device=/dev/video0 ! ...` are still plain sources.
- Unnamed entries keep their old feature names (`camera_00_0`,
  `camera_01_dev_video2`, ...). An unnamed RealSense records as
  `camera_<index>_realsense_<serial>` (or `camera_<index>_realsense` without a
  serial), where the index is its position in the list.
- `realsense` is case-insensitive. A token is a mode (`<W>x<H>[@<fps>]`) or a
  serial (4 to 32 letters and digits), each at most once.
- An explicit mode the camera does not offer is an error that lists the rgb8
  modes it does offer. Without an explicit mode, if 640x480 at 30 fps is not
  offered, the nearest rgb8 mode at 30 fps is used (else the nearest at any
  fps), and the choice is logged.
- With two or more RealSense entries, each needs a serial. Duplicate entries,
  names, serials, a name equal to another camera's depth feature
  (`wrist_depth`), and two entries that would record under the same feature
  name are rejected when the settings load, naming the entry.
- `+depth` on an OpenCV source is an error: depth needs a RealSense entry.

Each entry is a stream key; a `+depth` entry also yields `<entry>#depth`.
`CameraManager.read()` and `Settings.camera_streams` list every color stream
in configured order, then every depth stream in configured order.

Credentials in a source (`rtsp://user:password@host/...`, or `user-pw=`,
`password=`, `pwd=` values) are stripped from the stream key, so feature
names, status, logs, events, preflight reports and dataset metadata never
carry them, and error messages show `***` instead. Only the camera capture
sees them.

Under `SO101_BACKEND=sim`, `SO101_CAMERAS` is ignored (with one log line) and
`CameraManager` opens the simulated cameras in `SO101_SIM_CAMERAS` (`front`,
`wrist_cam`), which record under their own names. No OpenCV or RealSense
module is imported, and no real device is opened.

Do not configure the same camera twice. A RealSense also appears as ordinary
(UVC) camera indices, so `realsense:<serial>` and its OpenCV index would be
two clients of one camera. `make find-cameras` warns about this.

## Depth

- **Format.** HxWx1 `uint16` millimetres. 0 means no data.
- **Alignment.** Depth is aligned to the color frame (`rs.align` to color), so
  it has the color stream's width and height, and color and depth returned by
  one read come from the same frameset.
- **Units.** librealsense reports a depth scale in metres per unit, and the
  suite converts with it: `round(z * scale * 1000)`, clipped to 0..65535. It
  never assumes 1 mm units. librealsense's default is 0.001 m per unit, and
  0.0001 m (0.1 mm) for the D405 (`src/ds/d400/d400-device.cpp`, v2.56.5).
- **Mode.** Depth is z16 at the color fps: the color size if offered, else
  640x480, else 848x480, else the nearest size at that fps.
- **Live view.** Depth tiles are colorized: each frame is ranged on its own
  2nd to 98th percentile of valid pixels, from blue (near) to red (far), with
  black for no data. Colors show relative depth within one frame, not
  distance.

## Recording format (LeRobot 0.6.1)

What LeRobot 0.6.1 does with a depth feature:

- A feature of shape (H, W, 1) is a depth map when its `info` has
  `is_depth_map` (`datasets/dataset_metadata.py`); LeRobot's own feature
  helper sets it from the shape (`utils/feature_utils.py`). The suite's
  recorder sets `info.is_depth_map` on every (H, W, 1) `uint16` frame and
  keeps it `uint16`; a depth stream records as `<camera name>_depth`.
- `uint16` frames give `depth_unit` "mm" (`configs/video.py`,
  `infer_depth_unit`).
- Image mode writes each frame as a 16-bit TIFF with `compression="raw"`:
  lossless.
- Video mode quantizes depth to 12 bits on a log scale (`depth_min` 0.01 m,
  `depth_max` 10 m, `shift` 3.5 m, `use_log`), then encodes HEVC with libx265,
  `gray12le` and `x265-params lossless=1`. The quantization step at depth d is
  (d + 3500 mm) x ln(13500/3510) / 4095: about 1.25 mm at 0.3 m, 1.32 mm at
  0.5 m and 1.48 mm at 1.0 m. Running LeRobot's own `quantize_depth` and
  `dequantize_depth` over every millimetre value gave at most 1 mm round-trip
  error from 0.3 m to 4 m, and at most 2 mm up to 10 m.
- In video mode, no-data pixels (0) come back as 10 mm (`depth_min`), and
  anything beyond 10 m comes back as 10 m. Treat 10 mm or less as no data
  after a video round trip.
- PyAV 15.1.0 (LeRobot's `av>=15,<16`) bundles libx265 in its Windows and
  Linux wheels. Its 12-bit gray mode was exercised only on macOS. LeRobot
  sets up both encoders whenever it opens a dataset for writing; when PyAV
  lacks one, the recorder says so instead of LeRobot's bare "Unsupported
  video codec".

## Training

LeRobot 0.6.1 turns every image feature into a VISUAL policy input, so a
depth map would reach the presets' 3-channel image backbones as (1, H, W).
`so101 train` therefore leaves depth maps out of a local LeRobot dataset's
policy inputs (it passes `--policy.input_features` without them). Use
`--include-depth` (the API's `include_depth`) to keep them, for a policy that
can take them. A Hugging Face Hub dataset is not inspected.

At rollout, a LeRobot policy is given only the inputs its checkpoint
declares, so a color-only policy never sees a depth stream.

## USB bandwidth

Color and depth from one camera share its USB link. The suite does not
publish bandwidth figures. Give each camera its own USB 3 port where you can,
avoid unpowered hubs, and expect fewer and smaller modes on a USB 2 link.
`so101 find-cameras` and the preflight `cameras` check warn about a USB 2 link.

## Firmware

The preflight `cameras` check and `make find-cameras` warn when a camera's
firmware is older than the version librealsense recommends for it. To update:

- RealSense Viewer can update firmware, or
- `rs-fw-update -l` lists the cameras, then
  `rs-fw-update -s <serial> -f Signed_Image_UVC_<version>.bin` flashes one
  (`tools/fw-update/readme.md` in librealsense). On Ubuntu, Intel's apt package
  `librealsense2-utils` provides the tools.

A camera stuck in recovery mode enumerates with USB id 8086:0adb (or another
recovery id) and needs `rs-fw-update -r -f <image>`. The suite reports a
recovery-mode camera on the USB bus. RealSense's USB 2 support article says
firmware updates are not available over USB 2, so update on a USB 3 port.

## Troubleshooting

| Message | Fix |
|---|---|
| `pyrealsense2 is not importable` | Install the extra (above). |
| `no RealSense camera is attached; a RealSense is on the USB bus but not opened by librealsense` | Linux: install the udev rules, then unplug and replug the camera. |
| `... in recovery (DFU) mode on the USB bus` | Update the firmware (above). |
| `no RealSense camera is attached` and nothing on the USB bus | Check the cable and port; on Linux, `lsusb` should list an Intel `8086:` device. |
| `N RealSense cameras are attached and the entry names none` | Use one `realsense:<serial>` entry per camera; `make find-cameras` lists them. |
| `the camera has no rgb8 color mode ...; it offers ...` | Put one of the listed modes in the entry, or leave the mode out. |
| `USB 2 link` (warning or error hint) | Plug the camera straight into a USB 3 port with a USB 3 cable. |
| `Couldn't resolve requests` | The modes cannot stream together: use a smaller or slower color mode, or USB 3. |
| `busy`, or the start fails while another program runs | Each stream can be used by only one program at a time: close RealSense Viewer, another `so101` server, or other capture tools. |
| `RS2_USB_STATUS_ACCESS`, `Permission denied`, `failed to set power state` (Linux) | Install the udev rules; for the last one, also close other programs using the camera. |
| `the newest frame is N s old`, `the capture thread stopped` | The stream stalled: check the cable, hub and port. Connecting retries three times, with a hardware reset before the last try. |
| `only N% of depth pixels are valid` (preflight warning) | Aim the camera at the workspace from within its depth range, clear the lenses, and light the scene. |
| `SO101_BACKEND=sim: ... is not a simulated camera` | The sim backend never opens a real camera; its cameras are `SO101_SIM_CAMERAS`. Use `SO101_BACKEND=real` for real cameras. |
| `LeRobot could not set up its video encoders` | Install the PyAV LeRobot pins (`av>=15,<16`) in the suite's venv. |

`so101 find-cameras` prints both sources and any warnings. `so101 preflight`
probes each configured camera on its own (when no server is running) and
reports device facts and depth statistics in its `cameras` check.
