# Camera timing investigation — G15, 2026-10-27

Status: exposure contributes to the slowdown, but neither icSpring camera is
qualified at 30 delivered frames/s. A software or firmware fix has not yet been
established. Do not label the negotiated frame rate as measured throughput.

## Device evidence

| Feed | USB identity | Driver modes inspected | Link |
| --- | --- | --- | --- |
| Camera 1 | `32e6:9005`, no serial | YUYV, 640 × 480, advertises 30 fps | USB 2 high speed, 480 Mb/s |
| Camera 2 | `32e6:9221`, serial present | MJPEG up to 1920 × 1080; YUYV 640 × 480; advertises 30 fps | USB 2 high speed, 480 Mb/s |
| OBSBOT | Tiny 2 Lite | Current relay delivers about 30 fps | Measured separately |
| RealSense | D435i color | Current relay delivers about 30 fps | Current connection is USB 2 |

The icSpring cameras were on different host USB buses during isolation tests.
USB high speed is a link mode, not a guarantee of payload bandwidth. A VGA
YUYV stream at 30 fps needs 18.432 MB/s before USB overhead.

## Experiments that bypass the application

The raw probe sets format, requests 1/30 s, then sets manual exposure on the
same file descriptor. It starts streaming, allows four seconds to settle,
reads controls again and measures the following eleven seconds. It immediately
requeues buffers without decoding, copying images, JPEG encoding, HTTP, UI or
MCAP writing. Other camera relays are stopped for each isolated trial.

| Camera | Raw mode | Manual exposure request | Delivered fps |
| --- | --- | --- | --- |
| 1 | YUYV 640 × 480 | 1 (nominal 100 µs) | 22.59 |
| 1 | YUYV 640 × 480 | 30 (nominal 3 ms) | 22.59 |
| 2 | MJPEG 640 × 480 | 1 | 19.95 |
| 2 | MJPEG 320 × 240 | 1 | 19.95 |
| 2 | YUYV 640 × 480 | 1 | 19.95 |

All five read back manual exposure, the requested exposure value, disabled
anti-flicker and a 1/30 s negotiated interval after streaming began. No V4L2
sequence gaps or error flags appeared. Extremely short exposure produced dark
images in earlier direct capture trials, so it is a diagnostic setting, not
an acceptable final image profile. Original automatic exposure was restored.

Earlier automatic-exposure isolation tests delivered approximately 6.27 and
4.99 fps. Brighter workspace lighting raised them to approximately 9.4 and
10.1 fps. Temporarily disabling anti-flicker, changing 50/60 Hz, lowering
resolution and switching camera 2's MJPEG/YUYV modes did not reach 30 fps.

The probe source is [camera_v4l2_probe.c](../../scripts/diagnostics/camera_v4l2_probe.c).
Compile it with `cc -O2 -Wall scripts/diagnostics/camera_v4l2_probe.c -o /tmp/so101-camera-probe`.
Its arguments are `DEVICE FOURCC [WIDTH HEIGHT EXPOSURE]`. It changes camera
controls. Use an idle workstation, stop the camera's relay, preserve the
original controls, then restore those controls and the relay afterward.
Private trial results remain under `.artifacts/`, outside Git.

## Active USB endpoint budget

A subsequent isolated capture read the actual streaming alternate setting
while the raw probe ran. Other camera services were stopped; a temporary
exclusive control reservation prevented concurrent teleop. The robot buses
remained closed and torque off. Original camera controls and all four relays
were restored afterward, and the reservation was released.

| Camera/mode | Active alternate | Endpoint payload ceiling | Payload needed at 30 fps | Measured fps |
| --- | --- | --- | --- | --- |
| 1, YUYV VGA | 1 | 24.576 MB/s | 18.432 MB/s | 22.586 |
| 2, YUYV VGA | 1 | 24.576 MB/s | 18.432 MB/s | 19.949 |
| 2, MJPEG VGA | 5 | 8.192 MB/s | 0.210 MB/s in this dark diagnostic image | 19.952 |

Both raw streams selected three 1,024-byte transactions per high-speed
microframe. The compressed stream selected one. High-speed microframes are
125 microseconds; the descriptor therefore allows the payload ceilings above.
See Microsoft's [isochronous endpoint documentation](https://learn.microsoft.com/en-us/windows-hardware/drivers/usbcon/transfer-data-to-isochronous-endpoints)
and the kernel's [UVC bandwidth controls](https://docs.kernel.org/usb/gadget_uvc.html).
These are endpoint payload ceilings, not measured aggregate bus throughput.

The raw frames were complete 614,400-byte buffers. The compressed frames in
this deliberately short-exposure test averaged about 6,989 bytes, yet reached
the same approximately 20-fps ceiling as uncompressed capture. No V4L2 buffer
error or sequence gap appeared. Thus an undersized selected alternate setting
does not explain these trials, and JPEG compression workload/bandwidth does
not explain camera 2's fixed short-exposure cap. This does not establish that
the transport is error-free; camera-specific UVC statistics are still needed
to distinguish upstream frame/packet loss from sensor/firmware pacing.

## Interpretation and remaining evidence

Application rendering, OpenCV decode and recording cannot explain the raw
probe's shortfall. Competing active cameras did not explain the isolated
shortfall either. Exposure affects delivery, but the shortest accepted exposure
still misses 30 fps. Contiguous V4L2 sequence numbers only rule out losses visible
at that layer; they do not prove that the sensor, bridge firmware or USB
transport never discards a frame earlier.

IcSpring documents automatic low-light frame-rate adjustment and programmable
sensor clocks for its [CH9005](https://www.icspring.cn/CH9005.html) and
[CH9221](https://www.icspring.cn/CH9221.html) controller families. The latter also
has a JPEG encoder. This makes controller/sensor firmware scheduling a supported
hypothesis. It does **not** identify these cameras' actual controllers or sensors:
the vendor also permits configurable USB IDs. No applicable public Linux driver
replacement or verified firmware image was found.

The kernel documents [exposure units and automatic frame-rate controls](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html)
and warns that format changes can change [streaming parameters](https://docs.kernel.org/userspace-api/media/v4l/vidioc-g-parm.html).
Neither device exposed `exposure_auto_priority` in its queried control inventory.
Do not invent that control or apply global `uvcvideo` quirks as a workaround.

To isolate the remaining boundary, obtain the exact seller SKU and its advertised
resolution/format/30-fps conditions, then collect camera-specific UVC transport
statistics or compare the same camera/mode/exposure on another capture stack.
Driver debug statistics on this host require local administrator access. Until
that evidence exists, neither “the hub is too slow” nor “the camera is defective”
is a proven conclusion.

## Operator controls and appearance

Operate → Inspect live data → Camera settings queries supported controls from
each capture owner. It exposes exposure, white balance, anti-flicker and other
available image controls, verifies writes and saves a device-specific profile.
Profiles restore after reconnect. Status includes profile identity/hash, readback,
measured capture rate, mean luma, dark fraction and highlight clipping; MCAP
camera-status messages carry these fields on subsequent recordings.

Automatic-to-manual transitions must be applied before editing inactive manual
values. Re-enabling automatic mode removes the corresponding saved manual
override. Failed writes restore and verify prior settings; an unconfirmed
rollback remains visible. Settings changes are refused during teleop or recording.

Different views and sensors cannot be photometrically matched by assigning the
same numeric exposure or Kelvin value. Compare a common neutral reference under
the actual task lighting, choose usable shutter times, then tune each device.
Current visual differences and the icSpring frame-rate shortfall remain
qualification items; the controls alone do not prove they are solved.
