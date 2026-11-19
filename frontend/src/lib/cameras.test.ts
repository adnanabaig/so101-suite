import { describe, expect, test } from "vitest";
import { DEPTH_SUFFIX, baseKey, cameraLabel, cameraName, isDepthKey, realsenseLabel } from "./cameras";

describe("camera stream keys (docs/realsense.md)", () => {
  test("a depth key is its entry plus #depth", () => {
    expect(isDepthKey(`realsense:123456789012+depth${DEPTH_SUFFIX}`)).toBe(true);
    expect(isDepthKey("realsense:123456789012+depth")).toBe(false);
    expect(baseKey("front=realsense:front+depth#depth")).toBe("front=realsense:front+depth");
    expect(baseKey("0")).toBe("0");
  });

  test("a name is the text before the first = only when it matches the name grammar", () => {
    expect(cameraName("front=realsense:123456789012+depth")).toBe("front");
    expect(cameraName("wrist_cam=/dev/video2")).toBe("wrist_cam");
    // Not names: the whole entry is the source.
    expect(cameraName("rtsp://cam.local/stream?x=1")).toBeNull();
    expect(cameraName("v4l2src device=/dev/video0 ! videoconvert ! appsink")).toBeNull();
    expect(cameraName("2front=0")).toBeNull();
    expect(cameraName("realsense:123456789012")).toBeNull();
  });

  test("a tile is labelled with the name, else 'RealSense <serial>', else the entry, plus ' depth'", () => {
    expect(cameraLabel("0")).toBe("0");
    expect(cameraLabel("/dev/video2")).toBe("/dev/video2");
    expect(cameraLabel("rtsp://cam.local/stream?x=1")).toBe("rtsp://cam.local/stream?x=1");
    expect(cameraLabel("front=realsense:123456789012+depth")).toBe("front");
    expect(cameraLabel("front=realsense:123456789012+depth#depth")).toBe("front depth");
    expect(cameraLabel("realsense:123456789012")).toBe("RealSense 123456789012");
    expect(cameraLabel("realsense:front")).toBe("RealSense front");
    expect(cameraLabel("realsense:123456789012+depth#depth")).toBe("RealSense 123456789012 depth");
  });

  test("RealSense labels drop modes and +depth, in either token order", () => {
    expect(realsenseLabel("realsense:640x480@30:123456789012+depth")).toBe("RealSense 123456789012");
    expect(realsenseLabel("realsense:123456789012:1280x720")).toBe("RealSense 123456789012");
    expect(realsenseLabel("REALSENSE:F1234567")).toBe("RealSense F1234567");
    // Only one attached camera: no serial to show.
    expect(realsenseLabel("realsense")).toBe("RealSense");
    expect(realsenseLabel("realsense+depth")).toBe("RealSense");
    expect(realsenseLabel("realsense:848x480@15+depth")).toBe("RealSense");
    expect(cameraLabel("realsense:848x480@15+depth#depth")).toBe("RealSense depth");
  });

  test("anything the server's grammar would not call a RealSense keeps its entry", () => {
    expect(realsenseLabel("realsensefoo")).toBeNull();
    expect(realsenseLabel("realsense:a:b:c")).toBeNull();
    expect(realsenseLabel("realsense:ab")).toBeNull();
    expect(realsenseLabel("realsense+depth+depth")).toBeNull();
    expect(realsenseLabel("/dev/video0")).toBeNull();
    expect(cameraLabel("realsense:ab")).toBe("realsense:ab");
  });

  test("a spec, when the server exposes one, wins over the key", () => {
    expect(cameraLabel("realsense:123456789012+depth#depth", { name: "top" })).toBe("top depth");
    expect(cameraLabel("realsense:123456789012", { name: null, entry: "realsense:123456789012" })).toBe(
      "RealSense 123456789012",
    );
  });
});
