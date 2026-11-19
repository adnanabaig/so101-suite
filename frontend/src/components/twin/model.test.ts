import { describe, expect, test } from "vitest";
import { Vector3 } from "three";
import fixture from "./model-fixture.json";
import { modelFrames, type TwinModel } from "./model";
const model = fixture.model as unknown as TwinModel;

describe("vendored SO-101 transforms match backend FK", () => {
  for (const [index, sample] of fixture.cases.entries()) {
    test(`all body pivots match backend pose ${index}`, () => {
      const frames = modelFrames(model, sample.joints);
      for (const frame of sample.frames.filter((frame) => frame.name !== "gripperframe")) {
        const position = new Vector3().setFromMatrixPosition(frames.get(frame.name)!);
        position.toArray().forEach((coordinate, axis) => expect(coordinate).toBeCloseTo(frame.position_m[axis], 8));
      }
    });
  }
  test(".pos telemetry keys and unadorned keys produce the same pose", () => {
    const sample = fixture.cases[1].joints;
    const withSuffix = Object.fromEntries(Object.entries(sample).map(([joint, value]) => [`${joint}.pos`, value]));
    expect(modelFrames(model, sample)).toEqual(modelFrames(model, withSuffix));
  });
});
