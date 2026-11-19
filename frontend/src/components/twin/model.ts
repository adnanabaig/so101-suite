import { Matrix4, Quaternion, Vector3 } from "three";
import { jointValue, type TwinJoint } from "./types";

export type Triple = [number, number, number];
export type Wxyz = [number, number, number, number];
export interface ModelGeom {
  kind: string;
  mesh?: string;
  pos: Triple;
  quat: Wxyz;
  rgba: [number, number, number, number];
  size?: number[];
}
export interface ModelBody {
  name: string;
  parent: string | null;
  pos: Triple;
  quat: Wxyz;
  joint: { name: string; axis: Triple; range: [number, number] | null } | null;
  geoms: ModelGeom[];
}
export interface TwinModel {
  bodies: ModelBody[];
  meshes: Record<string, string>;
  joint_map: Record<string, { mjcf_joint: string; scale: number; offset: number }>;
}

export function quaternionWxyz(quat: Wxyz): Quaternion {
  return new Quaternion(quat[1], quat[2], quat[3], quat[0]);
}

export function bodyQuaternion(body: ModelBody, model: TwinModel, observation?: Readonly<Record<string, number>>): Quaternion {
  const base = quaternionWxyz(body.quat);
  if (!body.joint) return base;
  const mapping = Object.entries(model.joint_map).find(([, map]) => map.mjcf_joint === body.joint?.name);
  if (!mapping) throw new Error(`Missing joint mapping: ${body.joint.name}`);
  const [name, map] = mapping;
  const value = jointValue(observation, name as TwinJoint);
  const angle = value === null ? 0 : map.scale * value + map.offset;
  return base.multiply(new Quaternion().setFromAxisAngle(new Vector3(...body.joint.axis).normalize(), angle));
}

export function modelFrames(model: TwinModel, observation: Readonly<Record<string, number>>): Map<string, Matrix4> {
  const frames = new Map<string, Matrix4>();
  for (const body of model.bodies) {
    const local = new Matrix4().compose(new Vector3(...body.pos), bodyQuaternion(body, model, observation), new Vector3(1, 1, 1));
    if (body.parent) {
      const parent = frames.get(body.parent);
      if (!parent) throw new Error(`Missing model parent: ${body.parent}`);
      local.premultiply(parent);
    }
    frames.set(body.name, local);
  }
  return frames;
}
