import { useEffect, useMemo, useState } from "react";
import { useLoader, useThree } from "@react-three/fiber";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { Color, Vector3 } from "three";
import { Html, Line } from "@react-three/drei";
import { TWIN_JOINTS } from "./types";
import { bodyQuaternion, quaternionWxyz, modelFrames, type ModelBody, type ModelGeom, type TwinModel } from "./model";
import { PRESENCE_OPACITY, type TwinPresence } from "./kinematics";

let modelRequest: Promise<TwinModel> | undefined;
function loadModel() {
  modelRequest ??= fetch("/api/simulation/model").then((response) => {
    if (!response.ok) throw new Error("SO-101 geometry could not be loaded");
    return response.json() as Promise<TwinModel>;
  }).catch((error: unknown) => { modelRequest = undefined; throw error; });
  return modelRequest;
}

function Visual({ geom, model, presence }: { geom: ModelGeom; model: TwinModel; presence: TwinPresence }) {
  const geometry = useLoader(STLLoader, model.meshes[geom.mesh!]);
  const color = useMemo(() => presence === "none" ? new Color("#64748b") : new Color(...geom.rgba.slice(0, 3) as [number, number, number]), [geom.rgba, presence]);
  const opacity = PRESENCE_OPACITY[presence] * geom.rgba[3];
  return <mesh geometry={geometry} position={geom.pos} quaternion={quaternionWxyz(geom.quat)} castShadow receiveShadow>
    <meshStandardMaterial color={color} transparent={opacity < 1} opacity={opacity} depthWrite={opacity >= 1} roughness={0.55} metalness={0.12} />
  </mesh>;
}

function Body({ body, model, observation, presence }: { body: ModelBody; model: TwinModel; observation?: Readonly<Record<string, number>>; presence: TwinPresence }) {
  const invalidate = useThree((state) => state.invalidate);
  const quaternion = useMemo(() => bodyQuaternion(body, model, observation), [body, model, observation]);
  useEffect(() => { invalidate(); }, [quaternion, invalidate, presence]);
  return <group position={body.pos} quaternion={quaternion}>
    {body.joint && <>
      <mesh><sphereGeometry args={[0.006, 12, 12]} /><meshBasicMaterial color="#22d3ee" depthTest={false} /></mesh>
      <Html center position={[0, 0, 0.012]} style={{ pointerEvents: "none" }}><span className="rounded bg-slate-950/90 px-1 text-[10px] font-bold text-cyan-300">{TWIN_JOINTS.indexOf(body.joint.name as typeof TWIN_JOINTS[number]) + 1}</span></Html>
    </>}
    {body.geoms.filter((geom) => geom.kind === "mesh").map((geom, index) => <Visual key={index} geom={geom} model={model} presence={presence} />)}
    {model.bodies.filter((child) => child.parent === body.name).map((child) => <Body key={child.name} body={child} model={model} observation={observation} presence={presence} />)}
  </group>;
}

function MotorLines({ model, observation, presence }: { model: TwinModel; observation: Readonly<Record<string, number>>; presence: TwinPresence }) {
  const points = useMemo(() => {
    const frames = modelFrames(model, observation);
    return model.bodies.filter((body) => body.joint).map((body) => new Vector3().setFromMatrixPosition(frames.get(body.name)!));
  }, [model, observation]);
  return <Line points={points} color="#22d3ee" lineWidth={2} transparent opacity={PRESENCE_OPACITY[presence]} />;
}

export default function MjcfArm({ observation, presence }: { observation?: Readonly<Record<string, number>>; presence: TwinPresence }) {
  const invalidate = useThree((state) => state.invalidate);
  const [model, setModel] = useState<TwinModel | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [history, setHistory] = useState({ observation, known: observation ?? {} });
  if (observation !== history.observation) {
    setHistory({ observation, known: { ...history.known, ...Object.fromEntries(Object.entries(observation ?? {}).filter(([, value]) => Number.isFinite(value))) } });
  }
  useEffect(() => {
    let cancelled = false;
    void loadModel().then((value) => { if (!cancelled) { setModel(value); invalidate(); } }, (caught: unknown) => { if (!cancelled) setError(caught); });
    return () => { cancelled = true; };
  }, [invalidate]);
  if (error) throw error;
  if (!model) return <Html center><span className="text-xs text-slate-300">Loading arm geometry…</span></Html>;
  return <group scale={8} rotation={[-Math.PI / 2, 0, 0]}>
    <MotorLines model={model} observation={history.known} presence={presence} />
    {model.bodies.filter((body) => body.parent === null).map((body) => <Body key={body.name} body={body} model={model} observation={history.known} presence={presence} />)}
  </group>;
}
