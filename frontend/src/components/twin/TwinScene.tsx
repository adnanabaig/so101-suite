import { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import {
  ContactShadows,
  GizmoHelper,
  GizmoViewport,
  Line,
  OrbitControls,
} from "@react-three/drei";
import * as THREE from "three";
import type { TwinPresence } from "./kinematics";
import MjcfArm from "./MjcfArm";
import type {
  RobotTwinScenario,
  RobotTwinScenarioObject,
  RobotTwinTrajectoryPoint,
} from "./types";

export type TwinFrameloop = "demand" | "never";

export interface TwinSceneProps {
  fitArm?: boolean;
  observation?: Readonly<Record<string, number>>;
  mode: string;
  presence: TwinPresence;
  scenario?: RobotTwinScenario;
  trajectory?: readonly RobotTwinTrajectoryPoint[];
  /** "demand" draws only when something changed; "never" while hidden. */
  frameloop: TwinFrameloop;
  /** Snap to each new pose instead of easing toward it. */
  reducedMotion: boolean;
}

const SCENE_SCALE = 8;

function worldPosition(position: readonly number[]): [number, number, number] {
  // Backend scene/FK coordinates are x-forward, y-left, z-up. The R3F scene
  // uses y-up and looks down -z, so keep this conversion in one place.
  return [
    (position[0] ?? 0) * SCENE_SCALE,
    (position[2] ?? 0) * SCENE_SCALE,
    -(position[1] ?? 0) * SCENE_SCALE,
  ];
}

function ScenarioObject({ object }: { object: RobotTwinScenarioObject }) {
  const size = object.size_m;
  const pose = object.pose;
  const position = worldPosition(pose);
  const basis = new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0));
  const rotation = basis.clone().multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(
    THREE.MathUtils.degToRad(pose[3] ?? 0),
    THREE.MathUtils.degToRad(pose[4] ?? 0),
    THREE.MathUtils.degToRad(pose[5] ?? 0), "ZYX",
  ))).multiply(basis.clone().invert());
  const color = object.color ?? "#f59e0b";
  const targetZone = object.role === "target_zone";

  return (
    <group position={position} quaternion={rotation}>
      <mesh castShadow={!targetZone} receiveShadow={targetZone}>
        {object.shape === "cylinder" ? (
          <cylinderGeometry
            args={[
              (size[0] ?? 0.02) * SCENE_SCALE,
              (size[0] ?? 0.02) * SCENE_SCALE,
              (size[0] ?? 0.04) * SCENE_SCALE,
              28,
            ]}
          />
        ) : object.shape === "sphere" ? (
          <sphereGeometry args={[(size[0] ?? 0.02) * SCENE_SCALE, 24, 16]} />
        ) : (
          <boxGeometry
            args={[
              (size[1] ?? size[0] ?? 0.04) * SCENE_SCALE,
              (size[2] ?? size[0] ?? 0.04) * SCENE_SCALE,
              (size[1] ?? size[0] ?? 0.04) * SCENE_SCALE,
            ]}
          />
        )}
        <meshStandardMaterial
          color={color}
          metalness={targetZone ? 0.05 : 0.2}
          roughness={targetZone ? 0.75 : 0.42}
          transparent={targetZone}
          opacity={targetZone ? 0.68 : 1}
        />
      </mesh>
    </group>
  );
}

function ScenarioScene({
  scenario,
  trajectory,
}: {
  scenario?: RobotTwinScenario;
  trajectory?: readonly RobotTwinTrajectoryPoint[];
}) {
  const trajectoryPoints = useMemo(
    () =>
      (trajectory ?? [])
        .filter((point) => point.position_m.length >= 3)
        .map((point) => worldPosition(point.position_m)),
    [trajectory],
  );

  if (!scenario && trajectoryPoints.length < 2) return null;

  const surface = scenario?.surface;
  const size = surface?.size_m ?? [0.8, 0.6, 0.04];
  const thickness = Math.max(0.08, (size[2] ?? 0.04) * SCENE_SCALE);

  return (
    <group>
      {scenario && (
        <>
          <mesh receiveShadow position={[0, -thickness / 2, 0]}>
            <boxGeometry
              args={[
                (size[0] ?? 0.8) * SCENE_SCALE,
                thickness,
                (size[1] ?? 0.6) * SCENE_SCALE,
              ]}
            />
            <meshStandardMaterial
              color={surface?.color ?? "#172033"}
              metalness={0.12}
              roughness={0.78}
            />
          </mesh>
          <group>
            {(scenario.objects ?? []).map((object) => (
              <ScenarioObject key={object.id} object={object} />
            ))}
          </group>
        </>
      )}
      {trajectoryPoints.length >= 2 && (
        <Line
          points={trajectoryPoints}
          color="#a78bfa"
          lineWidth={2.4}
          transparent
          opacity={0.9}
        />
      )}
    </group>
  );
}

export default function TwinScene({
  observation,
  presence,
  scenario,
  trajectory,
  frameloop,
  reducedMotion,
  fitArm = false,
}: TwinSceneProps) {
  return (
    <Canvas
      frameloop={frameloop}
      shadows
      dpr={[1, 1.75]}
      camera={{ position: fitArm ? [3.5, 2.9, 4.3] : [5, 4, 6], fov: 45, near: 0.1, far: 100 }}
      gl={{ antialias: true, alpha: true }}
    >
      <color attach="background" args={["#070b14"]} />
      <fog attach="fog" args={["#070b14", 7, 14]} />
      <ambientLight intensity={0.65} />
      <hemisphereLight args={["#bae6fd", "#0f172a", 1.15]} />
      <directionalLight
        castShadow
        position={[4, 7, 5]}
        intensity={2.2}
        color="#e0f2fe"
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
      />
      <pointLight position={[-4, 3, -2]} color="#22d3ee" intensity={8} distance={8} />
      <Suspense fallback={null}>
        <ScenarioScene scenario={scenario} trajectory={trajectory} />
        <MjcfArm observation={observation} presence={presence} />
        {presence !== "none" && (
          <ContactShadows position={[0, 0.01, 0]} opacity={0.45} scale={8} blur={2.4} far={5} />
        )}
      </Suspense>
      <gridHelper args={[12, 24, "#164e63", "#172033"]} position={[0, 0, 0]} />
      <OrbitControls
        makeDefault
        target={fitArm ? [0.2, 1, 0] : [1, 1.2, 0]}
        minDistance={fitArm ? 0.8 : 3.2}
        maxDistance={9}
        minPolarAngle={0.25}
        maxPolarAngle={Math.PI / 2.02}
        enableDamping={!reducedMotion}
      />
      <GizmoHelper alignment="bottom-right" margin={[72, 72]}>
        <GizmoViewport axisColors={["#fb7185", "#4ade80", "#60a5fa"]} labelColor="#e2e8f0" />
      </GizmoHelper>
    </Canvas>
  );
}
