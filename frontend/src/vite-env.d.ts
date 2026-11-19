/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_MEDIAPIPE_WASM_URL?: string;
  readonly VITE_HAND_LANDMARKER_MODEL_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
