import * as ort from 'onnxruntime-web';

const MODEL_PATH = '/models/katana_points_detector.onnx';
const INPUT_SIZE = 640;
const CONF_THRESHOLD = 0.25;
const NUM_QUERIES = 300;

export type WeaponDetection = {
  cx: number;   // center x, normalized 0–1
  cy: number;   // center y, normalized 0–1
  score: number;
  classId: 0 | 1;  // 0 = katana_tip, 1 = katana_grip
};

let session: ort.InferenceSession | null = null;
let sessionLoading: Promise<void> | null = null;

export async function loadWeaponDetector(): Promise<void> {
  if (session) return;
  if (sessionLoading) return sessionLoading;

  sessionLoading = (async () => {
    ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/`;
    session = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ['wasm'],
    });
  })();
  return sessionLoading;
}

export function isWeaponDetectorReady(): boolean {
  return session !== null;
}

export async function detectWeapons(
  source: HTMLVideoElement | HTMLCanvasElement,
): Promise<{ tip: WeaponDetection | null; grip: WeaponDetection | null }> {
  if (!session) return { tip: null, grip: null };

  const canvas = document.createElement('canvas');
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(source, 0, 0, INPUT_SIZE, INPUT_SIZE);
  const pixels = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;

  const tensor = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const area = INPUT_SIZE * INPUT_SIZE;
  for (let i = 0; i < area; i++) {
    tensor[i]          = pixels[i * 4]     / 255;
    tensor[area + i]   = pixels[i * 4 + 1] / 255;
    tensor[area * 2 + i] = pixels[i * 4 + 2] / 255;
  }

  const input = new ort.Tensor('float32', tensor, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const results = await session.run({ images: input });
  const data = results['output0'].data as Float32Array;
  // shape [1, 300, 6]: each row = [cx, cy, w, h, score_tip, score_grip]

  let bestTip: WeaponDetection | null = null;
  let bestGrip: WeaponDetection | null = null;

  for (let i = 0; i < NUM_QUERIES; i++) {
    const base = i * 6;
    const cx        = data[base];
    const cy        = data[base + 1];
    const scoreTip  = data[base + 4];
    const scoreGrip = data[base + 5];

    if (scoreTip > CONF_THRESHOLD && (!bestTip || scoreTip > bestTip.score)) {
      bestTip = { cx, cy, score: scoreTip, classId: 0 };
    }
    if (scoreGrip > CONF_THRESHOLD && (!bestGrip || scoreGrip > bestGrip.score)) {
      bestGrip = { cx, cy, score: scoreGrip, classId: 1 };
    }
  }

  return { tip: bestTip, grip: bestGrip };
}
