import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import {
  DrawingUtils,
  FilesetResolver,
  PoseLandmarker,
  type PoseLandmarkerResult
} from '@mediapipe/tasks-vision';

type AngleItem = {
  key: string;
  label: string;
  value: number | null;
};

type LandmarkLike = {
  x: number;
  y: number;
  z?: number;
};

const WASM_FILE_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_ASSET_PATH =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';

const ANGLE_CONFIG = [
  { key: 'leftShoulder', label: '左肩', points: [13, 11, 23] },
  { key: 'rightShoulder', label: '右肩', points: [14, 12, 24] },
  { key: 'leftElbow', label: '左肘', points: [11, 13, 15] },
  { key: 'rightElbow', label: '右肘', points: [12, 14, 16] },
  { key: 'leftHip', label: '左股関節', points: [11, 23, 25] },
  { key: 'rightHip', label: '右股関節', points: [12, 24, 26] },
  { key: 'leftKnee', label: '左膝', points: [23, 25, 27] },
  { key: 'rightKnee', label: '右膝', points: [24, 26, 28] }
] as const;

function Spinner({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-slate-700 bg-slate-800/70 px-4 py-3 text-sm text-slate-200">
      <div className="h-5 w-5 animate-spin rounded-full border-2 border-green-400 border-t-transparent" />
      <span>{label}</span>
    </div>
  );
}

function loadImageElement(src: string) {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('画像の読み込みに失敗しました。'));
    image.src = src;
  });
}

function calculateAngle(a: LandmarkLike, b: LandmarkLike, c: LandmarkLike) {
  const vectorA = {
    x: a.x - b.x,
    y: a.y - b.y,
    z: (a.z ?? 0) - (b.z ?? 0)
  };
  const vectorB = {
    x: c.x - b.x,
    y: c.y - b.y,
    z: (c.z ?? 0) - (b.z ?? 0)
  };

  const dot = vectorA.x * vectorB.x + vectorA.y * vectorB.y + vectorA.z * vectorB.z;
  const magnitudeA = Math.hypot(vectorA.x, vectorA.y, vectorA.z);
  const magnitudeB = Math.hypot(vectorB.x, vectorB.y, vectorB.z);

  if (magnitudeA === 0 || magnitudeB === 0) {
    return null;
  }

  const cosine = Math.min(1, Math.max(-1, dot / (magnitudeA * magnitudeB)));
  return (Math.acos(cosine) * 180) / Math.PI;
}

function buildAngles(result: PoseLandmarkerResult): AngleItem[] {
  const landmarks = result.landmarks[0];

  if (!landmarks) {
    return ANGLE_CONFIG.map(({ key, label }) => ({ key, label, value: null }));
  }

  return ANGLE_CONFIG.map(({ key, label, points }) => {
    const [aIndex, bIndex, cIndex] = points;
    const a = landmarks[aIndex];
    const b = landmarks[bIndex];
    const c = landmarks[cIndex];

    if (!a || !b || !c) {
      return { key, label, value: null };
    }

    return {
      key,
      label,
      value: calculateAngle(a, b, c)
    };
  });
}

export default function PoseAnalyzer() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [poseLandmarker, setPoseLandmarker] = useState<PoseLandmarker | null>(null);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [angles, setAngles] = useState<AngleItem[]>(() =>
    ANGLE_CONFIG.map(({ key, label }) => ({ key, label, value: null }))
  );
  const [loadingModel, setLoadingModel] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let landmarkerInstance: PoseLandmarker | null = null;

    const initialize = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_FILE_PATH);
        landmarkerInstance = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: MODEL_ASSET_PATH
          },
          runningMode: 'IMAGE',
          numPoses: 1
        });

        if (!active) {
          landmarkerInstance.close();
          return;
        }

        setPoseLandmarker(landmarkerInstance);
      } catch (error) {
        if (active) {
          setErrorMessage(error instanceof Error ? error.message : 'モデルの初期化に失敗しました。');
        }
      } finally {
        if (active) {
          setLoadingModel(false);
        }
      }
    };

    void initialize();

    return () => {
      active = false;
      landmarkerInstance?.close();
    };
  }, []);

  useEffect(() => {
    if (!poseLandmarker || !imageDataUrl) {
      return;
    }

    let active = true;

    const analyzeImage = async () => {
      setIsAnalyzing(true);
      setErrorMessage(null);

      try {
        const image = await loadImageElement(imageDataUrl);
        if (!active) {
          return;
        }

        const result = poseLandmarker.detect(image);
        if (!active) {
          return;
        }

        setAngles(buildAngles(result));

        const canvas = canvasRef.current;
        const context = canvas?.getContext('2d');

        if (!canvas || !context) {
          return;
        }

        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        context.clearRect(0, 0, canvas.width, canvas.height);

        const drawingUtils = new (DrawingUtils as any)(context);
        const poseLandmarks = result.landmarks[0];

        if (!poseLandmarks) {
          return;
        }

        drawingUtils.drawConnectors(
          poseLandmarks,
          (PoseLandmarker as any).POSE_CONNECTIONS,
          {
            color: '#34d399',
            lineWidth: 4
          }
        );
        drawingUtils.drawLandmarks(poseLandmarks, {
          color: '#86efac',
          fillColor: '#16a34a',
          lineWidth: 2,
          radius: 3
        });
      } catch (error) {
        if (active) {
          setAngles(ANGLE_CONFIG.map(({ key, label }) => ({ key, label, value: null })));
          setErrorMessage(error instanceof Error ? error.message : '画像の解析に失敗しました。');
        }
      } finally {
        if (active) {
          setIsAnalyzing(false);
        }
      }
    };

    void analyzeImage();

    return () => {
      active = false;
    };
  }, [imageDataUrl, poseLandmarker]);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (!file) {
      return;
    }

    const reader = new FileReader();
    setErrorMessage(null);

    reader.onload = () => {
      const result = reader.result;
      if (typeof result === 'string') {
        setImageDataUrl(result);
      } else {
        setErrorMessage('画像データの変換に失敗しました。');
      }
    };

    reader.onerror = () => {
      setErrorMessage('画像ファイルの読み込みに失敗しました。');
    };

    reader.readAsDataURL(file);
  };

  return (
    <section className="space-y-5 rounded-2xl border border-slate-800 bg-slate-900 p-5 text-slate-100 shadow-2xl shadow-black/30 sm:p-6">

      {/* アップロードゾーン */}
      <label className={`flex min-h-[80px] cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-5 text-center transition-colors ${
        imageDataUrl
          ? 'border-slate-700 bg-slate-800/40 hover:border-slate-600'
          : 'border-slate-700 bg-slate-950/40 hover:border-green-500/50 hover:bg-green-500/5'
      }`}>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="h-6 w-6 text-slate-500" aria-hidden="true">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <circle cx="8.5" cy="8.5" r="1.5" />
          <path d="m21 15-5-5L5 21" />
        </svg>
        <span className="text-sm text-slate-400">
          {imageDataUrl ? '別の画像を選択' : 'クリックして画像を選択'}
        </span>
        <span className="text-xs text-slate-600">JPG, PNG, HEIC など</span>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="sr-only"
        />
      </label>

      {(loadingModel || isAnalyzing) && (
        <Spinner label={loadingModel ? 'モデル読み込み中...' : '画像を解析中...'} />
      )}

      {errorMessage && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/8 px-4 py-3 text-sm text-red-300">
          {errorMessage}
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-[minmax(0,2fr)_minmax(260px,1fr)]">
        {/* 画像 + 骨格オーバーレイ */}
        <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-3">
          {imageDataUrl ? (
            <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-black">
              <img
                src={imageDataUrl}
                alt="アップロードされた解析対象"
                className="block h-auto w-full"
              />
              <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
            </div>
          ) : (
            <div className="flex min-h-72 items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/50 text-sm text-slate-600">
              骨格検出結果がここに表示されます
            </div>
          )}
        </div>

        {/* 関節角度パネル */}
        <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
          <div className="mb-4 border-b border-slate-800 pb-3">
            <h3 className="text-sm font-semibold text-slate-200">関節角度</h3>
            <p className="mt-0.5 text-xs text-slate-500">8箇所の推定角度</p>
          </div>

          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-1">
            {angles.map((item) => (
              <div
                key={item.key}
                className={`rounded-xl border px-3 py-2.5 transition-colors ${
                  item.value != null
                    ? 'border-slate-700/60 bg-slate-900/60'
                    : 'border-slate-800 bg-slate-900/30'
                }`}
              >
                <div className="text-xs text-slate-500">{item.label}</div>
                <div className={`mt-0.5 text-xl font-semibold tabular-nums ${
                  item.value != null ? 'text-cyan-300' : 'text-slate-700'
                }`}>
                  {item.value == null ? '--' : `${item.value.toFixed(1)}°`}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
