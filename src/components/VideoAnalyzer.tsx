import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import {
  DrawingUtils,
  FilesetResolver,
  PoseLandmarker
} from '@mediapipe/tasks-vision';

const WASM_FILE_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_ASSET_PATH =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';

const FRAME_INTERVAL = 1 / 30;

// landmark 0-22: 上半身（頭・肩・腕・手首・腰）
// landmark 23-32: 下半身（膝・足首・踵・つま先）
const UPPER_BODY_MAX = 22;

type BudoMode = 'no-weapon' | 'weapon';
type LandmarkPoint = { x: number; y: number; z?: number };
type StoredFrame = { time: number; landmarks: LandmarkPoint[] | null };

const BUDO_MODES: { value: BudoMode; label: string; sub: string }[] = [
  { value: 'no-weapon', label: '武器なし', sub: '空手・テコンドー・太極拳' },
  { value: 'weapon',    label: '武器あり', sub: '居合道・剣道・薙刀・杖術' },
];

function Spinner({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-gray-700 bg-gray-800/70 px-4 py-3 text-sm text-gray-200">
      <div className="h-5 w-5 animate-spin rounded-full border-2 border-green-400 border-t-transparent" />
      <span>{label}</span>
    </div>
  );
}

export default function VideoAnalyzer() {
  const videoRef    = useRef<HTMLVideoElement | null>(null);
  const canvasRef   = useRef<HTMLCanvasElement | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const storedFramesRef   = useRef<StoredFrame[]>([]);
  const rafRef      = useRef<number | null>(null);
  const showTrailRef = useRef(true);
  const modeRef     = useRef<BudoMode>('no-weapon');

  const [videoUrl, setVideoUrl]       = useState<string | null>(null);
  const [loadingModel, setLoadingModel] = useState(true);
  const [isAnalyzing, setIsAnalyzing]   = useState(false);
  const [progress, setProgress]         = useState(0);
  const [done, setDone]                 = useState(false);
  const [showTrail, setShowTrail]       = useState(true);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [mode, setMode]                 = useState<BudoMode>('no-weapon');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  useEffect(() => { showTrailRef.current = showTrail; }, [showTrail]);
  useEffect(() => { modeRef.current = mode; }, [mode]);

  useEffect(() => {
    const video = videoRef.current;
    if (video) video.playbackRate = playbackRate;
  }, [playbackRate]);

  useEffect(() => {
    if (!done) return;
    const onKeyDown = (e: KeyboardEvent) => {
      const video = videoRef.current;
      if (!video) return;
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        video.pause();
        video.currentTime = Math.min(video.duration, video.currentTime + FRAME_INTERVAL);
      } else if (e.key === 'ArrowLeft') {
        e.preventDefault();
        video.pause();
        video.currentTime = Math.max(0, video.currentTime - FRAME_INTERVAL);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [done]);

  // モデル初期化
  useEffect(() => {
    let active = true;
    const init = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_FILE_PATH);
        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_ASSET_PATH },
          runningMode: 'VIDEO',
          numPoses: 1
        });
        if (!active) { landmarker.close(); return; }
        poseLandmarkerRef.current = landmarker;
      } catch (e) {
        if (active) setErrorMessage(e instanceof Error ? e.message : 'モデル初期化失敗');
      } finally {
        if (active) setLoadingModel(false);
      }
    };
    void init();
    return () => {
      active = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      poseLandmarkerRef.current?.close();
    };
  }, []);

  // ── 描画ヘルパー ─────────────────────────────────────

const drawFrame = (
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    landmarks: LandmarkPoint[] | null,
    frameIdx: number,
    budoMode: BudoMode,
    showTrailFlag: boolean
  ) => {
    ctx.clearRect(0, 0, w, h);

    if (landmarks) {
      const du = new (DrawingUtils as any)(ctx);
      const allConnections = (PoseLandmarker as any).POSE_CONNECTIONS as Array<{ start: number; end: number }>;
      const upperConns = allConnections.filter(c => c.start <= UPPER_BODY_MAX && c.end <= UPPER_BODY_MAX);
      const lowerConns = allConnections.filter(c => c.start > UPPER_BODY_MAX || c.end > UPPER_BODY_MAX);

      // 上半身: 両モード共通で通常表示
      du.drawConnectors(landmarks, upperConns, { color: '#34d399', lineWidth: 3 });
      du.drawLandmarks(
        landmarks.slice(0, UPPER_BODY_MAX + 1),
        { color: '#86efac', fillColor: '#16a34a', lineWidth: 2, radius: 3 }
      );

      // 下半身: 武器なし→通常、武器あり→半透明（袴で精度低）
      const lowerOpacity = budoMode === 'weapon' ? 0.4 : 1.0;
      du.drawConnectors(landmarks, lowerConns, {
        color: `rgba(52,211,153,${lowerOpacity})`, lineWidth: budoMode === 'weapon' ? 2 : 3
      });
      du.drawLandmarks(
        landmarks.slice(UPPER_BODY_MAX + 1),
        {
          color: `rgba(134,239,172,${lowerOpacity})`,
          fillColor: `rgba(22,163,74,${lowerOpacity})`,
          lineWidth: 1,
          radius: budoMode === 'weapon' ? 2 : 3
        }
      );
    }

    if (showTrailFlag) {
      drawTrailUpTo(ctx, w, h, frameIdx);
    }
  };

  const drawTrailUpTo = (
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    upToIdx: number
  ) => {
    const frames = storedFramesRef.current.slice(0, upToIdx + 1);
    const trailWidth = modeRef.current === 'weapon' ? 4 : 3;
    const drawLine = (points: (LandmarkPoint | undefined)[], color: string) => {
      const valid = points.filter((p): p is LandmarkPoint => !!p);
      if (valid.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = trailWidth;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.moveTo(valid[0].x * w, valid[0].y * h);
      for (let i = 1; i < valid.length; i++) {
        ctx.lineTo(valid[i].x * w, valid[i].y * h);
      }
      ctx.stroke();
    };
    drawLine(frames.map(f => f.landmarks?.[15]), '#67e8f9');
    drawLine(frames.map(f => f.landmarks?.[16]), '#fb923c');
  };

  // ── 再生同期ループ ────────────────────────────────────

  const startPlaybackLoop = () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    const loop = () => {
      const video  = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;
      const frames = storedFramesRef.current;
      if (frames.length > 0) {
        let idx = Math.round(video.currentTime / FRAME_INTERVAL);
        idx = Math.max(0, Math.min(idx, frames.length - 1));
        const frame = frames[idx];
        const ctx = canvas.getContext('2d');
        if (ctx && frame) {
          drawFrame(ctx, canvas.width, canvas.height, frame.landmarks, idx, modeRef.current, showTrailRef.current);
        }
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
  };

  // ── ファイル選択 ──────────────────────────────────────

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    storedFramesRef.current = [];
    setVideoUrl(URL.createObjectURL(file));
    setDone(false);
    setProgress(0);
    setErrorMessage(null);
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
  };

  // ── 解析 ─────────────────────────────────────────────

  const startAnalysis = async () => {
    const video    = videoRef.current;
    const canvas   = canvasRef.current;
    const landmarker = poseLandmarkerRef.current;
    if (!video || !canvas || !landmarker) return;

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    storedFramesRef.current = [];
    setIsAnalyzing(true);
    setDone(false);
    setProgress(0);
    setErrorMessage(null);

    await new Promise<void>((resolve) => {
      if (video.readyState >= 1 && isFinite(video.duration) && video.duration > 0) { resolve(); return; }
      const onMeta = () => { if (isFinite(video.duration) && video.duration > 0) resolve(); };
      video.addEventListener('loadedmetadata', onMeta, { once: true });
      video.addEventListener('durationchange', onMeta);
      video.load();
      setTimeout(() => { video.removeEventListener('durationchange', onMeta); resolve(); }, 10000);
    });

    const duration = video.duration;
    if (!duration || !isFinite(duration) || duration <= 0) {
      setErrorMessage(`動画の長さを取得できませんでした（duration: ${video.duration}）。`);
      setIsAnalyzing(false);
      return;
    }

    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 360;
    const ctx = canvas.getContext('2d');
    if (!ctx) { setIsAnalyzing(false); return; }

    video.pause();
    video.currentTime = 0;

    const totalFrames = Math.floor(duration / FRAME_INTERVAL);
    let frameIdx = 0;
    const currentMode = modeRef.current;

    const processFrame = () =>
      new Promise<void>((resolve, reject) => {
        const timer = setTimeout(() => {
          video.onseeked = null;
          reject(new Error(`フレーム${frameIdx}でシークタイムアウト`));
        }, 5000);

        video.onseeked = () => {
          clearTimeout(timer);
          video.onseeked = null;
          const timestampMs = video.currentTime * 1000;
          let landmarks: LandmarkPoint[] | null = null;
          try {
            const result = landmarker.detectForVideo(video, timestampMs);
            landmarks = result.landmarks[0] ?? null;
          } catch (_) { /* スキップ */ }

          storedFramesRef.current.push({ time: video.currentTime, landmarks });
          drawFrame(ctx, canvas.width, canvas.height, landmarks, storedFramesRef.current.length - 1, currentMode, true);

          frameIdx++;
          setProgress(Math.min(100, Math.round((frameIdx / totalFrames) * 100)));
          resolve();
        };
        video.currentTime = frameIdx * FRAME_INTERVAL;
      });

    try {
      while (frameIdx * FRAME_INTERVAL < duration) {
        await processFrame();
      }
    } catch (e) {
      setErrorMessage(e instanceof Error ? e.message : '解析中にエラーが発生しました。');
    }

    setProgress(100);
    setIsAnalyzing(false);
    setDone(true);
    startPlaybackLoop();
  };

  // ── UI ───────────────────────────────────────────────

  return (
    <section className="space-y-6 rounded-2xl border border-gray-800 bg-gray-900 p-6 text-white shadow-2xl shadow-black/30">
      <div className="space-y-2">
        <h2 className="text-2xl font-semibold">動画解析</h2>
        <p className="text-sm text-gray-400">
          動画をアップロードして骨格検出・手首軌跡を可視化します。
        </p>
      </div>

      {/* 武道種別選択 */}
      <div>
        <span className="mb-2 block text-sm font-medium text-gray-300">武道種別</span>
        <div className="flex gap-3">
          {BUDO_MODES.map(({ value, label, sub }) => (
            <button
              key={value}
              onClick={() => setMode(value)}
              className={`flex-1 rounded-xl border px-4 py-3 text-left transition-colors ${
                mode === value
                  ? 'border-green-500 bg-green-500/10 text-white'
                  : 'border-gray-700 bg-gray-800/50 text-gray-400 hover:border-gray-500'
              }`}
            >
              <div className="text-sm font-semibold">{label}</div>
              <div className="text-xs text-gray-500 mt-0.5">{sub}</div>
            </button>
          ))}
        </div>
        {mode === 'no-weapon' && (
          <p className="mt-2 text-xs text-yellow-500/80">
            ※ 袴着用時は膝・足首の精度が低下します（半透明で表示）
          </p>
        )}
        {mode === 'weapon' && (
          <p className="mt-2 text-xs text-gray-500">
            上半身・手首軌跡に特化して解析します。下半身は除外されます。
          </p>
        )}
      </div>

      {/* 動画ファイル */}
      <label className="block">
        <span className="mb-2 block text-sm font-medium text-gray-300">動画ファイル</span>
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          className="block w-full rounded-xl border border-gray-700 bg-gray-800 px-4 py-3 text-sm text-gray-100 file:mr-4 file:rounded-lg file:border-0 file:bg-green-500 file:px-4 file:py-2 file:text-sm file:font-medium file:text-gray-950 hover:file:bg-green-400"
        />
      </label>

      {/* 動画プレイヤー + キャンバスオーバーレイ */}
      {videoUrl && (
        <div className="relative rounded-xl border border-gray-800 bg-gray-950/80 overflow-hidden">
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            preload="auto"
            className="w-full block"
            onLoadedMetadata={() => {
              const v = videoRef.current;
              const c = canvasRef.current;
              if (v && c) { c.width = v.videoWidth; c.height = v.videoHeight; }
            }}
          />
          <canvas ref={canvasRef} className="absolute inset-0 w-full h-full pointer-events-none" />
        </div>
      )}

      {loadingModel && <Spinner label="モデル読み込み中..." />}
      {isAnalyzing && (
        <div className="space-y-2">
          <Spinner label={`解析中... ${progress}%`} />
          <div className="h-2 w-full overflow-hidden rounded-full bg-gray-700">
            <div className="h-full bg-green-400 transition-all duration-150" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {errorMessage && (
        <div className="rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-200">
          {errorMessage}
        </div>
      )}

      {videoUrl && !isAnalyzing && !loadingModel && (
        <button
          onClick={() => void startAnalysis()}
          className="rounded-xl bg-green-500 px-6 py-3 text-sm font-semibold text-gray-950 hover:bg-green-400"
        >
          {done ? '再解析' : '解析開始'}
        </button>
      )}

      {done && (
        <div className="space-y-3">
          {/* 再生速度 */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-gray-400 w-16">再生速度</span>
            {[0.25, 0.5, 1, 2].map((rate) => (
              <button
                key={rate}
                onClick={() => setPlaybackRate(rate)}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                  playbackRate === rate
                    ? 'bg-green-500 text-gray-950'
                    : 'border border-gray-600 text-gray-300 hover:bg-gray-800'
                }`}
              >
                {rate === 1 ? '×1' : rate === 0.25 ? '×¼' : rate === 0.5 ? '×½' : '×2'}
              </button>
            ))}
          </div>

          {/* コマ送り */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-gray-400 w-16">コマ送り</span>
            <button
              onClick={() => { const v = videoRef.current; if (!v) return; v.pause(); v.currentTime = Math.max(0, v.currentTime - FRAME_INTERVAL); }}
              className="rounded-lg border border-gray-600 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-800"
            >
              ◀ 1コマ戻る
            </button>
            <button
              onClick={() => { const v = videoRef.current; if (!v) return; v.pause(); v.currentTime = Math.min(v.duration, v.currentTime + FRAME_INTERVAL); }}
              className="rounded-lg border border-gray-600 px-3 py-1.5 text-xs text-gray-300 hover:bg-gray-800"
            >
              1コマ進む ▶
            </button>
            <span className="text-xs text-gray-500">（← → キーでも操作可）</span>
          </div>

          {/* 軌跡トグル + 凡例 */}
          <div className="flex flex-wrap items-center gap-4">
            <button
              onClick={() => setShowTrail((v) => !v)}
              className="rounded-xl border border-gray-600 px-5 py-2 text-sm text-gray-200 hover:bg-gray-800"
            >
              軌跡を{showTrail ? '非表示' : '表示'}
            </button>
            <div className="flex gap-4 text-sm text-gray-300">
              <span className="flex items-center gap-2">
                <span className="inline-block h-3 w-6 rounded-full bg-[#67e8f9]" />左手首
              </span>
              <span className="flex items-center gap-2">
                <span className="inline-block h-3 w-6 rounded-full bg-[#fb923c]" />右手首
              </span>
            </div>
          </div>
        </div>
      )}

      {!videoUrl && (
        <div className="flex min-h-64 items-center justify-center rounded-2xl border border-gray-800 bg-gray-950/80 text-sm text-gray-500">
          解析する動画を選択してください
        </div>
      )}

      {done && (
        <p className="text-xs text-gray-500">
          再生・スロー・コマ送り（← →キー）で確認できます。
        </p>
      )}
    </section>
  );
}
