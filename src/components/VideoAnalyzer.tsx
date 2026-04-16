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
    <div className="flex items-center gap-3 rounded-xl border border-slate-700 bg-slate-800/70 px-4 py-3 text-sm text-slate-200">
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
    <section className="space-y-5 rounded-2xl border border-slate-800 bg-slate-900 p-5 text-slate-100 shadow-2xl shadow-black/30 sm:p-6">

      {/* 武道種別選択（SegmentedControl風） */}
      <div>
        <span className="mb-2.5 block text-sm font-medium text-slate-300">武道種別</span>
        <div className="flex rounded-xl border border-slate-700 bg-slate-950/60 p-1 gap-1">
          {BUDO_MODES.map(({ value, label, sub }) => (
            <button
              key={value}
              onClick={() => setMode(value)}
              className={`flex-1 rounded-lg px-4 py-3 text-left transition-all duration-150 min-h-[44px] ${
                mode === value
                  ? 'bg-slate-800 shadow-sm border border-slate-700'
                  : 'hover:bg-slate-800/50'
              }`}
            >
              <div className={`text-sm font-semibold ${mode === value ? 'text-green-400' : 'text-slate-400'}`}>{label}</div>
              <div className="text-xs text-slate-500 mt-0.5">{sub}</div>
            </button>
          ))}
        </div>
        {mode === 'weapon' && (
          <p className="mt-2 text-xs text-cyan-400/80">
            上半身・手首軌跡に特化して解析します。下半身は半透明で表示されます。
          </p>
        )}
      </div>

      {/* 動画ファイルアップロード */}
      <div>
        <span className="mb-2.5 block text-sm font-medium text-slate-300">動画ファイル</span>
        <label className={`flex min-h-[80px] cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-5 text-center transition-colors ${
          videoUrl
            ? 'border-slate-700 bg-slate-800/40 hover:border-slate-600'
            : 'border-slate-700 bg-slate-950/40 hover:border-green-500/50 hover:bg-green-500/5'
        }`}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" className="h-6 w-6 text-slate-500" aria-hidden="true">
            <rect x="2" y="6" width="14" height="12" rx="2" />
            <path d="m16 10 6-3v10l-6-3" />
          </svg>
          <span className="text-sm text-slate-400">
            {videoUrl ? '別の動画を選択' : 'クリックして動画を選択'}
          </span>
          <span className="text-xs text-slate-600">MP4, MOV, WebM など</span>
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="sr-only"
          />
        </label>
      </div>

      {/* 動画プレイヤー + キャンバスオーバーレイ */}
      {videoUrl && (
        <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
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
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
            <div
              className="h-full rounded-full bg-gradient-to-r from-green-500 to-green-400 transition-all duration-150"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-right text-xs text-slate-500">{progress}%</p>
        </div>
      )}

      {errorMessage && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/8 px-4 py-3 text-sm text-red-300">
          {errorMessage}
        </div>
      )}

      {videoUrl && !isAnalyzing && !loadingModel && (
        <button
          onClick={() => void startAnalysis()}
          className="min-h-[44px] w-full rounded-xl bg-green-500 px-6 py-3 text-sm font-semibold text-slate-950 transition-colors hover:bg-green-400 active:bg-green-600 sm:w-auto"
        >
          {done ? '再解析' : '解析開始'}
        </button>
      )}

      {!videoUrl && !loadingModel && (
        <div className="flex min-h-48 items-center justify-center rounded-xl border border-slate-800 bg-slate-950/50 text-sm text-slate-600">
          解析する動画を選択してください
        </div>
      )}

      {/* 解析完了後のコントロール */}
      {done && (
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4 space-y-4">

          {/* 再生速度 */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="w-16 shrink-0 text-xs text-slate-400">再生速度</span>
            <div className="flex gap-1.5">
              {[0.25, 0.5, 1, 2].map((rate) => (
                <button
                  key={rate}
                  onClick={() => setPlaybackRate(rate)}
                  className={`min-h-[36px] rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                    playbackRate === rate
                      ? 'bg-green-500 text-slate-950'
                      : 'border border-slate-700 text-slate-300 hover:border-slate-600 hover:bg-slate-800'
                  }`}
                >
                  {rate === 1 ? '×1' : rate === 0.25 ? '×¼' : rate === 0.5 ? '×½' : '×2'}
                </button>
              ))}
            </div>
          </div>

          {/* コマ送り */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="w-16 shrink-0 text-xs text-slate-400">コマ送り</span>
            <div className="flex gap-1.5">
              <button
                onClick={() => { const v = videoRef.current; if (!v) return; v.pause(); v.currentTime = Math.max(0, v.currentTime - FRAME_INTERVAL); }}
                className="min-h-[36px] rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:border-slate-600 hover:bg-slate-800"
              >
                ◀ 戻る
              </button>
              <button
                onClick={() => { const v = videoRef.current; if (!v) return; v.pause(); v.currentTime = Math.min(v.duration, v.currentTime + FRAME_INTERVAL); }}
                className="min-h-[36px] rounded-lg border border-slate-700 px-3 py-1.5 text-xs text-slate-300 hover:border-slate-600 hover:bg-slate-800"
              >
                進む ▶
              </button>
            </div>
            <span className="text-xs text-slate-600">（← → キーでも操作可）</span>
          </div>

          {/* 軌跡トグル + 凡例 */}
          <div className="flex flex-wrap items-center gap-3 pt-1 border-t border-slate-800">
            <button
              onClick={() => setShowTrail((v) => !v)}
              className={`min-h-[36px] rounded-lg border px-4 py-1.5 text-xs font-medium transition-colors ${
                showTrail
                  ? 'border-cyan-500/40 bg-cyan-500/10 text-cyan-400'
                  : 'border-slate-700 text-slate-400 hover:bg-slate-800'
              }`}
            >
              軌跡 {showTrail ? 'ON' : 'OFF'}
            </button>
            <div className="flex gap-3 text-xs text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-5 rounded-full bg-[#67e8f9]" />左手首
              </span>
              <span className="flex items-center gap-1.5">
                <span className="inline-block h-2.5 w-5 rounded-full bg-[#fb923c]" />右手首
              </span>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
