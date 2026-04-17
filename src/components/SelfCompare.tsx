import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import {
  FilesetResolver,
  PoseLandmarker,
  type PoseLandmarkerResult
} from '@mediapipe/tasks-vision';
import {
  type VideoSnapshot,
  type VideoFrame,
  type FrameAngle,
  saveVideoSnapshot,
  getVideoSnapshotsByKata,
  getAllVideoKataNames,
  deleteVideoSnapshot,
} from '../lib/storage';

const WASM_FILE_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_ASSET_PATH =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';

const FRAME_INTERVAL = 1 / 30;

const ANGLE_CONFIG = [
  { key: 'leftShoulder',  label: '左肩',     points: [13, 11, 23] as const },
  { key: 'rightShoulder', label: '右肩',     points: [14, 12, 24] as const },
  { key: 'leftElbow',     label: '左肘',     points: [11, 13, 15] as const },
  { key: 'rightElbow',    label: '右肘',     points: [12, 14, 16] as const },
  { key: 'leftHip',       label: '左股関節', points: [11, 23, 25] as const },
  { key: 'rightHip',      label: '右股関節', points: [12, 24, 26] as const },
  { key: 'leftKnee',      label: '左膝',     points: [23, 25, 27] as const },
  { key: 'rightKnee',     label: '右膝',     points: [24, 26, 28] as const },
];

type LandmarkLike = { x: number; y: number; z?: number };
type Step = 'input' | 'analyze' | 'compare';

const STEP_LABELS: { key: Step; label: string }[] = [
  { key: 'input',   label: '型名入力' },
  { key: 'analyze', label: '動画解析' },
  { key: 'compare', label: '比較結果' },
];

function Spinner({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-slate-700 bg-slate-800/70 px-4 py-3 text-sm text-slate-200">
      <div className="h-5 w-5 animate-spin rounded-full border-2 border-green-400 border-t-transparent" />
      <span>{label}</span>
    </div>
  );
}

function calculateAngle(a: LandmarkLike, b: LandmarkLike, c: LandmarkLike): number | null {
  const vA = { x: a.x - b.x, y: a.y - b.y, z: (a.z ?? 0) - (b.z ?? 0) };
  const vB = { x: c.x - b.x, y: c.y - b.y, z: (c.z ?? 0) - (b.z ?? 0) };
  const dot = vA.x * vB.x + vA.y * vB.y + vA.z * vB.z;
  const magA = Math.hypot(vA.x, vA.y, vA.z);
  const magB = Math.hypot(vB.x, vB.y, vB.z);
  if (magA === 0 || magB === 0) return null;
  return Math.round((Math.acos(Math.min(1, Math.max(-1, dot / (magA * magB)))) * 180) / Math.PI * 10) / 10;
}

function buildAngles(result: PoseLandmarkerResult): FrameAngle[] {
  const landmarks = result.landmarks[0];
  if (!landmarks) return ANGLE_CONFIG.map(({ key, label }) => ({ key, label, value: null }));
  return ANGLE_CONFIG.map(({ key, label, points }) => {
    const [aIdx, bIdx, cIdx] = points;
    const a = landmarks[aIdx], b = landmarks[bIdx], c = landmarks[cIdx];
    if (!a || !b || !c) return { key, label, value: null };
    return { key, label, value: calculateAngle(a, b, c) };
  });
}

function formatDate(ts: number) {
  return new Date(ts).toLocaleString('ja-JP', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
  });
}

function AngleChart({ current, prev, angleKey }: {
  current: VideoSnapshot;
  prev: VideoSnapshot | null;
  angleKey: string;
}) {
  const W = 560, H = 160;
  const PL = 36, PR = 10, PT = 8, PB = 22;
  const plotW = W - PL - PR;
  const plotH = H - PT - PB;

  const maxDur = Math.max(current.duration, prev?.duration ?? 0);
  const xScale = (t: number) => PL + (t / maxDur) * plotW;
  const yScale = (v: number) => PT + (1 - v / 180) * plotH;

  const toPolyline = (frames: VideoFrame[]) => {
    const pts = frames
      .map(f => {
        const a = f.angles.find(a => a.key === angleKey);
        return a?.value != null ? `${xScale(f.time).toFixed(1)},${yScale(a.value).toFixed(1)}` : null;
      })
      .filter(Boolean) as string[];
    return pts.join(' ');
  };

  const gridAngles = [0, 45, 90, 135, 180];
  const xTicks = maxDur <= 10
    ? Array.from({ length: Math.ceil(maxDur) + 1 }, (_, i) => i).filter(t => t <= maxDur)
    : [0, Math.round(maxDur * 0.25), Math.round(maxDur * 0.5), Math.round(maxDur * 0.75), Math.round(maxDur)];

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" aria-label="角度時系列グラフ">
      {/* Y grid */}
      {gridAngles.map(v => (
        <g key={v}>
          <line x1={PL} y1={yScale(v)} x2={W - PR} y2={yScale(v)} stroke="#1e293b" strokeWidth="1" />
          <text x={PL - 4} y={yScale(v)} textAnchor="end" dominantBaseline="middle" fill="#475569" fontSize="9">{v}°</text>
        </g>
      ))}
      {/* X ticks */}
      {xTicks.map(t => (
        <text key={t} x={xScale(t)} y={H - 4} textAnchor="middle" fill="#475569" fontSize="9">{t}s</text>
      ))}
      {/* Previous line */}
      {prev && (
        <polyline
          points={toPolyline(prev.frames)}
          fill="none"
          stroke="#475569"
          strokeWidth="1.5"
          strokeDasharray="5,3"
        />
      )}
      {/* Current line */}
      <polyline
        points={toPolyline(current.frames)}
        fill="none"
        stroke="#34d399"
        strokeWidth="2"
      />
    </svg>
  );
}

export default function SelfCompare() {
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);
  const videoRef          = useRef<HTMLVideoElement | null>(null);

  const [loadingModel, setLoadingModel] = useState(true);
  const [isAnalyzing, setIsAnalyzing]   = useState(false);
  const [progress, setProgress]         = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [step, setStep]           = useState<Step>('input');
  const [kataName, setKataName]   = useState('');
  const [kataNames, setKataNames] = useState<string[]>([]);
  const [videoUrl, setVideoUrl]   = useState<string | null>(null);

  const [currentSnapshot, setCurrentSnapshot] = useState<VideoSnapshot | null>(null);
  const [prevSnapshot, setPrevSnapshot]       = useState<VideoSnapshot | null>(null);
  const [history, setHistory]                 = useState<VideoSnapshot[]>([]);
  const [showHistory, setShowHistory]         = useState(false);
  const [selectedJoint, setSelectedJoint]     = useState<string>(ANGLE_CONFIG[0].key);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_FILE_PATH);
        const lm = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_ASSET_PATH },
          runningMode: 'VIDEO',
          numPoses: 1,
          outputSegmentationMasks: false,
        });
        if (!active) { lm.close(); return; }
        poseLandmarkerRef.current = lm;
      } catch (e) {
        if (active) setErrorMessage(e instanceof Error ? e.message : 'モデルの初期化に失敗しました。');
      } finally {
        if (active) setLoadingModel(false);
      }
    })();
    return () => { active = false; poseLandmarkerRef.current?.close(); };
  }, []);

  useEffect(() => { setKataNames(getAllVideoKataNames()); }, []);

  const handleStartAnalyze = () => {
    if (!kataName.trim()) { setErrorMessage('型名を入力してください。'); return; }
    setErrorMessage(null);
    setStep('analyze');
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(URL.createObjectURL(file));
    setErrorMessage(null);
  };

  const startAnalysis = async () => {
    const video     = videoRef.current;
    const landmarker = poseLandmarkerRef.current;
    if (!video || !landmarker || !videoUrl) return;

    setIsAnalyzing(true);
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

    video.pause();
    video.currentTime = 0;

    const totalFrames = Math.floor(duration / FRAME_INTERVAL);
    let frameIdx = 0;
    const frames: VideoFrame[] = [];

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
          let angles: FrameAngle[] = ANGLE_CONFIG.map(({ key, label }) => ({ key, label, value: null }));

          try {
            const result = landmarker.detectForVideo(video, timestampMs);
            angles = buildAngles(result);
          } catch (_) { /* スキップ */ }

          frames.push({ time: Math.round(video.currentTime * 100) / 100, angles });
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
      setIsAnalyzing(false);
      return;
    }

    const existing = getVideoSnapshotsByKata(kataName);
    const prev = existing[0] ?? null;
    const snap: VideoSnapshot = {
      id: crypto.randomUUID(),
      kataName,
      timestamp: Date.now(),
      duration,
      frames,
    };

    saveVideoSnapshot(snap);
    setPrevSnapshot(prev);
    setCurrentSnapshot(snap);
    setHistory(getVideoSnapshotsByKata(kataName));
    setKataNames(getAllVideoKataNames());
    setProgress(100);
    setIsAnalyzing(false);
    setStep('compare');
  };

  const handleDeleteSnapshot = (id: string) => {
    deleteVideoSnapshot(id);
    setHistory(getVideoSnapshotsByKata(kataName));
  };

  const handleReset = () => {
    setStep('input');
    setKataName('');
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl(null);
    setCurrentSnapshot(null);
    setPrevSnapshot(null);
    setHistory([]);
    setShowHistory(false);
    setErrorMessage(null);
    setSelectedJoint(ANGLE_CONFIG[0].key);
    setKataNames(getAllVideoKataNames());
  };

  const stepIndex = STEP_LABELS.findIndex(s => s.key === step);

  return (
    <section className="space-y-5 rounded-2xl border border-slate-800 bg-slate-900 p-5 text-slate-100 shadow-2xl shadow-black/30 sm:p-6">

      {/* ステップインジケーター */}
      <div className="flex items-center gap-0">
        {STEP_LABELS.map((s, i) => {
          const isDone = i < stepIndex;
          const isActive = i === stepIndex;
          return (
            <div key={s.key} className="flex items-center flex-1 last:flex-none">
              <div className="flex flex-col items-center gap-1 min-w-[64px]">
                <div className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold transition-colors ${
                  isDone
                    ? 'bg-green-500 text-slate-950'
                    : isActive
                      ? 'border-2 border-green-400 text-green-400'
                      : 'border border-slate-700 text-slate-600'
                }`}>
                  {isDone ? (
                    <svg viewBox="0 0 16 16" fill="currentColor" className="h-3.5 w-3.5">
                      <path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.75.75 0 0 1 1.06-1.06L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0Z" />
                    </svg>
                  ) : i + 1}
                </div>
                <span className={`text-[10px] font-medium ${isActive ? 'text-green-400' : isDone ? 'text-slate-400' : 'text-slate-600'}`}>
                  {s.label}
                </span>
              </div>
              {i < STEP_LABELS.length - 1 && (
                <div className={`h-px flex-1 mx-1 transition-colors ${i < stepIndex ? 'bg-green-500' : 'bg-slate-800'}`} />
              )}
            </div>
          );
        })}
      </div>

      {loadingModel && <Spinner label="モデル読み込み中..." />}

      {errorMessage && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/8 px-4 py-3 text-sm text-red-300">
          {errorMessage}
        </div>
      )}

      {/* ── Step 1: 型名入力 ── */}
      {step === 'input' && !loadingModel && (
        <div className="space-y-4">
          <div>
            <label htmlFor="kata-input" className="mb-2 block text-sm font-medium text-slate-300">型名</label>
            <input
              id="kata-input"
              type="text"
              list="kata-datalist"
              value={kataName}
              onChange={e => setKataName(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') handleStartAnalyze(); }}
              placeholder="例: 平安初段"
              className="w-full rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-3 text-sm text-slate-100 placeholder-slate-500 transition-colors focus:border-green-500 focus:outline-none focus:bg-slate-800"
            />
            <datalist id="kata-datalist">
              {kataNames.map(n => <option key={n} value={n} />)}
            </datalist>
            {kataNames.length > 0 && (
              <p className="mt-1.5 text-xs text-slate-500">
                過去の記録: {kataNames.join(' / ')}
              </p>
            )}
          </div>
          <button
            type="button"
            onClick={handleStartAnalyze}
            disabled={!kataName.trim()}
            className="min-h-[44px] rounded-xl bg-green-500 px-6 py-2.5 text-sm font-semibold text-slate-950 transition-colors hover:bg-green-400 disabled:cursor-not-allowed disabled:opacity-40"
          >
            次へ →
          </button>
        </div>
      )}

      {/* ── Step 2: 動画アップロード・解析 ── */}
      {step === 'analyze' && !loadingModel && (
        <div className="space-y-4">
          <div className="flex items-center gap-2.5">
            <span className="rounded-lg border border-green-500/30 bg-green-500/8 px-3 py-1.5 text-sm font-medium text-green-400">
              {kataName}
            </span>
            <button type="button" onClick={() => setStep('input')} className="text-xs text-slate-500 hover:text-slate-300">
              変更
            </button>
          </div>

          {!isAnalyzing && (
            <label className={`flex min-h-[100px] cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-6 text-center transition-colors ${
              videoUrl
                ? 'border-slate-700 bg-slate-800/40 hover:border-slate-600'
                : 'border-slate-700 bg-slate-950/40 hover:border-green-500/50 hover:bg-green-500/5'
            }`}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="h-6 w-6 text-slate-500" aria-hidden="true">
                <rect x="2" y="6" width="14" height="12" rx="2" />
                <path d="m16 10 6-3v10l-6-3" />
              </svg>
              <span className="text-sm text-slate-400">
                {videoUrl ? '別の動画を選択' : 'クリックして動画を選択'}
              </span>
              <span className="text-xs text-slate-600">MP4, MOV, WebM など</span>
              <input type="file" accept="video/*" onChange={handleFileChange} className="sr-only" />
            </label>
          )}

          {videoUrl && !isAnalyzing && (
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              preload="auto"
              className="w-full rounded-xl border border-slate-800 bg-slate-950"
              onLoadedMetadata={() => {}}
            />
          )}

          {videoUrl && !isAnalyzing && (
            <button
              type="button"
              onClick={() => void startAnalysis()}
              className="min-h-[44px] w-full rounded-xl bg-green-500 px-6 py-3 text-sm font-semibold text-slate-950 transition-colors hover:bg-green-400 sm:w-auto"
            >
              解析開始
            </button>
          )}

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
        </div>
      )}

      {/* ── Step 3: 比較結果 ── */}
      {step === 'compare' && currentSnapshot && (
        <div className="space-y-5">

          {/* メタ情報 */}
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-green-500/25 bg-green-500/5 p-3">
              <div className="mb-1 flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-green-400" />
                <span className="text-xs font-semibold tracking-wider text-green-400">今回</span>
              </div>
              <div className="text-xs text-slate-400">{formatDate(currentSnapshot.timestamp)}</div>
              <div className="text-xs text-slate-500 mt-0.5">{currentSnapshot.duration.toFixed(1)}秒 · {currentSnapshot.frames.length}フレーム</div>
            </div>
            <div className="rounded-xl border border-slate-700/60 bg-slate-800/40 p-3">
              <div className="mb-1 flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-slate-500" />
                <span className="text-xs font-semibold tracking-wider text-slate-400">前回</span>
              </div>
              {prevSnapshot ? (
                <>
                  <div className="text-xs text-slate-400">{formatDate(prevSnapshot.timestamp)}</div>
                  <div className="text-xs text-slate-500 mt-0.5">{prevSnapshot.duration.toFixed(1)}秒 · {prevSnapshot.frames.length}フレーム</div>
                </>
              ) : (
                <div className="text-sm text-slate-500 mt-1">初回記録。次回から比較できます</div>
              )}
            </div>
          </div>

          {/* 関節セレクタ */}
          <div>
            <div className="mb-2 text-xs text-slate-400">関節を選択</div>
            <div className="flex flex-wrap gap-1.5">
              {ANGLE_CONFIG.map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => setSelectedJoint(key)}
                  className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors min-h-[32px] ${
                    selectedJoint === key
                      ? 'bg-green-500 text-slate-950'
                      : 'border border-slate-700 text-slate-400 hover:border-slate-600 hover:bg-slate-800'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* グラフ */}
          <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
            <div className="mb-2 flex items-center gap-4 text-xs">
              <span className="flex items-center gap-1.5 text-green-400">
                <span className="inline-block h-0.5 w-6 bg-green-400" />今回
              </span>
              {prevSnapshot && (
                <span className="flex items-center gap-1.5 text-slate-500">
                  <svg width="24" height="4" className="inline-block"><line x1="0" y1="2" x2="24" y2="2" stroke="#475569" strokeWidth="1.5" strokeDasharray="5,3" /></svg>
                  前回
                </span>
              )}
            </div>
            <AngleChart
              current={currentSnapshot}
              prev={prevSnapshot}
              angleKey={selectedJoint}
            />
          </div>

          <button
            type="button"
            onClick={handleReset}
            className="min-h-[44px] rounded-xl border border-slate-700 px-5 py-2.5 text-sm text-slate-300 transition-colors hover:border-slate-600 hover:bg-slate-800"
          >
            別の型を比較
          </button>

          {/* 過去の記録 */}
          <div>
            <button
              type="button"
              onClick={() => setShowHistory(h => !h)}
              className="flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-200"
            >
              <svg viewBox="0 0 16 16" fill="currentColor" className={`h-3.5 w-3.5 transition-transform ${showHistory ? 'rotate-180' : ''}`}>
                <path d="M4.22 6.22a.75.75 0 0 1 1.06 0L8 8.94l2.72-2.72a.75.75 0 1 1 1.06 1.06l-3.25 3.25a.75.75 0 0 1-1.06 0L4.22 7.28a.75.75 0 0 1 0-1.06Z" />
              </svg>
              過去の記録 ({history.length}件)
            </button>

            {showHistory && (
              <div className="mt-3 space-y-2">
                {history.slice(0, 5).map(snap => (
                  <div
                    key={snap.id}
                    className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-800/30 px-4 py-3"
                  >
                    <div>
                      <div className="text-sm text-slate-200">{formatDate(snap.timestamp)}</div>
                      <div className="text-xs text-slate-500 mt-0.5">{snap.duration.toFixed(1)}秒 · {snap.frames.length}フレーム</div>
                    </div>
                    <button
                      type="button"
                      onClick={() => handleDeleteSnapshot(snap.id)}
                      className="min-h-[36px] rounded-lg border border-red-500/30 px-3 py-1.5 text-xs text-red-400 transition-colors hover:bg-red-500/10"
                    >
                      削除
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
