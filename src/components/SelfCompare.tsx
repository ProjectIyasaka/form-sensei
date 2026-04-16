import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import {
  DrawingUtils,
  FilesetResolver,
  PoseLandmarker,
  type PoseLandmarkerResult
} from '@mediapipe/tasks-vision';
import {
  type AngleSnapshot,
  saveSnapshot,
  getSnapshotsByKata,
  getAllKataNames,
  deleteSnapshot,
  compressImageToDataUrl
} from '../lib/storage';

const WASM_FILE_PATH = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const MODEL_ASSET_PATH =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';

const ANGLE_CONFIG = [
  { key: 'leftShoulder',  label: '左肩',    points: [13, 11, 23] },
  { key: 'rightShoulder', label: '右肩',    points: [14, 12, 24] },
  { key: 'leftElbow',     label: '左肘',    points: [11, 13, 15] },
  { key: 'rightElbow',    label: '右肘',    points: [12, 14, 16] },
  { key: 'leftHip',       label: '左股関節', points: [11, 23, 25] },
  { key: 'rightHip',      label: '右股関節', points: [12, 24, 26] },
  { key: 'leftKnee',      label: '左膝',    points: [23, 25, 27] },
  { key: 'rightKnee',     label: '右膝',    points: [24, 26, 28] }
] as const;

type AngleItem = { key: string; label: string; value: number | null };
type LandmarkLike = { x: number; y: number; z?: number };
type Step = 'input' | 'analyze' | 'compare';

const STEP_LABELS: { key: Step; label: string }[] = [
  { key: 'input',   label: '型名入力' },
  { key: 'analyze', label: '写真解析' },
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
  return (Math.acos(Math.min(1, Math.max(-1, dot / (magA * magB)))) * 180) / Math.PI;
}

function buildAngles(result: PoseLandmarkerResult): AngleItem[] {
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

export default function SelfCompare() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [landmarker, setLandmarker] = useState<PoseLandmarker | null>(null);
  const [loadingModel, setLoadingModel] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [step, setStep] = useState<Step>('input');
  const [kataName, setKataName] = useState('');
  const [kataNames, setKataNames] = useState<string[]>([]);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [currentAngles, setCurrentAngles] = useState<AngleItem[]>([]);
  const [currentThumb, setCurrentThumb] = useState<string | null>(null);
  const [currentTs, setCurrentTs] = useState(0);
  const [prevSnapshot, setPrevSnapshot] = useState<AngleSnapshot | null>(null);
  const [history, setHistory] = useState<AngleSnapshot[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  // モデル初期化
  useEffect(() => {
    let active = true;
    let instance: PoseLandmarker | null = null;
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(WASM_FILE_PATH);
        instance = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_ASSET_PATH },
          runningMode: 'IMAGE',
          numPoses: 1
        });
        if (!active) { instance.close(); return; }
        setLandmarker(instance);
      } catch (e) {
        if (active) setErrorMessage(e instanceof Error ? e.message : 'モデルの初期化に失敗しました。');
      } finally {
        if (active) setLoadingModel(false);
      }
    })();
    return () => { active = false; instance?.close(); };
  }, []);

  useEffect(() => { setKataNames(getAllKataNames()); }, []);

  const handleStartAnalyze = () => {
    if (!kataName.trim()) { setErrorMessage('型名を入力してください。'); return; }
    setErrorMessage(null);
    setStep('analyze');
  };

  const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !landmarker) return;
    setErrorMessage(null);
    setIsAnalyzing(true);

    try {
      // 1. フルサイズ読み込み
      const dataUrl = await new Promise<string>((res, rej) => {
        const r = new FileReader();
        r.onload = () => res(r.result as string);
        r.onerror = () => rej(new Error('ファイルの読み込みに失敗しました。'));
        r.readAsDataURL(file);
      });
      setImageDataUrl(dataUrl);

      // 2. 解析
      const img = await new Promise<HTMLImageElement>((res, rej) => {
        const el = new Image();
        el.onload = () => res(el);
        el.onerror = () => rej(new Error('画像の読み込みに失敗しました。'));
        el.src = dataUrl;
      });
      const result = landmarker.detect(img);
      const angles = buildAngles(result);
      setCurrentAngles(angles);

      // 3. Canvas 描画
      const canvas = canvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (canvas && ctx) {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const du = new (DrawingUtils as any)(ctx);
        const lm = result.landmarks[0];
        if (lm) {
          du.drawConnectors(lm, (PoseLandmarker as any).POSE_CONNECTIONS, { color: '#34d399', lineWidth: 4 });
          du.drawLandmarks(lm, { color: '#86efac', fillColor: '#16a34a', lineWidth: 2, radius: 3 });
        }
      }

      // 4. サムネイル生成
      const thumb = await compressImageToDataUrl(file, 200);
      setCurrentThumb(thumb);
      const ts = Date.now();
      setCurrentTs(ts);

      // 5. 前回スナップを取得してから保存
      const existing = getSnapshotsByKata(kataName);
      setPrevSnapshot(existing[0] ?? null);

      saveSnapshot({ id: crypto.randomUUID(), kataName, timestamp: ts, imageDataUrl: thumb, angles });
      setHistory(getSnapshotsByKata(kataName));
      setKataNames(getAllKataNames());
      setStep('compare');
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : '解析に失敗しました。');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDeleteSnapshot = (id: string) => {
    deleteSnapshot(id);
    setHistory(getSnapshotsByKata(kataName));
  };

  const handleReset = () => {
    setStep('input');
    setKataName('');
    setImageDataUrl(null);
    setCurrentAngles([]);
    setCurrentThumb(null);
    setPrevSnapshot(null);
    setHistory([]);
    setShowHistory(false);
    setErrorMessage(null);
    setKataNames(getAllKataNames());
  };

  const summaryMessages = (): string[] => {
    if (!prevSnapshot || currentAngles.length === 0) return [];
    return currentAngles
      .map(curr => {
        const prev = prevSnapshot.angles.find(a => a.key === curr.key);
        if (curr.value == null || prev?.value == null) return null;
        return { label: curr.label, diff: curr.value - prev.value };
      })
      .filter((d): d is { label: string; diff: number } => d !== null)
      .sort((a, b) => Math.abs(b.diff) - Math.abs(a.diff))
      .slice(0, 2)
      .map(d =>
        Math.abs(d.diff) < 0.5
          ? `${d.label}角度が安定しています`
          : `${d.label}が前回より${Math.abs(d.diff).toFixed(1)}度${d.diff > 0 ? '深く' : '浅く'}なりました`
      );
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

      {(loadingModel || isAnalyzing) && (
        <Spinner label={loadingModel ? 'モデル読み込み中...' : '解析中...'} />
      )}

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

      {/* ── Step 2: 画像アップロード ── */}
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

          <label className={`flex min-h-[100px] cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-6 text-center transition-colors ${
            isAnalyzing
              ? 'border-slate-800 bg-slate-950/30 opacity-50 cursor-not-allowed'
              : imageDataUrl
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
              disabled={isAnalyzing}
              className="sr-only"
            />
          </label>

          {imageDataUrl && (
            <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-black">
              <img src={imageDataUrl} alt="解析対象" className="block h-auto w-full" />
              <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 h-full w-full" />
            </div>
          )}
        </div>
      )}

      {/* ── Step 3: 比較表示 ── */}
      {step === 'compare' && (
        <div className="space-y-5">
          {/* 今回 vs 前回 サムネイル */}
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-xl border border-green-500/25 bg-green-500/5 p-4">
              <div className="mb-2.5 flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-green-400" />
                <span className="text-xs font-semibold tracking-wider text-green-400">今回</span>
              </div>
              {currentThumb && (
                <img src={currentThumb} alt="今回" className="mb-2 h-20 w-auto rounded-lg object-cover" />
              )}
              <div className="text-xs text-slate-500">{formatDate(currentTs)}</div>
            </div>
            <div className="rounded-xl border border-slate-700/60 bg-slate-800/40 p-4">
              <div className="mb-2.5 flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-slate-500" />
                <span className="text-xs font-semibold tracking-wider text-slate-400">前回</span>
              </div>
              {prevSnapshot ? (
                <>
                  <img src={prevSnapshot.imageDataUrl} alt="前回" className="mb-2 h-20 w-auto rounded-lg object-cover" />
                  <div className="text-xs text-slate-500">{formatDate(prevSnapshot.timestamp)}</div>
                </>
              ) : (
                <div className="flex min-h-20 items-center text-sm text-slate-500">
                  初回記録です。<br />次回から比較できます
                </div>
              )}
            </div>
          </div>

          {/* 角度比較テーブル */}
          {prevSnapshot && (
            <>
              <div className="overflow-hidden rounded-xl border border-slate-800">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-800 bg-slate-800/50">
                      <th className="px-4 py-2.5 text-left text-xs font-medium text-slate-400">関節</th>
                      <th className="px-4 py-2.5 text-right text-xs font-medium text-green-400">今回</th>
                      <th className="px-4 py-2.5 text-right text-xs font-medium text-slate-400">前回</th>
                      <th className="px-4 py-2.5 text-right text-xs font-medium text-slate-400">差分</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentAngles.map(curr => {
                      const prev = prevSnapshot.angles.find(a => a.key === curr.key);
                      const diff = curr.value != null && prev?.value != null ? curr.value - prev.value : null;
                      const significant = diff != null && Math.abs(diff) >= 5;
                      const improved = diff != null && diff > 0;
                      return (
                        <tr
                          key={curr.key}
                          className={`border-b border-slate-800/60 ${
                            significant
                              ? improved
                                ? 'bg-green-500/5'
                                : 'bg-amber-500/5'
                              : ''
                          }`}
                        >
                          <td className={`px-4 py-2.5 text-slate-300 ${significant ? (improved ? 'border-l-2 border-l-green-500' : 'border-l-2 border-l-amber-400') : ''}`}>
                            {curr.label}
                          </td>
                          <td className="px-4 py-2.5 text-right font-medium text-cyan-300">
                            {curr.value != null ? `${curr.value.toFixed(1)}°` : '--'}
                          </td>
                          <td className="px-4 py-2.5 text-right text-slate-400">
                            {prev?.value != null ? `${prev.value.toFixed(1)}°` : '--'}
                          </td>
                          <td className={`px-4 py-2.5 text-right font-medium ${
                            diff == null
                              ? 'text-slate-600'
                              : significant
                                ? improved ? 'text-green-400' : 'text-amber-400'
                                : 'text-slate-500'
                          }`}>
                            {diff == null ? '--' : `${diff > 0 ? '+' : ''}${diff.toFixed(1)}°`}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {/* サマリーメッセージ */}
              {summaryMessages().length > 0 && (
                <div className="space-y-1.5 rounded-xl border border-green-500/15 bg-green-500/5 p-4">
                  {summaryMessages().map((msg, i) => (
                    <p key={i} className="text-sm text-green-300 flex items-start gap-2">
                      <span className="mt-0.5 h-1.5 w-1.5 shrink-0 rounded-full bg-green-400 mt-1.5" />
                      {msg}
                    </p>
                  ))}
                </div>
              )}
            </>
          )}

          <button
            type="button"
            onClick={handleReset}
            className="min-h-[44px] rounded-xl border border-slate-700 px-5 py-2.5 text-sm text-slate-300 transition-colors hover:border-slate-600 hover:bg-slate-800"
          >
            別の型を比較
          </button>

          {/* 過去の記録トグル */}
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
                {history.slice(0, 5).map(snap => {
                  const summary = snap.angles
                    .filter(a => a.value != null)
                    .slice(0, 4)
                    .map(a => `${a.label}: ${(a.value as number).toFixed(0)}°`)
                    .join(' · ');
                  return (
                    <div
                      key={snap.id}
                      className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-800/30 px-4 py-3"
                    >
                      <div>
                        <div className="text-sm text-slate-200">{formatDate(snap.timestamp)}</div>
                        <div className="text-xs text-slate-500 mt-0.5">{summary}</div>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleDeleteSnapshot(snap.id)}
                        className="min-h-[36px] rounded-lg border border-red-500/30 px-3 py-1.5 text-xs text-red-400 transition-colors hover:bg-red-500/10"
                      >
                        削除
                      </button>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
