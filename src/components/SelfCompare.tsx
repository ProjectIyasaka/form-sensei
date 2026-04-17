import { useEffect, useMemo, useRef, useState, type ChangeEvent, type Dispatch, type SetStateAction } from 'react';
import { PoseLandmarker } from '@mediapipe/tasks-vision';
import {
  type LandmarkPoint,
  type VideoFrame,
  type VideoSnapshot,
  createVideoThumbnailDataUrl,
  deleteVideoBlob,
  deleteVideoSnapshot,
  getAllVideoKataNames,
  getVideoBlob,
  getVideoSnapshotsByKata,
} from '../lib/storage';
import {
  ANGLE_CONFIG,
  CANVAS_H,
  CANVAS_W,
  DIFF_WINDOW_SECONDS,
  FRAME_INTERVAL,
  formatDate,
  type AngleDiffSummary,
  getAveragedAngleDiffs,
  getFrameAtTime,
  getInterpolatedFrame,
  getInterpolatedLandmarks,
  getTopDiffMoments,
} from '../lib/video-analysis';

type Step = 'input' | 'compare';
type CompareViewMode = 'split' | 'overlay';
type OverlayRenderMode = 'video' | 'skeleton';

const STEP_LABELS: { key: Step; label: string }[] = [
  { key: 'input', label: '記録を選ぶ' },
  { key: 'compare', label: '比較結果' },
];

function drawSkeletonOverlay(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement | null,
  landmarks: LandmarkPoint[] | null,
  primaryColor: string,
  label: string,
  highlightedAngleKeys: Set<string>,
  densityFactor = 1,
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.font = 'bold 13px sans-serif';
  const lw = ctx.measureText(label).width + 16;
  ctx.fillStyle = 'rgba(15, 23, 42, 0.75)';
  ctx.fillRect(8, 8, lw, 24);
  ctx.fillStyle = primaryColor;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, 16, 20);
  ctx.textBaseline = 'alphabetic';

  if (!landmarks || landmarks.length === 0) return;

  const videoWidth = video?.videoWidth ?? 0;
  const videoHeight = video?.videoHeight ?? 0;
  const canvasWidth = canvas.width;
  const canvasHeight = canvas.height;

  let drawWidth = canvasWidth;
  let drawHeight = canvasHeight;
  let offsetX = 0;
  let offsetY = 0;

  if (videoWidth > 0 && videoHeight > 0) {
    const videoAspect = videoWidth / videoHeight;
    const canvasAspect = canvasWidth / canvasHeight;
    if (videoAspect > canvasAspect) {
      drawWidth = canvasWidth;
      drawHeight = canvasWidth / videoAspect;
      offsetY = (canvasHeight - drawHeight) / 2;
    } else {
      drawHeight = canvasHeight;
      drawWidth = canvasHeight * videoAspect;
      offsetX = (canvasWidth - drawWidth) / 2;
    }
  }

  const project = (point: LandmarkPoint) => ({
    x: offsetX + point.x * drawWidth,
    y: offsetY + point.y * drawHeight,
  });

  const scaleBase = (Math.min(drawWidth, drawHeight) / 360) * densityFactor;
  const normalLineWidth = Math.max(1.3, Math.min(3.1, 2 * scaleBase));
  const highlightLineWidth = Math.max(normalLineWidth + 0.9, Math.min(4.7, 3.1 * scaleBase));
  const normalRadius = Math.max(2.2, Math.min(4.1, 2.9 * scaleBase));
  const highlightRadius = Math.max(normalRadius + 0.7, Math.min(5.3, 3.9 * scaleBase));

  const highlightedPointIndexes = new Set<number>();
  const highlightedConnections = new Set<string>();
  for (const angle of ANGLE_CONFIG) {
    if (!highlightedAngleKeys.has(angle.key)) continue;
    const [aIdx, bIdx, cIdx] = angle.points;
    highlightedPointIndexes.add(aIdx);
    highlightedPointIndexes.add(bIdx);
    highlightedPointIndexes.add(cIdx);
    highlightedConnections.add(`${Math.min(aIdx, bIdx)}-${Math.max(aIdx, bIdx)}`);
    highlightedConnections.add(`${Math.min(bIdx, cIdx)}-${Math.max(bIdx, cIdx)}`);
  }

  const allConns = (PoseLandmarker as any).POSE_CONNECTIONS as Array<{ start: number; end: number }>;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const conn of allConns) {
    const start = landmarks[conn.start];
    const end = landmarks[conn.end];
    if (!start || !end) continue;
    const a = project(start);
    const b = project(end);
    const connKey = `${Math.min(conn.start, conn.end)}-${Math.max(conn.start, conn.end)}`;
    const isHighlighted = highlightedConnections.has(connKey);
    ctx.beginPath();
    ctx.strokeStyle = isHighlighted ? '#fbbf24' : primaryColor;
    ctx.lineWidth = isHighlighted ? highlightLineWidth : normalLineWidth;
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  for (const [index, point] of landmarks.entries()) {
    const { x, y } = project(point);
    const isHighlighted = highlightedPointIndexes.has(index);
    ctx.beginPath();
    ctx.fillStyle = isHighlighted ? '#fbbf24' : primaryColor;
    ctx.arc(x, y, isHighlighted ? highlightRadius : normalRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.beginPath();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = Math.max(0.8, normalLineWidth * 0.45);
    ctx.arc(x, y, isHighlighted ? highlightRadius : normalRadius, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function Spinner({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-slate-700 bg-slate-800/70 px-4 py-3 text-sm text-slate-200">
      <div className="h-5 w-5 animate-spin rounded-full border-2 border-green-400 border-t-transparent" />
      <span>{label}</span>
    </div>
  );
}

export default function SelfCompare() {
  const currentOverlayRef = useRef<HTMLCanvasElement | null>(null);
  const prevOverlayRef = useRef<HTMLCanvasElement | null>(null);
  const mergeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const currentVideoRef = useRef<HTMLVideoElement | null>(null);
  const prevVideoRef = useRef<HTMLVideoElement | null>(null);
  const currentSnapshotRef = useRef<VideoSnapshot | null>(null);
  const prevSnapshotRef = useRef<VideoSnapshot | null>(null);
  const rafRef = useRef<number | null>(null);
  const virtualTimeRef = useRef(0);
  const isPlayingRef = useRef(false);
  const playbackRateRef = useRef(1);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const displayedDiffsUntilRef = useRef(0);

  const [step, setStep] = useState<Step>('input');
  const [kataName, setKataName] = useState('');
  const [kataNames, setKataNames] = useState<string[]>([]);
  const [history, setHistory] = useState<VideoSnapshot[]>([]);
  const [historyThumbnails, setHistoryThumbnails] = useState<Record<string, string>>({});
  const [currentSnapshot, setCurrentSnapshot] = useState<VideoSnapshot | null>(null);
  const [prevSnapshot, setPrevSnapshot] = useState<VideoSnapshot | null>(null);
  const [selectedCurrentId, setSelectedCurrentId] = useState<string | null>(null);
  const [selectedPrevId, setSelectedPrevId] = useState<string | null>(null);
  const [currentBlobUrl, setCurrentBlobUrl] = useState<string | null>(null);
  const [prevBlobUrl, setPrevBlobUrl] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isLoadingCompare, setIsLoadingCompare] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [virtualTime, setVirtualTime] = useState(0);
  const [isRecording, setIsRecording] = useState(false);
  const [compareViewMode, setCompareViewMode] = useState<CompareViewMode>('split');
  const [overlayOpacity, setOverlayOpacity] = useState(0.6);
  const [overlayRenderMode, setOverlayRenderMode] = useState<OverlayRenderMode>('video');
  const [displayedDiffs, setDisplayedDiffs] = useState<AngleDiffSummary[]>([]);

  useEffect(() => { currentSnapshotRef.current = currentSnapshot; }, [currentSnapshot]);
  useEffect(() => { prevSnapshotRef.current = prevSnapshot; }, [prevSnapshot]);
  useEffect(() => { isPlayingRef.current = isPlaying; }, [isPlaying]);
  useEffect(() => { playbackRateRef.current = playbackRate; }, [playbackRate]);
  useEffect(() => () => { if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl); }, [currentBlobUrl]);
  useEffect(() => () => { if (prevBlobUrl) URL.revokeObjectURL(prevBlobUrl); }, [prevBlobUrl]);

  useEffect(() => {
    setKataNames(getAllVideoKataNames());
    const params = new URLSearchParams(window.location.search);
    const queryKata = params.get('kata');
    if (queryKata) setKataName(queryKata);
  }, []);

  const typedKataHistory = useMemo(
    () => (kataName.trim() ? getVideoSnapshotsByKata(kataName.trim()) : []),
    [kataName],
  );
  const compareCandidates = history;
  const compareTargetCandidates = compareCandidates.filter(snap => snap.id !== selectedCurrentId);
  const currentFrameDiffs = getAveragedAngleDiffs(
    currentSnapshot?.frames ?? [],
    prevSnapshot?.frames ?? [],
    virtualTime,
    DIFF_WINDOW_SECONDS,
  );
  const topDiffMoments = getTopDiffMoments(currentSnapshot?.frames ?? [], prevSnapshot?.frames ?? []);
  const maxDuration = Math.max(currentSnapshot?.duration ?? 0, prevSnapshot?.duration ?? 0);

  useEffect(() => {
    if (!isPlaying) {
      setDisplayedDiffs(currentFrameDiffs);
      displayedDiffsUntilRef.current = virtualTime + 0.45;
      return;
    }
    if (virtualTime >= displayedDiffsUntilRef.current) {
      setDisplayedDiffs(currentFrameDiffs);
      displayedDiffsUntilRef.current = virtualTime + 0.45;
    }
  }, [currentFrameDiffs, isPlaying, virtualTime]);

  useEffect(() => {
    if (typedKataHistory.length === 0) {
      setHistoryThumbnails({});
      return;
    }
    let active = true;
    const preset = Object.fromEntries(
      typedKataHistory
        .filter(snap => snap.thumbnailDataUrl)
        .map(snap => [snap.id, snap.thumbnailDataUrl!] as const),
    );
    setHistoryThumbnails(preset);
    (async () => {
      for (const snap of typedKataHistory) {
        if (snap.thumbnailDataUrl) continue;
        try {
          const blob = await getVideoBlob(snap.id);
          if (!blob || !active) continue;
          const thumb = await createVideoThumbnailDataUrl(blob);
          if (!active) return;
          setHistoryThumbnails(current => current[snap.id] ? current : { ...current, [snap.id]: thumb });
        } catch {
          // 補助表示のため失敗は無視
        }
      }
    })();
    return () => { active = false; };
  }, [typedKataHistory]);

  useEffect(() => {
    if (step !== 'compare') {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      return;
    }
    const tick = () => {
      const t = virtualTimeRef.current;
      const cur = currentSnapshotRef.current;
      const prv = prevSnapshotRef.current;
      const curCanvas = currentOverlayRef.current;
      const prvCanvas = prevOverlayRef.current;
      const mergeCanvas = mergeCanvasRef.current;
      const curVideo = currentVideoRef.current;
      const prvVideo = prevVideoRef.current;
      const currentTime = curVideo ? curVideo.currentTime : virtualTimeRef.current;
      const densityFactor = compareViewMode === 'overlay' ? 0.9 : 0.78;
      virtualTimeRef.current = currentTime;
      if (isPlayingRef.current) {
        setVirtualTime(currentTime);
      }
      const highlighted = new Set(displayedDiffs.slice(0, 3).map(item => item.key));

      if (curCanvas) {
        drawSkeletonOverlay(curCanvas, curVideo, cur ? getInterpolatedLandmarks(cur.frames, currentTime) : null, '#34d399', '解析動画', highlighted, densityFactor);
      }
      if (prvCanvas) {
        drawSkeletonOverlay(prvCanvas, prvVideo, prv ? getInterpolatedLandmarks(prv.frames, currentTime) : null, '#94a3b8', '比較先', highlighted, densityFactor);
      }
      if (mergeCanvas && curCanvas && prvCanvas) {
        const ctx = mergeCanvas.getContext('2d');
        if (ctx) {
          ctx.fillStyle = '#0f172a';
          ctx.fillRect(0, 0, mergeCanvas.width, mergeCanvas.height);
          if (curVideo) ctx.drawImage(curVideo, 0, 0, CANVAS_W, CANVAS_H);
          if (prvVideo) ctx.drawImage(prvVideo, CANVAS_W, 0, CANVAS_W, CANVAS_H);
          ctx.drawImage(curCanvas, 0, 0, CANVAS_W, CANVAS_H);
          ctx.drawImage(prvCanvas, CANVAS_W, 0, CANVAS_W, CANVAS_H);
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [step, compareViewMode, displayedDiffs]);

  useEffect(() => {
    if (step !== 'compare') return;
    const currentVideo = currentVideoRef.current;
    if (!currentVideo) return;
    const prevVideo = prevVideoRef.current;

    const syncTime = () => {
      virtualTimeRef.current = currentVideo.currentTime;
      setVirtualTime(currentVideo.currentTime);
      if (prevVideo && Math.abs(prevVideo.currentTime - currentVideo.currentTime) > 0.05) {
        prevVideo.currentTime = currentVideo.currentTime;
      }
    };
    const handleEnded = () => {
      isPlayingRef.current = false;
      setIsPlaying(false);
      virtualTimeRef.current = currentVideo.duration || maxDuration;
      setVirtualTime(currentVideo.duration || maxDuration);
      prevVideo?.pause();
    };

    virtualTimeRef.current = currentVideo.currentTime;
    setVirtualTime(currentVideo.currentTime);
    currentVideo.addEventListener('timeupdate', syncTime);
    currentVideo.addEventListener('ended', handleEnded);
    return () => {
      currentVideo.removeEventListener('timeupdate', syncTime);
      currentVideo.removeEventListener('ended', handleEnded);
    };
  }, [step, maxDuration, currentBlobUrl, prevBlobUrl, compareViewMode]);

  const loadSnapshotBlob = async (
    snap: VideoSnapshot | null,
    setBlobUrl: Dispatch<SetStateAction<string | null>>,
  ) => {
    if (!snap) {
      setBlobUrl(current => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
      return;
    }
    try {
      const blob = await getVideoBlob(snap.id);
      setBlobUrl(current => {
        if (current) URL.revokeObjectURL(current);
        return blob ? URL.createObjectURL(blob) : null;
      });
    } catch {
      setBlobUrl(current => {
        if (current) URL.revokeObjectURL(current);
        return null;
      });
    }
  };

  const openCompare = async (nextHistory: VideoSnapshot[], currentId?: string | null, prevId?: string | null) => {
    const nextCurrent = nextHistory.find(snap => snap.id === currentId) ?? nextHistory[0] ?? null;
    const nextPrev =
      nextHistory.find(snap => snap.id === prevId && snap.id !== nextCurrent?.id)
      ?? nextHistory.find(snap => snap.id !== nextCurrent?.id)
      ?? null;

    setIsLoadingCompare(true);
    setErrorMessage(null);
    setHistory(nextHistory);
    setCurrentSnapshot(nextCurrent);
    setPrevSnapshot(nextPrev);
    setSelectedCurrentId(nextCurrent?.id ?? null);
    setSelectedPrevId(nextPrev?.id ?? null);
    setCompareViewMode('split');
    setOverlayRenderMode('video');
    setOverlayOpacity(0.6);
    setDisplayedDiffs([]);
    displayedDiffsUntilRef.current = 0;
    setIsPlaying(false);
    isPlayingRef.current = false;
    virtualTimeRef.current = 0;
    setVirtualTime(0);

    await Promise.all([
      loadSnapshotBlob(nextCurrent, setCurrentBlobUrl),
      loadSnapshotBlob(nextPrev, setPrevBlobUrl),
    ]);

    setStep('compare');
    setIsLoadingCompare(false);
  };

  const handleOpenKata = async () => {
    const trimmed = kataName.trim();
    if (!trimmed) {
      setErrorMessage('型名を入力してください。');
      return;
    }
    const nextHistory = getVideoSnapshotsByKata(trimmed);
    if (nextHistory.length === 0) {
      setErrorMessage('この型の保存済み解析結果がありません。先に「動画解析」で保存してください。');
      return;
    }
    await openCompare(nextHistory);
  };

  const handleOpenSavedSnapshot = async (snap: VideoSnapshot) => {
    setKataName(snap.kataName);
    const nextHistory = getVideoSnapshotsByKata(snap.kataName);
    await openCompare(nextHistory, snap.id, null);
  };

  const handleSelectCurrent = async (snap: VideoSnapshot) => {
    if (snap.id === selectedCurrentId) return;
    const nextPrev = selectedPrevId === snap.id ? null : prevSnapshot;
    setCurrentSnapshot(snap);
    setSelectedCurrentId(snap.id);
    if (selectedPrevId === snap.id) {
      setPrevSnapshot(null);
      setSelectedPrevId(null);
      await loadSnapshotBlob(null, setPrevBlobUrl);
    }
    await loadSnapshotBlob(snap, setCurrentBlobUrl);
    if (nextPrev) setPrevSnapshot(nextPrev);
    virtualTimeRef.current = 0;
    setVirtualTime(0);
    setIsPlaying(false);
  };

  const handleSelectPrev = async (snap: VideoSnapshot | null) => {
    setPrevSnapshot(snap);
    setSelectedPrevId(snap?.id ?? null);
    await loadSnapshotBlob(snap, setPrevBlobUrl);
    virtualTimeRef.current = 0;
    setVirtualTime(0);
    setIsPlaying(false);
  };

  const handleDeleteSnapshot = async (id: string) => {
    deleteVideoSnapshot(id);
    void deleteVideoBlob(id);
    const nextHistory = getVideoSnapshotsByKata(kataName.trim());
    setKataNames(getAllVideoKataNames());
    if (nextHistory.length === 0) {
      await loadSnapshotBlob(null, setCurrentBlobUrl);
      await loadSnapshotBlob(null, setPrevBlobUrl);
      setHistory([]);
      setCurrentSnapshot(null);
      setPrevSnapshot(null);
      setSelectedCurrentId(null);
      setSelectedPrevId(null);
      setStep('input');
      return;
    }
    await openCompare(nextHistory, selectedCurrentId === id ? null : selectedCurrentId, selectedPrevId === id ? null : selectedPrevId);
  };

  const handleReset = () => {
    setStep('input');
    setHistory([]);
    setCurrentSnapshot(null);
    setPrevSnapshot(null);
    setSelectedCurrentId(null);
    setSelectedPrevId(null);
    setCurrentBlobUrl(current => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
    setPrevBlobUrl(current => {
      if (current) URL.revokeObjectURL(current);
      return null;
    });
    setErrorMessage(null);
    setIsPlaying(false);
    setDisplayedDiffs([]);
    setVirtualTime(0);
    virtualTimeRef.current = 0;
    displayedDiffsUntilRef.current = 0;
  };

  const togglePlay = () => {
    const currentVideo = currentVideoRef.current;
    const prevVideo = prevVideoRef.current;
    if (virtualTime >= maxDuration && currentVideo) {
      currentVideo.currentTime = 0;
      if (prevVideo) prevVideo.currentTime = 0;
      virtualTimeRef.current = 0;
      setVirtualTime(0);
    }
    const next = !isPlaying;
    setIsPlaying(next);
    isPlayingRef.current = next;
    if (currentVideo) {
      currentVideo.playbackRate = playbackRateRef.current;
      if (next) {
        void currentVideo.play().catch(() => {
          isPlayingRef.current = false;
          setIsPlaying(false);
        });
      } else {
        currentVideo.pause();
      }
    }
    if (prevVideo) {
      prevVideo.playbackRate = playbackRateRef.current;
      if (next) void prevVideo.play().catch(() => {});
      else prevVideo.pause();
    }
  };

  const handleScrub = (e: ChangeEvent<HTMLInputElement>) => {
    const t = parseFloat(e.target.value);
    virtualTimeRef.current = t;
    setVirtualTime(t);
    isPlayingRef.current = false;
    setIsPlaying(false);
    if (currentVideoRef.current) {
      currentVideoRef.current.pause();
      currentVideoRef.current.currentTime = t;
    }
    if (prevVideoRef.current) {
      prevVideoRef.current.pause();
      prevVideoRef.current.currentTime = t;
    }
  };

  const jumpToTime = (time: number) => {
    const clamped = Math.max(0, Math.min(time, maxDuration || time));
    virtualTimeRef.current = clamped;
    setVirtualTime(clamped);
    isPlayingRef.current = false;
    setIsPlaying(false);
    if (currentVideoRef.current) {
      currentVideoRef.current.pause();
      currentVideoRef.current.currentTime = clamped;
    }
    if (prevVideoRef.current) {
      prevVideoRef.current.pause();
      prevVideoRef.current.currentTime = clamped;
    }
  };

  const handleStartRecording = () => {
    const canvas = mergeCanvasRef.current;
    if (!canvas) return;
    const stream = (canvas as any).captureStream(30) as MediaStream;
    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9') ? 'video/webm;codecs=vp9' : 'video/webm';
    const mr = new MediaRecorder(stream, { mimeType });
    recordedChunksRef.current = [];
    mr.ondataavailable = e => { if (e.data.size > 0) recordedChunksRef.current.push(e.data); };
    mr.onstop = () => {
      const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${kataName || 'compare'}_compare.webm`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      setIsRecording(false);
    };
    mr.start();
    mediaRecorderRef.current = mr;
    setIsRecording(true);
    virtualTimeRef.current = 0;
    setVirtualTime(0);
    isPlayingRef.current = true;
    setIsPlaying(true);
    if (currentVideoRef.current) {
      currentVideoRef.current.currentTime = 0;
      void currentVideoRef.current.play().catch(() => {});
    }
    if (prevVideoRef.current) {
      prevVideoRef.current.currentTime = 0;
      void prevVideoRef.current.play().catch(() => {});
    }
  };

  const handleStopRecording = () => {
    mediaRecorderRef.current?.stop();
    setIsPlaying(false);
    isPlayingRef.current = false;
    currentVideoRef.current?.pause();
    prevVideoRef.current?.pause();
  };

  const stepIndex = STEP_LABELS.findIndex(item => item.key === step);

  useEffect(() => {
    if (step !== 'input' || typedKataHistory.length === 0) return;
    const params = new URLSearchParams(window.location.search);
    const currentId = params.get('current');
    if (!currentId) return;
    const snap = typedKataHistory.find(item => item.id === currentId);
    if (!snap) return;
    void handleOpenSavedSnapshot(snap);
    params.delete('current');
    const nextQuery = params.toString();
    window.history.replaceState({}, '', nextQuery ? `/compare?${nextQuery}` : '/compare');
  }, [step, typedKataHistory]);

  return (
    <section className="space-y-5 rounded-2xl border border-slate-800 bg-slate-900 p-5 text-slate-100 shadow-2xl shadow-black/30 sm:p-6">
      <div className="flex items-center gap-0">
        {STEP_LABELS.map((item, index) => {
          const isDone = index < stepIndex;
          const isActive = index === stepIndex;
          return (
            <div key={item.key} className="flex flex-1 items-center last:flex-none">
              <div className="flex min-w-[64px] flex-col items-center gap-1">
                <div className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold transition-colors ${
                  isDone ? 'bg-green-500 text-slate-950' : isActive ? 'border-2 border-green-400 text-green-400' : 'border border-slate-700 text-slate-600'
                }`}>
                  {isDone ? '✓' : index + 1}
                </div>
                <span className={`text-[10px] font-medium ${isActive ? 'text-green-400' : isDone ? 'text-slate-400' : 'text-slate-600'}`}>
                  {item.label}
                </span>
              </div>
              {index < STEP_LABELS.length - 1 && (
                <div className={`mx-1 h-px flex-1 ${index < stepIndex ? 'bg-green-500' : 'bg-slate-800'}`} />
              )}
            </div>
          );
        })}
      </div>

      {errorMessage && (
        <div className="rounded-xl border border-red-500/30 bg-red-500/8 px-4 py-3 text-sm text-red-300">
          {errorMessage}
        </div>
      )}

      {step === 'input' && (
        <div className="space-y-4">
          <div className="rounded-xl border border-green-500/20 bg-green-500/5 px-4 py-3">
            <p className="text-sm font-medium text-green-400">先に動画解析で保存した結果を比較します</p>
            <p className="mt-1 text-xs text-slate-400">
              型名を入力すると、その型で保存済みの解析結果を選んで比較できます。
            </p>
          </div>

          <div>
            <label htmlFor="compare-kata" className="mb-2 block text-sm font-medium text-slate-300">型名</label>
            <input
              id="compare-kata"
              type="text"
              list="compare-kata-datalist"
              value={kataName}
              onChange={e => setKataName(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') void handleOpenKata(); }}
              placeholder="例: 平安初段"
              className="w-full rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-3 text-sm text-slate-100 placeholder-slate-500 transition-colors focus:border-green-500 focus:bg-slate-800 focus:outline-none"
            />
            <datalist id="compare-kata-datalist">
              {kataNames.map(name => <option key={name} value={name} />)}
            </datalist>
            {kataNames.length > 0 && (
              <p className="mt-1.5 text-xs text-slate-500">保存済みの型: {kataNames.join(' / ')}</p>
            )}
          </div>

          {kataName.trim() && typedKataHistory.length > 0 && (
            <div className="space-y-2 rounded-xl border border-slate-800 bg-slate-950/40 p-3">
              <div className="flex items-center justify-between gap-3">
                <p className="text-sm font-medium text-slate-300">保存済み解析結果</p>
                <span className="text-xs text-slate-500">{typedKataHistory.length}件</span>
              </div>
              <div className="space-y-2">
                {typedKataHistory.map(snap => (
                  <div key={snap.id} className="flex items-center justify-between gap-3 rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-2">
                    <div className="flex min-w-0 items-center gap-3">
                      <div className="flex h-14 w-20 shrink-0 items-center justify-center overflow-hidden rounded-md border border-slate-800 bg-slate-950">
                        {historyThumbnails[snap.id] ? (
                          <img src={historyThumbnails[snap.id]} alt="" className="h-full w-full object-cover" />
                        ) : (
                          <span className="text-[10px] text-slate-600">thumbnail</span>
                        )}
                      </div>
                      <div className="min-w-0">
                        <p className="text-sm text-slate-200">{formatDate(snap.timestamp)}</p>
                        <p className="text-xs text-slate-500">{snap.duration.toFixed(1)}秒</p>
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={() => void handleOpenSavedSnapshot(snap)}
                      className="min-h-[36px] shrink-0 rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-slate-600 hover:bg-slate-800"
                    >
                      この解析を開く
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {kataName.trim() && typedKataHistory.length === 0 && (
            <div className="rounded-xl border border-slate-800 bg-slate-950/50 px-4 py-3 text-sm text-slate-400">
              この型の保存済み解析結果がありません。先に「動画解析」で型名を付けて保存してください。
            </div>
          )}

          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => void handleOpenKata()}
              disabled={!kataName.trim() || typedKataHistory.length === 0}
              className="min-h-[44px] rounded-xl bg-green-500 px-6 py-2.5 text-sm font-semibold text-slate-950 transition-colors hover:bg-green-400 disabled:cursor-not-allowed disabled:opacity-40"
            >
              最新の解析を開く
            </button>
            <a
              href="/video"
              className="inline-flex min-h-[44px] items-center rounded-xl border border-slate-700 px-6 py-2.5 text-sm text-slate-300 transition-colors hover:border-slate-600 hover:bg-slate-800"
            >
              動画解析へ
            </a>
          </div>
        </div>
      )}

      {isLoadingCompare && <Spinner label="保存済み解析結果を読み込み中..." />}

      {step === 'compare' && currentSnapshot && !isLoadingCompare && (
        <div className="space-y-4">
          <div className="grid gap-3 sm:grid-cols-2">
            <label className="space-y-1.5">
              <span className="text-xs font-medium text-slate-400">解析した動画</span>
              <select
                value={selectedCurrentId ?? ''}
                onChange={e => {
                  const snap = compareCandidates.find(item => item.id === e.target.value);
                  if (snap) void handleSelectCurrent(snap);
                }}
                className="w-full rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-3 text-sm text-slate-100 transition-colors focus:border-green-500 focus:outline-none"
              >
                {compareCandidates.map(snap => (
                  <option key={snap.id} value={snap.id}>{formatDate(snap.timestamp)} / {snap.duration.toFixed(1)}秒</option>
                ))}
              </select>
            </label>
            <label className="space-y-1.5">
              <span className="text-xs font-medium text-slate-400">比較先動画</span>
              <select
                value={selectedPrevId ?? ''}
                onChange={e => {
                  const snap = compareTargetCandidates.find(item => item.id === e.target.value) ?? null;
                  void handleSelectPrev(snap);
                }}
                className="w-full rounded-xl border border-slate-700 bg-slate-800/80 px-4 py-3 text-sm text-slate-100 transition-colors focus:border-green-500 focus:outline-none"
              >
                <option value="">比較先を選択してください</option>
                {compareTargetCandidates.map(snap => (
                  <option key={snap.id} value={snap.id}>{formatDate(snap.timestamp)} / {snap.duration.toFixed(1)}秒</option>
                ))}
              </select>
            </label>
          </div>

          {prevSnapshot && (
            <div className="flex items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
              <div>
                <p className="text-sm font-medium text-slate-200">表示モード</p>
                <p className="text-xs text-slate-500">見比べやすい方に切り替えられます</p>
              </div>
              <div className="flex gap-1.5 rounded-lg border border-slate-800 bg-slate-900/80 p-1">
                {([
                  { key: 'split', label: '横並び' },
                  { key: 'overlay', label: '重ね表示' },
                ] as const).map(mode => (
                  <button
                    key={mode.key}
                    type="button"
                    onClick={() => setCompareViewMode(mode.key)}
                    className={`min-h-[34px] rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                      compareViewMode === mode.key ? 'bg-green-500 text-slate-950' : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                    }`}
                  >
                    {mode.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          {prevSnapshot && compareViewMode === 'overlay' && (
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3 space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-slate-200">重ね表示の設定</p>
                  <p className="text-xs text-slate-500">動画を出すか、骨格だけにするかを切り替えられます</p>
                </div>
                <div className="flex gap-1.5 rounded-lg border border-slate-800 bg-slate-900/80 p-1">
                  {([
                    { key: 'video', label: '動画込み' },
                    { key: 'skeleton', label: '骨格だけ' },
                  ] as const).map(mode => (
                    <button
                      key={mode.key}
                      type="button"
                      onClick={() => setOverlayRenderMode(mode.key)}
                      className={`min-h-[34px] rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                        overlayRenderMode === mode.key ? 'bg-green-500 text-slate-950' : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
                      }`}
                    >
                      {mode.label}
                    </button>
                  ))}
                </div>
              </div>
              {overlayRenderMode === 'video' && (
                <>
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-sm font-medium text-slate-200">比較先の濃さ</p>
                      <p className="text-xs text-slate-500">重ね表示の見え方を調整できます</p>
                    </div>
                    <span className="text-xs font-medium tabular-nums text-slate-400">{Math.round(overlayOpacity * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min={0.15}
                    max={0.9}
                    step={0.05}
                    value={overlayOpacity}
                    onChange={e => setOverlayOpacity(parseFloat(e.target.value))}
                    className="w-full accent-green-500"
                  />
                </>
              )}
            </div>
          )}

          {compareCandidates.length === 0 && (
            <div className="rounded-xl border border-slate-800 bg-slate-950/50 px-4 py-3 text-sm text-slate-400">
              比較データが見つかりませんでした。
            </div>
          )}

          {currentSnapshot && (
            compareViewMode === 'overlay' && prevSnapshot ? (
              <div className="space-y-2">
                <div className="flex flex-wrap items-center gap-3 text-xs">
                  <span className="inline-flex items-center gap-1.5 font-medium text-green-400"><span className="h-1.5 w-1.5 rounded-full bg-green-400" />解析した動画</span>
                  <span className="inline-flex items-center gap-1.5 font-medium text-slate-300"><span className="h-1.5 w-1.5 rounded-full bg-slate-300" />比較先動画</span>
                  <span className="text-slate-500">ズレの位置を同じ画面で確認できます</span>
                </div>
                <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950" style={{ aspectRatio: '16/9' }}>
                  {prevBlobUrl && (
                    <video
                      ref={prevVideoRef}
                      src={prevBlobUrl}
                      preload="metadata"
                      className={`absolute inset-0 h-full w-full object-contain ${overlayRenderMode === 'video' ? '' : 'opacity-0'}`}
                      style={overlayRenderMode === 'video' ? { opacity: overlayOpacity } : undefined}
                    />
                  )}
                  {currentBlobUrl ? (
                    <video
                      ref={currentVideoRef}
                      src={currentBlobUrl}
                      preload="metadata"
                      className={`absolute inset-0 h-full w-full object-contain ${overlayRenderMode === 'video' ? '' : 'opacity-0'}`}
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center text-sm text-slate-500">動画データなし</div>
                  )}
                  <canvas ref={prevOverlayRef} width={CANVAS_W} height={CANVAS_H} className="absolute inset-0 h-full w-full opacity-90" style={{ pointerEvents: 'none' }} />
                  <canvas ref={currentOverlayRef} width={CANVAS_W} height={CANVAS_H} className="absolute inset-0 h-full w-full" style={{ pointerEvents: 'none' }} />
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1.5">
                  <div className="flex items-center gap-1.5 text-xs font-medium text-green-400">
                    <span className="h-1.5 w-1.5 rounded-full bg-green-400" />
                    解析した動画 — {formatDate(currentSnapshot.timestamp)}
                  </div>
                  <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950" style={{ aspectRatio: '16/9' }}>
                    {currentBlobUrl ? (
                      <video ref={currentVideoRef} src={currentBlobUrl} preload="metadata" className="absolute inset-0 h-full w-full object-contain" />
                    ) : (
                      <div className="flex h-full items-center justify-center text-sm text-slate-500">動画データなし</div>
                    )}
                    <canvas ref={currentOverlayRef} width={CANVAS_W} height={CANVAS_H} className="absolute inset-0 h-full w-full" style={{ pointerEvents: 'none' }} />
                  </div>
                </div>
                <div className="space-y-1.5">
                  <div className="flex items-center gap-1.5 text-xs font-medium text-slate-400">
                    <span className="h-1.5 w-1.5 rounded-full bg-slate-500" />
                    {prevSnapshot ? `比較先動画 — ${formatDate(prevSnapshot.timestamp)}` : '比較先動画'}
                  </div>
                  <div className="relative overflow-hidden rounded-xl border border-slate-800 bg-slate-950" style={{ aspectRatio: '16/9' }}>
                    {prevBlobUrl ? (
                      <video ref={prevVideoRef} src={prevBlobUrl} preload="metadata" className="absolute inset-0 h-full w-full object-contain" />
                    ) : (
                      <div className="flex h-full items-center justify-center px-4 text-center text-sm text-slate-500">
                        {compareTargetCandidates.length === 0 ? '比較先に選べる動画がありません' : '比較先動画を選択してください'}
                      </div>
                    )}
                    <canvas ref={prevOverlayRef} width={CANVAS_W} height={CANVAS_H} className="absolute inset-0 h-full w-full" style={{ pointerEvents: 'none' }} />
                  </div>
                </div>
              </div>
            )
          )}

          <canvas ref={mergeCanvasRef} width={CANVAS_W * 2} height={CANVAS_H} className="hidden" />

          <div className="space-y-2 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={togglePlay}
                className="min-h-[36px] min-w-[80px] rounded-lg bg-green-500 px-4 py-1.5 text-xs font-semibold text-slate-950 transition-colors hover:bg-green-400"
              >
                {isPlaying ? '⏸ 停止' : '▶ 再生'}
              </button>
              <span className="text-xs tabular-nums text-slate-400">{virtualTime.toFixed(1)}s / {maxDuration.toFixed(1)}s</span>
            </div>
            <input type="range" min={0} max={maxDuration || 1} step={FRAME_INTERVAL} value={virtualTime} onChange={handleScrub} className="w-full accent-green-500" />
            <div className="flex items-center gap-2">
              <span className="text-xs text-slate-400">速度</span>
              <div className="flex gap-1.5">
                {([0.25, 0.5, 1] as const).map(rate => (
                  <button
                    key={rate}
                    type="button"
                    onClick={() => {
                      setPlaybackRate(rate);
                      playbackRateRef.current = rate;
                      if (currentVideoRef.current) currentVideoRef.current.playbackRate = rate;
                      if (prevVideoRef.current) prevVideoRef.current.playbackRate = rate;
                    }}
                    className={`min-h-[30px] rounded-lg px-3 py-1 text-xs font-medium transition-colors ${
                      playbackRate === rate ? 'bg-green-500 text-slate-950' : 'border border-slate-700 text-slate-400 hover:border-slate-600 hover:bg-slate-800'
                    }`}
                  >
                    {rate === 1 ? '×1' : rate === 0.5 ? '×½' : '×¼'}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="grid gap-3 lg:grid-cols-[1.2fr_1fr]">
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-slate-200">いま差が大きい部位</p>
                  <p className="text-xs text-slate-500">{virtualTime.toFixed(1)}秒時点</p>
                </div>
                {currentFrameDiffs[0] && (
                  <span className="rounded-lg border border-amber-400/30 bg-amber-400/10 px-2.5 py-1 text-xs font-medium text-amber-300">
                    最大差 {currentFrameDiffs[0].diff.toFixed(1)}°
                  </span>
                )}
              </div>
              {currentFrameDiffs.length > 0 ? (
                <div className="mt-3 grid gap-2 sm:grid-cols-3">
                  {displayedDiffs.slice(0, 3).map((item, index) => (
                    <div key={item.key} className="rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-2">
                      <p className="text-[11px] text-slate-500">#{index + 1}</p>
                      <p className="mt-1 text-sm font-medium text-slate-200">{item.label}</p>
                      <p className="mt-1 text-lg font-semibold text-amber-300">{item.diff.toFixed(1)}°</p>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="mt-3 text-sm text-slate-500">比較先動画を選ぶと差分が表示されます。</p>
              )}
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
              <p className="text-sm font-medium text-slate-200">ズレが大きい瞬間</p>
              <p className="mt-1 text-xs text-slate-500">気になる場面へすぐ移動できます</p>
              {topDiffMoments.length > 0 ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  {topDiffMoments.map(moment => (
                    <button
                      key={`${moment.time}-${moment.topLabel}`}
                      type="button"
                      onClick={() => jumpToTime(moment.time)}
                      className="rounded-lg border border-slate-700 bg-slate-900/80 px-3 py-2 text-left text-xs text-slate-300 transition-colors hover:border-green-500/40 hover:bg-slate-800"
                    >
                      <span className="block font-medium text-green-400">{moment.time.toFixed(1)}秒</span>
                      <span className="mt-1 block text-slate-400">{moment.topLabel} {moment.topDiff.toFixed(1)}°</span>
                    </button>
                  ))}
                </div>
              ) : (
                <p className="mt-3 text-sm text-slate-500">比較先動画を選ぶと表示されます。</p>
              )}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {!isRecording ? (
              <button
                type="button"
                onClick={handleStartRecording}
                className="min-h-[40px] rounded-xl border border-slate-700 px-4 py-2 text-sm text-slate-300 transition-colors hover:border-slate-600 hover:bg-slate-800"
              >
                ⬇ 比較動画を書き出す (WebM)
              </button>
            ) : (
              <button
                type="button"
                onClick={handleStopRecording}
                className="min-h-[40px] rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-2 text-sm text-red-400 transition-colors hover:bg-red-500/20"
              >
                ⏹ 停止してダウンロード
              </button>
            )}
            <button
              type="button"
              onClick={handleReset}
              className="min-h-[40px] rounded-xl border border-slate-700 px-4 py-2 text-sm text-slate-300 transition-colors hover:border-slate-600 hover:bg-slate-800"
            >
              別の型を比較
            </button>
          </div>

          <details className="rounded-xl border border-slate-800 bg-slate-950/60 p-4">
            <summary className="cursor-pointer list-none text-sm font-medium text-slate-200">
              保存済み解析結果を管理 <span className="ml-2 text-xs text-slate-500">({compareCandidates.length}件)</span>
            </summary>
            <div className="mt-3 space-y-2">
              {compareCandidates.map(snap => (
                <div key={snap.id} className="flex items-center justify-between gap-3 rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-2">
                  <div className="min-w-0">
                    <p className="text-sm text-slate-200">{formatDate(snap.timestamp)}</p>
                    <p className="text-xs text-slate-500">{snap.duration.toFixed(1)}秒</p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => void handleSelectCurrent(snap)}
                      className="min-h-[34px] rounded-lg border border-slate-700 px-3 py-1 text-xs text-slate-300 transition-colors hover:border-slate-600 hover:bg-slate-800"
                    >
                      左へ
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleDeleteSnapshot(snap.id)}
                      className="min-h-[34px] rounded-lg border border-red-500/30 px-3 py-1 text-xs text-red-300 transition-colors hover:bg-red-500/10"
                    >
                      削除
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </details>
        </div>
      )}
    </section>
  );
}
