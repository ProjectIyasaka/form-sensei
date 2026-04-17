export type AngleSnapshot = {
  id: string;
  kataName: string;
  timestamp: number;
  imageDataUrl: string;
  angles: {
    key: string;
    label: string;
    value: number | null;
  }[];
};

const STORAGE_KEY = 'formsensei_snapshots';
const MAX_PER_KATA = 10;
const MAX_TOTAL = 50;

function load(): AngleSnapshot[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as AngleSnapshot[]) : [];
  } catch {
    return [];
  }
}

function persist(snapshots: AngleSnapshot[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshots));
}

export function saveSnapshot(snapshot: AngleSnapshot): void {
  let snapshots = load();
  snapshots.push(snapshot);

  const kataItems = snapshots.filter(s => s.kataName === snapshot.kataName);
  if (kataItems.length > MAX_PER_KATA) {
    const sorted = [...kataItems].sort((a, b) => a.timestamp - b.timestamp);
    const removeIds = new Set(sorted.slice(0, kataItems.length - MAX_PER_KATA).map(s => s.id));
    snapshots = snapshots.filter(s => !removeIds.has(s.id));
  }

  if (snapshots.length > MAX_TOTAL) {
    snapshots.sort((a, b) => a.timestamp - b.timestamp);
    snapshots = snapshots.slice(snapshots.length - MAX_TOTAL);
  }

  persist(snapshots);
}

export function getSnapshotsByKata(kataName: string): AngleSnapshot[] {
  return load()
    .filter(s => s.kataName === kataName)
    .sort((a, b) => b.timestamp - a.timestamp);
}

export function getAllKataNames(): string[] {
  const snapshots = load();
  const seen = new Set<string>();
  const result: string[] = [];
  for (const s of [...snapshots].sort((a, b) => b.timestamp - a.timestamp)) {
    if (!seen.has(s.kataName)) {
      seen.add(s.kataName);
      result.push(s.kataName);
    }
  }
  return result;
}

export function deleteSnapshot(id: string): void {
  persist(load().filter(s => s.id !== id));
}

export function compressImageToDataUrl(file: File, maxWidth = 200): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const src = reader.result as string;
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, maxWidth / img.naturalWidth);
        const w = Math.round(img.naturalWidth * scale);
        const h = Math.round(img.naturalHeight * scale);
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        if (!ctx) { reject(new Error('Canvas context unavailable')); return; }
        ctx.drawImage(img, 0, 0, w, h);
        resolve(canvas.toDataURL('image/jpeg', 0.8));
      };
      img.onerror = () => reject(new Error('画像の読み込みに失敗しました。'));
      img.src = src;
    };
    reader.onerror = () => reject(new Error('ファイルの読み込みに失敗しました。'));
    reader.readAsDataURL(file);
  });
}

// ── Video snapshots ───────────────────────────────────

export type LandmarkPoint = { x: number; y: number; z?: number };
export type FrameAngle = { key: string; label: string; value: number | null };
export type VideoFrame = {
  time: number;
  angles: FrameAngle[];
  landmarks: LandmarkPoint[] | null;
};
export type VideoSnapshot = {
  id: string;
  kataName: string;
  timestamp: number;
  duration: number;
  frames: VideoFrame[];
  thumbnailDataUrl?: string;
};

const VIDEO_STORAGE_KEY = 'formsensei_video_snapshots';
const MAX_VIDEO_PER_KATA = 5;
const MAX_VIDEO_TOTAL = 20;

function loadVideo(): VideoSnapshot[] {
  try {
    const raw = localStorage.getItem(VIDEO_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as VideoSnapshot[]) : [];
  } catch {
    return [];
  }
}

function persistVideo(snapshots: VideoSnapshot[]): void {
  localStorage.setItem(VIDEO_STORAGE_KEY, JSON.stringify(snapshots));
}

export function saveVideoSnapshot(snapshot: VideoSnapshot): void {
  let snapshots = loadVideo();
  snapshots.push(snapshot);

  const kataItems = snapshots.filter(s => s.kataName === snapshot.kataName);
  if (kataItems.length > MAX_VIDEO_PER_KATA) {
    const sorted = [...kataItems].sort((a, b) => a.timestamp - b.timestamp);
    const removeIds = new Set(sorted.slice(0, kataItems.length - MAX_VIDEO_PER_KATA).map(s => s.id));
    snapshots = snapshots.filter(s => !removeIds.has(s.id));
  }

  if (snapshots.length > MAX_VIDEO_TOTAL) {
    snapshots.sort((a, b) => a.timestamp - b.timestamp);
    snapshots = snapshots.slice(snapshots.length - MAX_VIDEO_TOTAL);
  }

  persistVideo(snapshots);
}

export function updateVideoSnapshotThumbnail(id: string, thumbnailDataUrl: string): void {
  const snapshots = loadVideo().map(snapshot =>
    snapshot.id === id ? { ...snapshot, thumbnailDataUrl } : snapshot,
  );
  persistVideo(snapshots);
}

export function getVideoSnapshotsByKata(kataName: string): VideoSnapshot[] {
  return loadVideo()
    .filter(s => s.kataName === kataName)
    .sort((a, b) => b.timestamp - a.timestamp);
}

export function getAllVideoKataNames(): string[] {
  const snapshots = loadVideo();
  const seen = new Set<string>();
  const result: string[] = [];
  for (const s of [...snapshots].sort((a, b) => b.timestamp - a.timestamp)) {
    if (!seen.has(s.kataName)) {
      seen.add(s.kataName);
      result.push(s.kataName);
    }
  }
  return result;
}

export function deleteVideoSnapshot(id: string): void {
  persistVideo(loadVideo().filter(s => s.id !== id));
}

// ── Video blob storage (IndexedDB) ────────────────────────────────────
const VIDEO_BLOB_DB = 'formsensei_videoblobs';
const BLOB_STORE = 'blobs';

function openBlobDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(VIDEO_BLOB_DB, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(BLOB_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function saveVideoBlob(id: string, blob: Blob): Promise<void> {
  const db = await openBlobDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(BLOB_STORE, 'readwrite');
    tx.objectStore(BLOB_STORE).put(blob, id);
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onerror = () => { db.close(); reject(tx.error); };
  });
}

export async function getVideoBlob(id: string): Promise<Blob | null> {
  const db = await openBlobDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(BLOB_STORE, 'readonly');
    const req = tx.objectStore(BLOB_STORE).get(id);
    req.onsuccess = () => { db.close(); resolve((req.result as Blob) ?? null); };
    req.onerror = () => { db.close(); reject(req.error); };
  });
}

export async function deleteVideoBlob(id: string): Promise<void> {
  const db = await openBlobDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(BLOB_STORE, 'readwrite');
    tx.objectStore(BLOB_STORE).delete(id);
    tx.oncomplete = () => { db.close(); resolve(); };
    tx.onerror = () => { db.close(); reject(tx.error); };
  });
}

export function createVideoThumbnailDataUrl(blob: Blob, seekTime = 0.1, maxWidth = 240): Promise<string> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    const url = URL.createObjectURL(blob);
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error('動画サムネイル生成がタイムアウトしました。'));
    }, 4000);

    const cleanup = () => {
      clearTimeout(timeout);
      URL.revokeObjectURL(url);
      video.removeAttribute('src');
      video.load();
    };

    video.preload = 'metadata';
    video.muted = true;
    video.playsInline = true;

    video.onloadedmetadata = () => {
      const targetTime = Math.min(Math.max(seekTime, 0), Math.max((video.duration || 0) - 0.05, 0));
      video.currentTime = targetTime;
    };

    video.onseeked = () => {
      const width = video.videoWidth || 320;
      const height = video.videoHeight || 180;
      const scale = Math.min(1, maxWidth / width);
      const canvas = document.createElement('canvas');
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        cleanup();
        reject(new Error('Canvas context unavailable'));
        return;
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.82);
      cleanup();
      resolve(dataUrl);
    };

    video.onerror = () => {
      cleanup();
      reject(new Error('動画サムネイルの生成に失敗しました。'));
    };

    video.src = url;
  });
}
