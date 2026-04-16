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

  // 同じ型名は最大 MAX_PER_KATA 件（古い順に削除）
  const kataItems = snapshots.filter(s => s.kataName === snapshot.kataName);
  if (kataItems.length > MAX_PER_KATA) {
    const sorted = [...kataItems].sort((a, b) => a.timestamp - b.timestamp);
    const removeIds = new Set(sorted.slice(0, kataItems.length - MAX_PER_KATA).map(s => s.id));
    snapshots = snapshots.filter(s => !removeIds.has(s.id));
  }

  // 全体上限
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
