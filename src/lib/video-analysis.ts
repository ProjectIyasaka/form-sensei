import type { PoseLandmarkerResult } from '@mediapipe/tasks-vision';
import type { FrameAngle, LandmarkPoint, VideoFrame } from './storage';

export const FRAME_INTERVAL = 1 / 30;
export const ANALYSIS_INTERVAL = 1 / 15;
export const CANVAS_W = 640;
export const CANVAS_H = 360;
export const DIFF_WINDOW_SECONDS = 0.6;

export const ANGLE_CONFIG = [
  { key: 'leftShoulder', label: '左肩', points: [13, 11, 23] as const },
  { key: 'rightShoulder', label: '右肩', points: [14, 12, 24] as const },
  { key: 'leftElbow', label: '左肘', points: [11, 13, 15] as const },
  { key: 'rightElbow', label: '右肘', points: [12, 14, 16] as const },
  { key: 'leftHip', label: '左股関節', points: [11, 23, 25] as const },
  { key: 'rightHip', label: '右股関節', points: [12, 24, 26] as const },
  { key: 'leftKnee', label: '左膝', points: [23, 25, 27] as const },
  { key: 'rightKnee', label: '右膝', points: [24, 26, 28] as const },
] as const;

type LandmarkLike = { x: number; y: number; z?: number };

export type AngleDiffSummary = { key: string; label: string; diff: number };
export type DiffMoment = { time: number; totalDiff: number; topLabel: string; topDiff: number };

export function calculateAngle(a: LandmarkLike, b: LandmarkLike, c: LandmarkLike): number | null {
  const vA = { x: a.x - b.x, y: a.y - b.y, z: (a.z ?? 0) - (b.z ?? 0) };
  const vB = { x: c.x - b.x, y: c.y - b.y, z: (c.z ?? 0) - (b.z ?? 0) };
  const dot = vA.x * vB.x + vA.y * vB.y + vA.z * vB.z;
  const magA = Math.hypot(vA.x, vA.y, vA.z);
  const magB = Math.hypot(vB.x, vB.y, vB.z);
  if (magA === 0 || magB === 0) return null;
  return Math.round((Math.acos(Math.min(1, Math.max(-1, dot / (magA * magB)))) * 180) / Math.PI * 10) / 10;
}

export function buildAngles(result: PoseLandmarkerResult): FrameAngle[] {
  const landmarks = result.landmarks[0];
  if (!landmarks) return ANGLE_CONFIG.map(({ key, label }) => ({ key, label, value: null }));
  return ANGLE_CONFIG.map(({ key, label, points }) => {
    const [aIdx, bIdx, cIdx] = points;
    const a = landmarks[aIdx];
    const b = landmarks[bIdx];
    const c = landmarks[cIdx];
    if (!a || !b || !c) return { key, label, value: null };
    return { key, label, value: calculateAngle(a, b, c) };
  });
}

export function normalizeLandmarks(landmarks: LandmarkPoint[] | null): LandmarkPoint[] | null {
  if (!landmarks) return null;
  return landmarks.map(point => ({
    x: Math.round(point.x * 10000) / 10000,
    y: Math.round(point.y * 10000) / 10000,
  }));
}

export function formatDate(ts: number): string {
  return new Date(ts).toLocaleString('ja-JP', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function getFrameAtTime(frames: VideoFrame[], time: number): VideoFrame | null {
  if (frames.length === 0) return null;
  let nearest = frames[0];
  let nearestDistance = Math.abs(frames[0].time - time);
  for (let i = 1; i < frames.length; i++) {
    const distance = Math.abs(frames[i].time - time);
    if (distance < nearestDistance) {
      nearest = frames[i];
      nearestDistance = distance;
    }
  }
  return nearest;
}

export function getFrameLandmarks(frames: VideoFrame[], time: number): LandmarkPoint[] | null {
  return getFrameAtTime(frames, time)?.landmarks ?? null;
}

export function getInterpolatedLandmarks(frames: VideoFrame[], time: number): LandmarkPoint[] | null {
  if (frames.length === 0) return null;
  if (frames.length === 1) return frames[0].landmarks ?? null;

  let previous = frames[0];
  let next = frames[frames.length - 1];

  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    if (frame.time <= time) previous = frame;
    if (frame.time >= time) {
      next = frame;
      break;
    }
  }

  const previousLandmarks = previous.landmarks;
  const nextLandmarks = next.landmarks;
  if (!previousLandmarks || !nextLandmarks) {
    return previousLandmarks ?? nextLandmarks ?? null;
  }
  if (previous.time === next.time) return previousLandmarks;
  if (previousLandmarks.length !== nextLandmarks.length) return previousLandmarks;

  const ratio = Math.max(0, Math.min(1, (time - previous.time) / (next.time - previous.time)));
  return previousLandmarks.map((point, index) => {
    const nextPoint = nextLandmarks[index];
    if (!nextPoint) return point;
    return {
      x: point.x + (nextPoint.x - point.x) * ratio,
      y: point.y + (nextPoint.y - point.y) * ratio,
      z: point.z != null && nextPoint.z != null ? point.z + (nextPoint.z - point.z) * ratio : point.z,
    };
  });
}

export function getInterpolatedFrame(frames: VideoFrame[], time: number): VideoFrame | null {
  if (frames.length === 0) return null;
  if (frames.length === 1) return frames[0];

  let previous = frames[0];
  let next = frames[frames.length - 1];

  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];
    if (frame.time <= time) previous = frame;
    if (frame.time >= time) {
      next = frame;
      break;
    }
  }

  if (previous.time === next.time) return previous;

  const ratio = Math.max(0, Math.min(1, (time - previous.time) / (next.time - previous.time)));
  const landmarks = getInterpolatedLandmarks([previous, next], time);
  const angles = previous.angles.map((angle, index) => {
    const nextAngle = next.angles[index];
    if (!nextAngle || angle.value == null || nextAngle.value == null) {
      return {
        key: angle.key,
        label: angle.label,
        value: angle.value,
      };
    }
    return {
      key: angle.key,
      label: angle.label,
      value: Math.round((angle.value + (nextAngle.value - angle.value) * ratio) * 10) / 10,
    };
  });

  return {
    time,
    angles,
    landmarks,
  };
}

export function getAngleDiffs(currentFrame: VideoFrame | null, prevFrame: VideoFrame | null): AngleDiffSummary[] {
  if (!currentFrame || !prevFrame) return [];
  return currentFrame.angles
    .map((angle, index) => {
      const prevAngle = prevFrame.angles[index];
      if (!prevAngle || angle.value == null || prevAngle.value == null) return null;
      return {
        key: angle.key,
        label: angle.label,
        diff: Math.abs(angle.value - prevAngle.value),
      };
    })
    .filter((item): item is AngleDiffSummary => item != null)
    .sort((a, b) => b.diff - a.diff);
}

export function getAveragedAngleDiffs(
  currentFrames: VideoFrame[],
  prevFrames: VideoFrame[],
  centerTime: number,
  windowSeconds = 0.4,
): AngleDiffSummary[] {
  const halfWindow = windowSeconds / 2;
  const aggregate = new Map<string, { label: string; total: number; count: number }>();

  for (let time = Math.max(0, centerTime - halfWindow); time <= centerTime + halfWindow; time += FRAME_INTERVAL) {
    const currentFrame = getInterpolatedFrame(currentFrames, time);
    const prevFrame = getInterpolatedFrame(prevFrames, time);
    const diffs = getAngleDiffs(currentFrame, prevFrame);
    for (const diff of diffs) {
      const current = aggregate.get(diff.key) ?? { label: diff.label, total: 0, count: 0 };
      current.total += diff.diff;
      current.count += 1;
      aggregate.set(diff.key, current);
    }
  }

  return Array.from(aggregate.entries())
    .map(([key, value]) => ({
      key,
      label: value.label,
      diff: Math.round((value.total / Math.max(1, value.count)) * 10) / 10,
    }))
    .sort((a, b) => b.diff - a.diff);
}

export function getTopDiffMoments(currentFrames: VideoFrame[], prevFrames: VideoFrame[], limit = 4): DiffMoment[] {
  const total = Math.min(currentFrames.length, prevFrames.length);
  const moments: DiffMoment[] = [];
  for (let i = 0; i < total; i++) {
    const currentFrame = currentFrames[i];
    const prevFrame = prevFrames[i];
    const diffs = getAveragedAngleDiffs(currentFrames, prevFrames, currentFrame.time, DIFF_WINDOW_SECONDS);
    if (diffs.length === 0) continue;
    const totalDiff = diffs.reduce((sum, item) => sum + item.diff, 0);
    const top = diffs[0];
    moments.push({
      time: currentFrame.time,
      totalDiff: Math.round(totalDiff * 10) / 10,
      topLabel: top.label,
      topDiff: top.diff,
    });
  }
  return moments
    .sort((a, b) => b.totalDiff - a.totalDiff)
    .filter((moment, index, arr) => arr.findIndex(item => Math.abs(item.time - moment.time) < 0.3) === index)
    .slice(0, limit);
}
