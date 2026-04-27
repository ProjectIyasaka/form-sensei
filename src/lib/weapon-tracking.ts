import type { LandmarkPoint } from './storage';

export type WeaponTipPoint = { x: number; y: number; time: number };

export type WeaponConfig = {
  lengthMultiplier: number;
  dominantHand: 'right' | 'left' | 'auto';
  trailColor: string;
  trailWidth: number;
  trailFade: boolean;
};

export function estimateWeaponTip(
  landmarks: LandmarkPoint[],
  config: WeaponConfig,
  currentTime: number,
): { tip: WeaponTipPoint | null; hand: 'right' | 'left' | 'both' } {
  const lShoulder = landmarks[11];
  const rShoulder = landmarks[12];
  const lElbow    = landmarks[13];
  const rElbow    = landmarks[14];
  const lWrist    = landmarks[15];
  const rWrist    = landmarks[16];

  if (!lWrist || !rWrist || !lElbow || !rElbow) {
    return { tip: null, hand: 'right' };
  }

  const wristDist = Math.hypot(rWrist.x - lWrist.x, rWrist.y - lWrist.y);

  if (wristDist < 0.15) {
    const midWrist    = { x: (lWrist.x + rWrist.x) / 2,    y: (lWrist.y + rWrist.y) / 2 };
    const midShoulder = (lShoulder && rShoulder)
      ? { x: (lShoulder.x + rShoulder.x) / 2, y: (lShoulder.y + rShoulder.y) / 2 }
      : midWrist;
    const midElbow = { x: (lElbow.x + rElbow.x) / 2, y: (lElbow.y + rElbow.y) / 2 };

    const dx = midWrist.x - midShoulder.x;
    const dy = midWrist.y - midShoulder.y;
    const len = Math.hypot(dx, dy) || 1;
    const forearmLen = Math.hypot(midWrist.x - midElbow.x, midWrist.y - midElbow.y);

    return {
      tip: {
        x: midWrist.x + (dx / len) * forearmLen * config.lengthMultiplier,
        y: midWrist.y + (dy / len) * forearmLen * config.lengthMultiplier,
        time: currentTime,
      },
      hand: 'both',
    };
  }

  let wrist: LandmarkPoint;
  let elbow: LandmarkPoint;
  let hand: 'right' | 'left';

  if (config.dominantHand === 'right') {
    wrist = rWrist; elbow = rElbow; hand = 'right';
  } else if (config.dominantHand === 'left') {
    wrist = lWrist; elbow = lElbow; hand = 'left';
  } else {
    if (lWrist.y <= rWrist.y) {
      wrist = lWrist; elbow = lElbow; hand = 'left';
    } else {
      wrist = rWrist; elbow = rElbow; hand = 'right';
    }
  }

  return {
    tip: {
      x: wrist.x + (wrist.x - elbow.x) * config.lengthMultiplier,
      y: wrist.y + (wrist.y - elbow.y) * config.lengthMultiplier,
      time: currentTime,
    },
    hand,
  };
}

export function drawWeaponTrail(
  ctx: CanvasRenderingContext2D,
  points: WeaponTipPoint[],
  currentTime: number,
  config: WeaponConfig,
  canvasWidth: number,
  canvasHeight: number,
  fadeDuration = 3.0,
): void {
  if (points.length < 2) return;

  for (let i = 1; i < points.length; i++) {
    const prev = points[i - 1];
    const curr = points[i];

    let alpha = 1;
    if (config.trailFade) {
      const age = currentTime - curr.time;
      alpha = Math.max(0, 1 - age / fadeDuration);
    }
    if (alpha <= 0) continue;

    ctx.beginPath();
    ctx.strokeStyle = config.trailColor;
    ctx.lineWidth = config.trailWidth;
    ctx.globalAlpha = alpha;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.moveTo(prev.x * canvasWidth, prev.y * canvasHeight);
    ctx.lineTo(curr.x * canvasWidth, curr.y * canvasHeight);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

export function drawWeaponTip(
  ctx: CanvasRenderingContext2D,
  tip: { x: number; y: number } | null,
  config: WeaponConfig,
  canvasWidth: number,
  canvasHeight: number,
): void {
  if (!tip) return;
  ctx.beginPath();
  ctx.arc(tip.x * canvasWidth, tip.y * canvasHeight, 6, 0, Math.PI * 2);
  ctx.fillStyle = config.trailColor;
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();
}
