"""
動画からアノテーション用フレームを抽出するスクリプト。
OpenCV を使用（ffmpeg不要）。ブレたフレームを自動除外。

事前準備:
  pip install opencv-python

使い方:
  python scripts/extract_frames.py --input videos/iaido.mp4
  python scripts/extract_frames.py --input videos/iaido.mp4 --dest val
  python scripts/extract_frames.py --input videos/ --fps 2 --max 100
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
OUTPUT_SIZE = (640, 360)


def sharpness(frame: np.ndarray) -> float:
    """ラプラシアン分散によるシャープネス計算（高いほど鮮明）"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_frames(
    video_path: Path,
    out_dir: Path,
    fps: float,
    max_frames: int,
    min_sharpness: float,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  警告: {video_path.name} を開けません")
        return 0

    src_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval  = max(1, int(src_fps / fps))
    stem      = video_path.stem.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_idx = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            score = sharpness(frame)
            if score >= min_sharpness:
                resized  = cv2.resize(frame, OUTPUT_SIZE)
                out_path = out_dir / f"{stem}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1

        frame_idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description="動画からアノテーション用フレームを抽出")
    parser.add_argument("--input",         required=True, help="動画ファイルまたはディレクトリ")
    parser.add_argument("--dest",          default="train", choices=["train", "val"],
                        help="出力先 train or val（デフォルト: train）")
    parser.add_argument("--fps",           type=float, default=2.0,  help="抽出fps（デフォルト: 2）")
    parser.add_argument("--max",           type=int,   default=200,  help="1動画あたり最大フレーム数")
    parser.add_argument("--min-sharpness", type=float, default=50.0, help="最低シャープネス（低いほど緩い, デフォルト: 50）")
    parser.add_argument("--dataset",       default="dataset",        help="データセットルート")
    args = parser.parse_args()

    try:
        import cv2  # noqa: F401
    except ImportError:
        print("エラー: opencv-python が見つかりません。")
        print("  pip install opencv-python")
        sys.exit(1)

    input_path = Path(args.input)
    out_dir    = Path(args.dataset) / "images" / args.dest

    if input_path.is_file():
        videos = [input_path] if input_path.suffix.lower() in VIDEO_EXTS else []
    elif input_path.is_dir():
        videos = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    else:
        print(f"エラー: {args.input} が見つかりません")
        sys.exit(1)

    if not videos:
        print(f"動画ファイルが見つかりません: {args.input}")
        sys.exit(1)

    print(f"動画: {len(videos)} 件 / fps: {args.fps} / 出力先: {out_dir}")
    print(f"シャープネスフィルタ: {args.min_sharpness}以上のフレームのみ抽出")
    print()

    total = 0
    for i, video in enumerate(videos, 1):
        print(f"[{i}/{len(videos)}] {video.name} ...", end=" ", flush=True)
        n = extract_frames(video, out_dir, args.fps, args.max, args.min_sharpness)
        print(f"{n} フレーム")
        total += n

    print(f"\n完了: 合計 {total} 枚 → {out_dir.resolve()}")
    print()
    print("次のステップ:")
    print(f"  1. labelImg を起動してアノテーション:")
    print(f"       labelImg {out_dir} dataset/classes.txt")
    print(f"     Save Dir: dataset/labels/{args.dest}")
    print()
    print("  2. 学習:")
    print("       python scripts/train_rtdetr.py --epochs 100 --batch 4")


if __name__ == "__main__":
    main()
