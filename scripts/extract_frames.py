"""
動画からフレームを抽出してアノテーション用画像を生成するスクリプト。

使い方:
  python scripts/extract_frames.py --input videos/
  python scripts/extract_frames.py --input videos/iaido.mp4 --fps 2 --max 300
"""

import argparse
import subprocess
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("エラー: ffmpeg が見つかりません。")
        print("インストール: https://ffmpeg.org/download.html")
        sys.exit(1)

def extract_frames(video_path: Path, out_dir: Path, fps: float, max_frames: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem.replace(" ", "_")
    pattern = out_dir / f"{stem}_%04d.jpg"

    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps},scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2",
        "-q:v", "2",         # JPEG品質（低いほど高品質, 2=高品質）
        "-frames:v", str(max_frames),
        str(pattern),
        "-y", "-loglevel", "error",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  警告: {video_path.name} の抽出でエラー: {result.stderr.strip()}")
        return 0

    extracted = list(out_dir.glob(f"{stem}_*.jpg"))
    return len(extracted)

def main():
    parser = argparse.ArgumentParser(description="動画からアノテーション用フレームを抽出")
    parser.add_argument("--input",  required=True, help="動画ファイルまたは動画が入ったディレクトリ")
    parser.add_argument("--output", default="dataset/images", help="出力ディレクトリ（デフォルト: dataset/images）")
    parser.add_argument("--fps",    type=float, default=2.0, help="抽出フレームレート（デフォルト: 2fps）")
    parser.add_argument("--max",    type=int,   default=500, help="1動画あたりの最大フレーム数（デフォルト: 500）")
    args = parser.parse_args()

    check_ffmpeg()

    input_path = Path(args.input)
    out_dir = Path(args.output)

    if input_path.is_file():
        videos = [input_path] if input_path.suffix.lower() in VIDEO_EXTS else []
    elif input_path.is_dir():
        videos = [p for p in input_path.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    else:
        print(f"エラー: {args.input} が見つかりません")
        sys.exit(1)

    if not videos:
        print(f"動画ファイルが見つかりません: {args.input}")
        sys.exit(1)

    print(f"動画: {len(videos)} 件 / fps: {args.fps} / 最大フレーム: {args.max}枚/動画")
    print(f"出力先: {out_dir.resolve()}")
    print()

    total = 0
    for i, video in enumerate(sorted(videos), 1):
        print(f"[{i}/{len(videos)}] {video.name} を処理中...", end=" ", flush=True)
        n = extract_frames(video, out_dir, args.fps, args.max)
        print(f"{n} フレーム抽出")
        total += n

    print(f"\n完了: 合計 {total} 枚 → {out_dir.resolve()}")
    print()
    print("次のステップ:")
    print("  1. Roboflow (https://roboflow.com) にアップロードしてバウンディングボックスをアノテーション")
    print("     または")
    print("     Label Studio (pip install label-studio) でローカルアノテーション")
    print("  2. YOLO形式でエクスポート")
    print("  3. python scripts/train_rtdetr.py で学習開始")

if __name__ == "__main__":
    main()
