"""
RT-DETR fine-tuning スクリプト（刀検知用）。

事前準備:
  pip install ultralytics

データセット構成（Roboflow/YOLO形式）:
  dataset/
    images/train/   ← 学習用画像
    images/val/     ← 検証用画像
    labels/train/   ← YOLO形式ラベル (.txt)
    labels/val/
    katana.yaml     ← このスクリプトが自動生成

使い方:
  python scripts/train_rtdetr.py
  python scripts/train_rtdetr.py --epochs 150 --model rtdetr-l.pt
"""

import argparse
import subprocess
import sys
from pathlib import Path

YAML_TEMPLATE = """\
path: {dataset_path}
train: images/train
val: images/val

nc: 1
names:
  0: katana
"""

def check_ultralytics():
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("ultralytics が見つかりません。インストールします...")
        subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics"], check=True)

def main():
    parser = argparse.ArgumentParser(description="RT-DETR 刀検知モデルのfine-tuning")
    parser.add_argument("--dataset", default="dataset",    help="データセットのルートディレクトリ")
    parser.add_argument("--model",   default="rtdetr-l.pt", help="ベースモデル (rtdetr-l.pt / rtdetr-x.pt)")
    parser.add_argument("--epochs",  type=int, default=100, help="学習エポック数")
    parser.add_argument("--imgsz",   type=int, default=640, help="入力画像サイズ")
    parser.add_argument("--batch",   type=int, default=8,   help="バッチサイズ（VRAMに合わせて調整）")
    parser.add_argument("--export-only", action="store_true", help="学習済みモデルをONNXにエクスポートするだけ")
    args = parser.parse_args()

    check_ultralytics()
    from ultralytics import RTDETR

    dataset_path = Path(args.dataset).resolve()
    yaml_path = dataset_path / "katana.yaml"

    # YAML生成（なければ）
    if not yaml_path.exists():
        yaml_path.write_text(YAML_TEMPLATE.format(dataset_path=str(dataset_path)))
        print(f"データセット設定を生成: {yaml_path}")

    if args.export_only:
        # 最新の学習済みweightsをONNXにエクスポート
        runs_dir = Path("runs/detect")
        weight_candidates = sorted(runs_dir.rglob("best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not weight_candidates:
            print("エラー: 学習済みモデルが見つかりません (runs/detect/**/best.pt)")
            sys.exit(1)
        best_pt = weight_candidates[0]
        print(f"エクスポート: {best_pt}")
        model = RTDETR(str(best_pt))
        model.export(format="onnx", imgsz=args.imgsz, simplify=True)
        onnx_path = best_pt.with_suffix(".onnx")
        print(f"\nONNXモデル: {onnx_path}")
        print(f"サイズ: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
        print("\nブラウザ統合用にコピー:")
        print(f"  cp {onnx_path} public/models/katana_detector.onnx")
        return

    # 学習
    print(f"モデル: {args.model} / エポック: {args.epochs} / バッチ: {args.batch}")
    model = RTDETR(args.model)
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=30,       # 30エポック改善なしで早期終了
        save=True,
        plots=True,
        project="runs/detect",
        name="katana",
    )

    # 自動ONNX export
    print("\n学習完了。ONNXにエクスポート中...")
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    model_best = RTDETR(str(best_pt))
    model_best.export(format="onnx", imgsz=args.imgsz, simplify=True)

    onnx_path = best_pt.with_suffix(".onnx")
    print(f"\nONNXモデル: {onnx_path}")
    print(f"サイズ: {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("\n次のステップ:")
    print("  1. cp <onnx_path> public/models/katana_detector.onnx")
    print("  2. onnxruntime-web でブラウザ統合（実装予定）")

if __name__ == "__main__":
    main()
