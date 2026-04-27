"""
COCO 形式アノテーションを YOLO 形式へ変換するスクリプト。

想定入力:
  dataset/katana.v2i.coco/
    train/
      _annotations.coco.json
      *.jpg
    valid/
      _annotations.coco.json
      *.jpg

出力:
  dataset/
    images/train/
    images/val/
    labels/train/
    labels/val/

使い方:
  python scripts/convert_coco_to_yolo.py
  python scripts/convert_coco_to_yolo.py --input dataset/katana.v2i.coco --output dataset
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

EXPECTED_CLASSES = (
    "katana_tip",
    "katana_grip",
)

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def convert_bbox_to_yolo(bbox, image_width: int, image_height: int):
    x, y, w, h = bbox
    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    width = w / image_width
    height = h / image_height
    return (
        clamp(x_center, 0.0, 1.0),
        clamp(y_center, 0.0, 1.0),
        clamp(width, 0.0, 1.0),
        clamp(height, 0.0, 1.0),
    )


def ensure_dirs(output_root: Path):
    for rel in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_root / rel).mkdir(parents=True, exist_ok=True)


def validate_categories(categories):
    category_names = [item["name"] for item in categories]
    missing = [name for name in EXPECTED_CLASSES if name not in category_names]
    extra = [name for name in category_names if name not in EXPECTED_CLASSES]

    if missing or extra:
        print("エラー: COCO のカテゴリ定義が想定と一致しません。")
        print(f"想定: {', '.join(EXPECTED_CLASSES)}")
        print(f"実際: {', '.join(category_names) if category_names else '(none)'}")
        if missing:
            print(f"不足: {', '.join(missing)}")
        if extra:
            print(f"余分: {', '.join(extra)}")
        print()
        print("Roboflow 側のクラス名を見直してください。")
        sys.exit(1)


def convert_split(split_dir: Path, output_root: Path, split_name: str):
    annotations_path = split_dir / "_annotations.coco.json"
    if not annotations_path.exists():
        print(f"スキップ: {annotations_path} が見つかりません")
        return 0, 0

    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    validate_categories(data.get("categories", []))
    images = {item["id"]: item for item in data.get("images", [])}
    categories = {
        item["id"]: EXPECTED_CLASSES.index(item["name"])
        for item in data.get("categories", [])
    }
    image_annotations = {}

    for ann in data.get("annotations", []):
        image_annotations.setdefault(ann["image_id"], []).append(ann)

    image_out_dir = output_root / "images" / split_name
    label_out_dir = output_root / "labels" / split_name

    copied = 0
    labeled = 0

    for image_id, image_info in images.items():
        file_name = image_info["file_name"]
        source_image = split_dir / file_name
        target_image = image_out_dir / file_name
        target_label = label_out_dir / f"{Path(file_name).stem}.txt"

        if not source_image.exists():
            print(f"警告: 画像が見つかりません: {source_image}")
            continue

        shutil.copy2(source_image, target_image)
        copied += 1

        width = image_info["width"]
        height = image_info["height"]
        lines = []

        for ann in image_annotations.get(image_id, []):
            if ann.get("iscrowd"):
                continue
            bbox = ann.get("bbox")
            if not bbox or width <= 0 or height <= 0:
                continue
            class_id = categories.get(ann["category_id"], 0)
            x_center, y_center, box_w, box_h = convert_bbox_to_yolo(bbox, width, height)
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

        target_label.write_text("\n".join(lines), encoding="utf-8")
        if lines:
            labeled += 1

    return copied, labeled


def main():
    parser = argparse.ArgumentParser(description="COCO アノテーションを YOLO 形式へ変換")
    parser.add_argument("--input", default="dataset/katana.v2i.coco", help="COCO エクスポートのルート")
    parser.add_argument("--output", default="dataset", help="YOLO 形式の出力先")
    args = parser.parse_args()

    input_root = Path(args.input).resolve()
    output_root = Path(args.output).resolve()

    if not input_root.exists():
        print(f"エラー: 入力ディレクトリが見つかりません: {input_root}")
        sys.exit(1)

    ensure_dirs(output_root)

    total_images = 0
    total_labeled = 0

    for input_name, output_name in SPLIT_MAP.items():
        split_dir = input_root / input_name
        if not split_dir.exists():
            continue
        copied, labeled = convert_split(split_dir, output_root, output_name)
        total_images += copied
        total_labeled += labeled
        print(f"{input_name} -> {output_name}: {copied} images / {labeled} labeled")

    if total_images == 0:
        print("エラー: 変換対象の画像が見つかりませんでした。")
        sys.exit(1)

    print()
    print(f"完了: {total_images} images / {total_labeled} labeled")
    print(f"出力先: {output_root}")
    print("次のステップ:")
    print("  1. python scripts/train_rtdetr.py")
    print("  2. 学習完了後に ONNX を public/models 配下へ配置")


if __name__ == "__main__":
    main()
