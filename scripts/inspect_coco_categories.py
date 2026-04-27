"""
COCO アノテーションのカテゴリ定義を確認する軽量スクリプト。

使い方:
  python scripts/inspect_coco_categories.py
  python scripts/inspect_coco_categories.py --input dataset/katana.v2i.coco/train/_annotations.coco.json
"""

import argparse
import json
import sys
from pathlib import Path


EXPECTED_CLASSES = (
    "katana_tip",
    "katana_grip",
)


def load_categories(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("categories", [])


def main():
    parser = argparse.ArgumentParser(description="COCO カテゴリ定義の確認")
    parser.add_argument(
        "--input",
        default="dataset/katana.v2i.coco/train/_annotations.coco.json",
        help="確認対象の COCO annotation json",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    if not input_path.exists():
        print(f"エラー: ファイルが見つかりません: {input_path}")
        sys.exit(1)

    categories = load_categories(input_path)
    if not categories:
        print("カテゴリ定義が見つかりません。")
        sys.exit(1)

    category_names = [item.get("name", "(unknown)") for item in categories]
    print(f"file: {input_path}")
    print("categories:")
    for index, name in enumerate(category_names):
        print(f"  {index}: {name}")

    missing = [name for name in EXPECTED_CLASSES if name not in category_names]
    extra = [name for name in category_names if name not in EXPECTED_CLASSES]

    print()
    print(f"expected: {', '.join(EXPECTED_CLASSES)}")
    print("status:", "OK" if not missing and not extra else "MISMATCH")

    if missing:
        print(f"missing: {', '.join(missing)}")
    if extra:
        print(f"extra: {', '.join(extra)}")

    sys.exit(0 if not missing and not extra else 1)


if __name__ == "__main__":
    main()
