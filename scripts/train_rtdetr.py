"""
RT-DETR fine-tuning スクリプト（刀先端 / 柄基準点検知用）
License: Apache 2.0 (PekingU/rtdetr_r18vd + HuggingFace transformers)

事前準備:
  pip install transformers pillow onnx onnxslim

使い方:
  python scripts/train_rtdetr.py
  python scripts/train_rtdetr.py --epochs 100 --batch 4
  python scripts/train_rtdetr.py --export-only
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

CLASS_NAMES = ["katana_tip", "katana_grip"]
ID2LABEL    = {i: name for i, name in enumerate(CLASS_NAMES)}
LABEL2ID    = {name: i for i, name in enumerate(CLASS_NAMES)}
MODEL_ID    = "PekingU/rtdetr_r18vd"
INPUT_SIZE  = 640

REQUIRED_DIRS = ("images/train", "images/val", "labels/train", "labels/val")


def validate_dataset(dataset_path: Path):
    missing = [rel for rel in REQUIRED_DIRS if not (dataset_path / rel).exists()]
    if missing:
        print("エラー: データセット構成が不足しています。")
        for rel in missing:
            print(f"  - {rel}")
        sys.exit(1)
    train_imgs = list((dataset_path / "images/train").glob("*"))
    val_imgs   = list((dataset_path / "images/val").glob("*"))
    if not train_imgs:
        print("エラー: train データが空です。")
        sys.exit(1)
    print(f"データセット確認: train {len(train_imgs)} 枚 / val {len(val_imgs)} 枚")


class KatanaDataset(Dataset):
    def __init__(self, image_dir: Path, label_dir: Path, processor: RTDetrImageProcessor):
        self.label_dir = label_dir
        self.processor = processor
        self.images = sorted(p for ext in ("*.jpg", "*.jpeg", "*.png") for p in image_dir.glob(ext))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image    = Image.open(img_path).convert("RGB")
        w, h     = image.size

        label_path  = self.label_dir / (img_path.stem + ".txt")
        annotations = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id         = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                annotations.append({
                    "bbox":        [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
                    "category_id": cls_id,
                    "area":        bw * w * bh * h,
                    "iscrowd":     0,
                })

        encoding = self.processor(
            images=image,
            annotations={"image_id": idx, "annotations": annotations},
            return_tensors="pt",
        )
        result = {}
        for k, v in encoding.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.squeeze(0)
            elif isinstance(v, list) and len(v) == 1:
                result[k] = v[0]
            else:
                result[k] = v
        return result


def collate_fn(batch):
    pv = torch.stack([b["pixel_values"] for b in batch])
    if "pixel_mask" in batch[0]:
        pm = torch.stack([b["pixel_mask"] for b in batch])
    else:
        pm = torch.ones(pv.shape[0], pv.shape[2], pv.shape[3], dtype=torch.long)
    return {
        "pixel_values": pv,
        "pixel_mask":   pm,
        "labels": [
            {"class_labels": b["labels"]["class_labels"], "boxes": b["labels"]["boxes"]}
            for b in batch
        ],
    }


class RTDetrExportWrapper(torch.nn.Module):
    """ONNX export wrapper: [1,3,H,W] → [1, num_queries, 4+nc]"""
    def __init__(self, model: RTDetrForObjectDetection):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_mask = torch.ones(
            pixel_values.shape[0], pixel_values.shape[2], pixel_values.shape[3],
            dtype=torch.long, device=pixel_values.device,
        )
        out    = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        scores = torch.sigmoid(out.logits)  # [b, q, nc]
        boxes  = out.pred_boxes             # [b, q, 4]  cx cy w h normalized
        return torch.cat([boxes, scores], dim=-1)


def export_onnx(model: RTDetrForObjectDetection, output_path: Path):
    print("\nONNX エクスポート中...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = RTDetrExportWrapper(model)
    wrapper.eval()
    dummy = torch.zeros(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, str(output_path),
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes={"images": {0: "batch"}, "output0": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
    try:
        import onnxslim
        onnxslim.slim(str(output_path), str(output_path))
        print("onnxslim 適用済み")
    except ImportError:
        pass
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"ONNX: {output_path} ({size_mb:.1f} MB)")
    print(f"\n次のステップ: {output_path} を public/models/katana_points_detector.onnx として配置")


def main():
    parser = argparse.ArgumentParser(description="RT-DETR 刀先端・柄基準点 fine-tuning (Apache 2.0)")
    parser.add_argument("--dataset",     default="dataset")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch",       type=int,   default=4)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--checkpoint",  default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    validate_dataset(dataset_path)

    project_root = Path(__file__).resolve().parent.parent
    onnx_path    = project_root / "public" / "models" / "katana_points_detector.onnx"
    save_dir     = project_root / "runs" / "rtdetr_hf"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    processor = RTDetrImageProcessor.from_pretrained(MODEL_ID)

    if args.export_only:
        ckpt = args.checkpoint or str(save_dir / "best.pt")
        print(f"チェックポイント: {ckpt}")
        model = RTDetrForObjectDetection.from_pretrained(
            MODEL_ID, num_labels=len(CLASS_NAMES),
            id2label=ID2LABEL, label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        export_onnx(model, onnx_path)
        return

    train_ds = KatanaDataset(dataset_path / "images/train", dataset_path / "labels/train", processor)
    val_ds   = KatanaDataset(dataset_path / "images/val",   dataset_path / "labels/val",   processor)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = RTDetrForObjectDetection.from_pretrained(
        MODEL_ID, num_labels=len(CLASS_NAMES),
        id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    print(f"\n学習開始: {args.epochs} エポック / バッチ {args.batch} / lr {args.lr}")
    print(f"クラス: {', '.join(CLASS_NAMES)}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            pv  = batch["pixel_values"].to(device)
            pm  = batch["pixel_mask"].to(device)
            lbs = [{k: v.to(device) for k, v in lbl.items()} for lbl in batch["labels"]]
            loss = model(pixel_values=pv, pixel_mask=pm, labels=lbs).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pv  = batch["pixel_values"].to(device)
                pm  = batch["pixel_mask"].to(device)
                lbs = [{k: v.to(device) for k, v in lbl.items()} for lbl in batch["labels"]]
                val_loss += model(pixel_values=pv, pixel_mask=pm, labels=lbs).loss.item()

        scheduler.step()
        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        print(f"Epoch {epoch:3d}/{args.epochs}  train={avg_train:.4f}  val={avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), save_dir / "best.pt")
            print(f"  → best saved (val={best_loss:.4f})")

    print("\n学習完了")
    model.load_state_dict(torch.load(save_dir / "best.pt", map_location="cpu"))
    export_onnx(model, onnx_path)


if __name__ == "__main__":
    main()
