#!/usr/bin/env python
"""Fine-tune a pretrained Ultralytics YOLO segmentation model on the converted dataset.

Usage:
    python train.py --data datasets/mvtec_yolo/data.yaml --epochs 100
Small defects (screw, capsule) benefit from larger --imgsz (e.g. 800 or 1024).
"""
import argparse
import shutil

from config import DATASET_DIR, ROOT, WEIGHTS_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATASET_DIR / "data.yaml"))
    ap.add_argument("--model", default="yolo11n-seg.pt",
                    help="pretrained weights (yolo11n-seg.pt, yolo11s-seg.pt, yolov8n-seg.pt, ...)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default=None, help="0 for first GPU, cpu, mps ...")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="defect_seg")
    ap.add_argument("--flip", action="store_true",
                    help="enable flip augmentation (OFF by default: a mirrored metal_nut IS a defect in MVTec)")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, device=args.device, workers=args.workers,
        project=str(ROOT / "runs"), name=args.name, exist_ok=True, seed=0, plots=True,
        fliplr=0.5 if args.flip else 0.0, flipud=0.0,
        hsv_h=0.0,  # hue shifts would fight the separate colour-inspection branch
    )

    best = ROOT / "runs" / args.name / "weights" / "best.pt"
    if best.exists():
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, WEIGHTS_PATH)
        print(f"\nBest weights copied to {WEIGHTS_PATH}")
    else:
        print(f"[warn] {best} not found - check the training log.")


if __name__ == "__main__":
    main()
