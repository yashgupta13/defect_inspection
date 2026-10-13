#!/usr/bin/env python
"""Convert MVTec AD into a supervised YOLO-segmentation dataset (single class: 0 = defect).

Defective test images + their ground-truth masks become polygon labels. Defect-free
images are added as background (negative) samples so the model learns not to fire on
normal products. Splits are made per image and stratified by defect type.

NOTE: this is a *supervised segmentation experiment derived from MVTec AD annotations*,
not the original MVTec anomaly-detection protocol.

Usage:
    python prepare_dataset.py --mvtec /path/to/mvtec_anomaly_detection
"""
import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from config import DATASET_DIR, DEFAULT_CATEGORIES

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}


def mask_to_polygons(mask, min_area=4.0, eps_frac=0.0015):
    """Binary mask -> list of normalised polygons (Nx2 arrays in 0..1)."""
    binary = (mask > 127).astype(np.uint8)
    h, w = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        approx = cv2.approxPolyDP(c, eps_frac * cv2.arcLength(c, True), True)
        if len(approx) < 3:
            approx = c
        if len(approx) < 3:
            continue
        pts = approx.reshape(-1, 2).astype(np.float64)
        pts[:, 0] /= w
        pts[:, 1] /= h
        polys.append(np.clip(pts, 0.0, 1.0))
    return polys


def split_list(items, val_frac, test_frac, rng):
    """Shuffle and split into (train, val, test). Tiny groups go entirely to train."""
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    if n < 3:
        return items, [], []
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    if n_test + n_val >= n:
        n_test, n_val = 1, 1
    return items[n_val + n_test:], items[:n_val], items[n_val:n_val + n_test]


def list_images(folder):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXT)


def write_sample(img_path, out_root, split, name, polys, manifest, category, defect_type, label):
    img_dir = out_root / "images" / split
    lab_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    dst = img_dir / f"{name}{img_path.suffix.lower()}"
    shutil.copy2(img_path, dst)
    lines = []
    for poly in polys:
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
        lines.append(f"0 {coords}")
    (lab_dir / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    manifest.append({
        "split": split, "category": category, "defect_type": defect_type,
        "image": str(dst.relative_to(out_root)), "label": label,
    })


def process_category(mvtec, cat, out_root, args, rng, manifest, stats):
    cat_dir = mvtec / cat
    test_dir = cat_dir / "test"
    gt_dir = cat_dir / "ground_truth"
    if not test_dir.is_dir():
        print(f"[skip] {cat}: {test_dir} not found")
        return

    # ---- defective images (labelled) ----
    for dtype_dir in sorted(p for p in test_dir.iterdir() if p.is_dir() and p.name != "good"):
        dtype = dtype_dir.name
        samples = []
        for img_path in list_images(dtype_dir):
            mask_path = gt_dir / dtype / f"{img_path.stem}_mask.png"
            if not mask_path.exists():
                print(f"[warn] no mask for {img_path}")
                continue
            img = cv2.imread(str(img_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            if mask.shape != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            polys = mask_to_polygons(mask)
            if not polys:
                print(f"[warn] empty mask after filtering: {img_path.name}")
                continue
            samples.append((img_path, polys))
        tr, va, te = split_list(samples, args.val_frac, args.test_frac, rng)
        for split, group in (("train", tr), ("val", va), ("test", te)):
            for img_path, polys in group:
                write_sample(img_path, out_root, split, f"{cat}_{dtype}_{img_path.stem}",
                             polys, manifest, cat, dtype, 1)
                stats[(cat, split, "defect")] += 1

    # ---- defect-free images (background / negatives) ----
    test_good = list_images(test_dir / "good")
    train_good = list_images(cat_dir / "train" / "good")
    rng.shuffle(train_good)
    extra = train_good[:args.extra_eval_good]
    train_pool = train_good[args.extra_eval_good:][:args.train_good]
    eval_good = test_good + extra
    rng.shuffle(eval_good)
    half = len(eval_good) // 2
    for split, group in (("val", eval_good[:half]), ("test", eval_good[half:]), ("train", train_pool)):
        for img_path in group:
            tag = "tg" if img_path in test_good else "trg"
            write_sample(img_path, out_root, split, f"{cat}_good_{tag}_{img_path.stem}",
                         [], manifest, cat, "good", 0)
            stats[(cat, split, "good")] += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mvtec", required=True, help="MVTec AD root (contains bottle/, screw/, ...)")
    ap.add_argument("--out", default=str(DATASET_DIR), help="output YOLO dataset directory")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--val-frac", type=float, default=0.15, help="fraction of defective images for validation")
    ap.add_argument("--test-frac", type=float, default=0.15, help="fraction of defective images for test")
    ap.add_argument("--train-good", type=int, default=60, help="defect-free train images per category (negatives)")
    ap.add_argument("--extra-eval-good", type=int, default=10,
                    help="extra defect-free train/good images per category held out for val/test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true", help="delete the output directory first")
    args = ap.parse_args()

    mvtec, out_root = Path(args.mvtec), Path(args.out)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    manifest = []
    stats = defaultdict(int)
    for cat in args.categories:
        process_category(mvtec, cat, out_root, args, rng, manifest, stats)

    if not manifest:
        raise SystemExit("No samples written - check the --mvtec path and category names.")

    with open(out_root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "category", "defect_type", "image", "label"])
        w.writeheader()
        w.writerows(manifest)

    (out_root / "data.yaml").write_text(
        f"path: {out_root.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: defect\n"
    )

    print(f"\nDataset written to {out_root}")
    print(f"{'category':<12}{'split':<7}{'defect':>8}{'good':>8}")
    for cat in args.categories:
        for split in ("train", "val", "test"):
            print(f"{cat:<12}{split:<7}{stats[(cat, split, 'defect')]:>8}{stats[(cat, split, 'good')]:>8}")
    print(f"\nTrain with:  python train.py --data {out_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
