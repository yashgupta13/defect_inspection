#!/usr/bin/env python
"""Calibrate the CIELAB reference colour (and dE00 threshold) from DEFECT-FREE products.

From MVTec AD (one profile per category, named after the category):
    python calibrate_color.py --mvtec /path/to/mvtec_anomaly_detection

From your own good products (folder of images, a video, or the camera) under the
SAME lighting/camera setup you will inspect with:
    python calibrate_color.py --source good_samples/ --name mypart
    python calibrate_color.py --source 0 --name mypart --frames 40
"""
import argparse
from pathlib import Path

from color_analysis import ColorAnalyzer, bgr_to_lab, build_profile, find_product_mask, save_profile
from config import COLOR_REFERENCE_PATH, DEFAULT_CATEGORIES, TEXTURE_CATEGORIES
from utils import IMG_EXT, FrameSource

import cv2


def mean_lab(img, roi):
    mask, found = find_product_mask(img, roi)
    if not found:
        return None
    m, _, _ = ColorAnalyzer.measure(bgr_to_lab(img), mask)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mvtec")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--max-images", type=int, default=100)
    ap.add_argument("--source", help="folder / video / camera index of defect-free products")
    ap.add_argument("--name", help="profile name (required with --source)")
    ap.add_argument("--frames", type=int, default=40, help="frames to sample from a video/camera")
    ap.add_argument("--stride", type=int, default=3, help="frame stride for video/camera")
    ap.add_argument("--roi", default="auto", choices=["auto", "center", "full"])
    ap.add_argument("--min-threshold", type=float, default=3.0, help="floor for the dE00 threshold")
    ap.add_argument("--k", type=float, default=4.0, help="threshold = mean + k * std of dE00 on good samples")
    ap.add_argument("--out", default=str(COLOR_REFERENCE_PATH))
    args = ap.parse_args()

    jobs = []  # (name, list of mean-Lab)
    if args.mvtec:
        for cat in args.categories:
            folder = Path(args.mvtec) / cat / "train" / "good"
            files = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMG_EXT)[:args.max_images]
            if not files:
                print(f"[skip] {cat}: no images in {folder}")
                continue
            roi = "full" if cat in TEXTURE_CATEGORIES else args.roi
            labs = [m for m in (mean_lab(cv2.imread(str(p)), roi) for p in files) if m is not None]
            jobs.append((cat, labs))
    elif args.source:
        if not args.name:
            raise SystemExit("--name is required with --source")
        src, labs = FrameSource(args.source), []
        for idx, frame, _ in src:
            if src.files or idx % max(1, args.stride) == 0:
                m = mean_lab(frame, args.roi)
                if m is not None:
                    labs.append(m)
            if len(labs) >= args.frames:
                break
        src.close()
        jobs.append((args.name, labs))
    else:
        raise SystemExit("give --mvtec or --source")

    for name, labs in jobs:
        if len(labs) < 3:
            print(f"[skip] {name}: only {len(labs)} usable samples (need >= 3)")
            continue
        if len(labs) < 10:
            print(f"[warn] {name}: only {len(labs)} samples - threshold will be rough")
        prof = build_profile(labs, args.min_threshold, args.k)
        save_profile(args.out, name, prof)
        L, a, b = prof["mean_lab"]
        print(f"{name:<12} n={prof['n_images']:<4} ref Lab=({L:.1f}, {a:.1f}, {b:.1f})  "
              f"dE00 on good: mean {prof['de_mean']}, max {prof['de_max']}  ->  threshold {prof['threshold']}")
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
