#!/usr/bin/env python
"""Generate a tiny synthetic dataset in MVTec AD layout, for smoke-testing the code only."""
import sys
from pathlib import Path

import cv2
import numpy as np


def product(rng, size=256, tint=(60, 120, 180)):
    img = np.full((size, size, 3), 40, np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 3, tint, -1)
    noise = rng.normal(0, 4, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def defective(rng, kind, size=256, tint=(60, 120, 180)):
    img = product(rng, size, tint)
    mask = np.zeros((size, size), np.uint8)
    cx, cy = size // 2, size // 2
    if kind == "scratch":
        x = int(rng.integers(cx - 50, cx + 20))
        y = int(rng.integers(cy - 50, cy + 50))
        p1, p2 = (x, y), (x + int(rng.integers(30, 60)), y + int(rng.integers(-15, 15)))
        cv2.line(img, p1, p2, (10, 10, 10), 4)
        cv2.line(mask, p1, p2, 255, 4)
    else:  # blob
        x = int(rng.integers(cx - 45, cx + 45))
        y = int(rng.integers(cy - 45, cy + 45))
        r = int(rng.integers(8, 16))
        cv2.circle(img, (x, y), r, (230, 230, 230), -1)
        cv2.circle(mask, (x, y), r, 255, -1)
    return img, mask


def build(root, categories=("widget", "gizmo"), n_train=30, n_test_good=10, n_def=14, seed=0):
    root = Path(root)
    rng = np.random.default_rng(seed)
    for ci, cat in enumerate(categories):
        tint = [(60, 120, 180), (90, 160, 90)][ci % 2]
        for d in ("train/good", "test/good", "test/scratch", "test/blob",
                  "ground_truth/scratch", "ground_truth/blob"):
            (root / cat / d).mkdir(parents=True, exist_ok=True)
        for i in range(n_train):
            cv2.imwrite(str(root / cat / "train/good" / f"{i:03d}.png"), product(rng, tint=tint))
        for i in range(n_test_good):
            cv2.imwrite(str(root / cat / "test/good" / f"{i:03d}.png"), product(rng, tint=tint))
        for kind in ("scratch", "blob"):
            for i in range(n_def):
                img, mask = defective(rng, kind, tint=tint)
                cv2.imwrite(str(root / cat / "test" / kind / f"{i:03d}.png"), img)
                cv2.imwrite(str(root / cat / "ground_truth" / kind / f"{i:03d}_mask.png"), mask)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "fake_mvtec")
    print("done")
