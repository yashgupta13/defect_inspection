"""
Quick diagnostic: run YOLO model directly on an image at multiple confidence thresholds.
Usage:
    python test_model_image.py --image /tmp/test_frame.jpg
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from config import WEIGHTS_PATH


def run_test(image_path: str, conf_sweep=(0.01, 0.05, 0.1, 0.15, 0.25, 0.5)):
    from ultralytics import YOLO

    img = cv2.imread(image_path)
    if img is None:
        sys.exit(f"[ERROR] Cannot read image: {image_path}")
    h, w = img.shape[:2]
    print(f"\n{'='*60}")
    print(f"Image  : {image_path}  ({w}x{h})")
    print(f"Weights: {WEIGHTS_PATH}")
    print(f"{'='*60}")

    model = YOLO(str(WEIGHTS_PATH))
    model_names = model.names
    print(f"Model classes: {model_names}")
    print()

    # Run at very low conf to see ALL raw detections
    raw = model.predict(img, imgsz=640, conf=0.001, verbose=False)[0]
    print("--- ALL raw detections (conf >= 0.001) ---")
    if raw.boxes is not None and len(raw.boxes):
        confs = raw.boxes.conf.cpu().numpy()
        classes = raw.boxes.cls.cpu().numpy().astype(int)
        print(f"  Total instances found: {len(confs)}")
        for i, (c, cl) in enumerate(zip(confs, classes)):
            print(f"  [{i}] class={model_names[cl]} conf={c:.4f}")
    else:
        print("  No detections at all! Model sees nothing.")
    print()

    # Sweep confidence thresholds and draw results
    out_dir = Path("outputs/test_model_image")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("--- Confidence sweep + saved annotated images ---")
    for conf in conf_sweep:
        res = model.predict(img, imgsz=640, conf=conf, verbose=False)[0]
        vis = img.copy()
        count = 0
        if res.masks is not None and res.boxes is not None and len(res.boxes):
            boxes_conf = res.boxes.conf.cpu().numpy()
            for xy, bc in zip(res.masks.xy, boxes_conf):
                if len(xy) < 3:
                    continue
                pts = np.round(xy).astype(np.int32)
                overlay = vis.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 255))
                vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
                cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
                count += 1
            # also draw boxes
            for b, bc in zip(res.boxes.xyxy.cpu().numpy(), boxes_conf):
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(vis, f"defect {bc:.2f}", (x1, max(15, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        label = f"conf>={conf:.2f}  detections={count}"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 0), 4)
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2)
        fname = out_dir / f"result_conf{int(conf*100):03d}.jpg"
        cv2.imwrite(str(fname), vis)
        print(f"  conf={conf:.3f}  -> detections={count}  saved: {fname}")

    print(f"\nAll outputs saved to: {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/tmp/test_frame.jpg")
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    if args.weights:
        import config
        config.WEIGHTS_PATH = Path(args.weights)
        WEIGHTS_PATH = config.WEIGHTS_PATH

    run_test(args.image)
