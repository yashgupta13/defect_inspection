#!/usr/bin/env python
"""Evaluate the trained model and the full PASS/FAIL system on the held-out test split.

Reports (per category + overall): PASS/FAIL accuracy, precision, recall (defect detection
rate), F1, false-positive rate, false-negative rate, pixel IoU/Dice, latency, and Ultralytics
mask mAP. It also sweeps --conf / --min-area and recommends an operating point.

Usage:
    python evaluate.py --data-dir datasets/mvtec_yolo --weights models/best.pt
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from color_analysis import ColorAnalyzer, load_profiles
from config import (COLOR_REFERENCE_PATH, DATASET_DIR, OUTPUT_DIR, TEXTURE_CATEGORIES,
                    WEIGHTS_PATH)
from defect_analysis import analyze_defects


def read_gt_mask(label_path, h, w):
    m = np.zeros((h, w), np.uint8)
    if label_path.exists():
        for line in label_path.read_text().split("\n"):
            v = line.split()
            if len(v) >= 7:
                pts = np.array(v[1:], float).reshape(-1, 2) * [w, h]
                cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
    return m.astype(bool)


def union_area_pct(polys, confs, thr, shape):
    m = np.zeros(shape, np.uint8)
    hit = False
    for p, c in zip(polys, confs):
        if c >= thr:
            cv2.fillPoly(m, [p], 1)
            hit = True
    return (100.0 * m.sum() / (shape[0] * shape[1])) if hit else 0.0


def metrics(y, p):
    y, p = np.asarray(y, bool), np.asarray(p, bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": (tp + tn) / n if n else 0.0, "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "fpr": fp / (fp + tn) if fp + tn else 0.0, "fnr": fn / (fn + tp) if fn + tp else 0.0}


def main():
    from config import InspectionConfig as _IC
    _defaults = _IC()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(DATASET_DIR))
    ap.add_argument("--weights", default=str(WEIGHTS_PATH))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=_defaults.imgsz)
    ap.add_argument("--conf", type=float, default=_defaults.conf, help="operating-point confidence")
    ap.add_argument("--min-area", type=float, default=0.05, help="operating-point min area (%%)")
    ap.add_argument("--color-reference", default=str(COLOR_REFERENCE_PATH),
                    help="per-category profiles (from calibrate_color.py --mvtec) -> integrated PASS/FAIL")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(OUTPUT_DIR / "evaluation"))
    ap.add_argument("--skip-ultralytics-val", action="store_true")
    args = ap.parse_args()

    from ultralytics import YOLO

    root = Path(args.data_dir)
    rows = [r for r in csv.DictReader(open(root / "manifest.csv")) if r["split"] == args.split]
    if not rows:
        raise SystemExit(f"No '{args.split}' rows in {root / 'manifest.csv'}")
    model = YOLO(args.weights)
    model.predict(np.zeros((320, 320, 3), np.uint8), imgsz=args.imgsz, device=args.device, verbose=False)
    profiles = load_profiles(args.color_reference)

    recs, lat = [], []
    for i, r in enumerate(rows, 1):
        img_path = root / r["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        cat = r["category"]
        roi = "full" if cat in TEXTURE_CATEGORIES else "auto"
        analyzer = ColorAnalyzer(profiles.get(cat), roi_mode=roi)

        t0 = time.perf_counter()
        res = model.predict(img, imgsz=args.imgsz, conf=0.05, device=args.device, verbose=False)[0]
        low = analyze_defects(res, (h, w), conf_thr=0.05, min_area_pct=0.0)          # for the sweep
        op = analyze_defects(res, (h, w), args.conf, args.min_area)                  # operating point
        pmask, _ = analyzer.product_mask(img)
        color = analyzer.analyze(img, pmask, op["mask"])
        lat.append((time.perf_counter() - t0) * 1000)

        gt = read_gt_mask(root / "labels" / args.split / (img_path.stem + ".txt"), h, w)
        recs.append({
            "image": r["image"], "category": cat, "defect_type": r["defect_type"],
            "y": int(r["label"]), "polys": low["polygons"], "confs": low["confidences"],
            "shape": (h, w), "op_defect": op["detected"], "op_conf": op["confidence"],
            "op_area": op["area_percent"], "color_fail": color["color_defect"],
            "color_checked": color["color_checked"], "delta_e": color["delta_e"],
            "pred_mask": op["mask"] if op["detected"] else np.zeros((h, w), bool), "gt": gt,
        })
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}")

    y = [r["y"] for r in recs]
    p_def = [r["op_defect"] for r in recs]
    p_all = [r["op_defect"] or r["color_fail"] for r in recs]
    used_color = any(r["color_checked"] for r in recs)

    # ---- operating-point report ----
    def pix(rs):
        ious, dices, tp, fp, fn = [], [], 0, 0, 0
        for r in rs:
            g, p = r["gt"], r["pred_mask"]
            inter, uni = (g & p).sum(), (g | p).sum()
            tp += inter; fp += (p & ~g).sum(); fn += (g & ~p).sum()
            if r["y"] == 1:
                ious.append(inter / uni if uni else 0.0)
                dices.append(2 * inter / (g.sum() + p.sum()) if (g.sum() + p.sum()) else 0.0)
        return {"mean_iou_defective": float(np.mean(ious)) if ious else 0.0,
                "mean_dice_defective": float(np.mean(dices)) if dices else 0.0,
                "pixel_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
                "pixel_recall": float(tp / (tp + fn)) if tp + fn else 0.0}

    report = {"split": args.split, "operating_point": {"conf": args.conf, "min_area_pct": args.min_area},
              "images": len(recs), "color_branch_used": used_color,
              "latency_ms": {"mean": float(np.mean(lat)), "p95": float(np.percentile(lat, 95))},
              "defect_branch": metrics(y, p_def), "integrated_pass_fail": metrics(y, p_all),
              "pixel_level": pix(recs), "per_category": {}}
    for cat in sorted({r["category"] for r in recs}):
        rs = [r for r in recs if r["category"] == cat]
        yy = [r["y"] for r in rs]
        report["per_category"][cat] = {
            "defect_branch": metrics(yy, [r["op_defect"] for r in rs]),
            "integrated_pass_fail": metrics(yy, [r["op_defect"] or r["color_fail"] for r in rs]),
            "pixel_level": pix(rs)}

    # ---- sweep ----
    confs_grid = [round(c, 2) for c in np.arange(0.05, 0.951, 0.05)]
    areas_grid = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
    sweep = []
    for c in confs_grid:
        areas = [union_area_pct(r["polys"], r["confs"], c, r["shape"]) for r in recs]
        for a in areas_grid:
            m = metrics(y, [ar > 0 and ar >= a for ar in areas])
            sweep.append({"conf": c, "min_area_pct": a, **{k: round(v, 4) if isinstance(v, float) else v
                                                             for k, v in m.items()}})
    best = max(sweep, key=lambda s: (s["f1"], -s["fpr"]))
    report["recommended_operating_point"] = {"conf": best["conf"], "min_area_pct": best["min_area_pct"],
                                             "f1": best["f1"], "fpr": best["fpr"], "fnr": best["fnr"]}

    # ---- Ultralytics mask metrics ----
    if not args.skip_ultralytics_val:
        try:
            v = model.val(data=str(root / "data.yaml"), split=args.split, imgsz=args.imgsz,
                          device=args.device, verbose=False, plots=False)
            report["ultralytics_mask_metrics"] = {
                "mAP50": float(v.seg.map50), "mAP50-95": float(v.seg.map),
                "precision": float(v.seg.mp), "recall": float(v.seg.mr)}
        except Exception as e:  # noqa: BLE001
            report["ultralytics_mask_metrics"] = f"unavailable: {e}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    with open(out / "sweep.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        wr.writeheader()
        wr.writerows(sweep)
    with open(out / "per_image.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "category", "defect_type", "truth", "defect_pred", "defect_conf",
                     "defect_area_pct", "delta_e", "color_fail", "final"])
        for r in recs:
            wr.writerow([r["image"], r["category"], r["defect_type"], "FAIL" if r["y"] else "PASS",
                         int(r["op_defect"]), f"{r['op_conf']:.3f}", f"{r['op_area']:.3f}",
                         "" if r["delta_e"] is None else f"{r['delta_e']:.2f}", int(r["color_fail"]),
                         "FAIL" if (r["op_defect"] or r["color_fail"]) else "PASS"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sel = [s for s in sweep if s["min_area_pct"] == best["min_area_pct"]]
        plt.figure(figsize=(6, 4))
        for key, lab in (("fpr", "False-positive rate"), ("fnr", "False-negative rate"), ("f1", "F1")):
            plt.plot([s["conf"] for s in sel], [s[key] for s in sel], label=lab)
        plt.axvline(best["conf"], ls="--", c="gray")
        plt.xlabel("confidence threshold"); plt.legend(); plt.grid(alpha=.3)
        plt.title(f"Threshold sweep (min area {best['min_area_pct']}%)")
        plt.tight_layout(); plt.savefig(out / "threshold_sweep.png", dpi=130); plt.close()
    except Exception:  # noqa: BLE001
        pass

    # ---- console summary ----
    d, ip = report["defect_branch"], report["integrated_pass_fail"]
    print(f"\n=== {args.split} split: {len(recs)} images | conf>={args.conf}, area>={args.min_area}% ===")
    print(f"{'category':<12}{'n':>5}{'acc':>8}{'recall':>8}{'FPR':>8}{'FNR':>8}{'IoU':>8}")
    for cat, v in report["per_category"].items():
        m = v["integrated_pass_fail"]
        print(f"{cat:<12}{m['n']:>5}{m['accuracy']:>8.3f}{m['recall']:>8.3f}{m['fpr']:>8.3f}{m['fnr']:>8.3f}"
              f"{v['pixel_level']['mean_iou_defective']:>8.3f}")
    print(f"{'ALL':<12}{ip['n']:>5}{ip['accuracy']:>8.3f}{ip['recall']:>8.3f}{ip['fpr']:>8.3f}{ip['fnr']:>8.3f}"
          f"{report['pixel_level']['mean_iou_defective']:>8.3f}")
    print(f"Defect-branch only: acc {d['accuracy']:.3f}, FPR {d['fpr']:.3f}, FNR {d['fnr']:.3f}"
          + ("" if used_color else "   (no colour profiles -> integrated == defect-only)"))
    print(f"Latency: {report['latency_ms']['mean']:.1f} ms mean, {report['latency_ms']['p95']:.1f} ms p95")
    rb = report["recommended_operating_point"]
    print(f"Recommended: --conf {rb['conf']} --min-area {rb['min_area_pct']}  (F1 {rb['f1']}, FPR {rb['fpr']}, FNR {rb['fnr']})")
    print(f"Reports in {out}")


if __name__ == "__main__":
    main()
