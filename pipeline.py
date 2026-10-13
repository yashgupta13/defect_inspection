"""End-to-end inspection pipeline: frame -> YOLO segmentation + colour analysis -> PASS/FAIL.

`InspectionPipeline.inspect(frame)` is the single entry point used by the camera/video
inspector, the evaluator and (later) the Phase-2 hardware/ESP32 integration.
"""
import time

import cv2
import numpy as np

from color_analysis import ColorAnalyzer
from config import InspectionConfig
from decision import decide
from defect_analysis import analyze_defects

GREEN, RED, GRAY = (0, 150, 0), (0, 0, 200), (90, 90, 90)


class Inspection:
    """Result container: JSON-friendly `record` plus the arrays needed for drawing."""

    def __init__(self, record, defect, color, product_mask):
        self.record = record
        self.defect = defect
        self.color = color
        self.product_mask = product_mask


def _empty_defect(h, w):
    return {"detected": False, "confidence": 0.0, "area_percent": 0.0,
            "area_percent_of_product": 0.0, "count": 0, "location": None, "severity": None,
            "mask": np.zeros((h, w), bool), "polygons": [], "boxes": [], "confidences": []}


class InspectionPipeline:
    def __init__(self, weights, cfg=None, color_profile=None):
        from ultralytics import YOLO

        self.cfg = cfg or InspectionConfig()
        self.model = YOLO(str(weights))
        self.color = ColorAnalyzer(
            color_profile, roi_mode=self.cfg.roi_mode, threshold=self.cfg.color_threshold,
            local_pixel_thr=self.cfg.local_pixel_thr, local_area_limit=self.cfg.local_area_limit,
        )

    def warmup(self, shape=(480, 640, 3)):
        self.model.predict(np.zeros(shape, np.uint8), imgsz=self.cfg.imgsz,
                           device=self.cfg.device, verbose=False)

    def inspect(self, frame, product_id="P0001"):
        cfg = self.cfg
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        product_mask, found = self.color.product_mask(frame)
        present = found or not cfg.require_product

        if present:
            r = self.model.predict(frame, imgsz=cfg.imgsz, conf=cfg.conf, device=cfg.device,
                                   verbose=False)[0]
            defect = analyze_defects(
                r, (h, w), cfg.conf, cfg.min_area_pct, product_mask,
                cfg.severity_minor_below, cfg.severity_moderate_below)
            color = self.color.analyze(frame, product_mask, defect["mask"])
        else:
            defect = _empty_defect(h, w)
            color = {"color_checked": False, "color_defect": False, "delta_e": None,
                     "threshold": self.color.threshold, "mean_lab": None, "local_area_percent": None}

        result, reasons = decide(defect, color, present)
        ms = (time.perf_counter() - t0) * 1000.0

        record = {
            "product_id": product_id,
            "defect_detected": defect["detected"],
            "defect_confidence": round(defect["confidence"], 4),
            "defect_area_percent": round(defect["area_percent"], 3),
            "defect_count": defect["count"],
            "defect_location": defect["location"],
            "defect_severity": defect["severity"],
            "color_checked": color["color_checked"],
            "color_defect": color["color_defect"],
            "delta_e": None if color["delta_e"] is None else round(color["delta_e"], 3),
            "color_threshold": None if color["threshold"] is None else round(float(color["threshold"]), 3),
            "final_result": result,
            "fail_reasons": reasons,
            "inference_ms": round(ms, 1),
        }
        return Inspection(record, defect, color, product_mask)

    # ------------------------------------------------------------------ drawing
    def annotate(self, frame, insp, status, fps=None):
        """Draw masks, boxes, PASS/FAIL banner and metrics on a copy of the frame."""
        vis = frame.copy()
        h, w = vis.shape[:2]
        scale = max(0.6, w / 900.0)
        th = max(1, int(round(scale * 2)))

        d = insp.defect
        if d["polygons"]:
            overlay = vis.copy()
            for pts in d["polygons"]:
                cv2.fillPoly(overlay, [pts], (0, 0, 255))
            vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
            for pts in d["polygons"]:
                cv2.polylines(vis, [pts], True, (0, 0, 255), th)
            for b, c in zip(d["boxes"], d["confidences"]):
                x1, y1, x2, y2 = (int(v) for v in b)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), th)
                cv2.putText(vis, f"defect {c:.2f}", (x1, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (0, 255, 255), th)

        banner_h = int(46 * scale)
        colour = {"PASS": GREEN, "FAIL": RED}.get(status, GRAY)
        cv2.rectangle(vis, (0, 0), (w, banner_h), colour, -1)
        label = {"PASS": "PASS", "FAIL": "FAIL", "NO_PRODUCT": "NO PRODUCT"}.get(status, status)
        cv2.putText(vis, label, (10, int(banner_h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0 * scale, (255, 255, 255), th + 1)
        if fps is not None:
            txt = f"{fps:.1f} FPS"
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, th)
            cv2.putText(vis, txt, (w - tw - 10, int(banner_h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 * scale, (255, 255, 255), th)

        rec = insp.record
        lines = []
        if rec["final_result"] != "NO_PRODUCT":
            if rec["defect_detected"]:
                lines.append(f"Defect: conf {rec['defect_confidence']:.2f} | area {rec['defect_area_percent']:.2f}% "
                             f"| {rec['defect_location']} | {rec['defect_severity']}")
            else:
                lines.append("Defect: none")
            if rec["color_checked"]:
                tag = "COLOR DEFECT" if rec["color_defect"] else "ok"
                lines.append(f"Color: dE00 {rec['delta_e']:.2f} / {rec['color_threshold']:.1f} {tag}")
            else:
                lines.append("Color: not calibrated")
        y = banner_h + int(22 * scale)
        for ln in lines:
            cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (0, 0, 0), th + 1)
            cv2.putText(vis, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (255, 255, 255), th)
            y += int(22 * scale)
        return vis
