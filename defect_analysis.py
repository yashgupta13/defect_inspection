"""Turn raw YOLO segmentation output into defect presence / confidence / area / location."""
import cv2
import numpy as np

from utils import region_name


def severity_from_area(area_pct, minor_below=0.5, moderate_below=2.0):
    if area_pct < minor_below:
        return "minor"
    if area_pct < moderate_below:
        return "moderate"
    return "severe"


def analyze_defects(result, shape, conf_thr=0.25, min_area_pct=0.05, product_mask=None,
                    minor_below=0.5, moderate_below=2.0):
    """Analyse one Ultralytics `Results` object.

    Returns a dict with a boolean `mask` (union of kept instances, image resolution),
    polygons/boxes/confidences and the summary numbers.
    """
    h, w = shape
    union = np.zeros((h, w), np.uint8)
    polys, confs, boxes = [], [], []

    if result is not None and result.masks is not None and result.boxes is not None and len(result.boxes):
        cs = result.boxes.conf.cpu().numpy()
        bxs = result.boxes.xyxy.cpu().numpy()
        for xy, c, b in zip(result.masks.xy, cs, bxs):
            if c < conf_thr or len(xy) < 3:
                continue
            pts = np.round(xy).astype(np.int32)
            cv2.fillPoly(union, [pts], 1)
            polys.append(pts)
            confs.append(float(c))
            boxes.append([float(v) for v in b])

    area_px = int(union.sum())
    area_pct = 100.0 * area_px / float(h * w)
    prod_px = int(product_mask.sum()) if product_mask is not None else h * w
    area_pct_product = 100.0 * area_px / prod_px if prod_px > 0 else 0.0
    detected = bool(polys) and area_pct >= min_area_pct

    location = severity = None
    if detected:
        m = cv2.moments(union, binaryImage=True)
        if m["m00"] > 0:
            location = region_name(m["m10"] / m["m00"], m["m01"] / m["m00"], w, h)
        severity = severity_from_area(area_pct, minor_below, moderate_below)

    return {
        "detected": detected,
        "confidence": max(confs) if (detected and confs) else 0.0,
        "area_percent": area_pct if detected else 0.0,
        "area_percent_of_product": area_pct_product if detected else 0.0,
        "count": len(polys) if detected else 0,
        "location": location,
        "severity": severity,
        "mask": union.astype(bool),
        "polygons": polys,
        "boxes": boxes,
        "confidences": confs,
    }
