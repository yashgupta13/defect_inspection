"""CIELAB / CIEDE2000 colour inspection (independent of the YOLO model).

Pipeline: frame -> product region -> RGB->CIELAB -> compare with a calibrated
reference colour using CIEDE2000 -> threshold -> colour normal / colour defect.
"""
import json
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# CIEDE2000
# --------------------------------------------------------------------------- #
def ciede2000(lab1, lab2):
    """CIEDE2000 colour difference (Sharma et al. 2005). Inputs (...,3) arrays."""
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cbar7 = ((C1 + C2) / 2.0) ** 7
    G = 0.5 * (1.0 - np.sqrt(Cbar7 / (Cbar7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, dh)
    dh = np.where(dh < -180, dh + 360, dh)
    dh = np.where(C1p * C2p == 0, 0.0, dh)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2.0))

    Lbar = (L1 + L2) / 2.0
    Cbar = (C1p + C2p) / 2.0
    hsum = h1p + h2p
    hbar = np.where(np.abs(h1p - h2p) <= 180, hsum / 2.0,
                    np.where(hsum < 360, (hsum + 360) / 2.0, (hsum - 360) / 2.0))
    hbar = np.where(C1p * C2p == 0, hsum, hbar)

    T = (1 - 0.17 * np.cos(np.radians(hbar - 30)) + 0.24 * np.cos(np.radians(2 * hbar))
         + 0.32 * np.cos(np.radians(3 * hbar + 6)) - 0.20 * np.cos(np.radians(4 * hbar - 63)))
    dtheta = 30.0 * np.exp(-(((hbar - 275) / 25.0) ** 2))
    Rc = 2.0 * np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbar - 50) ** 2 / np.sqrt(20 + (Lbar - 50) ** 2)
    Sc = 1 + 0.045 * Cbar
    Sh = 1 + 0.015 * Cbar * T
    Rt = -np.sin(np.radians(2 * dtheta)) * Rc

    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


def bgr_to_lab(img_bgr):
    """uint8 BGR -> float32 CIELAB (L 0..100, a/b about -128..127)."""
    return cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


# --------------------------------------------------------------------------- #
# Product region
# --------------------------------------------------------------------------- #
def find_product_mask(img_bgr, mode="auto", min_frac=0.02):
    """Return (bool mask, found). 'auto' uses Otsu + largest blob; 'full'/'center' are fixed."""
    h, w = img_bgr.shape[:2]
    if mode == "full":
        return np.ones((h, w), bool), True
    if mode == "center":
        m = np.zeros((h, w), bool)
        m[int(.2 * h):int(.8 * h), int(.2 * w):int(.8 * w)] = True
        return m, True

    gray = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([th[0, :], th[-1, :], th[:, 0], th[:, -1]])
    bg_val = 255 if border.mean() > 127 else 0
    fg = (th != bg_val).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n <= 1:
        return np.zeros((h, w), bool), False
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == biggest).astype(np.uint8)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, cnts, -1, 1, -1)
    frac = filled.mean()
    if frac < min_frac:
        return filled.astype(bool), False
    if frac > 0.95:  # object fills the frame
        return np.ones((h, w), bool), True
    return filled.astype(bool), True


# --------------------------------------------------------------------------- #
# Reference profiles
# --------------------------------------------------------------------------- #
def load_profiles(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text()).get("profiles", {})


def save_profile(path, name, profile):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(path)
    profiles[name] = profile
    path.write_text(json.dumps({"profiles": profiles}, indent=2))


# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #
class ColorAnalyzer:
    def __init__(self, profile=None, roi_mode="auto", threshold=None,
                 local_pixel_thr=12.0, local_area_limit=None):
        self.profile = profile
        self.roi_mode = roi_mode
        self.local_pixel_thr = local_pixel_thr
        self.local_area_limit = local_area_limit
        self.ref_lab = np.array(profile["mean_lab"], np.float64) if profile else None
        self.threshold = threshold if threshold is not None else (profile or {}).get("threshold")

    @property
    def enabled(self):
        return self.ref_lab is not None and self.threshold is not None

    def product_mask(self, img_bgr):
        return find_product_mask(img_bgr, self.roi_mode)

    @staticmethod
    def measure(lab, product_mask, exclude=None):
        """Median Lab of the product region (edges eroded, defect pixels excluded)."""
        valid = product_mask.astype(np.uint8)
        eroded = cv2.erode(valid, np.ones((7, 7), np.uint8), iterations=2)
        if eroded.sum() > 200:
            valid = eroded
        valid = valid.astype(bool)
        if exclude is not None and exclude.any():
            grown = cv2.dilate(exclude.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
            if (valid & ~grown).sum() > 200:
                valid &= ~grown
        n = int(valid.sum())
        if n < 100:
            return None, n, valid
        return np.median(lab[valid], axis=0).astype(np.float64), n, valid

    def analyze(self, img_bgr, product_mask, defect_mask=None):
        out = {"color_checked": False, "color_defect": False, "delta_e": None,
               "threshold": self.threshold, "mean_lab": None, "local_area_percent": None}
        if not self.enabled:
            return out
        lab = bgr_to_lab(img_bgr)
        mean_lab, n, valid = self.measure(lab, product_mask, defect_mask)
        if mean_lab is None:
            return out
        de = float(ciede2000(mean_lab, self.ref_lab))
        out.update(color_checked=True, delta_e=de, mean_lab=[float(x) for x in mean_lab])
        defect = de > self.threshold

        if self.local_area_limit is not None:
            blurred = cv2.GaussianBlur(lab, (15, 15), 0)
            de_map = ciede2000(blurred[valid], self.ref_lab)
            frac = 100.0 * float((de_map > self.local_pixel_thr).mean())
            out["local_area_percent"] = frac
            defect = defect or frac > self.local_area_limit
        out["color_defect"] = bool(defect)
        return out


def build_profile(mean_labs, min_threshold=3.0, k=4.0):
    """Reference = median of per-image mean Lab; threshold = mean + k*std of dE00 (>= min)."""
    arr = np.array(mean_labs, np.float64)
    ref = np.median(arr, axis=0)
    des = ciede2000(arr, ref[None, :])
    thr = max(min_threshold, float(des.mean() + k * des.std()))
    return {"mean_lab": [float(x) for x in ref], "threshold": round(thr, 3),
            "n_images": int(len(arr)), "de_mean": round(float(des.mean()), 3),
            "de_std": round(float(des.std()), 3), "de_max": round(float(des.max()), 3)}
