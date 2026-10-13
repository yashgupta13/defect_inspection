"""Frame sources (camera / video / image / folder) and small helpers."""
from pathlib import Path

import cv2

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}


class FrameSource:
    """Iterate over frames from a camera index, video file, image or image folder.

    Yields (index, frame_bgr, label).
    """

    def __init__(self, source, width=None, height=None):
        s = str(source)
        self.name = "camera" if s.isdigit() else Path(s).stem
        self.live = False
        self.is_video = False
        self.files = []
        self.cap = None
        self.fps = 30.0

        if s.isdigit() or s.lower().startswith(("rtsp://", "http://", "https://")):
            self.live = True
            self.cap = cv2.VideoCapture(int(s) if s.isdigit() else s)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Could not open camera/stream '{s}'. Check the index (try 0, 1, 2) "
                    "and that no other program is using the camera."
                )
            if width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            for _ in range(5):  # let auto-exposure settle
                self.cap.read()
        else:
            p = Path(s)
            if p.is_dir():
                self.files = sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXT)
                if not self.files:
                    raise FileNotFoundError(f"No images found in folder {p}")
            elif p.suffix.lower() in IMG_EXT and p.exists():
                self.files = [p]
            elif p.exists():
                self.is_video = True
                self.cap = cv2.VideoCapture(str(p))
                if not self.cap.isOpened():
                    raise RuntimeError(f"Could not open video {p}")
            else:
                raise FileNotFoundError(f"Source not found: {p}")

        if self.cap is not None:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps and 1 <= fps <= 240:
                self.fps = float(fps)

    def __iter__(self):
        if self.files:
            for i, p in enumerate(self.files):
                img = cv2.imread(str(p))
                if img is None:
                    print(f"[warn] could not read {p}")
                    continue
                yield i, img, p.stem
        else:
            i = 0
            while True:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    break
                yield i, frame, f"frame{i:05d}"
                i += 1

    def close(self):
        if self.cap is not None:
            self.cap.release()


def region_name(cx, cy, w, h):
    """Name the 3x3 grid cell containing (cx, cy)."""
    col = "left" if cx < w / 3 else "right" if cx > 2 * w / 3 else "center"
    row = "top" if cy < h / 3 else "bottom" if cy > 2 * h / 3 else "middle"
    if row == "middle" and col == "center":
        return "center"
    if row == "middle":
        return col
    if col == "center":
        return row
    return f"{row}-{col}"
