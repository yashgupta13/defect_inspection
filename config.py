"""Central configuration and defaults for the defect-inspection pipeline."""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DEFAULT_CATEGORIES = ["bottle", "metal_nut", "screw", "capsule", "tile"]

# MVTec texture categories: the "product" fills the whole frame, so the
# automatic product-region (ROI) detection must not be used for them.
TEXTURE_CATEGORIES = {"carpet", "grid", "leather", "tile", "wood"}

DATASET_DIR = ROOT / "datasets" / "mvtec_yolo"
WEIGHTS_PATH = ROOT / "models" / "best.pt"
COLOR_REFERENCE_PATH = ROOT / "models" / "color_reference.json"
OUTPUT_DIR = ROOT / "outputs"


@dataclass
class InspectionConfig:
    # --- YOLO inference ---
    # imgsz should match the native resolution of input images/video.
    # Downsampling a 1024x1024 source to 640 crushes confidence scores well
    # below the detection threshold (best conf drops from ~0.59 -> ~0.20).
    imgsz: int = 1024
    conf: float = 0.15            # minimum confidence for a defect instance to count
    min_area_pct: float = 0.05    # minimum total defect area (% of image) to call a defect
    # --- severity bands (defect area, % of image) ---
    severity_minor_below: float = 0.5
    severity_moderate_below: float = 2.0
    # --- colour branch ---
    roi_mode: str = "auto"        # auto | center | full
    color_threshold: float | None = None   # None -> use calibrated profile threshold
    local_pixel_thr: float = 12.0          # per-pixel dE00 for the optional local check
    local_area_limit: float | None = None  # % of product area; None disables local check
    # --- temporal smoothing (video / camera) ---
    smooth_window: int = 5
    smooth_votes: int = 3
    # --- misc ---
    require_product: bool = True  # report NO_PRODUCT when no object is found in frame
    device: str | None = None
