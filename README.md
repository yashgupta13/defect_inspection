# AI-Assisted Defect Detection — Phase 1 AI/ML Pipeline

Frame (camera / video / image) → **YOLO segmentation** (defects) + **CIELAB/CIEDE2000** (colour) → **Decision engine** → **PASS / FAIL**

```
config.py            defaults and thresholds
prepare_dataset.py   MVTec AD  -> YOLO-seg dataset (masks -> polygons, single class "defect")
train.py             fine-tune Ultralytics YOLO segmentation
calibrate_color.py   build CIELAB reference + dE00 threshold from defect-free products
pipeline.py          InspectionPipeline.inspect(frame) -> JSON record + overlay drawing
defect_analysis.py   presence / confidence / area % / location / severity from the masks
color_analysis.py    product-region finder, RGB->CIELAB, CIEDE2000, reference comparison
decision.py          PASS/FAIL rule + temporal smoothing for video
inspector.py         live camera / uploaded video / image / folder  -> PASS/FAIL
evaluate.py          system metrics, per-category table, threshold sweep
train_and_inspect.ipynb  notebook: prepare -> train -> calibrate -> evaluate -> use the model in code
tests/               unit tests + synthetic MVTec-style data generator
```

## 0. Notebook (easiest way)
Open `train_and_inspect.ipynb` (Jupyter / VS Code / Google Colab with a GPU) and run it top to bottom. Set `MVTEC_DIR` to your extracted
MVTec AD folder. Set `DEMO_MODE = True` first for a few-minute smoke test on synthetic data. It trains, calibrates colour, evaluates,
and then runs the trained model in code (`InspectionPipeline`) on test images and an uploaded video.

## 1. Setup
```bash
pip install -r requirements.txt
```
A GPU is strongly recommended for training (CPU works for inference).

## 2. Prepare data and train
```bash
python prepare_dataset.py --mvtec /path/to/mvtec_anomaly_detection --clean
python train.py --data datasets/mvtec_yolo/data.yaml --epochs 100 --imgsz 640 --device 0
```
Defaults use the five categories from the synopsis (bottle, metal_nut, screw, capsule, tile).
Add more with `--categories ...`. Best weights are copied to `models/best.pt`.
Small defects (screw, capsule) usually need `--imgsz 800` or `1024`.

Notes
- This is a **supervised segmentation experiment derived from MVTec AD annotations**, not the original anomaly-detection protocol (the report should say so).
- Defect-free images are added as background samples so the model learns not to fire on normal parts.
- Flip augmentation is **off** by default: a mirrored `metal_nut` is itself an MVTec defect ("flip").

## 3. Colour calibration
```bash
python calibrate_color.py --mvtec /path/to/mvtec_anomaly_detection            # one profile per MVTec category
python calibrate_color.py --source good_samples/ --name mypart                 # your own good parts (folder/video)
python calibrate_color.py --source 0 --name mypart --frames 40                 # from the camera
```
The threshold is `mean + 4·std` of dE00 over defect-free samples (floor 3.0). Calibrate under the same lighting and camera as inspection.

## 4. Run inspection (PASS / FAIL)
```bash
python inspector.py --source 0                                 # webcam;  q quit, s snapshot, space pause
python inspector.py --source my_video.mp4                      # uploaded video
python inspector.py --source samples/ --no-display             # image or folder
python inspector.py --source 0 --color-profile mypart          # with colour check
```
Outputs go to `outputs/<source-name>/`: annotated video/images, `inspection_log.csv`, `summary.json`.
For video/camera, a majority vote over the last 5 frames (3 FAIL votes) removes flicker; one contiguous FAIL run counts as one **FAIL event**, and the overall verdict is FAIL if any event occurred.
Frames with no product in view are reported as `NO PRODUCT` (disable with `--allow-empty` or `--roi full`).

Per-frame record (same fields as the synopsis example):
```json
{"product_id":"F00042","defect_detected":true,"defect_confidence":0.96,"defect_area_percent":2.31,
 "color_defect":false,"delta_e":2.14,"final_result":"FAIL", "...": "..."}
```

## 5. Evaluate
```bash
python evaluate.py --data-dir datasets/mvtec_yolo --weights models/best.pt
```
Prints a per-category table (accuracy, recall, FPR, FNR, pixel IoU), latency, mask mAP, and recommends `--conf` / `--min-area` from a sweep on the held-out split.
**Choose thresholds on `--split val`, then report final numbers once on `--split test`.**

## Practical notes
- Camera: keep a fixed distance, steady lighting and a plain, contrasting background. Auto-exposure/white-balance shifts show up as colour differences — lock them if your camera allows.
- Models trained on MVTec images will not transfer to your own physical product unchanged; for the Phase-2 prototype, collect good/defective images of your own parts and fine-tune (the same scripts apply).
- `InspectionPipeline.inspect()` returns the PASS/FAIL record, so Phase 2 only needs to send `final_result` to the ESP32 (serial/HTTP/MQTT).
