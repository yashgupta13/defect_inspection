#!/usr/bin/env python
"""Run the PASS/FAIL inspection on a live camera, an uploaded video, an image or a folder.

Examples:
    python inspector.py --source 0                         # webcam (q to quit, s = snapshot, space = pause)
    python inspector.py --source product_video.mp4         # uploaded video  -> annotated video + CSV + summary
    python inspector.py --source samples/ --no-display     # folder of images
    python inspector.py --source 0 --color-profile mypart  # with calibrated colour reference
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2

from color_analysis import load_profiles
from config import COLOR_REFERENCE_PATH, OUTPUT_DIR, WEIGHTS_PATH, InspectionConfig
from decision import TemporalSmoother
from pipeline import InspectionPipeline
from utils import FrameSource


def parse_args():
    _defaults = InspectionConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="camera index (0), video file, image file or folder")
    ap.add_argument("--weights", default=str(WEIGHTS_PATH))
    ap.add_argument("--imgsz", type=int, default=_defaults.imgsz)
    ap.add_argument("--conf", type=float, default=_defaults.conf, help="min confidence for a defect instance")
    ap.add_argument("--min-area", type=float, default=0.05, help="min defect area, %% of image")
    ap.add_argument("--device", default=None)
    # colour
    ap.add_argument("--color-reference", default=str(COLOR_REFERENCE_PATH))
    ap.add_argument("--color-profile", default=None, help="profile name inside the colour reference file")
    ap.add_argument("--color-thr", type=float, default=None, help="override dE00 threshold")
    ap.add_argument("--local-color-area", type=float, default=None,
                    help="also FAIL if this %% of the product area deviates strongly in colour (off by default)")
    ap.add_argument("--no-color", action="store_true", help="disable the colour branch")
    ap.add_argument("--roi", default="auto", choices=["auto", "center", "full"],
                    help="product region: auto (Otsu), fixed centre box, or whole frame")
    # video behaviour
    ap.add_argument("--smooth", type=int, default=5, help="temporal vote window (video/camera)")
    ap.add_argument("--votes", type=int, default=3, help="FAIL votes needed inside the window")
    ap.add_argument("--stride", type=int, default=1, help="inspect every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--cam-width", type=int, default=None)
    ap.add_argument("--cam-height", type=int, default=None)
    ap.add_argument("--allow-empty", action="store_true",
                    help="inspect every frame even if no product is detected in it")
    # output
    ap.add_argument("--out", default=str(OUTPUT_DIR))
    ap.add_argument("--save", action=argparse.BooleanOptionalAction, default=None,
                    help="save annotated video/images (default: on for files, off for camera)")
    ap.add_argument("--no-display", action="store_true", help="headless: do not open a window")
    return ap.parse_args()


def pick_profile(args):
    if args.no_color:
        return None
    profiles = load_profiles(args.color_reference)
    if not profiles:
        print(f"[color] no calibration at {args.color_reference} -> colour check DISABLED. "
              "Run calibrate_color.py to enable it.")
        return None
    if args.color_profile:
        if args.color_profile in profiles:
            return profiles[args.color_profile]
        print(f"[color] profile '{args.color_profile}' not found (have: {', '.join(profiles)}) -> colour DISABLED")
        return None
    if len(profiles) == 1:
        name = next(iter(profiles))
        print(f"[color] using the only profile: '{name}'")
        return profiles[name]
    print(f"[color] several profiles ({', '.join(profiles)}); pass --color-profile NAME -> colour DISABLED")
    return None


def main():
    args = parse_args()
    if not Path(args.weights).exists():
        raise SystemExit(f"Weights not found: {args.weights}\nTrain first (python train.py) or pass --weights.")

    cfg = InspectionConfig(
        imgsz=args.imgsz, conf=args.conf, min_area_pct=args.min_area, roi_mode=args.roi,
        color_threshold=args.color_thr, local_area_limit=args.local_color_area,
        smooth_window=args.smooth, smooth_votes=args.votes,
        require_product=not args.allow_empty and args.roi == "auto", device=args.device)

    pipe = InspectionPipeline(args.weights, cfg, pick_profile(args))
    pipe.warmup()

    try:
        src = FrameSource(args.source, args.cam_width, args.cam_height)
    except (RuntimeError, FileNotFoundError) as e:
        raise SystemExit(f"Error: {e}")
    temporal = src.live or src.is_video
    save = args.save if args.save is not None else (not src.live)
    out_dir = Path(args.out) / src.name
    out_dir.mkdir(parents=True, exist_ok=True)  # CSV log + summary are always written

    smoother = TemporalSmoother(cfg.smooth_window, cfg.smooth_votes) if temporal else None
    display = not args.no_display
    writer = None
    log_rows, ema_fps, paused_frame = [], None, None
    n_inspected = n_empty = raw_fail = stable_fail = events = 0
    prev_stable = None
    t_start = time.time()
    n_seen = 0

    print(f"Inspecting '{args.source}'  (conf>={cfg.conf}, area>={cfg.min_area_pct}%, "
          f"color={'on' if pipe.color.enabled else 'off'})")
    try:
        for idx, frame, label in src:
            n_seen += 1
            if args.max_frames and n_seen > args.max_frames:
                break
            if idx % max(1, args.stride):
                continue
            t0 = time.perf_counter()
            pid = f"F{idx:05d}" if temporal else label
            insp = pipe.inspect(frame, pid)
            raw = insp.record["final_result"]
            stable = smoother.update(raw) if smoother else raw

            if raw == "NO_PRODUCT":
                n_empty += 1
            else:
                n_inspected += 1
                raw_fail += raw == "FAIL"
            if stable == "FAIL":
                stable_fail += 1
                if prev_stable != "FAIL":
                    events += 1
            prev_stable = stable

            dt = time.perf_counter() - t0
            ema_fps = (1 / dt) if ema_fps is None else 0.9 * ema_fps + 0.1 * (1 / dt)
            log_rows.append({**insp.record, "frame": idx, "stable_result": stable,
                             "fail_reasons": "+".join(insp.record["fail_reasons"])})

            vis = pipe.annotate(frame, insp, stable, ema_fps)

            if save:
                if src.files:
                    cv2.imwrite(str(out_dir / f"{label}_{stable}.jpg"), vis)
                else:
                    if writer is None:
                        vh, vw = vis.shape[:2]
                        fps_out = max(1.0, (src.fps if not src.live else 20.0) / max(1, args.stride))
                        vpath = out_dir / f"{src.name}_inspected.mp4"
                        writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), fps_out, (vw, vh))
                    writer.write(vis)

            if src.files:
                r = insp.record
                print(f"{label:<40} {stable:<10} defect={r['defect_detected']!s:<5} conf={r['defect_confidence']:.2f} "
                      f"area={r['defect_area_percent']:.2f}% dE={r['delta_e']}")
            elif idx % 30 == 0:
                print(f"frame {idx:>6}  {stable:<10} {ema_fps:5.1f} FPS")

            if display:
                try:
                    cv2.imshow("Defect Inspection  (q quit | s snapshot | space pause)", vis)
                    wait = 0 if src.files and len(src.files) == 1 else 1
                    key = cv2.waitKey(wait) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("s"):
                        snap = out_dir / f"snapshot_{int(time.time())}.jpg"
                        cv2.imwrite(str(snap), vis)
                        print(f"saved {snap}")
                    if key == ord(" "):
                        cv2.waitKey(0)
                except cv2.error:
                    print("[info] no display available -> continuing headless (use --no-display to silence)")
                    display = False
    except KeyboardInterrupt:
        print("\nstopped by user")
    finally:
        src.close()
        if writer is not None:
            writer.release()
        if display:
            cv2.destroyAllWindows()

    # ------------------------------------------------------------------ summary
    if log_rows:
        with open(out_dir / "inspection_log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)

    if temporal:
        overall = "FAIL" if events > 0 else ("PASS" if n_inspected else "NO_PRODUCT")
    else:
        fails = sum(r["stable_result"] == "FAIL" for r in log_rows)
        overall = "FAIL" if fails else ("PASS" if n_inspected else "NO_PRODUCT")
        events = fails
    ms = [r["inference_ms"] for r in log_rows if r["final_result"] != "NO_PRODUCT"]
    summary = {
        "source": str(args.source), "overall_result": overall,
        "frames_inspected": n_inspected, "frames_without_product": n_empty,
        "raw_fail_frames": raw_fail, "stable_fail_frames": stable_fail,
        "fail_events": events,
        "avg_inference_ms": round(sum(ms) / len(ms), 1) if ms else None,
        "elapsed_s": round(time.time() - t_start, 1),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n" + "=" * 52)
    print(f" OVERALL RESULT : {overall}")
    print(f" inspected      : {n_inspected} frames  ({n_empty} without product)")
    print(f" FAIL events    : {events}")
    print(f" outputs        : {out_dir}")
    print("=" * 52)


if __name__ == "__main__":
    main()
