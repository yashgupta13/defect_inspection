"""Unit tests for the parts that need no trained model. Run: python -m pytest tests -q"""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from color_analysis import ColorAnalyzer, bgr_to_lab, build_profile, ciede2000, find_product_mask  # noqa: E402
from decision import TemporalSmoother, decide  # noqa: E402
from make_fake_mvtec import build, product  # noqa: E402
from prepare_dataset import mask_to_polygons  # noqa: E402


def test_ciede2000_reference_pairs():
    # Sharma, Wu & Dalal (2005) test data
    assert abs(float(ciede2000([50, 2.6772, -79.7751], [50, 0, -82.7485])) - 2.0425) < 1e-3
    assert abs(float(ciede2000([50, 2.5, 0], [73, 25, -18])) - 27.1492) < 1e-3
    assert float(ciede2000([60, 10, 10], [60, 10, 10])) == 0.0


def test_mask_to_polygons_roundtrip():
    m = np.zeros((100, 200), np.uint8)
    cv2.rectangle(m, (20, 30), (60, 50), 255, -1)
    polys = mask_to_polygons(m)
    assert len(polys) == 1
    assert polys[0][:, 0].max() <= 1 and polys[0][:, 1].max() <= 1
    back = np.zeros_like(m)
    cv2.fillPoly(back, [np.round(polys[0] * [200, 100]).astype(np.int32)], 255)
    inter, uni = ((back > 0) & (m > 0)).sum(), ((back > 0) | (m > 0)).sum()
    assert inter / uni > 0.9
    assert mask_to_polygons(np.zeros((50, 50), np.uint8)) == []


def test_decision_engine():
    assert decide({"detected": True}, {"color_defect": False})[0] == "FAIL"
    assert decide({"detected": False}, {"color_defect": True})[0] == "FAIL"
    assert decide({"detected": False}, {"color_defect": False})[0] == "PASS"
    assert decide({"detected": True}, {"color_defect": True})[1] == ["defect", "color"]
    assert decide({"detected": True}, {"color_defect": True}, product_present=False)[0] == "NO_PRODUCT"


def test_temporal_smoother():
    s = TemporalSmoother(window=5, votes=3)
    out = [s.update(r) for r in ["PASS", "FAIL", "PASS", "FAIL", "FAIL"]]
    assert out == ["PASS", "PASS", "PASS", "PASS", "FAIL"]
    assert s.update("NO_PRODUCT") == "NO_PRODUCT"
    assert s.update("FAIL") == "PASS"  # history cleared


def test_color_analyzer_detects_shift():
    rng = np.random.default_rng(0)
    good = [product(rng) for _ in range(15)]
    labs = []
    for g in good:
        mask, found = find_product_mask(g)
        assert found
        labs.append(ColorAnalyzer.measure(bgr_to_lab(g), mask)[0])
    prof = build_profile(labs)
    an = ColorAnalyzer(prof)
    mask, _ = an.product_mask(good[0])
    assert not an.analyze(good[0], mask)["color_defect"]

    shifted = product(rng, tint=(60, 120, 230))  # redder product
    mask, _ = an.product_mask(shifted)
    res = an.analyze(shifted, mask)
    assert res["color_defect"] and res["delta_e"] > prof["threshold"]


def test_product_mask_empty_scene():
    empty = np.full((240, 320, 3), 40, np.uint8)
    _, found = find_product_mask(empty)
    assert not found


def test_fake_dataset_builds():
    with tempfile.TemporaryDirectory() as d:
        build(d, categories=("widget",), n_train=3, n_test_good=2, n_def=3)
        assert len(list(Path(d, "widget", "test", "scratch").glob("*.png"))) == 3
