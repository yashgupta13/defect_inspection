"""Decision engine: combine defect + colour results into PASS / FAIL, plus video smoothing."""
from collections import deque


def decide(defect, color, product_present=True):
    """FAIL if a defect passed the criteria OR the colour difference exceeded its threshold.

    Returns (result, reasons) where result is PASS | FAIL | NO_PRODUCT.
    """
    if not product_present:
        return "NO_PRODUCT", []
    reasons = []
    if defect.get("detected"):
        reasons.append("defect")
    if color.get("color_defect"):
        reasons.append("color")
    return ("FAIL" if reasons else "PASS"), reasons


class TemporalSmoother:
    """Majority vote over the last `window` frames to suppress single-frame flicker.

    FAIL is reported once at least `votes` of the last `window` frames failed.
    An empty frame (NO_PRODUCT) clears the history so each new product starts fresh.
    """

    def __init__(self, window=5, votes=3):
        self.window = max(1, window)
        self.votes = max(1, min(votes, self.window))
        self.history = deque(maxlen=self.window)

    def update(self, result):
        if result == "NO_PRODUCT":
            self.history.clear()
            return "NO_PRODUCT"
        self.history.append(result == "FAIL")
        return "FAIL" if sum(self.history) >= self.votes else "PASS"
