# tracking.py
import cv2
import numpy as np

from .config import USE_CV_TRACKER, CV_TRACKER_TYPE
from .utils import box_center, clip_box_to_frame

def compute_hist(image, bbox_xyxy):
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox_xyxy]
    x2 = min(x2, image.shape[1])
    y2 = min(y2, image.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    den = (np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / den)

def make_cv_tracker(name=CV_TRACKER_TYPE):
    legacy = getattr(cv2, "legacy", None)
    n = (name or "CSRT").upper()
    if n == "CSRT":
        return legacy.TrackerCSRT_create() if legacy and hasattr(legacy, "TrackerCSRT_create") else cv2.TrackerCSRT_create()
    if n == "KCF":
        return legacy.TrackerKCF_create() if legacy and hasattr(legacy, "TrackerKCF_create") else cv2.TrackerKCF_create()
    if n == "MOSSE":
        return legacy.TrackerMOSSE_create() if legacy and hasattr(legacy, "TrackerMOSSE_create") else cv2.TrackerMOSSE_create()
    return legacy.TrackerCSRT_create() if legacy and hasattr(legacy, "TrackerCSRT_create") else cv2.TrackerCSRT_create()

class GapPredictor:
    def __init__(self, frame_size, max_gap=12, damping=0.88):
        self.W, self.H = frame_size
        self.max_gap = int(max_gap)
        self.damping = float(damping)
        self.reset()

    def reset(self):
        self.last_box = None
        self.last_center = None
        self.vel = (0.0, 0.0)
        self.gap = 0
        self.active = False

    def update_with_detection(self, box_xyxy):
        x1, y1, x2, y2 = map(float, box_xyxy)
        c = box_center(x1, y1, x2, y2)
        if self.last_center is not None:
            vx = c[0] - self.last_center[0]
            vy = c[1] - self.last_center[1]
            self.vel = (0.6 * vx + 0.4 * self.vel[0],
                        0.6 * vy + 0.4 * self.vel[1])
        self.last_box = (x1, y1, x2, y2)
        self.last_center = c
        self.gap = 0
        self.active = False

    def predict_next(self):
        if self.last_box is None:
            return None
        if self.gap >= self.max_gap:
            self.reset()
            return None

        vx, vy = self.vel
        x1, y1, x2, y2 = self.last_box
        grow = 1.0 + 0.008 * (self.gap + 1)
        cx, cy = box_center(x1, y1, x2, y2)
        w = (x2 - x1) * grow
        h = (y2 - y1) * grow
        cx, cy = cx + vx, cy + vy

        nx1, ny1 = cx - w / 2, cy - h / 2
        nx2, ny2 = cx + w / 2, cy + h / 2
        nb = clip_box_to_frame((nx1, ny1, nx2, ny2), self.W, self.H)

        self.vel = (0.88 * vx, 0.88 * vy)
        self.last_box = nb
        self.last_center = box_center(*nb)
        self.gap += 1
        self.active = True
        return nb
