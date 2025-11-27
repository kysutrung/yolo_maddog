import os
import cv2
import numpy as np
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import queue
import time  # timers

# -------- Optional dependencies --------
try:
    import pygame
except ImportError:
    print("You need 'pygame' (pip install pygame).")
    pygame = None

try:
    import vgamepad as vg
except ImportError:
    print("You need 'vgamepad' (pip install vgamepad) and ViGEmBus on Windows.")
    vg = None

# -------- ReID + Hungarian dependencies --------
try:
    import torch
    from torchvision import models, transforms
    TORCH_OK = True
except ImportError:
    print("You need 'torch' and 'torchvision' (pip install torch torchvision) for ResNet50 Re-ID.")
    torch = None
    models = None
    transforms = None
    TORCH_OK = False

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except ImportError:
    print("You need 'scipy' (pip install scipy) for Hungarian assignment. Fallback = greedy matching.")
    linear_sum_assignment = None
    SCIPY_OK = False

# =================== Configuration ===================
WEIGHTS        = "yolo_weights/yolov8m.engine"
USE_GPU        = True
CONF_THRES     = 0.30
CLASSES        = [0, 2, 3]
TRACKER_CFG    = "bytetrack.yaml"  # giữ lại cho đủ config, không dùng trực tiếp
PERSIST_ID     = True
SAVE_OUTPUT    = False
CAM_INDEX      = 0
TARGET_FPS     = 30

AX_LX, AX_LY, AX_RY, AX_RX = 0, 1, 2, 3
DEADZONE, ARM_EPS = 0.08, 0.05
SWAP_REAL_LR  = True

WIN_TITLE        = "YOLO Mad Dog Control Panel By TrungTauLua"
START_FULLSCREEN = False
VIDEO_W, VIDEO_H = 1024, 768
RIGHT_PANEL_W    = 320
THEME_PAD        = 10
# =====================================================

# ======== Enhanced Follow config ========
PRED_MAX_GAP        = 12
REACQ_IOU_THR       = 0.32
HIST_SIM_THR        = 0.50
USE_CV_TRACKER      = True
CV_TRACKER_TYPE     = "CSRT"
MIN_AREA_REACQ      = 200
# =====================================================

# ======== Navigation without on-frame text ===========
NAV_DEAD_ZONE_PX      = 110
CROSSHAIR_CENTER_SIZE = 6
CROSSHAIR_TARGET_SIZE = 6
# =====================================================

def dz(v, d=DEADZONE): return 0.0 if abs(v) < d else v

class ForwardState:
    OFF, ARMING, ON = "Auto", "Arming", "Manual"

# ================== Math & utility ==================
def next_higher(lst, cur): return next((x for x in lst if x > cur), None)

def resize_with_letterbox(bgr, target_w, target_h):
    h, w = bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(bgr, (new_w, new_h), cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), np.uint8)
    x, y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas

def clamp(v, lo, hi): return max(lo, min(hi, v))

def box_center(x1, y1, x2, y2): return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def clip_box_to_frame(b, W, H):
    x1, y1, x2, y2 = b
    return (
        clamp(int(x1), 0, W - 1),
        clamp(int(y1), 0, H - 1),
        clamp(int(x2), 0, W - 1),
        clamp(int(y2), 0, H - 1),
    )

def cosine_sim(a, b):
    if a is None or b is None:
        return 0.0
    den = (np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-9:
        return 0.0
    return float(np.dot(a, b) / den)

# ================== Drawing helpers ==================
def draw_tracks(frame, tracks, selected_id, names, show=True, allowed_classes=None):
    """
    tracks: list of dict {'id', 'cls', 'conf', 'bbox'}
    """
    if not show or tracks is None:
        return frame

    for trk in tracks:
        if trk.get("bbox") is None:
            continue
        x1, y1, x2, y2 = map(int, trk["bbox"])
        cls_i = int(trk.get("cls", 0))
        if (allowed_classes is not None) and (cls_i not in allowed_classes):
            continue
        label = names.get(cls_i, str(cls_i)) if isinstance(names, dict) else str(cls_i)
        track_id = int(trk.get("id", -1))
        conf = float(trk.get("conf", 1.0))
        color, thick = ((0, 0, 255), 3) if selected_id == track_id else ((0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        txt = f"{label} ID:{track_id if track_id != -1 else 'NA'} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(0, y1 - 8)
        cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 2), (0, 0, 0), -1)
        cv2.putText(frame, txt, (x1 + 3, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def draw_predicted_box(frame, box, label="PRED", color=(255, 200, 0)):
    if box is None:
        return frame
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + 90, max(0, y1 - 2)), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 4, max(10, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def draw_crosshair_center(frame, size=CROSSHAIR_CENTER_SIZE, color=(255, 0, 0)):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1, cv2.LINE_AA)

def draw_crosshair_at(frame, x, y, size=CROSSHAIR_TARGET_SIZE, color=(0, 255, 255)):
    x, y = int(x), int(y)
    cv2.line(frame, (x - size, y), (x + size, y), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y - size), (x, y + size), color, 1, cv2.LINE_AA)

# =============== Appearance & CV Tracker helpers (giữ để không phá code cũ) ===============
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

# ================== Gap Predictor (không dùng chính, giữ cho backward compat) ==================
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
            self.vel = (0.6 * vx + 0.4 * self.vel[0], 0.6 * vy + 0.4 * self.vel[1])
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

# ================== Gamepad bridge ==================
class GamepadBridge:
    def __init__(self):
        self.pad_name = "N/A"
        self.pygame_ok = self.vpad_ok = False
        self.state = ForwardState.ON   # DEFAULT = Manual
        self.v_lx = self.v_ly = self.v_rx = self.v_ry = 0.0
        self.r_lx = self.r_ly = self.r_rx = self.r_ry = 0.0
        if pygame:
            try:
                pygame.init()
                pygame.joystick.init()
                if pygame.joystick.get_count() > 0:
                    self.js = pygame.joystick.Joystick(0)
                    self.js.init()
                    self.pad_name = self.js.get_name()
                    self.pygame_ok = True
            except Exception as e:
                print(e)
        if vg:
            try:
                self.vpad = vg.VX360Gamepad()
                self.vpad_ok = True
            except Exception as e:
                print(e)
    def read_axes_real(self):
        if not self.pygame_ok:
            return (0, 0, 0, 0)
        try:
            pygame.event.pump()
            if SWAP_REAL_LR:
                lx, ly, rx, ry = [dz(self.js.get_axis(a)) for a in [AX_RX, AX_RY, AX_LX, AX_LY]]
            else:
                lx, ly, rx, ry = [dz(self.js.get_axis(a)) for a in [AX_LX, AX_LY, AX_RX, AX_RY]]
        except Exception:
            lx = ly = rx = ry = 0.0
        self.r_lx, self.r_ly, self.r_rx, self.r_ry = [max(-1, min(1, float(v))) for v in [lx, ly, rx, ry]]
        return self.r_lx, self.r_ly, self.r_rx, self.r_ry
    def send_to_virtual(self, lx, ly, rx, ry):
        self.v_lx, self.v_ly, self.v_rx, self.v_ry = lx, ly, rx, ry
        if not self.vpad_ok:
            return
        self.vpad.left_joystick_float(lx, ly)
        self.vpad.right_joystick_float(rx, ry)
        self.vpad.update()

# ================== ReID + Kalman + Hungarian ==================

class ReIDExtractor:
    """
    Dùng ResNet50 (torchvision) để trích đặc trưng Re-ID.
    """
    def __init__(self, device="cpu"):
        self.enabled = TORCH_OK
        if not TORCH_OK:
            self.model = None
            self.transform = None
            return
        self.device = torch.device(device)
        try:
            # Với torch >= 2.0
            base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            # Fallback cho version cũ
            base = models.resnet50(pretrained=True)
        modules = list(base.children())[:-1]  # bỏ layer FC
        self.model = torch.nn.Sequential(*modules).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, frame_bgr, bbox_xyxy):
        if not self.enabled or self.model is None or self.transform is None:
            return None
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        h, w = frame_bgr.shape[:2]
        x1 = clamp(x1, 0, w - 1)
        x2 = clamp(x2, 0, w - 1)
        y1 = clamp(y1, 0, h - 1)
        y2 = clamp(y2, 0, h - 1)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(img)
        feat = feat.view(feat.size(0), -1)
        feat = feat / (feat.norm(p=2, dim=1, keepdim=True) + 1e-12)
        return feat[0].cpu().numpy()


class KalmanBoxTracker:
    """
    Kalman Filter cho bbox: state = [cx, cy, w, h, vx, vy, vw, vh]
    """
    def __init__(self, box_xyxy, dt=1.0):
        self.dt = dt
        self.dim_x = 8
        self.dim_z = 4

        self.F = np.eye(self.dim_x, dtype=float)
        for i in range(4):
            self.F[i, i + 4] = self.dt

        self.H = np.zeros((self.dim_z, self.dim_x), dtype=float)
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        self.P = np.eye(self.dim_x, dtype=float) * 10.0
        self.Q = np.eye(self.dim_x, dtype=float) * 0.01
        self.R = np.eye(self.dim_z, dtype=float) * 1.0

        self.x = self._box_to_state(box_xyxy)

    def _box_to_state(self, box):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        x = np.zeros((self.dim_x, 1), dtype=float)
        x[0, 0] = cx
        x[1, 0] = cy
        x[2, 0] = w
        x[3, 0] = h
        return x

    def _state_to_box(self):
        cx = self.x[0, 0]
        cy = self.x[1, 0]
        w = max(1.0, self.x[2, 0])
        h = max(1.0, self.x[3, 0])
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return (x1, y1, x2, y2)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_state_bbox()

    def update(self, box_xyxy):
        x1, y1, x2, y2 = box_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        z = np.array([[cx], [cy], [w], [h]], dtype=float)

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ self.x)
        self.x = self.x + K @ y
        I = np.eye(self.dim_x, dtype=float)
        self.P = (I - K @ self.H) @ self.P

    def get_state_bbox(self):
        return self._state_to_box()


class Track:
    _next_id = 0

    def __init__(self, box, cls_id, score, feature=None):
        self.track_id = Track._next_id
        Track._next_id += 1
        self.cls_id = int(cls_id)
        self.score = float(score)
        self.kf = KalmanBoxTracker(box)
        self.last_box = box
        self.feature = feature
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

    def predict(self):
        self.last_box = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.last_box

    def update(self, box, cls_id, score, feature=None):
        self.time_since_update = 0
        self.hits += 1
        self.cls_id = int(cls_id)
        self.score = float(score)
        if box is not None:
            self.kf.update(box)
            self.last_box = self.kf.get_state_bbox()
        if feature is not None:
            if self.feature is None:
                self.feature = feature
            else:
                self.feature = 0.8 * self.feature + 0.2 * feature


def hungarian_assignment(cost_matrix):
    """
    Wrapper dùng Hungarian (scipy) nếu có, không có thì dùng greedy fallback.
    """
    if linear_sum_assignment is not None:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return list(zip(row_ind, col_ind))

    # Fallback: greedy matching
    num_rows, num_cols = cost_matrix.shape
    used_rows = set()
    used_cols = set()
    matches = []
    while True:
        min_cost = np.inf
        min_r = min_c = None
        for r in range(num_rows):
            if r in used_rows:
                continue
            for c in range(num_cols):
                if c in used_cols:
                    continue
                if cost_matrix[r, c] < min_cost:
                    min_cost = cost_matrix[r, c]
                    min_r, min_c = r, c
        if min_r is None:
            break
        matches.append((min_r, min_c))
        used_rows.add(min_r)
        used_cols.add(min_c)
    return matches


class MultiObjectTracker:
    """
    Tracking-by-detection:
        - YOLO: detection
        - Kalman Filter: motion
        - ResNet50: appearance (Re-ID)
        - Hungarian: assignment
    """
    def __init__(self, reid_extractor=None,
                 max_age=30,
                 min_hits=1,
                 iou_threshold=0.3,
                 appearance_weight=0.5,
                 max_cost=0.7):
        self.reid = reid_extractor
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_weight = appearance_weight if (reid_extractor is not None and reid_extractor.enabled) else 0.0
        self.max_cost = max_cost
        self.tracks = []

    def _start_new_track(self, det, feat):
        box = det["box"]
        cls_id = det["cls"]
        score = det["score"]
        t = Track(box, cls_id, score, feat)
        self.tracks.append(t)

    def update(self, frame_bgr, detections):
        """
        detections: list[{"box": (x1,y1,x2,y2), "score": float, "cls": int}]
        """
        # Step 0: predict tất cả track hiện tại
        for t in self.tracks:
            t.predict()

        # Step 1: chuẩn bị feature cho detection (ReID)
        det_features = []
        for det in detections:
            if self.reid is not None and self.reid.enabled:
                feat = self.reid.extract(frame_bgr, det["box"])
            else:
                feat = None
            det_features.append(feat)

        if len(detections) == 0:
            # chỉ giữ lại track chưa quá cũ
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return list(self.tracks)

        if len(self.tracks) == 0:
            for det, feat in zip(detections, det_features):
                self._start_new_track(det, feat)
            return list(self.tracks)

        # Step 2: tạo cost matrix cho Hungarian
        N = len(self.tracks)
        M = len(detections)
        cost_matrix = np.zeros((N, M), dtype=np.float32)

        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                dbox = det["box"]
                tbox = track.last_box
                iou_val = iou(tbox, dbox)
                if iou_val < self.iou_threshold:
                    one_minus_iou = 1.0
                else:
                    one_minus_iou = 1.0 - iou_val

                if self.appearance_weight > 0.0 and track.feature is not None and det_features[di] is not None:
                    sim = cosine_sim(track.feature, det_features[di])
                    sim = max(0.0, sim)
                    app_cost = 1.0 - sim
                else:
                    app_cost = 1.0

                cost_matrix[ti, di] = (1.0 - self.appearance_weight) * one_minus_iou + \
                                      self.appearance_weight * app_cost

        # Step 3: Hungarian assignment
        matches = hungarian_assignment(cost_matrix)

        matched_tracks = set()
        matched_dets = set()

        for ti, di in matches:
            if ti < 0 or ti >= N or di < 0 or di >= M:
                continue
            if cost_matrix[ti, di] > self.max_cost:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            det = detections[di]
            feat = det_features[di]
            self.tracks[ti].update(det["box"], det["cls"], det["score"], feat)

        # Step 4: unmatched detections -> new tracks
        unmatched_det_indices = [di for di in range(M) if di not in matched_dets]
        for di in unmatched_det_indices:
            det = detections[di]
            feat = det_features[di]
            self._start_new_track(det, feat)

        # Step 5: xóa track quá cũ
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return list(self.tracks)

# ================== App ==================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(WIN_TITLE)
        self.geometry("1420x900")
        if START_FULLSCREEN:
            self.attributes("-fullscreen", True)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.selected_id = None
        self.available_ids = []
        self.running = True
        self.frame_lock = threading.Lock()
        self.latest_bgr = None
        self.video_ready = False
        self.gp = GamepadBridge()

        # device cho YOLO + ReID
        if TORCH_OK and USE_GPU and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.model = YOLO(WEIGHTS)
        self.names = self.model.names

        # ReID + Tracker mới
        self.reid = ReIDExtractor(device=self.device) if TORCH_OK else None
        self.tracker = MultiObjectTracker(self.reid)

        # Re-ID signature cho mục tiêu được chọn (giữ để re-ID khi quay lại)
        self.selected_identity_feat = None
        self.reid_reacq_thr = 0.70  # ngưỡng Re-ID để nhận lại mục tiêu
        self.target_lost_flag = False

        # hiển thị & chọn mục tiêu
        self.show_boxes = False
        self.bbox_btn_var = tk.StringVar(value="Object Detection: OFF")
        self.lock_target = False
        self.lock_btn_var = tk.StringVar(value="Target Lock: OFF")

        # Target Follow
        self.target_follow = False
        self.follow_btn_var = tk.StringVar(value="Target Follow: OFF")

        # FPS
        self._fps_frame_count = 0
        self._fps_last_time = time.monotonic()
        self._fps_value = 0.0
        self.fps_var = tk.StringVar(value="FPS: 0.0")

        # lọc theo lớp
        self.filter_var = tk.StringVar(value="All")
        self.display_allowed = None

        # flight buttons
        self._flight_btns = []

        # reselect / lock
        self.lock_signature = None
        self.reselect_signature = None
        self.last_tracks = {}   # id -> cls_id

        # log queue
        self.log_queue = queue.Queue()

        # styles
        self._init_styles()

        # ---- Auto ramp settings ----
        self.RY_CAP = 0.300
        self.RY_STEP = 0.05
        self.RX_CAP = 0.15
        self.RX_STEP = 0.05
        self.LX_CAP = 0.15
        self.LX_STEP = 0.05

        # Giá trị hiện tại (Auto) gửi ra vgamepad
        self.auto_ry = 0.0
        self.auto_rx = 0.0
        self.auto_lx = 0.0

        # Ramp state (manual holds)
        self._ramp_job = None
        self._forward_holding = False
        self._back_holding = False
        self._left_holding = False
        self._right_holding = False
        self._yaw_left_holding = False
        self._yaw_right_holding = False

        # Auto-hold flags/timers
        self._auto_left = False
        self._auto_right = False
        self._auto_yaw_left = False
        self._auto_yaw_right = False
        self._auto_back = False
        self._auto_forward = False
        self._nav_left_until = 0.0
        self._nav_right_until = 0.0
        self._nav_back_until = 0.0
        self._nav_forward_until = 0.0

        # giữ LY khi vào Auto
        self.auto_ly_hold = 0.0

        # ===== Enhanced follow states (giữ cấu trúc cũ) =====
        self.PRED_IOU_THR = REACQ_IOU_THR
        self.PRED_MAX_GAP = PRED_MAX_GAP
        self.pred = GapPredictor((VIDEO_W, VIDEO_H), max_gap=self.PRED_MAX_GAP, damping=0.88)
        self.predicted_box = None

        self.sel_hist = None
        self.cv_tracker = None
        self.hist_sim_thr = HIST_SIM_THR

        # ===== Navigation (logs only) =====
        self.nav_dead_zone_px = NAV_DEAD_ZONE_PX
        self.last_nav_cmd = None

        # ===== Distance Estimation =====
        self.ref_rect = None
        self.ref_area = None
        self.last_distance_state = None
        self.dist_thr_near_low = 0.70
        self.dist_thr_near_high = 1.30
        self.draw_ref_rect = True

        self._build_ui()

        # key bindings
        for key, func in {
            "<Escape>": self.on_close,
            "<s>": self.on_key_s,
            "<S>": self.on_key_s,
            "<a>": self.on_key_a,
            "<A>": self.on_key_a,
            "<d>": self.on_key_d,
            "<D>": self.on_key_d
        }.items():
            self.bind(key, lambda e, f=func: f())

        threading.Thread(target=self._loop_worker, daemon=True).start()
        self.after(33, self._update_ui)

        self.log("Application started")
        self.log(f"Model: {WEIGHTS} | GPU: {'ON' if USE_GPU else 'OFF'} | Device: {self.device}")
        self.log(f"Gamepad: {self.gp.pad_name}")
        self.log(f"ReID (ResNet50): {'ON' if (self.reid is not None and self.reid.enabled) else 'OFF'}")
        self._update_target_controls_state()
        self._update_flight_mode_controls()

    # ---------- Styles ----------
    def _init_styles(self):
        style = ttk.Style(self)
        style.configure("Compact.TButton", font=("Segoe UI", 9), padding=(6, 2))
        style.configure("Compact.TLabel", font=("Segoe UI", 9))
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Flight.TButton", font=("Segoe UI", 9), padding=(6, 2))
        style.configure("Foot.TLabel", font=("Segoe UI", 9))

    # ---------- Utility ----------
    def log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{ts}] {message}\n")

    def _confirm(self, title, message) -> bool:
        return messagebox.askyesno(title=title, message=message, parent=self)

    def _update_target_controls_state(self):
        enabled_select = (self.show_boxes and not self.lock_target)
        for b in (self.btn_select, self.btn_switch_target):
            b.configure(state=("normal" if enabled_select else "disabled"))
        # Follow & Approach chỉ khi Auto + có target + OD ON
        follow_enabled = (self.show_boxes and self.selected_id is not None and self.gp.state == ForwardState.OFF)
        self.btn_follow.configure(state=("normal" if follow_enabled else "disabled"))
        self.btn_approach.configure(state=("normal" if follow_enabled else "disabled"))

    def _get_class_for_id(self, tid):
        if tid is None:
            return None, None
        cls_id = self.last_tracks.get(int(tid), None)
        if cls_id is None:
            return None, None
        if isinstance(self.names, dict):
            cls_name = self.names.get(int(cls_id), str(cls_id))
        else:
            cls_name = str(cls_id)
        return int(cls_id), cls_name

    def _update_flight_mode_controls(self):
        enable = (self.gp.state == ForwardState.OFF)
        state = "normal" if enable else "disabled"
        for b in self._flight_btns:
            b.configure(state=state)
        self.btn_fx.configure(state=state)
        self._update_target_controls_state()

    # ---------- Filter helpers ----------
    def _filter_label_to_allowed(self, label: str):
        label = (label or "").strip().lower()
        if label in ("all", "tất cả"):
            return None
        if label in ("person", "người"):
            return {0}
        if label in ("car", "xe ô tô", "oto", "ô tô"):
            return {2}
        if label in ("motorcycle", "xe máy", "xe may", "motorbike"):
            return {3}
        return None

    def _rebuild_available_ids(self):
        if self.display_allowed is None:
            self.available_ids = sorted(self.last_tracks.keys())
        else:
            self.available_ids = sorted([tid for tid, cls in self.last_tracks.items() if cls in self.display_allowed])
        if self.selected_id is not None and self.selected_id not in self.available_ids:
            self.log(f"Current target ID={self.selected_id} is hidden by filter -> deselect")
            self.selected_id = None
            self.pred.reset()
            self.predicted_box = None
            self.cv_tracker = None
            self.sel_hist = None
            self.last_nav_cmd = None
        self._update_target_controls_state()

    def on_filter_change(self, *_):
        self.display_allowed = self._filter_label_to_allowed(self.filter_var.get())
        self.log(f"Object filter -> {self.filter_var.get()}")
        self._rebuild_available_ids()

    # ---------- UI ----------
    def _build_ui(self):
        root = ttk.Frame(self, padding=THEME_PAD)
        root.pack(fill="both", expand=True)

        ttk.Frame(root).pack(side="left", fill="both", expand=True)

        # LEFT video
        left = ttk.Frame(root, width=VIDEO_W, height=VIDEO_H)
        left.pack(side="left")
        left.pack_propagate(False)
        self.video_canvas = tk.Canvas(left, width=VIDEO_W, height=VIDEO_H, bg="black", highlightthickness=0)
        self.video_canvas.pack()

        ttk.Frame(root).pack(side="left", fill="both", expand=True)

        # RIGHT panel with right-edge gap
        right = ttk.Frame(root, width=RIGHT_PANEL_W)
        right.pack(side="left", fill="y", padx=(0, 50))
        right.pack_propagate(False)

        # ---- 1) Flight Mode (top)
        flight = ttk.Frame(right)
        self.state_var = tk.StringVar(value=f"Flight Mode: {self.gp.state}")
        ttk.Label(flight, textvariable=self.state_var, style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Button(flight, text="Switch Mode", command=self.toggle_forward,
                   style="Compact.TButton").pack(fill="x")

        # ---- 2) Autonomous Control group
        ac = ttk.Frame(right)
        ttk.Label(ac, text="Autonomous Control", style="Section.TLabel").pack(anchor="w", pady=(0, 6))

        nav_border = tk.Frame(ac, highlightbackground="#888", highlightthickness=1, bd=0)
        nav_border.pack(fill="x", pady=(0, 6))
        fm = ttk.Frame(nav_border, padding=6)
        fm.pack(fill="x")

        mk = lambda t, cmd: ttk.Button(fm, text=t, command=cmd, style="Flight.TButton", width=10)
        self.btn_forward = mk("Forward", self.cmd_forward)
        self.btn_back = mk("Backward", self.cmd_back)
        self.btn_up = mk("Up", self.cmd_up)
        self.btn_left = mk("Left", self.cmd_left)
        self.btn_right = mk("Right", self.cmd_right)
        self.btn_down = mk("Down", self.cmd_down)
        self.btn_forward.grid(row=0, column=0, padx=(0, 4), pady=(0, 4), sticky="ew")
        self.btn_back.grid(row=0, column=1, padx=4, pady=(0, 4), sticky="ew")
        self.btn_up.grid(row=0, column=2, padx=(4, 0), pady=(0, 4), sticky="ew")
        self.btn_left.grid(row=1, column=0, padx=(0, 4), sticky="ew")
        self.btn_right.grid(row=1, column=1, padx=4, sticky="ew")
        self.btn_down.grid(row=1, column=2, padx=(4, 0), sticky="ew")
        self.btn_yaw_left = mk("Yaw Left", lambda: None)
        self.btn_yaw_right = mk("Yaw Right", lambda: None)
        self.btn_yaw_left.grid(row=2, column=0, padx=(0, 4), pady=(0, 4), sticky="ew")
        self.btn_yaw_right.grid(row=2, column=1, padx=4, pady=(0, 4), sticky="ew")
        for c in range(3):
            fm.grid_columnconfigure(c, weight=1, uniform="fm")
        self._flight_btns = [
            self.btn_forward, self.btn_back, self.btn_left, self.btn_right,
            self.btn_up, self.btn_down, self.btn_yaw_left, self.btn_yaw_right
        ]

        # ---- Target control row
        tc = ttk.Frame(ac)
        tc.pack(fill="x", pady=(0, 4))
        self.btn_select = ttk.Button(tc, text="Select", command=self.on_key_s,
                                     style="Compact.TButton", width=10)
        self.btn_switch_target = ttk.Button(tc, text="Switch Target",
                                            command=self.on_switch_target,
                                            style="Compact.TButton", width=14)
        self.btn_select.grid(row=0, column=0, padx=(0, 4), pady=(0, 2), sticky="ew")
        self.btn_switch_target.grid(row=0, column=1, padx=(4, 0), pady=(0, 2), sticky="ew")
        tc.grid_columnconfigure(0, weight=1, uniform="tc")
        tc.grid_columnconfigure(1, weight=1, uniform="tc")

        # Toggles row
        toggles = ttk.Frame(ac)
        toggles.pack(fill="x", pady=(0, 6))
        self.btn_bbox = ttk.Button(toggles, textvariable=self.bbox_btn_var,
                                   command=self.toggle_bboxes, style="Compact.TButton")
        self.btn_lock = ttk.Button(toggles, textvariable=self.lock_btn_var,
                                   command=self.toggle_lock, style="Compact.TButton")
        self.btn_bbox.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.btn_lock.grid(row=0, column=1, padx=(4, 0), sticky="ew")
        toggles.grid_columnconfigure(0, weight=1, uniform="tog")
        toggles.grid_columnconfigure(1, weight=1, uniform="tog")

        # Filter row
        filt_row = ttk.Frame(ac)
        filt_row.pack(fill="x", pady=(0, 6))
        ttk.Label(filt_row, text="Show:", style="Compact.TLabel", width=8).grid(row=0, column=0, sticky="w")
        self.filter_combo = ttk.Combobox(
            filt_row,
            textvariable=self.filter_var,
            values=["All", "Person", "Car", "Motorcycle"],
            state="readonly",
            width=18
        )
        self.filter_combo.grid(row=0, column=1, sticky="ew")
        filt_row.grid_columnconfigure(1, weight=1)
        self.filter_combo.bind("<<ComboboxSelected>>", self.on_filter_change)

        # Follow + buttons
        self.btn_follow = ttk.Button(ac, textvariable=self.follow_btn_var,
                                     command=self.toggle_follow, style="Compact.TButton")
        self.btn_follow.pack(fill="x", pady=(0, 4))

        self.btn_approach = ttk.Button(ac, text="Target Aproach",
                                       command=self.cmd_target_approach, style="Compact.TButton")
        self.btn_approach.pack(fill="x", pady=(0, 4))

        self.btn_fx = ttk.Button(ac, text="Function X",
                                 command=self.cmd_function_x, style="Compact.TButton")
        self.btn_fx.pack(fill="x")

        # Bind press/release cho ramp (Auto)
        self.btn_forward.bind("<ButtonPress-1>", self._forward_press)
        self.btn_forward.bind("<ButtonRelease-1>", self._forward_release)
        self.btn_back.bind("<ButtonPress-1>", self._back_press)
        self.btn_back.bind("<ButtonRelease-1>", self._back_release)
        self.btn_left.bind("<ButtonPress-1>", self._left_press)
        self.btn_left.bind("<ButtonRelease-1>", self._left_release)
        self.btn_right.bind("<ButtonPress-1>", self._right_press)
        self.btn_right.bind("<ButtonRelease-1>", self._right_release)
        self.btn_yaw_left.bind("<ButtonPress-1>", self._yaw_left_press)
        self.btn_yaw_left.bind("<ButtonRelease-1>", self._yaw_left_release)
        self.btn_yaw_right.bind("<ButtonPress-1>", self._yaw_right_press)
        self.btn_yaw_right.bind("<ButtonRelease-1>", self._yaw_right_release)

        # ---- 3) Control Parameters
        params = ttk.Frame(right)
        ttk.Label(params, text="Control Parameters", style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        telem = ttk.Frame(params)
        telem.pack(anchor="w", fill="x")

        mono_font = ("Consolas", 10) if os.name == "nt" else ("Menlo", 10)

        ttk.Label(telem, text="Gamepad:", style="Compact.TLabel", width=10) \
            .grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        self.pad_name_var = tk.StringVar(value=self.gp.pad_name)
        ttk.Label(telem, textvariable=self.pad_name_var, style="Compact.TLabel") \
            .grid(row=0, column=1, columnspan=2, sticky="w", pady=2)

        ttk.Label(telem, text="Axis", style="Compact.TLabel", width=6) \
            .grid(row=1, column=0, sticky="w", padx=(0, 6), pady=(6, 2))
        ttk.Label(telem, text="Physical", style="Compact.TLabel", width=10) \
            .grid(row=1, column=1, sticky="w", padx=(0, 6), pady=(6, 2))
        ttk.Label(telem, text="Virtual", style="Compact.TLabel", width=10) \
            .grid(row=1, column=2, sticky="w", padx=(0, 0),  pady=(6, 2))

        axes = ["LX", "LY", "RX", "RY"]
        self.real_vars = [tk.StringVar(value="+0.000") for _ in range(4)]
        self.virt_vars = [tk.StringVar(value="+0.000") for _ in range(4)]

        for i, name in enumerate(axes):
            r = 2 + i
            ttk.Label(telem, text=f"{name}:", style="Compact.TLabel", width=6) \
                .grid(row=r, column=0, sticky="w", padx=(0, 6))
            tk.Label(telem, textvariable=self.real_vars[i], font=mono_font, width=7, anchor="e") \
                .grid(row=r, column=1, sticky="w")
            tk.Label(telem, textvariable=self.virt_vars[i], font=mono_font, width=7, anchor="e") \
                .grid(row=r, column=2, sticky="w")

        telem.grid_columnconfigure(0, weight=0)
        telem.grid_columnconfigure(1, weight=1)
        telem.grid_columnconfigure(2, weight=1)

        # ---- 4) Logs
        logs = ttk.Frame(right)
        ttk.Label(logs, text="Logs", style="Section.TLabel").pack(anchor="w", pady=(0, 4))
        log_holder = ttk.Frame(logs, height=110)
        log_holder.pack(fill="x", expand=False)
        log_holder.pack_propagate(False)
        self.log_text = tk.Text(log_holder, height=5, wrap="word", state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_holder, orient="vertical", command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        # ---- 5) FPS (footer)
        fps_panel = ttk.Frame(right)
        ttk.Label(fps_panel, textvariable=self.fps_var, style="Foot.TLabel").pack(anchor="w")

        sections = [flight, ac, params, logs, fps_panel]
        row = 0
        for sect in sections:
            spacer = ttk.Frame(right)
            spacer.grid(row=row, column=0, sticky="nsew")
            right.grid_rowconfigure(row, weight=1)
            row += 1
            sect.grid(row=row, column=0, sticky="nsew", padx=0)
            right.grid_rowconfigure(row, weight=0)
            row += 1
        spacer = ttk.Frame(right)
        spacer.grid(row=row, column=0, sticky="nsew")
        right.grid_rowconfigure(row, weight=1)
        right.grid_columnconfigure(0, weight=1)

    # ---------- Logic ----------
    def toggle_forward(self):
        prev = self.gp.state
        self.gp.state = ForwardState.ON if prev == ForwardState.ARMING \
            else (ForwardState.ARMING if prev == ForwardState.OFF else ForwardState.OFF)
        self.state_var.set(f"Flight Mode: {self.gp.state}")
        self._update_flight_mode_controls()
        self.log(f"Flight Mode -> {self.gp.state}")

        if prev == ForwardState.OFF and self.gp.state in (ForwardState.ARMING, ForwardState.ON):
            if self.target_follow:
                self.target_follow = False
                self.follow_btn_var.set("Target Follow: OFF")
                self.log("Target Follow -> OFF (leaving Auto)")
            self._clear_auto_holds()
            self._stop_ramp()

        if self.gp.state == ForwardState.OFF:
            self.auto_ly_hold = float(self.gp.v_ly)
            self.log(f"Auto mode: hold LY = {self.auto_ly_hold:+.3f}")

        self._update_target_controls_state()

    def _clear_auto_holds(self):
        changed = any([self._auto_left, self._auto_right, self._auto_yaw_left,
                       self._auto_yaw_right, self._auto_back, self._auto_forward])
        self._auto_left = self._auto_right = False
        self._nav_left_until = self._nav_right_until = 0.0
        if self._auto_yaw_left or self._auto_yaw_right or self._auto_back or self._auto_forward:
            changed = True
        self._nav_back_until = 0.0
        self._nav_forward_until = 0.0
        self._auto_back = False
        self._auto_forward = False
        if changed:
            self._start_ramp_loop()

    def toggle_bboxes(self):
        if self.show_boxes and self.lock_target:
            self.log("Blocked: Target Lock is active, cannot turn OFF Object Detection")
            self.bbox_btn_var.set("Object Detection: ON")
            return
        prev = self.show_boxes
        self.show_boxes = not self.show_boxes
        self.bbox_btn_var.set(f"Object Detection: {'ON' if self.show_boxes else 'OFF'}")
        self.log(f"Object Detection -> {'ON' if self.show_boxes else 'OFF'}")
        if prev and not self.show_boxes:
            if self.selected_id is not None:
                self.log(f"Object Detection turned OFF -> deselected ID={self.selected_id}")
            self.selected_id = None
            self.reselect_signature = None
            self.selected_identity_feat = None
            self.target_lost_flag = False
            if self.target_follow:
                self.target_follow = False
                self.follow_btn_var.set("Target Follow: OFF")
                self.log("Target Follow -> OFF (Object Detection OFF)")
            self.pred.reset()
            self.predicted_box = None
            self.cv_tracker = None
            self.sel_hist = None
            self.last_nav_cmd = None
            self._clear_auto_holds()
            self.last_distance_state = None
            self.log("No target is currently selected")
        self._update_target_controls_state()

    def toggle_lock(self):
        if not self.show_boxes:
            self.lock_target = False
            self.lock_btn_var.set("Target Lock: OFF")
            self.log("Blocked: Object Detection is OFF -> cannot enable Target Lock")
            self.log("No target is currently selected")
            self._update_target_controls_state()
            return
        if self.selected_id is None and not self.lock_target:
            self.log("Blocked: Select a target before enabling Target Lock")
            self.lock_target = False
            self.lock_btn_var.set("Target Lock: OFF")
            self._update_target_controls_state()
            return
        if not self.lock_target:
            cls_id, cls_name = self._get_class_for_id(self.selected_id)
            if cls_id is None:
                self.log("Blocked: Cannot read target class -> lock aborted")
                return
            self.lock_signature = (int(self.selected_id), int(cls_id), str(cls_name))
            self.lock_target = True
            self.lock_btn_var.set("Target Lock: ON")
            self.log(f"Target Lock -> ON (ID={self.lock_signature[0]}, class={self.lock_signature[2]})")
        else:
            self.lock_target = False
            self.lock_btn_var.set("Target Lock: OFF")
            self.log("Target Lock -> OFF")
            self.lock_signature = None
        self._update_target_controls_state()

    def toggle_follow(self):
        if self.gp.state != ForwardState.OFF:
            self.target_follow = False
            self.follow_btn_var.set("Target Follow: OFF")
            self.log("Blocked: Target Follow requires Flight Mode = Auto")
            self._update_target_controls_state()
            return
        if not self.show_boxes:
            self.target_follow = False
            self.follow_btn_var.set("Target Follow: OFF")
            self.log("Blocked: Object Detection is OFF -> cannot enable Target Follow")
            self._update_target_controls_state()
            return
        if self.selected_id is None:
            self.target_follow = False
            self.follow_btn_var.set("Target Follow: OFF")
            self.log("Blocked: No target selected -> cannot enable Target Follow")
            self._update_target_controls_state()
            return

        if not self.target_follow:
            ok = self._confirm("Warning !", "Target Follow?")
            if not ok:
                self.log("Target Follow -> cancelled by user")
                return

        self.target_follow = not self.target_follow
        self.follow_btn_var.set(f"Target Follow: {'ON' if self.target_follow else 'OFF'}")
        if self.target_follow:
            cls_id, cls_name = self._get_class_for_id(self.selected_id)
            self.log(f"Target Follow -> ON (ID={self.selected_id}{', class=' + str(cls_name) if cls_name else ''})")
            self.last_nav_cmd = None
        else:
            self.log("Target Follow -> OFF")
            self.last_nav_cmd = None
            self._clear_auto_holds()

    def cmd_target_approach(self):
        if self.gp.state != ForwardState.OFF or not self.show_boxes or self.selected_id is None:
            self.log("Target Aproach: yêu cầu Auto mode + OD ON + đã chọn target")
            return
        if not self._confirm("Warning !", "Target Aproach?"):
            self.log("Target Aproach -> cancelled by user")
            return
        self.log(f"Target Aproach pressed (ID={self.selected_id}) - placeholder")

    def cmd_function_x(self):
        if self.gp.state != ForwardState.OFF:
            self.log("Function X: yêu cầu Flight Mode = Auto")
            return
        if not self._confirm("Xác nhận", "Thực thi Function X?"):
            self.log("Function X -> cancelled by user")
            return
        self.log("Function X pressed - placeholder for future code")

    # --- Selection handlers / Switch Target ---
    def on_key_s(self):
        if not self.show_boxes:
            self.log("Blocked: Object Detection is OFF (S)")
            self.log("No target is currently selected")
            return
        if self.lock_target:
            self.log("Blocked: Target Lock is active (S)")
            return
        if self.available_ids:
            if self.selected_id is None:
                # select target
                self.selected_id = self.available_ids[0]
                cls_id, cls_name = self._get_class_for_id(self.selected_id)
                self.reselect_signature = (int(self.selected_id), int(cls_id), str(cls_name)) if cls_id is not None else None
                self.selected_identity_feat = None
                self.target_lost_flag = False
                self.log(f"Selected target ID={self.selected_id}")
                self.last_nav_cmd = None
            else:
                # deselect target
                self.log("Target deselected")
                self.selected_id = None
                self.reselect_signature = None
                self.selected_identity_feat = None
                self.target_lost_flag = False
                if self.target_follow:
                    self.target_follow = False
                    self.follow_btn_var.set("Target Follow: OFF")
                    self.log("Target Follow -> OFF (no target selected)")
                self.pred.reset()
                self.predicted_box = None
                self.cv_tracker = None
                self.sel_hist = None
                self.last_nav_cmd = None
                self._clear_auto_holds()
                self.last_distance_state = None
        else:
            self.log("No available targets to select")
        self._update_target_controls_state()

    def on_switch_target(self):
        if not self.show_boxes:
            self.log("Blocked: Object Detection is OFF (Switch Target)")
            self.log("No target is currently selected")
            return
        if self.lock_target:
            self.log("Blocked: Target Lock is active (Switch Target)")
            return
        if not self.available_ids:
            self.log("No available targets")
            return

        self.selected_identity_feat = None
        self.target_lost_flag = False

        if self.selected_id is None:
            self.selected_id = self.available_ids[0]
        else:
            nh = next_higher(self.available_ids, self.selected_id)
            self.selected_id = nh if nh is not None else self.available_ids[0]

        cls_id, cls_name = self._get_class_for_id(self.selected_id)
        self.reselect_signature = (int(self.selected_id), int(cls_id), str(cls_name)) if cls_id is not None else None
        self.log(f"Selected target ID={self.selected_id}")
        self.last_nav_cmd = None
        self._clear_auto_holds()
        self.last_distance_state = None
        self._update_target_controls_state()

    def on_key_d(self):
        if not self.show_boxes:
            self.log("Blocked: Object Detection is OFF (D)")
            self.log("No target is currently selected")
            return
        if self.lock_target:
            self.log("Blocked: Target Lock is active (D)")
            return
        if self.available_ids:
            self.selected_identity_feat = None
            self.target_lost_flag = False
            if self.selected_id is None:
                self.selected_id = self.available_ids[0]
            else:
                nh = next_higher(self.available_ids, self.selected_id)
                self.selected_id = nh if nh is not None else self.available_ids[0]
            cls_id, cls_name = self._get_class_for_id(self.selected_id)
            self.reselect_signature = (int(self.selected_id), int(cls_id), str(cls_name)) if cls_id is not None else None
            self.log(f"Selected target ID={self.selected_id}")
            self.last_nav_cmd = None
            self._clear_auto_holds()
            self.last_distance_state = None
        else:
            self.log("No available targets")
        self._update_target_controls_state()

    def on_key_a(self):
        if not self.show_boxes:
            self.log("Blocked: Object Detection is OFF (A)")
            self.log("No target is currently selected")
            return
        if self.lock_target:
            self.log("Blocked: Target Lock is active (A)")
            return
        if self.available_ids:
            self.selected_identity_feat = None
            self.target_lost_flag = False
            self.selected_id = self.available_ids[0]
            cls_id, cls_name = self._get_class_for_id(self.selected_id)
            self.reselect_signature = (int(self.selected_id), int(cls_id), str(cls_name)) if cls_id is not None else None
            self.log(f"Selected target ID={self.selected_id}")
            self.last_nav_cmd = None
            self._clear_auto_holds()
            self.last_distance_state = None
        else:
            self.log("No available targets")
        self._update_target_controls_state()

    # ----- Flight Mode 6-button commands (only act in Auto) -----
    def _flight_cmd_guard(self, name: str) -> bool:
        if self.gp.state != ForwardState.OFF:
            self.log(f"Ignored ({name}): Flight Mode is not Auto")
            return False
        return True
    def cmd_forward(self):
        if not self._flight_cmd_guard("forward"):
            return
        self.log("Flight command: forward")
    def cmd_back(self):
        if not self._flight_cmd_guard("back"):
            return
        self.log("Flight command: back")
    def cmd_left(self):
        if not self._flight_cmd_guard("left"):
            return
        self.log("Flight command: left")
    def cmd_right(self):
        if not self._flight_cmd_guard("right"):
            return
        self.log("Flight command: right")
    def cmd_up(self):
        if not self._flight_cmd_guard("up"):
            return
        self.log("Flight command: up")
    def cmd_down(self):
        if not self._flight_cmd_guard("down"):
            return
        self.log("Flight command: down")

    # ----- Ramp helpers (manual) -----
    def _forward_press(self, *_):
        if not self._flight_cmd_guard("forward"):
            return
        self._forward_holding = True
        self._start_ramp_loop()
    def _forward_release(self, *_):
        self._forward_holding = False
        self._start_ramp_loop()
    def _back_press(self, *_):
        if not self._flight_cmd_guard("back"):
            return
        self._back_holding = True
        self._start_ramp_loop()
    def _back_release(self, *_):
        self._back_holding = False
        self._start_ramp_loop()
    def _left_press(self, *_):
        if not self._flight_cmd_guard("left"):
            return
        self._left_holding = True
        self._start_ramp_loop()
    def _left_release(self, *_):
        self._left_holding = False
        self._start_ramp_loop()
    def _right_press(self, *_):
        if not self._flight_cmd_guard("right"):
            return
        self._right_holding = True
        self._start_ramp_loop()
    def _right_release(self, *_):
        self._right_holding = False
        self._start_ramp_loop()
    def _yaw_left_press(self, *_):
        if not self._flight_cmd_guard("yaw left"):
            return
        self._yaw_left_holding = True
        self._start_ramp_loop()
    def _yaw_left_release(self, *_):
        self._yaw_left_holding = False
        self._start_ramp_loop()
    def _yaw_right_press(self, *_):
        if not self._flight_cmd_guard("yaw right"):
            return
        self._yaw_right_holding = True
        self._start_ramp_loop()
    def _yaw_right_release(self, *_):
        self._yaw_right_holding = False
        self._start_ramp_loop()

    def _stop_ramp(self):
        if self._ramp_job is not None:
            try:
                self.after_cancel(self._ramp_job)
            except Exception:
                pass
            self._ramp_job = None

    def _start_ramp_loop(self):
        if self._ramp_job is None and self.gp.state == ForwardState.OFF:
            self._ramp_job = self.after(33, self._ramp_tick)

    def _ramp_tick(self):
        self._ramp_job = None
        if self.gp.state != ForwardState.OFF:
            return

        right_eff = self._right_holding
        left_eff = self._left_holding

        yaw_r_eff = self._yaw_right_holding or self._auto_yaw_right
        yaw_l_eff = self._yaw_left_holding or self._auto_yaw_left

        back_eff = self._back_holding or self._auto_back
        fwd_eff = self._forward_holding or self._auto_forward

        if right_eff and not left_eff:
            target_rx = +self.RX_CAP
        elif left_eff and not right_eff:
            target_rx = -self.RX_CAP
        else:
            target_rx = 0.0

        if yaw_r_eff and not yaw_l_eff:
            target_lx = +self.LX_CAP
        elif yaw_l_eff and not yaw_r_eff:
            target_lx = -self.LX_CAP
        else:
            target_lx = 0.0

        if fwd_eff and not back_eff:
            target_ry = +self.RY_CAP
        elif back_eff and not fwd_eff:
            target_ry = -self.RY_CAP
        else:
            target_ry = 0.0

        if self.auto_ry < target_ry:
            self.auto_ry = min(target_ry, self.auto_ry + self.RY_STEP)
        elif self.auto_ry > target_ry:
            self.auto_ry = max(target_ry, self.auto_ry - self.RY_STEP)

        if self.auto_rx < target_rx:
            self.auto_rx = min(target_rx, self.auto_rx + self.RX_STEP)
        elif self.auto_rx > target_rx:
            self.auto_rx = max(target_rx, self.auto_rx - self.RX_STEP)

        if self.auto_lx < target_lx:
            self.auto_lx = min(target_lx, self.auto_lx + self.LX_STEP)
        elif self.auto_lx > target_lx:
            self.auto_lx = max(target_lx, self.auto_lx - self.LX_STEP)

        need_more = (
            (abs(self.auto_ry - target_ry) > 1e-6) or fwd_eff or back_eff or
            (abs(self.auto_rx - target_rx) > 1e-6) or right_eff or left_eff or
            (abs(self.auto_lx - target_lx) > 1e-6) or yaw_r_eff or yaw_l_eff
        )
        if need_more:
            self._ramp_job = self.after(33, self._ramp_tick)

    # ===== Navigation (auto-hold) =====
    def _apply_auto_nav_holds(self):
        now = time.monotonic()
        if self.gp.state != ForwardState.OFF or not self.target_follow:
            if any([self._auto_left, self._auto_right, self._auto_yaw_left, self._auto_yaw_right, self._auto_back, self._auto_forward]):
                self._auto_left = self._auto_right = False
                self._auto_yaw_left = self._auto_yaw_right = False
                self._auto_back = self._auto_forward = False
            return

        desired_auto_left = (now < self._nav_left_until)
        desired_auto_right = (now < self._nav_right_until)
        desired_auto_back = (now < self._nav_back_until)
        desired_auto_forward = (now < self._nav_forward_until)

        if desired_auto_left and desired_auto_right:
            if self._nav_left_until >= self._nav_right_until:
                desired_auto_right = False
            else:
                desired_auto_left = False

        if desired_auto_forward and desired_auto_back:
            if self._nav_forward_until >= self._nav_back_until:
                desired_auto_back = False
            else:
                desired_auto_forward = False

        changed = False
        if desired_auto_left:
            if not self._auto_yaw_left:
                self._auto_yaw_left = True
                changed = True
            if self._auto_yaw_right:
                self._auto_yaw_right = False
                changed = True
        else:
            if self._auto_yaw_left:
                self._auto_yaw_left = False
                changed = True

        if desired_auto_right:
            if not self._auto_yaw_right:
                self._auto_yaw_right = True
                changed = True
            if self._auto_yaw_left:
                self._auto_yaw_left = False
                changed = True
        else:
            if self._auto_yaw_right:
                self._auto_yaw_right = False
                changed = True

        if desired_auto_forward:
            if not self._auto_forward:
                self._auto_forward = True
                changed = True
            if self._auto_back:
                self._auto_back = False
                changed = True
        else:
            if self._auto_forward:
                self._auto_forward = False
                changed = True

        if desired_auto_back:
            if not self._auto_back:
                self._auto_back = True
                changed = True
            if self._auto_forward:
                self._auto_forward = False
                changed = True
        else:
            if self._auto_back:
                self._auto_back = False
                changed = True

        if changed:
            self._start_ramp_loop()

    # ===== Distance Estimation helpers =====
    def _init_reference_rect(self, frame_w, frame_h):
        w, h = int(frame_w * 0.3), int(frame_h * 0.5)
        cx, cy = frame_w // 2, frame_h // 2
        x1, y1 = cx - w // 2, cy - h // 2
        x2, y2 = cx + w // 2, cy + h // 2
        self.ref_rect = (x1, y1, x2, y2)
        self.ref_area = w * h
        self.log(f"Reference rectangle initialized: area={self.ref_area}")

    def _estimate_distance_state(self, current_box):
        if self.ref_rect is None or current_box is None:
            return None, None
        x1, y1, x2, y2 = current_box
        area = max(0, (x2 - x1)) * max(0, (y2 - y1))
        ratio = area / (self.ref_area + 1e-6)
        if ratio < self.dist_thr_near_low:
            state = "Far"
        elif ratio <= self.dist_thr_near_high:
            state = "Near"
        else:
            state = "Too Close"
        return state, ratio

    def _draw_reference_rect(self, frame, color=(200, 200, 200)):
        if not self.draw_ref_rect or self.ref_rect is None:
            return frame
        x1, y1, x2, y2 = self.ref_rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        return frame

    def _loop_worker(self):
        cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            self.log(f"Cannot open camera index {CAM_INDEX}")
            return

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            # ===== FPS counting =====
            self._fps_frame_count += 1
            now = time.monotonic()
            dt = now - self._fps_last_time
            if dt >= 1.0:
                self._fps_value = self._fps_frame_count / dt
                self._fps_frame_count = 0
                self._fps_last_time = now

            # ===== YOLO detection =====
            try:
                results = self.model.predict(
                    frame,
                    conf=CONF_THRES,
                    device=self.device,
                    classes=CLASSES,
                    verbose=False
                )
                res = results[0]
            except Exception as e:
                self.log(f"YOLO inference error: {e}")
                continue

            detections = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                for box, conf, cls_id in zip(xyxy, confs, clss):
                    x1, y1, x2, y2 = box
                    detections.append({
                        "box": (float(x1), float(y1), float(x2), float(y2)),
                        "score": float(conf),
                        "cls": int(cls_id)
                    })

            # ===== Advanced tracker: YOLO + ReID + Kalman + Hungarian =====
            tracks = self.tracker.update(frame, detections)

            # Build maps cho UI
            self.last_tracks = {}
            id_to_box = {}
            id_to_area = {}
            for t in tracks:
                if t.last_box is None:
                    continue
                x1, y1, x2, y2 = t.last_box
                tid = int(t.track_id)
                self.last_tracks[tid] = int(t.cls_id)
                id_to_box[tid] = (float(x1), float(y1), float(x2), float(y2))
                id_to_area[tid] = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

            self._rebuild_available_ids()

            # Reacquire theo Lock (ID-based, trong khi track còn sống)
            if self.lock_target and self.lock_signature is not None and self.selected_id is None and self.show_boxes:
                sig_id, sig_cls_id, sig_cls_name = self.lock_signature
                cond_in_filter = (self.display_allowed is None) or (sig_cls_id in self.display_allowed)
                if cond_in_filter and sig_id in self.last_tracks and self.last_tracks[sig_id] == sig_cls_id:
                    self.selected_id = sig_id
                    self.reselect_signature = (sig_id, sig_cls_id, sig_cls_name)
                    self.selected_identity_feat = None
                    self.target_lost_flag = False
                    self.log(f"Reacquired locked target ID={sig_id} class={sig_cls_name}")

            # Restore theo reselect_signature khi ID vẫn tồn tại
            if (not self.lock_target) and self.reselect_signature is not None and self.selected_id is None and self.show_boxes:
                sig_id, sig_cls_id, sig_cls_name = self.reselect_signature
                cond_in_filter = (self.display_allowed is None) or (sig_cls_id in self.display_allowed)
                if cond_in_filter and sig_id in self.last_tracks and self.last_tracks[sig_id] == sig_cls_id:
                    self.selected_id = sig_id
                    self.selected_identity_feat = None
                    self.target_lost_flag = False
                    self.log(f"Restored selection for target ID={sig_id} class={sig_cls_name}")

            # Handle selected target + predicted_box, update Re-ID signature
            current_target_box = None
            is_predicted_only = False
            if self.show_boxes and self.selected_id is not None:
                sel_id = int(self.selected_id)

                # Cập nhật Re-ID feature khi target đang nhìn thấy
                if sel_id in id_to_box:
                    for t in tracks:
                        if int(t.track_id) == sel_id and t.feature is not None:
                            if self.selected_identity_feat is None:
                                self.selected_identity_feat = t.feature.copy()
                            else:
                                self.selected_identity_feat = 0.9 * self.selected_identity_feat + 0.1 * t.feature
                            break

                if sel_id in id_to_box:
                    current_target_box = tuple(map(int, id_to_box[sel_id]))
                    for t in tracks:
                        if int(t.track_id) == sel_id:
                            is_predicted_only = (t.time_since_update > 0)
                            break
                else:
                    # Track của mục tiêu đã biến mất (ra khỏi FOV quá lâu -> bị xóa)
                    if not self.target_lost_flag:
                        self.log(f"TARGET LOST (ID={sel_id})")
                        self.target_lost_flag = True
                    cls_id, cls_name = self._get_class_for_id(sel_id)
                    if cls_id is not None:
                        self.reselect_signature = (int(sel_id), int(cls_id), str(cls_name))
                    self.selected_id = None
                    self.last_nav_cmd = None
                    self._clear_auto_holds()
                    self.last_distance_state = None
                    if self.target_follow:
                        self.target_follow = False
                        self.follow_btn_var.set("Target Follow: OFF")
                        self.log("Target Follow -> OFF (track removed)")
                    current_target_box = None

            self.predicted_box = current_target_box if (current_target_box is not None and is_predicted_only) else None

            # ===== Long-term Re-ID based reacquisition =====
            if (self.selected_id is None
                and self.show_boxes
                and self.selected_identity_feat is not None
                and len(tracks) > 0):

                best_id = None
                best_cls = None
                best_sim = 0.0
                for t in tracks:
                    if t.feature is None:
                        continue
                    sim = cosine_sim(self.selected_identity_feat, t.feature)
                    if sim > best_sim:
                        best_sim = sim
                        best_id = int(t.track_id)
                        best_cls = int(t.cls_id)
                if best_id is not None and best_sim >= self.reid_reacq_thr:
                    self.selected_id = best_id
                    cls_name = self.names.get(best_cls, str(best_cls)) if isinstance(self.names, dict) else str(best_cls)
                    self.reselect_signature = (best_id, best_cls, cls_name)
                    if self.lock_target:
                        self.lock_signature = (best_id, best_cls, cls_name)
                    self.last_nav_cmd = None
                    self.last_distance_state = None
                    self.target_lost_flag = False
                    self.log(f"Re-ID reacquired target -> ID={best_id}, class={cls_name}, sim={best_sim:.2f}")

            # Manual control I/O
            r_lx, r_ly, r_rx, r_ry = self.gp.read_axes_real()
            if self.gp.state == ForwardState.ARMING:
                if all(abs(a - b) <= ARM_EPS for a, b in zip(
                        [r_lx, r_ly, r_rx, r_ry], [self.gp.v_lx, self.gp.v_ly, self.gp.v_rx, self.gp.v_ry])):
                    self.gp.state = ForwardState.ON
                    self.gp.send_to_virtual(r_lx, r_ly, r_rx, r_ry)
                    self.log("Flight Mode: ARMING -> ON")
            elif self.gp.state == ForwardState.ON:
                self.gp.send_to_virtual(r_lx, r_ly, r_rx, r_ry)

            # ===== Vẽ frame =====
            frame_draw = frame.copy()
            frame_h, frame_w = frame_draw.shape[:2]

            if self.ref_rect is None:
                self._init_reference_rect(frame_w, frame_h)

            tracks_for_draw = []
            for t in tracks:
                if t.last_box is None:
                    continue
                tracks_for_draw.append({
                    "id": int(t.track_id),
                    "cls": int(t.cls_id),
                    "conf": float(t.score),
                    "bbox": t.last_box
                })

            frame_draw = draw_tracks(frame_draw, tracks_for_draw, self.selected_id, self.names,
                                     show=self.show_boxes, allowed_classes=self.display_allowed)

            if self.show_boxes and self.selected_id is not None and self.predicted_box is not None:
                frame_draw = draw_predicted_box(frame_draw, self.predicted_box, label="KF-PRED")

            frame_draw = self._draw_reference_rect(frame_draw, color=(220, 220, 220))
            draw_crosshair_center(frame_draw, size=CROSSHAIR_CENTER_SIZE, color=(255, 0, 0))

            if self.target_follow and self.gp.state == ForwardState.OFF and current_target_box is not None:
                x1, y1, x2, y2 = current_target_box
                cx_obj, cy_obj = int((x1 + x2) // 2), int((y1 + y2) // 2)
                draw_crosshair_at(frame_draw, cx_obj, cy_obj, size=CROSSHAIR_TARGET_SIZE, color=(0, 255, 255))
                dx = cx_obj - (frame_w // 2)

                horiz = None
                if abs(dx) > self.nav_dead_zone_px:
                    horiz = "Right" if dx > 0 else "Left"

                cmd_text = horiz if horiz else "Hold"
                if cmd_text != self.last_nav_cmd:
                    self.log(f"NAV CMD -> {cmd_text}")
                    self.last_nav_cmd = cmd_text

                now2 = time.monotonic()
                if horiz == "Right":
                    self._nav_right_until = now2 + 1.0
                    self._nav_left_until = 0.0
                elif horiz == "Left":
                    self._nav_left_until = now2 + 1.0
                    self._nav_right_until = 0.0

                dist_state, ratio = self._estimate_distance_state(current_target_box)
                if dist_state is not None:
                    if dist_state != self.last_distance_state:
                        self.last_distance_state = dist_state
                        self.log(f"Distance state -> {dist_state} (ratio={ratio:.2f})")
                    if dist_state == "Far":
                        self._nav_forward_until = now2 + 1.0
                        self._nav_back_until = 0.0
                    elif dist_state == "Too Close":
                        self._nav_back_until = now2 + 1.0
                        self._nav_forward_until = 0.0
            else:
                if self.last_nav_cmd is not None and (not self.target_follow or self.gp.state != ForwardState.OFF):
                    self.last_nav_cmd = None

            fixed = resize_with_letterbox(frame_draw, VIDEO_W, VIDEO_H)
            with self.frame_lock:
                self.latest_bgr = fixed
                self.video_ready = True

        cap.release()

    def _render_placeholder(self):
        img = np.zeros((VIDEO_H, VIDEO_W, 3), np.uint8)
        cv2.putText(img, "Running...", (VIDEO_W // 2 - 100, VIDEO_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

    def _update_ui(self):
        self.state_var.set(f"Flight Mode: {self.gp.state}")
        self._update_flight_mode_controls()
        self.pad_name_var.set(self.gp.pad_name)

        self.fps_var.set(f"FPS: {self._fps_value:0.1f}")

        self._apply_auto_nav_holds()

        vals = [self.gp.r_lx, self.gp.r_ly, self.gp.r_rx, self.gp.r_ry]
        for var, v in zip(self.real_vars, vals):
            var.set(f"{v:+.3f}")

        if self.gp.state == ForwardState.OFF:
            ry_float = max(-self.RY_CAP, min(self.RY_CAP, self.auto_ry))
            rx_float = max(-self.RX_CAP, min(self.RX_CAP, self.auto_rx))
            lx_float = max(-self.LX_CAP, min(self.LX_CAP, self.auto_lx))
            self.gp.send_to_virtual(lx_float, self.auto_ly_hold, rx_float, ry_float)

        vals = [self.gp.v_lx, self.gp.v_ly, self.gp.v_rx, self.gp.v_ry]
        for var, v in zip(self.virt_vars, vals):
            var.set(f"{v:+.3f}")

        while not self.log_queue.empty():
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.configure(state="normal")
            self.log_text.insert("end", line)
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        with self.frame_lock:
            frame = self.latest_bgr.copy() if self.video_ready and self.latest_bgr is not None else self._render_placeholder()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_canvas.imgtk = img
        self.video_canvas.create_image(0, 0, anchor="nw", image=img)
        if self.running:
            self.after(33, self._update_ui)

    def on_close(self):
        self.running = False
        try:
            self.log("Application closing")
        except:
            pass
        self.destroy()
        if pygame:
            try:
                pygame.quit()
            except:
                pass

if __name__ == "__main__":
    app = App()
    app.mainloop()
