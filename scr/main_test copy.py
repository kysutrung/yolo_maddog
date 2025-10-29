# yolo_gamepad_forward_tk_spacer_enhanced.py
# GUI + Persistent Tracking (YOLO detections + OpenCV tracker fallback)

import os
import cv2
import numpy as np
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import queue
import time
import random
from typing import Optional, List, Tuple, Dict, Any

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

# =================== Configuration ===================
WEIGHTS        = "yolo_weights/yolov8n.pt"
USE_GPU        = True
CONF_THRES     = 0.30
# Chỉ quét 3 lớp: person(0), car(2), motorcycle(3)
CLASSES        = [0, 2, 3]

# --- Persistent tracking params (mới) ---
IMG_SIZE             = 640
IOU_THRESHOLD        = 0.35
HIST_SIM_THRESHOLD   = 0.50
MAX_LOST             = 30       # số frame cho phép mất detection trước khi xóa track
MIN_AREA             = 800      # lọc box quá nhỏ
CV_TRACKER_TYPE      = "CSRT"   # "CSRT" | "KCF" | "MOSSE"

SAVE_OUTPUT    = False
OUT_PATH       = "tracked_output.mp4"
CAM_INDEX      = 0
TARGET_FPS     = 30

AX_LX, AX_LY, AX_RY, AX_RX = 0, 1, 2, 3
DEADZONE, ARM_EPS = 0.08, 0.05
SWAP_REAL_LR  = True

WIN_TITLE        = "YOLO Mad Dog Control Panel"
START_FULLSCREEN = False
VIDEO_W, VIDEO_H = 960, 720
RIGHT_PANEL_W    = 320
THEME_PAD        = 10
# =====================================================

def dz(v, d=DEADZONE): return 0.0 if abs(v) < d else v

class ForwardState:
    OFF, ARMING, ON = "Auto", "Arming", "Manual"

# ================= Persistent Tracker (module inline) =================
def _iou(boxA, boxB) -> float:
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    areaA = max(0, xa2-xa1) * max(0, ya2-ya1)
    areaB = max(0, xb2-xb1) * max(0, yb2-yb1)
    union = areaA + areaB - inter_area
    if union <= 0: return 0.0
    return inter_area / union

def _xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

def _compute_hist(image, bbox):
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
    x2 = min(x2, image.shape[1]); y2 = min(y2, image.shape[0])
    if x2 <= x1 or y2 <= y1: return None
    crop = image[y1:y2, x1:x2]
    if crop.size == 0: return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [50,60], [0,180,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def _cosine_sim(a, b):
    if a is None or b is None: return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0: return 0.0
    return float(np.dot(a, b) / denom)

def _make_cv_tracker(name: str):
    legacy = getattr(cv2, "legacy", None)
    name = (name or "CSRT").upper()
    def pick(legacy_name, modern_name):
        if legacy and hasattr(legacy, legacy_name):
            return getattr(legacy, legacy_name)()
        if hasattr(cv2, modern_name):
            return getattr(cv2, modern_name)()
        raise RuntimeError(f"OpenCV tracker {name} not available.")
    if name == "CSRT":  return pick("TrackerCSRT_create",  "TrackerCSRT_create")
    if name == "KCF":   return pick("TrackerKCF_create",   "TrackerKCF_create")
    if name == "MOSSE": return pick("TrackerMOSSE_create", "TrackerMOSSE_create")
    return pick("TrackerCSRT_create", "TrackerCSRT_create")

class _Track:
    _next_id = 0
    def __init__(self, frame, bbox_xyxy, cls_id, conf,
                 use_cv_tracker: bool, tracker_type: str):
        self.id = _Track._next_id; _Track._next_id += 1
        self.bbox = tuple(int(v) for v in bbox_xyxy)
        self.cls  = int(cls_id) if cls_id is not None else -1
        self.conf = float(conf) if conf is not None else 1.0
        self.hist = _compute_hist(frame, self.bbox)
        self.last_seen = 0
        self.lost = 0
        self.color = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
        self.cv_tracker = None
        if use_cv_tracker:
            try:
                self.cv_tracker = _make_cv_tracker(tracker_type)
                self.cv_tracker.init(frame, _xyxy_to_xywh(self.bbox))
            except Exception:
                self.cv_tracker = None

    def update_with_detection(self, frame, bbox_xyxy, cls_id, conf,
                              use_cv_tracker: bool, tracker_type: str):
        self.bbox = tuple(int(v) for v in bbox_xyxy)
        if cls_id is not None: self.cls = int(cls_id)
        if conf   is not None: self.conf = float(conf)
        self.hist = _compute_hist(frame, self.bbox)
        self.lost = 0
        if use_cv_tracker:
            try:
                self.cv_tracker = _make_cv_tracker(tracker_type)
                self.cv_tracker.init(frame, _xyxy_to_xywh(self.bbox))
            except Exception:
                self.cv_tracker = None

    def predict_with_cvtracker(self, frame) -> bool:
        if self.cv_tracker is None: return False
        ok, box = self.cv_tracker.update(frame)
        if not ok: return False
        x, y, w, h = box
        self.bbox = (int(x), int(y), int(x+w), int(y+h))
        return True

class PersistentTracker:
    """YOLO detection + OpenCV tracker fallback to persist IDs when detector misses."""
    def __init__(self,
                 weights=WEIGHTS,
                 conf=CONF_THRES,
                 classes: Optional[List[int]] = None,
                 img_size=IMG_SIZE,
                 iou_threshold=IOU_THRESHOLD,
                 hist_sim_threshold=HIST_SIM_THRESHOLD,
                 max_lost=MAX_LOST,
                 min_area=MIN_AREA,
                 use_cv_tracker=True,
                 cv_tracker_type=CV_TRACKER_TYPE,
                 device: Optional[str] = None):
        self.model = YOLO(weights)
        self.conf = conf
        self.classes = classes
        self.img_size = img_size
        self.iou_thr = iou_threshold
        self.hist_thr = hist_sim_threshold
        self.max_lost = max_lost
        self.min_area = min_area
        self.use_cv_tracker = use_cv_tracker
        self.cv_tracker_type = cv_tracker_type
        self.device = device
        # names
        self.names = self.model.model.names if hasattr(self.model, "model") else getattr(self.model, "names", {})
        self._tracks: List[_Track] = []
        self._frame_idx = 0

    def _detect(self, frame):
        results = self.model.predict(source=frame, imgsz=self.img_size, conf=self.conf,
                                     classes=self.classes, device=self.device, verbose=False)
        dets = []
        if not results: return dets
        r = results[0]
        if r.boxes is None: return dets
        xyxy = r.boxes.xyxy.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else np.zeros(len(xyxy))
        confs= r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones(len(xyxy))
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            if (x2-x1)*(y2-y1) < self.min_area: continue
            dets.append(((x1,y1,x2,y2), int(clss[i]), float(confs[i])))
        return dets

    def process_frame(self, frame, draw=True):
        self._frame_idx += 1
        detections = self._detect(frame)  # list of ((x1,y1,x2,y2), cls, conf)

        unmatched = set(range(len(detections)))
        matched_tracks = set()

        # IoU matching
        if self._tracks and detections:
            iou_mat = np.zeros((len(self._tracks), len(detections)), dtype=float)
            for ti, tr in enumerate(self._tracks):
                for di, (bb, _, _) in enumerate(detections):
                    iou_mat[ti, di] = _iou(tr.bbox, bb)
            while True:
                ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[ti, di] <= self.iou_thr: break
                bb, cls_id, conf = detections[di]
                self._tracks[ti].update_with_detection(frame, bb, cls_id, conf,
                                                       self.use_cv_tracker, self.cv_tracker_type)
                self._tracks[ti].last_seen = self._frame_idx
                matched_tracks.add(ti)
                if di in unmatched: unmatched.remove(di)
                iou_mat[ti,:] = -1; iou_mat[:,di] = -1

        # Appearance match
        if unmatched and self._tracks:
            for di in list(unmatched):
                bb, cls_id, conf = detections[di]
                d_hist = _compute_hist(frame, bb)
                best_sim, best_ti = 0.0, None
                for ti, tr in enumerate(self._tracks):
                    if ti in matched_tracks: continue
                    sim = _cosine_sim(d_hist, tr.hist)
                    if sim > best_sim:
                        best_sim, best_ti = sim, ti
                if best_ti is not None and best_sim >= self.hist_thr:
                    self._tracks[best_ti].update_with_detection(frame, bb, cls_id, conf,
                                                                self.use_cv_tracker, self.cv_tracker_type)
                    self._tracks[best_ti].last_seen = self._frame_idx
                    matched_tracks.add(best_ti)
                    unmatched.remove(di)

        # New tracks
        for di in list(unmatched):
            bb, cls_id, conf = detections[di]
            t = _Track(frame, bb, cls_id, conf, self.use_cv_tracker, self.cv_tracker_type)
            t.last_seen = self._frame_idx
            self._tracks.append(t)

        # Fallback with cv-tracker if missing this frame
        for tr in self._tracks:
            if tr.last_seen == self._frame_idx: continue
            predicted = tr.predict_with_cvtracker(frame) if self.use_cv_tracker else False
            tr.lost += 1
            if predicted and tr.lost < 3:
                tr.hist = _compute_hist(frame, tr.bbox)

        # prune
        self._tracks = [tr for tr in self._tracks if tr.lost <= self.max_lost]

        # output
        out = frame.copy()
        if draw:
            for tr in self._tracks:
                x1,y1,x2,y2 = tr.bbox
                color = tr.color
                cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
                name = self.names.get(tr.cls, str(tr.cls)) if isinstance(self.names, dict) else str(tr.cls)
                cv2.putText(out, f"{name} ID:{tr.id} L{tr.lost}",
                            (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # tracks list (dict)
        tracks = [{"id": tr.id, "bbox": tr.bbox, "lost": tr.lost, "cls": tr.cls,
                   "conf": tr.conf, "color": tr.color} for tr in self._tracks]
        return tracks, out

    def get_names(self):
        return self.names

# ================= Drawing helper for our tracks =================
def draw_tracks_from_list(frame, tracks, selected_id, names, show=True, allowed_classes=None):
    if not show: return frame
    for t in tracks:
        cls_i = int(t["cls"])
        if (allowed_classes is not None) and (cls_i not in allowed_classes): continue
        x1,y1,x2,y2 = t["bbox"]
        track_id = int(t["id"])
        color, thick = ((0,0,255),3) if selected_id == track_id else (t["color"],2)
        cv2.rectangle(frame, (x1,y1),(x2,y2), color, thick)
        label = names.get(cls_i, str(cls_i)) if isinstance(names, dict) else str(cls_i)
        txt = f"{label} ID:{track_id} L{t['lost']}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(0, y1-8)
        cv2.rectangle(frame,(x1,y_text-th-6),(x1+tw+6,y_text+2),(0,0,0),-1)
        cv2.putText(frame, txt, (x1+3,y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
    return frame

# =================== Rest of your app ===================
def next_higher(lst, cur): return next((x for x in lst if x > cur), None)
def next_lower(lst, cur):
    prev = None
    for x in lst:
        if x >= cur:
            return prev
        prev = x
    return prev

def resize_with_letterbox(bgr, target_w, target_h):
    h, w = bgr.shape[:2]
    scale = min(target_w/w, target_h/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(bgr, (new_w, new_h), cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), np.uint8)
    x, y = (target_w-new_w)//2, (target_h-new_h)//2
    canvas[y:y+new_h, x:x+new_w] = resized
    return canvas

class GamepadBridge:
    def __init__(self):
        self.pad_name = "N/A"; self.pygame_ok = self.vpad_ok = False
        self.state = ForwardState.ON   # DEFAULT = Manual
        self.v_lx = self.v_ly = self.v_rx = self.v_ry = 0.0
        self.r_lx = self.r_ly = self.r_rx = self.r_ry = 0.0
        if pygame:
            try:
                pygame.init(); pygame.joystick.init()
                if pygame.joystick.get_count()>0:
                    self.js = pygame.joystick.Joystick(0); self.js.init()
                    self.pad_name = self.js.get_name(); self.pygame_ok = True
            except Exception as e:
                print(e)
        if vg:
            try:
                self.vpad = vg.VX360Gamepad(); self.vpad_ok = True
            except Exception as e:
                print(e)

    def read_axes_real(self):
        if not self.pygame_ok:
            return (0,0,0,0)
        try:
            pygame.event.pump()
            if SWAP_REAL_LR:
                lx, ly, rx, ry = [dz(self.js.get_axis(a)) for a in [AX_RX, AX_RY, AX_LX, AX_LY]]
            else:
                lx, ly, rx, ry = [dz(self.js.get_axis(a)) for a in [AX_LX, AX_LY, AX_RX, AX_RY]]
        except Exception:
            lx = ly = rx = ry = 0.0
        self.r_lx, self.r_ly, self.r_rx, self.r_ry = [max(-1,min(1,float(v))) for v in [lx,ly,rx,ry]]
        return self.r_lx, self.r_ly, self.r_rx, self.r_ry

    def send_to_virtual(self, lx, ly, rx, ry):
        self.v_lx, self.v_ly, self.v_rx, self.v_ry = lx, ly, rx, ry
        if not self.vpad_ok: return
        self.vpad.left_joystick_float(lx, ly)
        self.vpad.right_joystick_float(rx, ry)
        self.vpad.update()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(WIN_TITLE)
        self.geometry("1420x820")
        if START_FULLSCREEN: self.attributes("-fullscreen", True)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.selected_id = None
        self.available_ids = []
        self.running = True
        self.frame_lock = threading.Lock()
        self.latest_bgr = None
        self.video_ready = False
        self.gp = GamepadBridge()

        # PersistentTracker thay cho model.track()
        self.tracker = PersistentTracker(
            weights=WEIGHTS,
            conf=CONF_THRES,
            classes=CLASSES,
            img_size=IMG_SIZE,
            iou_threshold=IOU_THRESHOLD,
            hist_sim_threshold=HIST_SIM_THRESHOLD,
            max_lost=MAX_LOST,
            min_area=MIN_AREA,
            use_cv_tracker=True,
            cv_tracker_type=CV_TRACKER_TYPE,
            device=("cuda" if USE_GPU else "cpu")
        )
        self.names = self.tracker.get_names()

        # trạng thái hiển thị & chọn mục tiêu
        self.show_boxes = True
        self.bbox_btn_var = tk.StringVar(value="Object Detection: ON")
        self.lock_target = False
        self.lock_btn_var = tk.StringVar(value="Target Lock: OFF")

        # NEW: Target Follow state
        self.target_follow = False
        self.follow_btn_var = tk.StringVar(value="Target Follow: OFF")

        # lọc theo lớp
        self.filter_var = tk.StringVar(value="All")
        self.display_allowed = None  # None => All

        self._flight_btns = []
        self.lock_signature = None
        self.reselect_signature = None
        self.last_tracks = {}   # id -> cls_id
        self.log_queue = queue.Queue()
        self._init_styles()

        # ---- Auto ramp settings ----
        self.RY_CAP = 0.200; self.RY_STEP = 0.006
        self.RX_CAP = 0.200; self.RX_STEP = 0.006
        self.LX_CAP = 0.200; self.LX_STEP = 0.006
        self.auto_ry = 0.0; self.auto_rx = 0.0; self.auto_lx = 0.0
        self._ramp_job = None
        self._forward_holding = self._back_holding = False
        self._left_holding = self._right_holding = False
        self._yaw_left_holding = self._yaw_right_holding = False
        self.auto_ly_hold = 0.0

        self._build_ui()

        for key, func in {"<Escape>": self.on_close, "<s>": self.on_key_s, "<S>": self.on_key_s,
                          "<a>": self.on_key_a, "<A>": self.on_key_a, "<d>": self.on_key_d, "<D>": self.on_key_d}.items():
            self.bind(key, lambda e, f=func: f())

        threading.Thread(target=self._loop_worker, daemon=True).start()
        self.after(33, self._update_ui)

        self.log("Application started")
        self.log(f"Model: {WEIGHTS} | GPU: {'ON' if USE_GPU else 'OFF'} | Tracker: {CV_TRACKER_TYPE}")
        self.log(f"Gamepad: {self.gp.pad_name}")
        self._update_target_controls_state()
        self._update_flight_mode_controls()

    # ---------- Styles ----------
    def _init_styles(self):
        style = ttk.Style(self)
        style.configure("Compact.TButton", font=("Segoe UI", 9), padding=(6, 2))
        style.configure("Compact.TLabel", font=("Segoe UI", 9))
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))
        style.configure("Flight.TButton", font=("Segoe UI", 9), padding=(6, 2))

    # ---------- Utility ----------
    def log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{ts}] {message}\n")

    def _update_target_controls_state(self):
        enabled = (self.show_boxes and not self.lock_target)
        state = "normal" if enabled else "disabled"
        for b in (self.btn_prev, self.btn_select, self.btn_next):
            b.configure(state=state)
        follow_enabled = (self.show_boxes and self.selected_id is not None)
        self.btn_follow.configure(state=("normal" if follow_enabled else "disabled"))

    def _get_class_for_id(self, tid):
        if tid is None: return None, None
        cls_id = self.last_tracks.get(int(tid), None)
        if cls_id is None: return None, None
        if isinstance(self.names, dict):
            cls_name = self.names.get(int(cls_id), str(cls_id))
        else:
            cls_name = str(cls_id)
        return int(cls_id), cls_name

    def _update_flight_mode_controls(self):
        enable = (self.gp.state == ForwardState.OFF)
        state = "normal" if enable else "disabled"
        for b in self._flight_btns: b.configure(state=state)

    # ---------- Filter helpers ----------
    def _filter_label_to_allowed(self, label: str):
        label = (label or "").strip().lower()
        if label in ("all", "tất cả"): return None
        if label in ("person", "người"): return {0}
        if label in ("car", "xe ô tô", "oto", "ô tô"): return {2}
        if label in ("motorcycle", "xe máy", "xe may", "motorbike"): return {3}
        return None

    def _rebuild_available_ids(self):
        if self.display_allowed is None:
            self.available_ids = sorted(self.last_tracks.keys())
        else:
            self.available_ids = sorted([tid for tid, cls in self.last_tracks.items() if cls in self.display_allowed])
        if self.selected_id is not None and self.selected_id not in self.available_ids:
            self.log(f"Current target ID={self.selected_id} is hidden by filter -> deselect")
            self.selected_id = None
        self._update_target_controls_state()

    def on_filter_change(self, *_):
        self.display_allowed = self._filter_label_to_allowed(self.filter_var.get())
        self.log(f"Object filter -> {self.filter_var.get()}")
        self._rebuild_available_ids()

    # ---------- UI ----------
    def _build_ui(self):
        root = ttk.Frame(self, padding=THEME_PAD); root.pack(fill="both", expand=True)
        ttk.Frame(root).pack(side="left", fill="both", expand=True)

        left = ttk.Frame(root, width=VIDEO_W, height=VIDEO_H)
        left.pack(side="left"); left.pack_propagate(False)
        self.video_canvas = tk.Canvas(left, width=VIDEO_W, height=VIDEO_H, bg="black", highlightthickness=0)
        self.video_canvas.pack()

        ttk.Frame(root).pack(side="left", fill="both", expand=True)

        right = ttk.Frame(root, width=RIGHT_PANEL_W)
        right.pack(side="left", fill="y"); right.pack_propagate(False)
        ttk.Frame(root).pack(side="left", fill="both", expand=True)

        # ===== 1) Target Control =====
        ttk.Label(right, text="Target Tracking", style="Section.TLabel").pack(anchor="w", pady=(0,4))
        tc = ttk.Frame(right); tc.pack(fill="x", pady=(0,4))
        self.btn_prev   = ttk.Button(tc, text="◀ A",   command=self.on_key_a, style="Compact.TButton", width=8)
        self.btn_select = ttk.Button(tc, text="Select", command=self.on_key_s, style="Compact.TButton", width=10)
        self.btn_next   = ttk.Button(tc, text="D ▶",   command=self.on_key_d, style="Compact.TButton", width=8)
        self.btn_prev.grid(row=0, column=0, padx=(0,4), pady=(0,2), sticky="ew")
        self.btn_select.grid(row=0, column=1, padx=2,    pady=(0,2), sticky="ew")
        self.btn_next.grid(row=0, column=2, padx=(4,0),  pady=(0,2), sticky="ew")
        tc.grid_columnconfigure(0, weight=1, uniform="tc")
        tc.grid_columnconfigure(1, weight=1, uniform="tc")
        tc.grid_columnconfigure(2, weight=1, uniform="tc")

        toggles = ttk.Frame(right); toggles.pack(fill="x", pady=(0,6))
        self.btn_bbox = ttk.Button(toggles, textvariable=self.bbox_btn_var,
                                   command=self.toggle_bboxes, style="Compact.TButton", width=16)
        self.btn_lock = ttk.Button(toggles, textvariable=self.lock_btn_var,
                                   command=self.toggle_lock, style="Compact.TButton", width=14)
        self.btn_bbox.grid(row=0, column=0, padx=(0,4), sticky="ew")
        self.btn_lock.grid(row=0, column=1, padx=(4,0), sticky="ew")
        toggles.grid_columnconfigure(0, weight=1, uniform="tog")
        toggles.grid_columnconfigure(1, weight=1, uniform="tog")

        # Combobox lọc
        filt_row = ttk.Frame(right); filt_row.pack(fill="x", pady=(0,6))
        ttk.Label(filt_row, text="Show:", style="Compact.TLabel", width=8).grid(row=0, column=0, sticky="w")
        self.filter_combo = ttk.Combobox(
            filt_row, textvariable=self.filter_var,
            values=["All", "Person", "Car", "Motorcycle"], state="readonly", width=18
        )
        self.filter_combo.grid(row=0, column=1, sticky="ew")
        filt_row.grid_columnconfigure(1, weight=1)
        self.filter_combo.bind("<<ComboboxSelected>>", self.on_filter_change)

        # ===== 2) Flight Mode =====
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)
        self.state_var = tk.StringVar(value=f"Flight Mode: {self.gp.state}")
        ttk.Label(right, textvariable=self.state_var, style="Section.TLabel").pack(anchor="w", pady=(0,4))
        ttk.Button(right, text="Switch Mode", command=self.toggle_forward,
                   style="Compact.TButton").pack(fill="x", pady=(0,6))

        fm = ttk.Frame(right); fm.pack(fill="x", pady=(0,6))
        mk = lambda t, cmd: ttk.Button(fm, text=t, command=cmd, style="Flight.TButton", width=10)
        self.btn_forward = mk("Forward", self.cmd_forward)
        self.btn_back    = mk("Backward", self.cmd_back)
        self.btn_up      = mk("Up",      self.cmd_up)
        self.btn_left    = mk("Left",    self.cmd_left)
        self.btn_right   = mk("Right",   self.cmd_right)
        self.btn_down    = mk("Down",    self.cmd_down)
        self.btn_forward.grid(row=0, column=0, padx=(0,4), pady=(0,4), sticky="ew")
        self.btn_back.grid(   row=0, column=1, padx=4,     pady=(0,4), sticky="ew")
        self.btn_up.grid(     row=0, column=2, padx=(4,0), pady=(0,4), sticky="ew")
        self.btn_left.grid(   row=1, column=0, padx=(0,4), sticky="ew")
        self.btn_right.grid(  row=1, column=1, padx=4,     sticky="ew")
        self.btn_down.grid(   row=1, column=2, padx=(4,0), sticky="ew")

        self.btn_yaw_left  = mk("Yaw Left",  lambda: None)
        self.btn_yaw_right = mk("Yaw Right", lambda: None)
        self.btn_yaw_left.grid( row=2, column=0, padx=(0,4), pady=(0,4), sticky="ew")
        self.btn_yaw_right.grid(row=2, column=1, padx=4,     pady=(0,4), sticky="ew")

        for c in range(3): fm.grid_columnconfigure(c, weight=1, uniform="fm")
        self._flight_btns = [
            self.btn_forward, self.btn_back, self.btn_left, self.btn_right, self.btn_up, self.btn_down,
            self.btn_yaw_left, self.btn_yaw_right
        ]

        ttk.Frame(right, height=8).pack(fill="x")
        self.btn_follow = ttk.Button(right, textvariable=self.follow_btn_var,
                                     command=self.toggle_follow, style="Compact.TButton")
        self.btn_follow.pack(fill="x", pady=(0,6))

        # Bind press/release để ramp khi ở Auto
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

        # ===== 3) Control Parameters =====
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(right, text="Control Parameters", style="Section.TLabel").pack(anchor="w", pady=(0,4))
        telem = ttk.Frame(right); telem.pack(anchor="w", fill="x")
        mono_font = ("Consolas", 10) if os.name=="nt" else ("Menlo", 10)

        ttk.Label(telem, text="Gamepad:", style="Compact.TLabel", width=10).grid(row=0, column=0, sticky="w", padx=(0,6), pady=2)
        self.pad_name_var = tk.StringVar(value=self.gp.pad_name)
        ttk.Label(telem, textvariable=self.pad_name_var, style="Compact.TLabel").grid(row=0, column=1, sticky="w", pady=2)

        ttk.Label(telem, text="Physical", style="Compact.TLabel", width=10).grid(row=1, column=0, sticky="w", padx=(0,6), pady=(6,2))
        ttk.Separator(telem, orient="horizontal").grid(row=1, column=1, sticky="ew", pady=(6,2))
        for i, name in enumerate(["LX","LY","RX","RY"], start=2):
            ttk.Label(telem, text=f"{name}:", style="Compact.TLabel", width=6).grid(row=i, column=0, sticky="w")
        self.real_vars = [tk.StringVar(value="+0.000") for _ in range(4)]
        for i, v in enumerate(self.real_vars, start=2):
            tk.Label(telem, textvariable=v, font=mono_font, width=7, anchor="e").grid(row=i, column=1, sticky="w")

        ttk.Label(telem, text="Virtual", style="Compact.TLabel", width=10).grid(row=6, column=0, sticky="w", padx=(0,6), pady=(6,2))
        ttk.Separator(telem, orient="horizontal").grid(row=6, column=1, sticky="ew", pady=(6,2))
        for i, name in enumerate(["LX","LY","RX","RY"], start=7):
            ttk.Label(telem, text=f"{name}:", style="Compact.TLabel", width=6).grid(row=i, column=0, sticky="w")
        self.virt_vars = [tk.StringVar(value="+0.000") for _ in range(4)]
        for i, v in enumerate(self.virt_vars, start=7):
            tk.Label(telem, textvariable=v, font=mono_font, width=7, anchor="e").grid(row=i, column=1, sticky="w")

        telem.grid_columnconfigure(0, weight=0)
        telem.grid_columnconfigure(1, weight=1)

        # ===== 4) Logs =====
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(right, text="Logs", style="Section.TLabel").pack(anchor="w", pady=(0,4))
        log_holder = ttk.Frame(right, height=110); log_holder.pack(fill="x", expand=False); log_holder.pack_propagate(False)
        self.log_text = tk.Text(log_holder, height=5, wrap="word", state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_holder, orient="vertical", command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set)

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(
            right,
            text="ESC: Quit | S: select/clear | A/D: prev/next | Toggles: OD / Lock / Follow",
            style="Compact.TLabel",
            wraplength=RIGHT_PANEL_W-16, justify="left"
        ).pack(anchor="w")

    # ---------- Logic ----------
    def toggle_forward(self):
        prev = self.gp.state
        self.gp.state = ForwardState.ON if prev == ForwardState.ARMING \
                        else (ForwardState.ARMING if prev == ForwardState.OFF else ForwardState.OFF)
        self.state_var.set(f"Flight Mode: {self.gp.state}")
        self._update_flight_mode_controls()
        self.log(f"Flight Mode -> {self.gp.state}")
        if prev == ForwardState.OFF and self.gp.state in (ForwardState.ARMING, ForwardState.ON):
            self._stop_ramp()
        if self.gp.state == ForwardState.OFF:
            self.auto_ly_hold = float(self.gp.v_ly)
            self.log(f"Auto mode: hold LY = {self.auto_ly_hold:+.3f}")

    def toggle_bboxes(self):
        if self.show_boxes and self.lock_target:
            self.log("Blocked: Target Lock is active, cannot turn OFF Object Detection")
            self.bbox_btn_var.set("Object Detection: ON"); return
        prev = self.show_boxes
        self.show_boxes = not self.show_boxes
        self.bbox_btn_var.set(f"Object Detection: {'ON' if self.show_boxes else 'OFF'}")
        self.log(f"Object Detection -> {'ON' if self.show_boxes else 'OFF'}")
        if prev and not self.show_boxes:
            if self.selected_id is not None:
                self.log(f"Object Detection turned OFF -> deselected ID={self.selected_id}")
            self.selected_id = None
            self.reselect_signature = None
            if self.target_follow:
                self.target_follow = False
                self.follow_btn_var.set("Target Follow: OFF")
                self.log("Target Follow -> OFF (Object Detection OFF)")
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
            self.lock_target = False; self.lock_btn_var.set("Target Lock: OFF")
            self._update_target_controls_state(); return
        if not self.lock_target:
            cls_id, cls_name = self._get_class_for_id(self.selected_id)
            if cls_id is None:
                self.log("Blocked: Cannot read target class -> lock aborted"); return
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
        if not self.show_boxes:
            self.target_follow = False; self.follow_btn_var.set("Target Follow: OFF")
            self.log("Blocked: Object Detection is OFF -> cannot enable Target Follow")
            self._update_target_controls_state(); return
        if self.selected_id is None:
            self.target_follow = False; self.follow_btn_var.set("Target Follow: OFF")
            self.log("Blocked: No target selected -> cannot enable Target Follow")
            self._update_target_controls_state(); return
        self.target_follow = not self.target_follow
        self.follow_btn_var.set(f"Target Follow: {'ON' if self.target_follow else 'OFF'}")
        if self.target_follow:
            cls_id, cls_name = self._get_class_for_id(self.selected_id)
            if cls_name is not None: self.log(f"Target Follow -> ON (ID={self.selected_id}, class={cls_name})")
            else: self.log(f"Target Follow -> ON (ID={self.selected_id})")
        else:
            self.log("Target Follow -> OFF")

    # --- Selection handlers ---
    def on_key_s(self):
        if not self.show_boxes:
            self.log("Blocked: Object Detection is OFF (S)"); self.log("No target is currently selected"); return
        if self.lock_target:
            self.log("Blocked: Target Lock is active (S)"); return
        if self.available_ids:
            if self.selected_id is None:
                self.selected_id = self.available_ids[0]
                cls_id, cls_name = self._get_class_for_id(self.selected_id)
                self.reselect_signature = (int(self.selected_id), int(cls_id), str(cls_name)) if cls_id is not None else None
                self.log(f"Selected target ID={self.selected_id}")
            else:
                self.log("Target deselected")
                self.selected_id = None; self.reselect_signature = None
                if self.target_follow:
                    self.target_follow = False
                    self.follow_btn_var.set("Target Follow: OFF")
                    self.log("Target Follow -> OFF (no target selected)")
        else:
            self.log("No available targets to select")
        self._update_target_controls_state()

    def on_key_d(self):
        if not self.show_boxes:
            self.log("Blocked: Object Detection is OFF (D)"); self.log("No target is currently selected"); return
        if self.lock_target:
            self.log("Blocked: Target Lock is active (D)"); return
        if self.available_ids:
            if self.selected_id is None:
                self.selected_id = self.available_ids[0]
            else:
                nh = next_higher(self.available_ids, self.selected_id)
                if nh: self.selected_id = nh
                else: self.log("No higher ID available"); return
            cls_id, cls_name = self._get_class_for_id(self.selected_id)
            self.reselect_signature = (int(self.selected_id), int(cls_id), str(cls_name)) if cls_id is not None else None
            self.log(f"Selected target ID={self.selected_id}")
        else:
            self.log("No available targets")
        self._update_target_controls_state()

    def on_key_a(self):
        if not self.show_boxes:
            self.log("Blocked: Object Detection is OFF (A)"); self.log("No target is currently selected"); return
        if self.lock_target:
            self.log("Blocked: Target Lock is active (A)"); return
        if self.available_ids:
            if self.selected_id is None:
                self.selected_id = self.available_ids[0]
            else:
                nl = next_lower(self.available_ids, self.selected_id)
                if nl: self.selected_id = nl
                else: self.log("No lower ID available"); return
            cls_id, cls_name = self._get_class_for_id(self.selected_id)
            self.reselect_signature = (int(self.selected_id), int(cls_id), str(cls_name)) if cls_id is not None else None
            self.log(f"Selected target ID={self.selected_id}")
        else:
            self.log("No available targets")
        self._update_target_controls_state()

    # ----- Flight Mode commands -----
    def _flight_cmd_guard(self, name: str) -> bool:
        if self.gp.state != ForwardState.OFF:
            self.log(f"Ignored ({name}): Flight Mode is not Auto")
            return False
        return True

    def cmd_forward(self):  # placeholder
        if not self._flight_cmd_guard("forward"): return
        self.log("Flight command: forward")

    def cmd_back(self):
        if not self._flight_cmd_guard("back"): return
        self.log("Flight command: back")

    def cmd_left(self):
        if not self._flight_cmd_guard("left"): return
        self.log("Flight command: left")

    def cmd_right(self):
        if not self._flight_cmd_guard("right"): return
        self.log("Flight command: right")

    def cmd_up(self):
        if not self._flight_cmd_guard("up"): return
        self.log("Flight command: up")

    def cmd_down(self):
        if not self._flight_cmd_guard("down"): return
        self.log("Flight command: down")

    # ----- Ramp helpers -----
    def _forward_press(self, *_):
        if not self._flight_cmd_guard("forward"): return
        self._forward_holding = True; self._start_ramp_loop()
    def _forward_release(self, *_):
        self._forward_holding = False; self._start_ramp_loop()
    def _back_press(self, *_):
        if not self._flight_cmd_guard("back"): return
        self._back_holding = True; self._start_ramp_loop()
    def _back_release(self, *_):
        self._back_holding = False; self._start_ramp_loop()
    def _left_press(self, *_):
        if not self._flight_cmd_guard("left"): return
        self._left_holding = True; self._start_ramp_loop()
    def _left_release(self, *_):
        self._left_holding = False; self._start_ramp_loop()
    def _right_press(self, *_):
        if not self._flight_cmd_guard("right"): return
        self._right_holding = True; self._start_ramp_loop()
    def _right_release(self, *_):
        self._right_holding = False; self._start_ramp_loop()
    def _yaw_left_press(self, *_):
        if not self._flight_cmd_guard("yaw left"): return
        self._yaw_left_holding = True; self._start_ramp_loop()
    def _yaw_left_release(self, *_):
        self._yaw_left_holding = False; self._start_ramp_loop()
    def _yaw_right_press(self, *_):
        if not self._flight_cmd_guard("yaw right"): return
        self._yaw_right_holding = True; self._start_ramp_loop()
    def _yaw_right_release(self, *_):
        self._yaw_right_holding = False; self._start_ramp_loop()

    def _stop_ramp(self):
        if self._ramp_job is not None:
            try: self.after_cancel(self._ramp_job)
            except Exception: pass
            self._ramp_job = None

    def _start_ramp_loop(self):
        if self._ramp_job is None and self.gp.state == ForwardState.OFF:
            self._ramp_job = self.after(33, self._ramp_tick)

    def _ramp_tick(self):
        self._ramp_job = None
        if self.gp.state != ForwardState.OFF: return
        target_ry = +self.RY_CAP if (self._forward_holding and not self._back_holding) else \
                    (-self.RY_CAP if (self._back_holding and not self._forward_holding) else 0.0)
        target_rx = +self.RX_CAP if (self._right_holding and not self._left_holding) else \
                    (-self.RX_CAP if (self._left_holding and not self._right_holding) else 0.0)
        target_lx = +self.LX_CAP if (self._yaw_right_holding and not self._yaw_left_holding) else \
                    (-self.LX_CAP if (self._yaw_left_holding and not self._yaw_right_holding) else 0.0)
        # ramp
        self.auto_ry = min(target_ry, self.auto_ry + self.RY_STEP) if self.auto_ry < target_ry else \
                       max(target_ry, self.auto_ry - self.RY_STEP)
        self.auto_rx = min(target_rx, self.auto_rx + self.RX_STEP) if self.auto_rx < target_rx else \
                       max(target_rx, self.auto_rx - self.RX_STEP)
        self.auto_lx = min(target_lx, self.auto_lx + self.LX_STEP) if self.auto_lx < target_lx else \
                       max(target_lx, self.auto_lx - self.LX_STEP)
        need_more = (
            (abs(self.auto_ry - target_ry) > 1e-6) or self._forward_holding or self._back_holding or
            (abs(self.auto_rx - target_rx) > 1e-6) or self._left_holding   or self._right_holding or
            (abs(self.auto_lx - target_lx) > 1e-6) or self._yaw_left_holding or self._yaw_right_holding
        )
        if need_more:
            self._ramp_job = self.after(33, self._ramp_tick)

    def _loop_worker(self):
        cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            self.log("Cannot open camera"); return

        writer = None
        if SAVE_OUTPUT:
            fps0 = cap.get(cv2.CAP_PROP_FPS)
            if not fps0 or np.isnan(fps0) or fps0 < 1: fps0 = TARGET_FPS
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(OUT_PATH, fourcc, fps0, (w, h))

        t_prev = time.perf_counter()
        frame_interval = 1.0 / max(1, TARGET_FPS)

        while self.running:
            ok, frame = cap.read()
            if not ok: continue

            # --- Tracking (nâng cấp): YOLO + OpenCV fallback ---
            tracks, _ = self.tracker.process_frame(frame, draw=False)

            # Cập nhật map id->cls cho UI
            self.last_tracks = {int(t["id"]): int(t["cls"]) for t in tracks}
            self._rebuild_available_ids()

            # Reacquire theo Lock
            if self.lock_target and self.lock_signature is not None and self.selected_id is None and self.show_boxes:
                sig_id, sig_cls_id, sig_cls_name = self.lock_signature
                cond_in_filter = (self.display_allowed is None) or (sig_cls_id in self.display_allowed)
                if cond_in_filter and (sig_id in self.last_tracks) and (self.last_tracks[sig_id] == sig_cls_id):
                    self.selected_id = sig_id
                    self.reselect_signature = (sig_id, sig_cls_id, sig_cls_name)
                    self.log(f"Reacquired locked target ID={sig_id} class={sig_cls_name}")

            # Restore theo reselect_signature
            if (not self.lock_target) and self.reselect_signature is not None and self.selected_id is None and self.show_boxes:
                sig_id, sig_cls_id, sig_cls_name = self.reselect_signature
                cond_in_filter = (self.display_allowed is None) or (sig_cls_id in self.display_allowed)
                if cond_in_filter and (sig_id in self.last_tracks) and (self.last_tracks[sig_id] == sig_cls_id):
                    self.selected_id = sig_id
                    self.log(f"Restored selection for target ID={sig_id} class={sig_cls_name}")

            if self.selected_id is not None and self.selected_id not in self.last_tracks:
                self.log(f"Lost/hidden target ID={self.selected_id}")
                cls_id, cls_name = self._get_class_for_id(self.selected_id)
                if cls_id is not None:
                    self.reselect_signature = (int(self.selected_id), int(cls_id), str(cls_name))
                self.selected_id = None
                if self.target_follow:
                    self.target_follow = False
                    self.follow_btn_var.set("Target Follow: OFF")
                    self.log("Target Follow -> OFF (target lost)")

            # Manual control I/O
            r_lx, r_ly, r_rx, r_ry = self.gp.read_axes_real()
            if self.gp.state == ForwardState.ARMING:
                if all(abs(a-b) <= ARM_EPS for a,b in zip(
                        [r_lx,r_ly,r_rx,r_ry], [self.gp.v_lx,self.gp.v_ly,self.gp.v_rx,self.gp.v_ry])):
                    self.gp.state = ForwardState.ON
                    self.gp.send_to_virtual(r_lx,r_ly,r_rx,r_ry)
                    self.log("Flight Mode: ARMING -> ON")
            elif self.gp.state == ForwardState.ON:
                self.gp.send_to_virtual(r_lx,r_ly,r_rx,r_ry)

            # Vẽ overlay theo danh sách tracks (thay cho draw_tracks cũ)
            frame_drawn = frame.copy()
            frame_drawn = draw_tracks_from_list(
                frame_drawn, tracks, self.selected_id, self.names,
                show=self.show_boxes, allowed_classes=self.display_allowed
            )

            fixed = resize_with_letterbox(frame_drawn, VIDEO_W, VIDEO_H)
            with self.frame_lock:
                self.latest_bgr = fixed
                self.video_ready = True

            if writer is not None:
                writer.write(frame_drawn)

            # Giới hạn FPS của luồng xử lý (đỡ nóng máy)
            t_now = time.perf_counter()
            elapsed = t_now - t_prev
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            t_prev = time.perf_counter()

        if writer is not None:
            writer.release()
        cap.release()

    def _render_placeholder(self):
        img = np.zeros((VIDEO_H, VIDEO_W, 3), np.uint8)
        cv2.putText(img, "Running...", (VIDEO_W//2-100, VIDEO_H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return img

    def _update_ui(self):
        self.state_var.set(f"Flight Mode: {self.gp.state}")
        self._update_flight_mode_controls()
        self.pad_name_var.set(self.gp.pad_name)

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
        try: self.log("Application closing")
        except: pass
        self.destroy()
        if pygame:
            try: pygame.quit()
            except: pass

if __name__ == "__main__":
    app = App()
    app.mainloop()
