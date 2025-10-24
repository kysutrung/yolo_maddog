# yolo_gamepad_forward_tk_spacer.py
# GUI with flexible spacing and compact log panel (English version, compact controls + restore selection on reappear)

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
CLASSES        = [0, 39]
TRACKER_CFG    = "bytetrack.yaml"
PERSIST_ID     = True
SAVE_OUTPUT    = False
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
    OFF, ARMING, ON = "Auto", "ARMING", "Manual"

def draw_tracks(frame, res, selected_id, names, show=True):
    if not show:
        return frame
    if res.boxes is None or len(res.boxes) == 0:
        return frame
    boxes = res.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
    ids  = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else None
    clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        cls_i = int(clss[i]) if i < len(clss) else 0
        label = names.get(cls_i, str(cls_i)) if isinstance(names, dict) else str(cls_i)
        track_id = int(ids[i]) if ids is not None and i < len(ids) else -1
        color, thick = ((0,0,255),3) if selected_id == track_id else ((0,255,0),2)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,thick)
        txt = f"{label} ID:{track_id if track_id!=-1 else 'NA'} {confs[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(0, y1-8)
        cv2.rectangle(frame,(x1,y_text-th-6),(x1+tw+6,y_text+2),(0,0,0),-1)
        cv2.putText(frame, txt, (x1+3,y_text), cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2,cv2.LINE_AA)
    return frame

def get_available_ids(res):
    if res.boxes is None or res.boxes.id is None:
        return []
    return sorted(set(res.boxes.id.detach().cpu().numpy().astype(int).tolist()))

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
        self.state = ForwardState.OFF
        self.v_lx = self.v_ly = self.v_rx = self.v_ry = 0.0
        self.r_lx = self.r_ly = self.r_rx = self.r_ry = 0.0
        if pygame:
            try:
                pygame.init(); pygame.joystick.init()
                if pygame.joystick.get_count()>0:
                    self.js = pygame.joystick.Joystick(0)
                    self.js.init()
                    self.pad_name = self.js.get_name()
                    self.pygame_ok = True
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
        except:
            lx = ly = rx = ry = 0
        self.r_lx, self.r_ly, self.r_rx, self.r_ry = [max(-1,min(1,v)) for v in [lx,ly,rx,ry]]
        return self.r_lx, self.r_ly, self.r_rx, self.r_ry

    def send_to_virtual(self, lx, ly, rx, ry):
        self.v_lx, self.v_ly, self.v_rx, self.v_ry = lx, ly, rx, ry
        if not self.vpad_ok:
            return
        self.vpad.left_joystick_float(lx, ly)
        self.vpad.right_joystick_float(rx, ry)
        self.vpad.update()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(WIN_TITLE)
        self.geometry("1420x820")
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
        self.model = YOLO(WEIGHTS)
        self.names = self.model.model.names if hasattr(self.model, "model") else self.model.names

        # state variables
        self.show_boxes = False
        self.bbox_btn_var = tk.StringVar(value="Object Detection: OFF")
        self.lock_target = False
        self.lock_btn_var = tk.StringVar(value="Target Lock: OFF")

        # Lock signature (for hard lock) and reselect signature (for auto-restore without lock)
        self.lock_signature = None              # tuple(id, cls_id, cls_name)
        self.reselect_signature = None          # tuple(id, cls_id, cls_name) saved on last natural selection
        self.last_tracks = {}                   # id -> cls_id for current frame

        # log queue
        self.log_queue = queue.Queue()

        # styles
        self._init_styles()

        self._build_ui()

        # key bindings
        for key, func in {"<Escape>": self.on_close, "<s>": self.on_key_s, "<S>": self.on_key_s,
                          "<a>": self.on_key_a, "<A>": self.on_key_a, "<d>": self.on_key_d, "<D>": self.on_key_d}.items():
            self.bind(key, lambda e, f=func: f())

        threading.Thread(target=self._loop_worker, daemon=True).start()
        self.after(33, self._update_ui)

        self.log("Application started")
        self.log(f"Model: {WEIGHTS} | GPU: {'ON' if USE_GPU else 'OFF'}")
        self.log(f"Gamepad: {self.gp.pad_name}")
        self._update_target_controls_state()

    # ---------- Styles ----------
    def _init_styles(self):
        style = ttk.Style(self)
        style.configure("Compact.TButton", font=("Segoe UI", 9), padding=(6, 2))
        style.configure("Compact.TLabel", font=("Segoe UI", 9))
        style.configure("Section.TLabel", font=("Segoe UI", 10, "bold"))

    # ---------- Utility ----------
    def log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{ts}] {message}\n")

    def _update_target_controls_state(self):
        enabled = (self.show_boxes and not self.lock_target)
        state = "normal" if enabled else "disabled"
        for b in (self.btn_prev, self.btn_select, self.btn_next):
            b.configure(state=state)

    def _get_class_for_id(self, tid):
        if tid is None: return None, None
        cls_id = self.last_tracks.get(int(tid), None)
        if cls_id is None:
            return None, None
        if isinstance(self.names, dict):
            cls_name = self.names.get(int(cls_id), str(cls_id))
        else:
            cls_name = str(cls_id)
        return int(cls_id), cls_name

    # ---------- UI ----------
    def _build_ui(self):
        root = ttk.Frame(self, padding=THEME_PAD)
        root.pack(fill="both", expand=True)

        ttk.Frame(root).pack(side="left", fill="both", expand=True)

        left = ttk.Frame(root, width=VIDEO_W, height=VIDEO_H)
        left.pack(side="left")
        left.pack_propagate(False)
        self.video_canvas = tk.Canvas(left, width=VIDEO_W, height=VIDEO_H, bg="black", highlightthickness=0)
        self.video_canvas.pack()

        ttk.Frame(root).pack(side="left", fill="both", expand=True)

        right = ttk.Frame(root, width=RIGHT_PANEL_W)
        right.pack(side="left", fill="y")
        right.pack_propagate(False)

        ttk.Frame(root).pack(side="left", fill="both", expand=True)

        # ===== 1) Target Control (compact) =====
        ttk.Label(right, text="Target Control", style="Section.TLabel").pack(anchor="w", pady=(0,4))
        tc = ttk.Frame(right)
        tc.pack(fill="x", pady=(0,4))
        self.btn_prev   = ttk.Button(tc, text="◀ A",   command=self.on_key_a, style="Compact.TButton", width=8)
        self.btn_select = ttk.Button(tc, text="Select", command=self.on_key_s, style="Compact.TButton", width=10)
        self.btn_next   = ttk.Button(tc, text="D ▶",   command=self.on_key_d, style="Compact.TButton", width=8)
        self.btn_prev.grid(row=0, column=0, padx=(0,4), pady=(0,2), sticky="ew")
        self.btn_select.grid(row=0, column=1, padx=2,    pady=(0,2), sticky="ew")
        self.btn_next.grid(row=0, column=2, padx=(4,0),  pady=(0,2), sticky="ew")
        tc.grid_columnconfigure(0, weight=1, uniform="tc")
        tc.grid_columnconfigure(1, weight=1, uniform="tc")
        tc.grid_columnconfigure(2, weight=1, uniform="tc")

        toggles = ttk.Frame(right)
        toggles.pack(fill="x", pady=(0,6))
        self.btn_bbox = ttk.Button(toggles, textvariable=self.bbox_btn_var,
                                   command=self.toggle_bboxes, style="Compact.TButton", width=16)
        self.btn_lock = ttk.Button(toggles, textvariable=self.lock_btn_var,
                                   command=self.toggle_lock, style="Compact.TButton", width=14)
        self.btn_bbox.grid(row=0, column=0, padx=(0,4), sticky="ew")
        self.btn_lock.grid(row=0, column=1, padx=(4,0), sticky="ew")
        toggles.grid_columnconfigure(0, weight=1, uniform="tog")
        toggles.grid_columnconfigure(1, weight=1, uniform="tog")

        # ===== 2) Manual Control =====
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)
        self.state_var = tk.StringVar(value=f"Manual Control: {self.gp.state}")
        ttk.Label(right, textvariable=self.state_var, style="Section.TLabel").pack(anchor="w", pady=(0,4))
        ttk.Button(right, text="Switch Mode", command=self.toggle_forward,
                   style="Compact.TButton").pack(fill="x", pady=(0,6))

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

        # ===== 4) Shortcuts =====
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(
            right,
            text="ESC: Quit | S: select/clear | A/D: prev/next | Toggles: OD / Lock",
            style="Compact.TLabel",
            wraplength=RIGHT_PANEL_W-16, justify="left"
        ).pack(anchor="w")

        # ===== 5) Logs (Compact) =====
        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(right, text="Logs", style="Section.TLabel").pack(anchor="w", pady=(0,4))
        log_holder = ttk.Frame(right, height=110)
        log_holder.pack(fill="x", expand=False)
        log_holder.pack_propagate(False)
        self.log_text = tk.Text(log_holder, height=5, wrap="word", state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_holder, orient="vertical", command=self.log_text.yview)
        log_scroll.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scroll.set)

    # ---------- Logic ----------
    def toggle_forward(self):
        s = self.gp.state
        self.gp.state = ForwardState.ON if s == ForwardState.ARMING \
                        else (ForwardState.ARMING if s == ForwardState.OFF else ForwardState.OFF)
        self.state_var.set(f"Manual Control: {self.gp.state}")
        self.log(f"Manual Control -> {self.gp.state}")

    def toggle_bboxes(self):
        # Prevent turning OFF OD while lock is active
        if self.show_boxes and self.lock_target:
            self.log("Blocked: Target Lock is active, cannot turn OFF Object Detection")
            self.bbox_btn_var.set("Object Detection: ON")
            return

        prev = self.show_boxes
        self.show_boxes = not self.show_boxes
        self.bbox_btn_var.set(f"Object Detection: {'ON' if self.show_boxes else 'OFF'}")
        self.log(f"Object Detection -> {'ON' if self.show_boxes else 'OFF'}")

        # When turning OD OFF: deselect current target and clear reselect_signature (user intent)
        if prev and not self.show_boxes:
            if self.selected_id is not None:
                self.log(f"Object Detection turned OFF -> deselected ID={self.selected_id}")
            self.selected_id = None
            self.reselect_signature = None   # do not auto-restore after user turns OD off
            self.log("No target is currently selected")

        self._update_target_controls_state()

    def toggle_lock(self):
        # Cannot enable lock while OD is OFF
        if not self.show_boxes:
            self.lock_target = False
            self.lock_btn_var.set("Target Lock: OFF")
            self.log("Blocked: Object Detection is OFF -> cannot enable Target Lock")
            self.log("No target is currently selected")
            self._update_target_controls_state()
            return

        # Require a selected target to lock
        if self.selected_id is None and not self.lock_target:
            self.log("Blocked: Select a target before enabling Target Lock")
            self.lock_target = False
            self.lock_btn_var.set("Target Lock: OFF")
            self._update_target_controls_state()
            return

        # Toggle
        if not self.lock_target:
            # Turning ON -> capture lock signature from current selection
            cls_id, cls_name = self._get_class_for_id(self.selected_id)
            if cls_id is None:
                self.log("Blocked: Cannot read target class -> lock aborted")
                return
            self.lock_signature = (int(self.selected_id), int(cls_id), str(cls_name))
            self.lock_target = True
            self.lock_btn_var.set("Target Lock: ON")
            self.log(f"Target Lock -> ON (ID={self.lock_signature[0]}, class={self.lock_signature[2]})")
        else:
            # Turning OFF -> clear signature
            self.lock_target = False
            self.lock_btn_var.set("Target Lock: OFF")
            self.log("Target Lock -> OFF")
            self.lock_signature = None

        self._update_target_controls_state()

    # Selection handlers (blocked when OD off or Lock on)
    def on_key_s(self):
        if not self.show_boxes:
            self.log("Blocked: Object Detection is OFF (S)")
            self.log("No target is currently selected")
            return
        if self.lock_target:
            self.log("Blocked: Target Lock is active (S)")
            return
        if self.selected_id is None:
            self.selected_id = self.available_ids[0] if self.available_ids else None
            if self.selected_id is not None:
                cls_id, cls_name = self._get_class_for_id(self.selected_id)
                self.reselect_signature = (int(self.selected_id), int(cls_id), str(cls_name)) if cls_id is not None else None
                self.log(f"Selected target ID={self.selected_id}")
            else:
                self.log("No available targets to select")
        else:
            # Manual deselect -> clear reselect signature (user intent)
            self.log("Target deselected")
            self.selected_id = None
            self.reselect_signature = None

    def on_key_d(self):
        if not self.show_boxes:
            self.log("Blocked: Object Detection is OFF (D)")
            self.log("No target is currently selected")
            return
        if self.lock_target:
            self.log("Blocked: Target Lock is active (D)")
            return
        if self.available_ids:
            if self.selected_id is None:
                self.selected_id = self.available_ids[0]
            else:
                nh = next_higher(self.available_ids, self.selected_id)
                if nh:
                    self.selected_id = nh
                else:
                    self.log("No higher ID available"); return
            cls_id, cls_name = self._get_class_for_id(self.selected_id)
            self.reselect_signature = (int(self.selected_id), int(cls_id), str(cls_name)) if cls_id is not None else None
            self.log(f"Selected target ID={self.selected_id}")
        else:
            self.log("No available targets")

    def on_key_a(self):
        if not self.show_boxes:
            self.log("Blocked: Object Detection is OFF (A)")
            self.log("No target is currently selected")
            return
        if self.lock_target:
            self.log("Blocked: Target Lock is active (A)")
            return
        if self.available_ids:
            if self.selected_id is None:
                self.selected_id = self.available_ids[0]
            else:
                nl = next_lower(self.available_ids, self.selected_id)
                if nl:
                    self.selected_id = nl
                else:
                    self.log("No lower ID available"); return
            cls_id, cls_name = self._get_class_for_id(self.selected_id)
            self.reselect_signature = (int(self.selected_id), int(cls_id), str(cls_name)) if cls_id is not None else None
            self.log(f"Selected target ID={self.selected_id}")
        else:
            self.log("No available targets")

    def _loop_worker(self):
        gen = self.model.track(source=CAM_INDEX, conf=CONF_THRES,
                               device="cuda" if USE_GPU else "cpu",
                               classes=CLASSES, tracker=TRACKER_CFG,
                               persist=PERSIST_ID, stream=True, verbose=False)
        while self.running:
            try:
                res = next(gen)
            except:
                continue

            # Build last_tracks: id -> cls_id for this frame
            self.last_tracks = {}
            if res.boxes is not None and len(res.boxes) > 0:
                ids = res.boxes.id.detach().cpu().numpy().astype(int) if res.boxes.id is not None else None
                clss = res.boxes.cls.detach().cpu().numpy().astype(int) if res.boxes.cls is not None else None
                if ids is not None and clss is not None:
                    for i in range(min(len(ids), len(clss))):
                        self.last_tracks[int(ids[i])] = int(clss[i])

            frame = res.orig_img.copy()
            self.available_ids = sorted(self.last_tracks.keys())

            # ===== Reacquire logic =====
            # Priority 1: Target Lock signature
            if self.lock_target and self.lock_signature is not None and self.selected_id is None and self.show_boxes:
                sig_id, sig_cls_id, sig_cls_name = self.lock_signature
                if sig_id in self.last_tracks and self.last_tracks[sig_id] == sig_cls_id:
                    self.selected_id = sig_id
                    # also update reselect signature to this locked target
                    self.reselect_signature = (sig_id, sig_cls_id, sig_cls_name)
                    self.log(f"Reacquired locked target ID={sig_id} class={sig_cls_name}")

            # Priority 2: Auto-restore previous selection (no lock required)
            if (not self.lock_target) and self.reselect_signature is not None and self.selected_id is None and self.show_boxes:
                sig_id, sig_cls_id, sig_cls_name = self.reselect_signature
                if sig_id in self.last_tracks and self.last_tracks[sig_id] == sig_cls_id:
                    self.selected_id = sig_id
                    self.log(f"Restored selection for target ID={sig_id} class={sig_cls_name}")

            # If selected target no longer present -> deselect (keep reselect_signature for auto-restore)
            if self.selected_id is not None and self.selected_id not in self.available_ids:
                self.log(f"Lost target ID={self.selected_id} (out of view)")
                # Save signature for restore if we can read class
                cls_id, cls_name = self._get_class_for_id(self.selected_id)
                if cls_id is not None:
                    self.reselect_signature = (int(self.selected_id), int(cls_id), str(cls_name))
                self.selected_id = None  # do not clear reselect_signature here

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

            frame = draw_tracks(frame, res, self.selected_id, self.names, show=self.show_boxes)
            fixed = resize_with_letterbox(frame, VIDEO_W, VIDEO_H)
            with self.frame_lock:
                self.latest_bgr = fixed
                self.video_ready = True

    def _render_placeholder(self):
        img = np.zeros((VIDEO_H, VIDEO_W, 3), np.uint8)
        cv2.putText(img, "Running...", (VIDEO_W//2-100, VIDEO_H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        return img

    def _update_ui(self):
        self.state_var.set(f"Flight Mode: {self.gp.state}")
        self.pad_name_var.set(self.gp.pad_name)
        vals = [self.gp.r_lx, self.gp.r_ly, self.gp.r_rx, self.gp.r_ry]
        for var, v in zip(self.real_vars, vals):
            var.set(f"{v:+.3f}")
        vals = [self.gp.v_lx, self.gp.v_ly, self.gp.v_rx, self.gp.v_ry]
        for var, v in zip(self.virt_vars, vals):
            var.set(f"{v:+.3f}")

        # flush logs
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
