# yolo_gamepad_forward_tk_fixed.py
# Single-window GUI (Tkinter) with a FIXED-SIZE OpenCV/YOLO video panel.
# - Video panel is fixed to VIDEO_W x VIDEO_H (no auto-zoom).
# - No gamepad text is drawn over the video (clean feed).
# - Window starts in fullscreen (you can turn this off by setting START_FULLSCREEN=False).
#
# Hotkeys:
#   ESC : Quit
#   S   : Select smallest ID / press S again to clear
#   D   : Next higher ID
#   A   : Next lower ID
#
# Requirements:
#   pip install ultralytics opencv-python pillow pygame vgamepad numpy
#   (Windows) Install ViGEmBus for vgamepad

import os
import cv2
import numpy as np
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

from ultralytics import YOLO

# -------- Optional deps for controllers --------
try:
    import pygame
except ImportError:
    print("Bạn cần 'pygame' (pip install pygame).")
    pygame = None

try:
    import vgamepad as vg
except ImportError:
    print("Bạn cần 'vgamepad' (pip install vgamepad) và ViGEmBus trên Windows.")
    vg = None

# =================== Configuration ===================
# YOLO / Tracking
WEIGHTS        = "yolo_weights/yolov8m.pt"
USE_GPU        = True
CONF_THRES     = 0.30
CLASSES        = [0, 39]          # ví dụ: person (0), bottle (39). Để None nếu muốn tất cả
TRACKER_CFG    = "bytetrack.yaml"
PERSIST_ID     = True
SAVE_OUTPUT    = False
CAM_INDEX      = 0
TARGET_FPS     = 30

# Gamepad axes mapping (theo note: A00,A01,A02,A03)
AX_LX = 0  # A00
AX_LY = 1  # A01
AX_RY = 2  # A02
AX_RX = 3  # A03

DEADZONE      = 0.08
ARM_EPS       = 0.05
SWAP_REAL_LR  = True         # fix tay cầm thật trái/phải theo môi trường của bạn
INVERT_Y_DEFAULT = False

# GUI
WIN_TITLE        = "YOLO + Gamepad Control (Fixed Video Size)"
START_FULLSCREEN = False       # True: mở lên fullscreen; False: cửa sổ bình thường
VIDEO_W          = 960       # Kích thước cố định của panel video
VIDEO_H          = 720
RIGHT_PANEL_W    = 360
THEME_PAD        = 10
# =====================================================

def dz(v, d=DEADZONE):
    return 0.0 if abs(v) < d else v

class ForwardState:
    OFF    = "OFF"
    ARMING = "ARMING"
    ON     = "ON"

def draw_tracks(frame, res, selected_id, names):
    # Chỉ vẽ bbox/caption (không vẽ HUD tay cầm)
    if res.boxes is None or len(res.boxes) == 0:
        return frame
    boxes = res.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
    ids  = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else None
    clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        cls_i = int(clss[i]) if i < len(clss) else 0
        if isinstance(names, dict):
            label = names.get(cls_i, str(cls_i))
        else:
            label = names[cls_i] if 0 <= cls_i < len(names) else str(cls_i)
        track_id = int(ids[i]) if ids is not None and i < len(ids) else -1
        if selected_id is not None and track_id == selected_id:
            color, thickness = (0, 0, 255), 3  # đỏ cho ID đang chọn
        else:
            color, thickness = (0, 255, 0), 2  # xanh cho còn lại
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        caption = f"{label} ID:{track_id if track_id!=-1 else 'NA'} {confs[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(0, y1 - 8)
        cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 2), (0, 0, 0), -1)
        cv2.putText(frame, caption, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

def get_available_ids(res):
    if res.boxes is None or res.boxes.id is None:
        return []
    ids = res.boxes.id.detach().cpu().numpy().astype(int).tolist()
    return sorted(list(set(ids)))

def next_higher(sorted_ids, current):
    for x in sorted_ids:
        if x > current:
            return x
    return None

def next_lower(sorted_ids, current):
    prev = None
    for x in sorted_ids:
        if x >= current:
            return prev
        prev = x
    return prev

def resize_with_letterbox(bgr, target_w, target_h):
    """Giữ tỉ lệ, thêm viền đen nếu cần để khớp kích thước cố định."""
    h, w = bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas[y:y+new_h, x:x+new_w] = resized
    return canvas

class GamepadBridge:
    def __init__(self):
        self.pygame_ok = False
        self.vpad_ok   = False
        self.pad_name  = "N/A"
        self.invert_y  = INVERT_Y_DEFAULT
        self.state     = ForwardState.OFF

        # Ảo (v)
        self.v_lx = 0.0
        self.v_ly = 0.0
        self.v_rx = 0.0
        self.v_ry = 0.0

        # Thật (r) — lưu giá trị mới nhất để hiển thị UI
        self.r_lx = 0.0
        self.r_ly = 0.0
        self.r_rx = 0.0
        self.r_ry = 0.0

        # init pygame
        if pygame is not None:
            try:
                pygame.init()
                pygame.joystick.init()
                if pygame.joystick.get_count() > 0:
                    self.js = pygame.joystick.Joystick(0)
                    self.js.init()
                    self.pad_name = self.js.get_name()
                    self.pygame_ok = True
                else:
                    print("[WARN] Không tìm thấy tay cầm thật (pygame).")
            except Exception as e:
                print(f"[WARN] Lỗi khởi tạo pygame joystick: {e}")

        # init vgamepad
        if vg is not None:
            try:
                self.vpad = vg.VX360Gamepad()
                self.vpad_ok = True
            except Exception as e:
                print(f"[WARN] Không khởi tạo được vgamepad: {e}")

    def read_axes_real(self):
        if not self.pygame_ok:
            self.r_lx = self.r_ly = self.r_rx = self.r_ry = 0.0
            return self.r_lx, self.r_ly, self.r_rx, self.r_ry
        try:
            pygame.event.pump()
            if SWAP_REAL_LR:
                lx = dz(self.js.get_axis(AX_RX))
                ly = dz(self.js.get_axis(AX_RY))
                rx = dz(self.js.get_axis(AX_LX))
                ry = dz(self.js.get_axis(AX_LY))
            else:
                lx = dz(self.js.get_axis(AX_LX))
                ly = dz(self.js.get_axis(AX_LY))
                rx = dz(self.js.get_axis(AX_RX))
                ry = dz(self.js.get_axis(AX_RY))
        except Exception:
            lx = ly = rx = ry = 0.0

        if self.invert_y:
            ly = -ly
            ry = -ry

        self.r_lx = max(-1.0, min(1.0, lx))
        self.r_ly = max(-1.0, min(1.0, ly))
        self.r_rx = max(-1.0, min(1.0, rx))
        self.r_ry = max(-1.0, min(1.0, ry))
        return self.r_lx, self.r_ly, self.r_rx, self.r_ry

    def send_to_virtual(self, lx, ly, rx, ry):
        self.v_lx, self.v_ly, self.v_rx, self.v_ry = lx, ly, rx, ry
        if not self.vpad_ok:
            return
        try:
            self.vpad.left_joystick_float(x_value_float=lx, y_value_float=ly)
            self.vpad.right_joystick_float(x_value_float=rx, y_value_float=ry)
            self.vpad.update()
        except Exception as e:
            print(f"[WARN] Lỗi gửi sang vgamepad: {e}")
            self.state = ForwardState.OFF

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(WIN_TITLE)
        if START_FULLSCREEN:
            self.attributes("-fullscreen", True)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # State
        self.selected_id   = None
        self.available_ids = []
        self.running       = True
        self.frame_lock    = threading.Lock()
        self.latest_bgr    = np.zeros((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)  # black init

        # Gamepad
        self.gp = GamepadBridge()

        # YOLO
        device = "cuda" if USE_GPU else "cpu"
        self.model = YOLO(WEIGHTS)
        self.names = self.model.model.names if hasattr(self.model, "model") else self.model.names

        # UI
        self._build_ui()

        # Key bindings
        self.bind("<Escape>", lambda e: self.on_close())
        self.bind("<s>", lambda e: self.on_key_s())
        self.bind("<S>", lambda e: self.on_key_s())
        self.bind("<d>", lambda e: self.on_key_d())
        self.bind("<D>", lambda e: self.on_key_d())
        self.bind("<a>", lambda e: self.on_key_a())
        self.bind("<A>", lambda e: self.on_key_a())

        # Background worker (YOLO + controller)
        self.worker = threading.Thread(target=self._loop_worker, daemon=True)
        self.worker.start()

        # UI refresh timer
        self.after(33, self._update_ui)

    def _build_ui(self):
        root = ttk.Frame(self, padding=THEME_PAD)
        root.pack(fill="both", expand=True)

        # Left: FIXED-SIZE Video panel
        left = ttk.Frame(root, width=VIDEO_W, height=VIDEO_H)
        left.pack(side="left", padx=(0, THEME_PAD), pady=0)
        left.pack_propagate(False)  # giữ kích thước cố định, không co giãn theo con

        # Dùng Canvas để chủ động set kích thước cố định
        self.video_canvas = tk.Canvas(left, width=VIDEO_W, height=VIDEO_H, bg="black", highlightthickness=0)
        self.video_canvas.pack()

        # Right: Controls & Telemetry (không overlay lên video)
        right = ttk.Frame(root, width=RIGHT_PANEL_W)
        right.pack(side="right", fill="y")
        right.pack_propagate(True)

        # Forward state + buttons
        self.state_var = tk.StringVar(value=f"Forward: {self.gp.state}")
        ttk.Label(right, textvariable=self.state_var, font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 6))

        ttk.Button(right, text="Toggle Forward", command=self.toggle_forward).pack(fill="x", pady=4)

        self.invert_var = tk.BooleanVar(value=self.gp.invert_y)
        ttk.Checkbutton(right, text="Invert Y", variable=self.invert_var, command=self.toggle_invert).pack(anchor="w", pady=4)

        # ID controls
        ttk.Label(right, text="Target ID Controls", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10,4))
        ttk.Button(right, text="Select Smallest ID (S)", command=self.on_key_s).pack(fill="x", pady=2)
        ttk.Button(right, text="Prev ID (A)", command=self.on_key_a).pack(fill="x", pady=2)
        ttk.Button(right, text="Next ID (D)", command=self.on_key_d).pack(fill="x", pady=2)

        # Telemetry (chỉ UI, không đè lên video)
        ttk.Label(right, text="Telemetry", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10,4))
        self.pad_name_var = tk.StringVar(value=f"Pad: {self.gp.pad_name}")
        ttk.Label(right, textvariable=self.pad_name_var).pack(anchor="w")

        self.real_axes_var = tk.StringVar(value="Real  LX:+0.000  LY:+0.000  RX:+0.000  RY:+0.000")
        self.v_axes_var    = tk.StringVar(value="Virt  LX:+0.000  LY:+0.000  RX:+0.000  RY:+0.000")
        ttk.Label(right, textvariable=self.real_axes_var).pack(anchor="w")
        ttk.Label(right, textvariable=self.v_axes_var).pack(anchor="w")

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)
        ttk.Label(right, text="ESC: Quit | S: select/clear | A/D: prev/next ID", wraplength=RIGHT_PANEL_W-20, justify="left").pack(anchor="w")

    def toggle_forward(self):
        if self.gp.state == ForwardState.OFF:
            self.gp.state = ForwardState.ARMING
        elif self.gp.state == ForwardState.ARMING:
            self.gp.state = ForwardState.OFF
        elif self.gp.state == ForwardState.ON:
            self.gp.state = ForwardState.OFF
        self.state_var.set(f"Forward: {self.gp.state}")

    def toggle_invert(self):
        self.gp.invert_y = self.invert_var.get()

    def on_key_s(self):
        if self.selected_id is None:
            if self.available_ids:
                self.selected_id = self.available_ids[0]
            else:
                self.selected_id = None
        else:
            self.selected_id = None

    def on_key_d(self):
        if self.available_ids:
            if self.selected_id is None:
                self.selected_id = self.available_ids[0]
            else:
                nh = next_higher(self.available_ids, self.selected_id)
                if nh is not None:
                    self.selected_id = nh

    def on_key_a(self):
        if self.available_ids:
            if self.selected_id is None:
                self.selected_id = self.available_ids[0]
            else:
                nl = next_lower(self.available_ids, self.selected_id)
                if nl is not None:
                    self.selected_id = nl

    def _loop_worker(self):
        device = "cuda" if USE_GPU else "cpu"
        results_gen = self.model.track(
            source=CAM_INDEX,
            conf=CONF_THRES,
            device=device,
            classes=CLASSES,
            tracker=TRACKER_CFG,
            persist=PERSIST_ID,
            stream=True,
            verbose=False
        )

        out_dir = "runs/webcam_tracks"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"track_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        writer = None

        while self.running:
            try:
                res = next(results_gen)
            except StopIteration:
                break
            except Exception:
                continue

            frame = res.orig_img.copy()
            self.available_ids = get_available_ids(res)

            # Read gamepad (real)
            r_lx, r_ly, r_rx, r_ry = self.gp.read_axes_real()

            # Forward state machine
            if self.gp.state == ForwardState.ARMING:
                ok = (abs(r_lx - self.gp.v_lx) <= ARM_EPS and
                      abs(r_ly - self.gp.v_ly) <= ARM_EPS and
                      abs(r_rx - self.gp.v_rx) <= ARM_EPS and
                      abs(r_ry - self.gp.v_ry) <= ARM_EPS)
                if ok:
                    self.gp.state = ForwardState.ON
                    self.gp.send_to_virtual(r_lx, r_ly, r_rx, r_ry)
            elif self.gp.state == ForwardState.ON:
                self.gp.send_to_virtual(r_lx, r_ly, r_rx, r_ry)

            # Draw ONLY tracking info on the camera frame
            frame = draw_tracks(frame, res, self.selected_id, self.names)

            # FIXED-SIZE render with letterbox
            fixed_frame = resize_with_letterbox(frame, VIDEO_W, VIDEO_H)

            # Optional save
            if SAVE_OUTPUT:
                if writer is None:
                    h0, w0 = fixed_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_path, fourcc, TARGET_FPS, (w0, h0))
                    if not writer.isOpened():
                        print("[WARN] Không mở được VideoWriter, sẽ không lưu video.")
                        writer = None
                if writer is not None:
                    writer.write(fixed_frame)

            # Save last frame for UI thread
            with self.frame_lock:
                self.latest_bgr = fixed_frame

        if writer is not None:
            writer.release()
            print(f"[INFO] Đã lưu video: {out_path}")

    def _update_ui(self):
        # Update labels
        self.state_var.set(f"Forward: {self.gp.state}")
        self.pad_name_var.set(f"Pad: {self.gp.pad_name}")

        # Real & Virtual axes
        self.real_axes_var.set(
            f"Real  LX:{self.gp.r_lx:+.3f}  LY:{self.gp.r_ly:+.3f}  RX:{self.gp.r_rx:+.3f}  RY:{self.gp.r_ry:+.3f}"
        )
        self.v_axes_var.set(
            f"Virt  LX:{self.gp.v_lx:+.3f}  LY:{self.gp.v_ly:+.3f}  RX:{self.gp.v_rx:+.3f}  RY:{self.gp.v_ry:+.3f}"
        )

        # Draw latest fixed-size frame to Canvas
        with self.frame_lock:
            frame = None if self.latest_bgr is None else self.latest_bgr.copy()

        if frame is not None:
            # Convert to PhotoImage and draw at (0,0)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            # keep a reference
            self.video_canvas.imgtk = imgtk
            self.video_canvas.create_image(0, 0, anchor="nw", image=imgtk)

        if self.running:
            self.after(33, self._update_ui)

    def on_close(self):
        self.running = False
        self.destroy()
        if pygame is not None:
            try:
                pygame.quit()
            except Exception:
                pass

if __name__ == "__main__":
    app = App()
    app.mainloop()
