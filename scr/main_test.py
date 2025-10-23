# yolo_gamepad_forward_tk_spacer.py
# Giao diện: có khoảng trống linh hoạt giữa video và khu vực nút bấm

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
RIGHT_PANEL_W    = 360
THEME_PAD        = 10
# =====================================================

def dz(v, d=DEADZONE): return 0.0 if abs(v) < d else v

class ForwardState:
    OFF, ARMING, ON = "OFF", "ARMING", "ON"

def draw_tracks(frame, res, selected_id, names):
    if res.boxes is None or len(res.boxes) == 0: return frame
    boxes = res.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
    ids  = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else None
    clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        cls_i = int(clss[i]) if i < len(clss) else 0
        label = names[cls_i] if isinstance(names, (list, dict)) and cls_i in names else str(cls_i)
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
    if res.boxes is None or res.boxes.id is None: return []
    return sorted(set(res.boxes.id.detach().cpu().numpy().astype(int).tolist()))

def next_higher(lst, cur): return next((x for x in lst if x > cur), None)
def next_lower(lst, cur):
    prev = None
    for x in lst:
        if x >= cur: return prev
        prev = x
    return prev

def resize_with_letterbox(bgr, target_w, target_h):
    h, w = bgr.shape[:2]
    scale = min(target_w/w, target_h/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized = cv2.resize(bgr,(new_w,new_h),cv2.INTER_AREA)
    canvas = np.zeros((target_h,target_w,3),np.uint8)
    x, y = (target_w-new_w)//2, (target_h-new_h)//2
    canvas[y:y+new_h,x:x+new_w] = resized
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
                    self.js=pygame.joystick.Joystick(0); self.js.init()
                    self.pad_name=self.js.get_name(); self.pygame_ok=True
            except Exception as e: print(e)
        if vg:
            try: self.vpad=vg.VX360Gamepad(); self.vpad_ok=True
            except Exception as e: print(e)
    def read_axes_real(self):
        if not self.pygame_ok: return (0,0,0,0)
        try:
            pygame.event.pump()
            if SWAP_REAL_LR:
                lx,ly,rx,ry=[dz(self.js.get_axis(a)) for a in [AX_RX,AX_RY,AX_LX,AX_LY]]
            else:
                lx,ly,rx,ry=[dz(self.js.get_axis(a)) for a in [AX_LX,AX_LY,AX_RX,AX_RY]]
        except: lx=ly=rx=ry=0
        self.r_lx,self.r_ly,self.r_rx,self.r_ry=[max(-1,min(1,v)) for v in [lx,ly,rx,ry]]
        return self.r_lx,self.r_ly,self.r_rx,self.r_ry
    def send_to_virtual(self,lx,ly,rx,ry):
        self.v_lx,self.v_ly,self.v_rx,self.v_ry=lx,ly,rx,ry
        if not self.vpad_ok: return
        self.vpad.left_joystick_float(lx,ly)
        self.vpad.right_joystick_float(rx,ry)
        self.vpad.update()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(WIN_TITLE)
        self.geometry("1450x780")
        if START_FULLSCREEN: self.attributes("-fullscreen",True)
        self.protocol("WM_DELETE_WINDOW",self.on_close)
        self.selected_id=None; self.available_ids=[]
        self.running=True; self.frame_lock=threading.Lock()
        self.latest_bgr=None; self.video_ready=False
        self.gp=GamepadBridge()
        self.model=YOLO(WEIGHTS)
        self.names=self.model.model.names if hasattr(self.model,"model") else self.model.names
        self._build_ui()
        for key,func in {"<Escape>":self.on_close,"<s>":self.on_key_s,"<S>":self.on_key_s,
                         "<a>":self.on_key_a,"<A>":self.on_key_a,"<d>":self.on_key_d,"<D>":self.on_key_d}.items():
            self.bind(key,lambda e,f=func:f())
        threading.Thread(target=self._loop_worker,daemon=True).start()
        self.after(33,self._update_ui)

    def _build_ui(self):
        root=ttk.Frame(self,padding=THEME_PAD)
        root.pack(fill="both",expand=True)

        # Spacer trái (giữ căn giữa)
        ttk.Frame(root).pack(side="left", fill="both", expand=True)

        # Video
        left=ttk.Frame(root,width=VIDEO_W,height=VIDEO_H)
        left.pack(side="left",padx=0,pady=0)
        left.pack_propagate(False)
        self.video_canvas=tk.Canvas(left,width=VIDEO_W,height=VIDEO_H,bg="black",highlightthickness=0)
        self.video_canvas.pack()

        # --- Spacer LINH HOẠT giữa video và panel nút ---
        ttk.Frame(root).pack(side="left", fill="both", expand=True)  # <== khoảng trống giãn linh hoạt

        # Panel nút
        right=ttk.Frame(root,width=RIGHT_PANEL_W)
        right.pack(side="left",fill="y")
        right.pack_propagate(False)

        # Spacer phải
        ttk.Frame(root).pack(side="left", fill="both", expand=True)

        # Nội dung panel nút
        self.state_var=tk.StringVar(value=f"Forward: {self.gp.state}")
        ttk.Label(right,textvariable=self.state_var,font=("Segoe UI",13,"bold")).pack(anchor="w",pady=(0,6))
        ttk.Button(right,text="Toggle Forward",command=self.toggle_forward).pack(fill="x",pady=4)
        ttk.Label(right,text="Target ID Controls",font=("Segoe UI",10,"bold")).pack(anchor="w",pady=(10,4))
        ttk.Button(right,text="Select Smallest ID (S)",command=self.on_key_s).pack(fill="x",pady=2)
        ttk.Button(right,text="Prev ID (A)",command=self.on_key_a).pack(fill="x",pady=2)
        ttk.Button(right,text="Next ID (D)",command=self.on_key_d).pack(fill="x",pady=2)

        ttk.Label(right,text="Control Parameters",font=("Segoe UI",10,"bold")).pack(anchor="w",pady=(10,4))
        telem=ttk.Frame(right); telem.pack(anchor="w",fill="x")
        mono_font=("Consolas",11) if os.name=="nt" else ("Menlo",11)
        ttk.Label(telem,text="Gamepad Device:",width=8,anchor="w").grid(row=0,column=0,sticky="w",padx=(0,6),pady=2)
        self.pad_name_var=tk.StringVar(value=self.gp.pad_name)
        ttk.Label(telem,textvariable=self.pad_name_var,anchor="w").grid(row=0,column=1,sticky="w",pady=2)
        ttk.Label(telem,text="Phisical Controller",width=8,anchor="w").grid(row=1,column=0,sticky="w",padx=(0,6),pady=(8,2))
        ttk.Separator(telem,orient="horizontal").grid(row=1,column=1,sticky="ew",pady=(8,2))
        for i,name in enumerate(["LX","LY","RX","RY"],start=2):
            ttk.Label(telem,text=f"{name}:",width=8,anchor="w").grid(row=i,column=0,sticky="w",padx=(0,6))
        self.real_vars=[tk.StringVar(value="+0.000") for _ in range(4)]
        for i,v in enumerate(self.real_vars,start=2):
            tk.Label(telem,textvariable=v,font=mono_font,width=8,anchor="e").grid(row=i,column=1,sticky="w")
        ttk.Label(telem,text="Virtual Controller",width=8,anchor="w").grid(row=6,column=0,sticky="w",padx=(0,6),pady=(8,2))
        ttk.Separator(telem,orient="horizontal").grid(row=6,column=1,sticky="ew",pady=(8,2))
        for i,name in enumerate(["LX","LY","RX","RY"],start=7):
            ttk.Label(telem,text=f"{name}:",width=8,anchor="w").grid(row=i,column=0,sticky="w",padx=(0,6))
        self.virt_vars=[tk.StringVar(value="+0.000") for _ in range(4)]
        for i,v in enumerate(self.virt_vars,start=7):
            tk.Label(telem,textvariable=v,font=mono_font,width=8,anchor="e").grid(row=i,column=1,sticky="w")
        ttk.Separator(right,orient="horizontal").pack(fill="x",pady=8)
        ttk.Label(right,text="ESC: Quit | S: select/clear | A/D: prev/next ID",
                  wraplength=RIGHT_PANEL_W-20,justify="left").pack(anchor="w")
        telem.grid_columnconfigure(0,weight=0); telem.grid_columnconfigure(1,weight=1)

    def toggle_forward(self):
        s=self.gp.state
        self.gp.state = ForwardState.ON if s==ForwardState.ARMING else (ForwardState.ARMING if s==ForwardState.OFF else ForwardState.OFF)
        self.state_var.set(f"Forward: {self.gp.state}")

    def on_key_s(self):
        if self.selected_id is None: self.selected_id=self.available_ids[0] if self.available_ids else None
        else: self.selected_id=None
    def on_key_d(self):
        if self.available_ids:
            if self.selected_id is None: self.selected_id=self.available_ids[0]
            else:
                nh=next_higher(self.available_ids,self.selected_id)
                if nh: self.selected_id=nh
    def on_key_a(self):
        if self.available_ids:
            if self.selected_id is None: self.selected_id=self.available_ids[0]
            else:
                nl=next_lower(self.available_ids,self.selected_id)
                if nl: self.selected_id=nl

    def _loop_worker(self):
        gen=self.model.track(source=CAM_INDEX,conf=CONF_THRES,device="cuda" if USE_GPU else "cpu",
                             classes=CLASSES,tracker=TRACKER_CFG,persist=PERSIST_ID,stream=True,verbose=False)
        while self.running:
            try: res=next(gen)
            except: continue
            frame=res.orig_img.copy()
            self.available_ids=get_available_ids(res)
            r_lx,r_ly,r_rx,r_ry=self.gp.read_axes_real()
            if self.gp.state==ForwardState.ARMING:
                if all(abs(a-b)<=ARM_EPS for a,b in zip([r_lx,r_ly,r_rx,r_ry],
                                                        [self.gp.v_lx,self.gp.v_ly,self.gp.v_rx,self.gp.v_ry])):
                    self.gp.state=ForwardState.ON
                    self.gp.send_to_virtual(r_lx,r_ly,r_rx,r_ry)
            elif self.gp.state==ForwardState.ON:
                self.gp.send_to_virtual(r_lx,r_ly,r_rx,r_ry)
            frame=draw_tracks(frame,res,self.selected_id,self.names)
            fixed=resize_with_letterbox(frame,VIDEO_W,VIDEO_H)
            with self.frame_lock: self.latest_bgr=fixed; self.video_ready=True

    def _render_placeholder(self):
        img=np.zeros((VIDEO_H,VIDEO_W,3),np.uint8)
        cv2.putText(img,"Running...",(VIDEO_W//2-100,VIDEO_H//2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        return img

    def _update_ui(self):
        self.state_var.set(f"Forward: {self.gp.state}")
        self.pad_name_var.set(self.gp.pad_name)
        vals=[self.gp.r_lx,self.gp.r_ly,self.gp.r_rx,self.gp.r_ry]
        for var,v in zip(self.real_vars,vals): var.set(f"{v:+.3f}")
        vals=[self.gp.v_lx,self.gp.v_ly,self.gp.v_rx,self.gp.v_ry]
        for var,v in zip(self.virt_vars,vals): var.set(f"{v:+.3f}")
        with self.frame_lock:
            frame=self.latest_bgr.copy() if self.video_ready and self.latest_bgr is not None else self._render_placeholder()
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img=ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_canvas.imgtk=img
        self.video_canvas.create_image(0,0,anchor="nw",image=img)
        if self.running: self.after(33,self._update_ui)

    def on_close(self):
        self.running=False; self.destroy()
        if pygame: 
            try: pygame.quit()
            except: pass

if __name__=="__main__":
    app=App()
    app.mainloop()
