# app.py
import os
import cv2
import numpy as np
import threading
import queue
import time
from datetime import datetime

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO

from .config import (
    WEIGHTS, USE_GPU, CONF_THRES, CLASSES, TRACKER_CFG, PERSIST_ID,
    CAM_INDEX, VIDEO_W, VIDEO_H, RIGHT_PANEL_W, THEME_PAD,
    WIN_TITLE, START_FULLSCREEN,
    PRED_MAX_GAP, REACQ_IOU_THR, HIST_SIM_THR, MIN_AREA_REACQ,
    NAV_DEAD_ZONE_PX,
)
from .gamepad import GamepadBridge, ForwardState
from .utils import next_higher, resize_with_letterbox
from .drawing import (
    draw_tracks,
    draw_predicted_box,
    draw_crosshair_center,
    draw_crosshair_at,
)
from .tracking import (
    GapPredictor,
    compute_hist,
    cosine_sim,
    make_cv_tracker,
)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        # ... PHẦN CÒN LẠI GIỮ GẦN NHƯ NGUYÊN ...
        # chỉ cần sửa chỗ dùng GamepadBridge, ForwardState, GapPredictor, các hàm util/drawing
