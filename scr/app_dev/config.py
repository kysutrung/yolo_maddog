# config.py

# =================== Configuration ===================
WEIGHTS        = "yolo_weights/yolov8m.engine"
USE_GPU        = True
CONF_THRES     = 0.30
CLASSES        = [0, 2, 3]
TRACKER_CFG    = "bytetrack.yaml"
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

# ======== Enhanced Follow config ========
PRED_MAX_GAP        = 30
REACQ_IOU_THR       = 0.32
HIST_SIM_THR        = 0.50
USE_CV_TRACKER      = True
CV_TRACKER_TYPE     = "CSRT"
MIN_AREA_REACQ      = 200

# ======== Navigation without on-frame text ========
NAV_DEAD_ZONE_PX      = 110
CROSSHAIR_CENTER_SIZE = 6
CROSSHAIR_TARGET_SIZE = 6
