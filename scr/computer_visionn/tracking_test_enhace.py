# yolo_persistent_tracking_fixed.py
# pip install ultralytics opencv-python numpy

import cv2
import numpy as np
from ultralytics import YOLO
import time
import random

# ================= Config =================
YOLO_WEIGHTS   = "yolo_weights/yolov8n.pt"
CONF_THRES     = 0.35
CLASSES        = [39]          # 0=person
INPUT_SRC      = 0            # camera index hoặc đường dẫn video
IMG_SIZE       = 640

IOU_THRESHOLD        = 0.35   # ngưỡng khớp IoU
HIST_SIM_THRESHOLD   = 0.5    # 0..1 (1 là giống hệt)
MAX_LOST             = 30     # số frame cho phép mất detection
MIN_AREA             = 400   # bỏ box quá nhỏ
USE_CV_TRACKER       = True   # fallback bằng OpenCV tracker
CV_TRACKER_TYPE      = "CSRT" # CSRT/KCF/MOSSE

# Lưu video đầu ra
SAVE_VIDEO = True
OUT_PATH   = "tracked_output.mp4"
OUT_FPS    = 30  # nếu nguồn là camera; nếu là file video sẽ cố lấy fps gốc

# ================ Helpers =================
def iou(boxA, boxB):
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
    if union <= 0:
        return 0.0
    return inter_area / union

def box_xyxy_to_xywh(b):
    x1, y1, x2, y2 = b
    return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

def compute_hist(image, bbox):
    x1, y1, x2, y2 = [max(0, int(v)) for v in bbox]
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
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def make_cv_tracker(name=CV_TRACKER_TYPE):
    # Tương thích cả cv2.legacy.* (một số bản opencv >=4.5)
    legacy = getattr(cv2, "legacy", None)
    if name.upper() == "CSRT":
        if legacy and hasattr(legacy, "TrackerCSRT_create"):
            return legacy.TrackerCSRT_create()
        return cv2.TrackerCSRT_create()
    if name.upper() == "KCF":
        if legacy and hasattr(legacy, "TrackerKCF_create"):
            return legacy.TrackerKCF_create()
        return cv2.TrackerKCF_create()
    if name.upper() == "MOSSE":
        if legacy and hasattr(legacy, "TrackerMOSSE_create"):
            return legacy.TrackerMOSSE_create()
        return cv2.TrackerMOSSE_create()
    # mặc định
    if legacy and hasattr(legacy, "TrackerCSRT_create"):
        return legacy.TrackerCSRT_create()
    return cv2.TrackerCSRT_create()

# ================ Track class ==============
class Track:
    _next_id = 0
    def __init__(self, frame, bbox_xyxy, appearance_hist=None):
        self.id = Track._next_id
        Track._next_id += 1
        self.bbox = tuple(int(v) for v in bbox_xyxy)  # (x1,y1,x2,y2)
        self.hist = appearance_hist
        self.last_seen = 0
        self.lost = 0
        self.color = (random.randint(50,255), random.randint(50,255), random.randint(50,255))
        self.cv_tracker = None
        if USE_CV_TRACKER:
            try:
                self.cv_tracker = make_cv_tracker()
                self.cv_tracker.init(frame, box_xyxy_to_xywh(self.bbox))
            except Exception:
                self.cv_tracker = None

    def update_with_detection(self, frame, bbox_xyxy):
        self.bbox = tuple(int(v) for v in bbox_xyxy)
        self.hist = compute_hist(frame, self.bbox)
        self.lost = 0
        if USE_CV_TRACKER:
            try:
                self.cv_tracker = make_cv_tracker()
                self.cv_tracker.init(frame, box_xyxy_to_xywh(self.bbox))
            except Exception:
                self.cv_tracker = None

    def predict_with_cvtracker(self, frame):
        if self.cv_tracker is None:
            return False
        ok, box = self.cv_tracker.update(frame)
        if not ok:
            return False
        x, y, w, h = box
        self.bbox = (int(x), int(y), int(x + w), int(y + h))
        return True

# ================ Main =====================
def main():
    model = YOLO(YOLO_WEIGHTS)

    cap = cv2.VideoCapture(INPUT_SRC)
    if not cap.isOpened():
        print("Không mở được camera/video.")
        return

    # Chuẩn bị VideoWriter nếu cần
    writer = None
    if SAVE_VIDEO:
        # cố gắng lấy fps gốc nếu là file; nếu thất bại dùng OUT_FPS
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or np.isnan(src_fps) or src_fps <= 1:
            src_fps = OUT_FPS
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUT_PATH, fourcc, src_fps, (w, h))

    tracks = []
    frame_idx = 0
    fps = 0.0                        # <-- FIX: khởi tạo trước khi dùng
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # --- YOLO detect (per-frame) ---
        results = model.predict(source=frame, imgsz=IMG_SIZE, conf=CONF_THRES, classes=CLASSES, verbose=False)
        detections = []
        if results:
            res = results[0]
            try:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.empty((0,4))
            except Exception:
                boxes_xyxy = np.empty((0,4))
            for box in boxes_xyxy:
                x1, y1, x2, y2 = [int(v) for v in box[:4]]
                if (x2 - x1) * (y2 - y1) >= MIN_AREA:
                    detections.append((x1, y1, x2, y2))

        # --- Match by IoU ---
        unmatched_dets = set(range(len(detections)))
        matched_tracks = set()

        if tracks and detections:
            iou_mat = np.zeros((len(tracks), len(detections)), dtype=float)
            for ti, tr in enumerate(tracks):
                for di, det in enumerate(detections):
                    iou_mat[ti, di] = iou(tr.bbox, det)

            # greedy
            while True:
                ti, di = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
                if iou_mat[ti, di] <= IOU_THRESHOLD:
                    break
                tracks[ti].update_with_detection(frame, detections[di])
                tracks[ti].last_seen = frame_idx
                matched_tracks.add(ti)
                if di in unmatched_dets:
                    unmatched_dets.remove(di)
                iou_mat[ti, :] = -1
                iou_mat[:, di] = -1

        # --- Match phần còn lại theo appearance (hist) ---
        if unmatched_dets and tracks:
            for di in list(unmatched_dets):
                det = detections[di]
                det_hist = compute_hist(frame, det)
                best_sim, best_ti = 0.0, None
                for ti, tr in enumerate(tracks):
                    if ti in matched_tracks:
                        continue
                    sim = cosine_sim(det_hist, tr.hist)
                    if sim > best_sim:
                        best_sim, best_ti = sim, ti
                if best_ti is not None and best_sim >= HIST_SIM_THRESHOLD:
                    tracks[best_ti].update_with_detection(frame, det)
                    tracks[best_ti].last_seen = frame_idx
                    matched_tracks.add(best_ti)
                    unmatched_dets.remove(di)

        # --- Tạo track mới cho các detection còn lại ---
        for di in list(unmatched_dets):
            det = detections[di]
            hist = compute_hist(frame, det)
            new_tr = Track(frame, det, appearance_hist=hist)
            new_tr.last_seen = frame_idx
            tracks.append(new_tr)

        # --- Fallback khi mất detection ---
        for tr in tracks:
            if tr.last_seen == frame_idx:
                continue
            predicted = tr.predict_with_cvtracker(frame) if USE_CV_TRACKER else False
            tr.lost += 1
            if predicted and tr.lost < 3:
                tr.hist = compute_hist(frame, tr.bbox)

        # --- Xóa track mất quá lâu ---
        tracks = [tr for tr in tracks if tr.lost <= MAX_LOST]

        # --- Vẽ ---
        for tr in tracks:
            x1, y1, x2, y2 = tr.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), tr.color, 2)
            cv2.putText(frame, f"ID {tr.id} L{tr.lost}", (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, tr.color, 2)

        # --- FPS (EMA để mượt hơn) ---
        t_now = time.perf_counter()
        inst_fps = 1.0 / max(1e-6, (t_now - t_prev))
        t_prev = t_now
        fps = 0.9 * fps + 0.1 * inst_fps  # mượt
        cv2.putText(frame, f"FPS ~ {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- Hiển thị / Lưu ---
        cv2.imshow("Persistent YOLO Tracking", frame)
        if writer is not None:
            writer.write(frame)

        if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
            break

    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
