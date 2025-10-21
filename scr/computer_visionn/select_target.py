# yolo_webcam_tracking_select_id_nav_toggle_fullscreen.py
import cv2
import os
from datetime import datetime
from ultralytics import YOLO
import numpy as np

# --------- Cấu hình nhanh ----------
WEIGHTS = "yolo_weights/yolov8m.pt"
USE_GPU = True
CONF_THRES = 0.30
CLASSES = [0, 39]                 # None nếu muốn tất cả lớp
TRACKER_CFG = "bytetrack.yaml"
PERSIST_ID = True
SHOW_WINDOW = True
SAVE_OUTPUT = False
CAM_INDEX = 0
TARGET_FPS = 30
# ------------------------------------

def draw_tracks(frame, res, selected_id, names):
    if res.boxes is None or len(res.boxes) == 0:
        return frame
    boxes = res.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
    ids = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else None
    clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        cls_i = int(clss[i]) if i < len(clss) else 0
        label = names.get(cls_i, str(cls_i)) if isinstance(names, dict) else (names[cls_i] if cls_i < len(names) else str(cls_i))
        track_id = int(ids[i]) if ids is not None and i < len(ids) else -1
        if selected_id is not None and track_id == selected_id:
            color, thickness = (0, 0, 255), 3   # đỏ cho ID đang chọn
        else:
            color, thickness = (0, 255, 0), 2   # xanh lá cho còn lại
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

def main():
    device = "cuda" if USE_GPU else "cpu"
    model = YOLO(WEIGHTS)
    names = model.model.names if hasattr(model, "model") else model.names

    out_dir = "runs/webcam_tracks"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"track_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
    writer = None

    selected_id = None
    is_fullscreen = False

    print("[INFO] Bắt đầu tracking từ webcam...")
    print("Phím: ESC thoát | S chọn ID nhỏ nhất / nhấn lại S để hủy chọn | A/D: qua ID nhỏ hơn/lớn hơn | P: Fullscreen ON/OFF")

    results_gen = model.track(
        source=CAM_INDEX,
        conf=CONF_THRES,
        device=device,
        classes=CLASSES,
        tracker=TRACKER_CFG,
        persist=PERSIST_ID,
        stream=True,
        verbose=False
    )

    window_title = f"YOLOv8 Tracking (tracker={os.path.basename(TRACKER_CFG)})"
    if SHOW_WINDOW:
        # Tạo cửa sổ có thể đổi kích thước và bật/tắt fullscreen
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    for res in results_gen:
        available_ids = get_available_ids(res)

        frame = res.orig_img.copy()
        frame = draw_tracks(frame, res, selected_id, names)

        if SAVE_OUTPUT and writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, TARGET_FPS, (w, h))
            if not writer.isOpened():
                print("[WARN] Không mở được VideoWriter, sẽ không lưu video.")
                writer = None

        if writer is not None:
            writer.write(frame)

        if SHOW_WINDOW:
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC -> thoát
                break

            elif key in (ord('p'), ord('P')):
                # Toggle fullscreen
                is_fullscreen = not is_fullscreen
                cv2.setWindowProperty(
                    window_title,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                )

            elif key in (ord('s'), ord('S')):
                if selected_id is None:
                    if available_ids:
                        selected_id = available_ids[0]
                        print(f"[KEY] Chọn ID nhỏ nhất: {selected_id}")
                    else:
                        print("[KEY] Không có ID nào trong khung để chọn.")
                else:
                    selected_id = None
                    print("[KEY] Hủy chọn: không còn bbox đỏ.")

            elif key in (ord('d'), ord('D')):
                if available_ids:
                    if selected_id is None:
                        selected_id = available_ids[0]
                        print(f"[KEY] Chưa chọn, auto chọn ID nhỏ nhất: {selected_id}")
                    else:
                        nh = next_higher(available_ids, selected_id)
                        if nh is not None:
                            selected_id = nh
                            print(f"[KEY] Sang ID lớn hơn: {selected_id}")
                        else:
                            print("[KEY] Đang ở ID lớn nhất hiện có.")
                else:
                    print("[KEY] Không có ID để chuyển.")

            elif key in (ord('a'), ord('A')):
                if available_ids:
                    if selected_id is None:
                        selected_id = available_ids[0]
                        print(f"[KEY] Chưa chọn, auto chọn ID nhỏ nhất: {selected_id}")
                    else:
                        nl = next_lower(available_ids, selected_id)
                        if nl is not None:
                            selected_id = nl
                            print(f"[KEY] Sang ID nhỏ hơn: {selected_id}")
                        else:
                            print("[KEY] Đang ở ID nhỏ nhất hiện có.")
                else:
                    print("[KEY] Không có ID để chuyển.")

    if writer is not None:
        writer.release()
        print(f"[INFO] Đã lưu video: {out_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
