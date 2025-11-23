# drawing.py
import cv2
import numpy as np

from config import CROSSHAIR_CENTER_SIZE, CROSSHAIR_TARGET_SIZE

def draw_tracks(frame, res, selected_id, names, show=True, allowed_classes=None):
    if not show or res.boxes is None or len(res.boxes) == 0:
        return frame

    boxes = res.boxes
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
    ids  = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else None
    clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else np.zeros(len(xyxy), dtype=int)
    confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))

    for i, (x1, y1, x2, y2) in enumerate(xyxy):
        cls_i = int(clss[i]) if i < len(clss) else 0
        if (allowed_classes is not None) and (cls_i not in allowed_classes):
            continue

        label = names.get(cls_i, str(cls_i)) if isinstance(names, dict) else str(cls_i)
        track_id = int(ids[i]) if ids is not None and i < len(ids) else -1

        color, thick = ((0, 0, 255), 3) if selected_id == track_id else ((0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)

        txt = f"{label} ID:{track_id if track_id != -1 else 'NA'} {confs[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(0, y1 - 8)
        cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 2), (0, 0, 0), -1)
        cv2.putText(frame, txt, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)

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
