# utils.py
import numpy as np

def next_higher(lst, cur):
    """Return the first element in lst that is > cur, or None."""
    return next((x for x in lst if x > cur), None)

def resize_with_letterbox(bgr, target_w, target_h):
    """Resize image with letterbox (black bars) to fit into target size."""
    import cv2
    h, w = bgr.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(bgr, (new_w, new_h), cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w, 3), np.uint8)
    x, y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def box_center(x1, y1, x2, y2):
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

def iou(a, b):
    """Intersection-over-union of 2 boxes in (x1,y1,x2,y2) format."""
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
    """Clip box (x1,y1,x2,y2) into image frame size WxH."""
    x1, y1, x2, y2 = b
    return (
        clamp(int(x1), 0, W - 1),
        clamp(int(y1), 0, H - 1),
        clamp(int(x2), 0, W - 1),
        clamp(int(y2), 0, H - 1),
    )
