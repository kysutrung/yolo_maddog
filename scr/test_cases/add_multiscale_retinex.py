import cv2
import numpy as np
import time

# ================== Fast Multiscale Retinex (V-channel) ==================
def msr_gray(gray_u8: np.ndarray, scales=(15, 80, 250)) -> np.ndarray:
    """
    Multiscale Retinex cho ảnh xám:
        R = mean_s [ log(I) - log(blur_s(I)) ]
    Blur dùng boxFilter để nhanh hơn GaussianBlur.
    """
    eps = 1e-6
    I = gray_u8.astype(np.float32) + 1.0
    logI = np.log(I)

    acc = np.zeros_like(I, dtype=np.float32)
    for s in scales:
        k = int(max(3, (int(s) // 2) * 2 + 1))  # kernel odd
        blur = cv2.boxFilter(I, ddepth=-1, ksize=(k, k), normalize=True)
        acc += (logI - np.log(blur + eps))
    acc /= float(len(scales))

    out = cv2.normalize(acc, None, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.uint8)

def msr_bgr_vchannel(bgr: np.ndarray, scales=(15, 80, 250), gain=1.15, clahe_clip=2.0) -> np.ndarray:
    """
    Retinex xử lý kênh sáng (V) trong HSV để nhanh:
    - MSR trên V
    - CLAHE nhẹ tăng chi tiết
    - Gain tăng sáng
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v_ret = msr_gray(v, scales=scales)

    # CLAHE nhẹ
    if clahe_clip and clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        v_ret = clahe.apply(v_ret)

    # Gain
    v_ret = cv2.convertScaleAbs(v_ret, alpha=float(gain), beta=0)

    hsv2 = cv2.merge([h, s, v_ret])
    return cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)

# ================== Main ==================
def main():
    cam_index = 0
    cap = cv2.VideoCapture(cam_index)

    # Nếu muốn ép độ phân giải để test tốc độ:
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Không mở được camera.")
        return

    # Presets (nhanh -> mạnh)
    presets = {
        1: (15, 80),          # nhanh
        2: (15, 80, 200),     # vừa
        3: (15, 80, 250),     # mạnh (chậm hơn)
    }
    preset_id = 2
    scales = presets[preset_id]

    gain = 1.15
    clahe_clip = 2.0

    # Skip: chỉ xử lý 1 lần / N frame để tăng FPS
    retinex_skip = 1
    frame_count = 0
    cached_ret = None

    # FPS đo riêng
    last_time = time.monotonic()
    fps = 0.0
    fps_count = 0

    print("Phím tắt:")
    print("  1/2/3: preset scales (nhanh/vừa/mạnh)")
    print("  [ ]  : giảm/tăng gain")
    print("  - =  : tăng/giảm retinex_skip")
    print("  q/ESC: thoát")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame_count += 1

        # Retinex với skip + cache
        if cached_ret is None or (frame_count % max(1, retinex_skip) == 0):
            cached_ret = msr_bgr_vchannel(frame, scales=scales, gain=gain, clahe_clip=clahe_clip)

        out = cached_ret

        # Ghép 2 khung hình cạnh nhau
        # Đảm bảo cùng kích thước
        if out.shape != frame.shape:
            out = cv2.resize(out, (frame.shape[1], frame.shape[0]))

        combo = np.hstack([frame, out])

        # FPS
        fps_count += 1
        now = time.monotonic()
        if now - last_time >= 1.0:
            fps = fps_count / (now - last_time)
            fps_count = 0
            last_time = now

        # Overlay text
        info = f"Preset:{preset_id} scales={scales} | gain={gain:.2f} | skip={retinex_skip} | FPS={fps:.1f}"
        cv2.putText(combo, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(combo, "Left: Original | Right: MSR", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Original vs Multiscale Retinex", combo)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):  # ESC or q
            break
        elif key == ord('1'):
            preset_id = 1
            scales = presets[preset_id]
        elif key == ord('2'):
            preset_id = 2
            scales = presets[preset_id]
        elif key == ord('3'):
            preset_id = 3
            scales = presets[preset_id]
        elif key == ord('['):
            gain = max(0.50, gain - 0.05)
        elif key == ord(']'):
            gain = min(3.00, gain + 0.05)
        elif key == ord('-'):
            retinex_skip = min(10, retinex_skip + 1)  # tăng skip -> nhanh hơn
        elif key == ord('='):
            retinex_skip = max(1, retinex_skip - 1)   # giảm skip -> đẹp hơn

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
