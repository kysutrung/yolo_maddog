# Coded by Duc Le and TrungTauLua (updated for larger window + always-on crosshair + outlined text)

import threading
import cv2
from ultralytics import YOLO
import time

print("RUNNING...")

# -------------------------------
# Cấu hình
# -------------------------------
model = YOLO("yolo_weights/yolov8m.pt")

# Tăng độ phân giải camera (nếu camera hỗ trợ)
CAM_ID = 0
WIDTH, HEIGHT = 1280, 720      # <-- chỉnh ở đây để ảnh thật lớn hơn
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2

DEAD_ZONE = 20
PX_PER_CM = 20  # 20px ~ 1cm
auto_tracking = True  # tracking mặc định bật


# -------------------------------
# Hàm vẽ grid chữ thập giữa khung
# -------------------------------
def draw_crosshair(frame):
    # Crosshair full khung hình
    cv2.line(frame, (0, CENTER_Y), (WIDTH, CENTER_Y), (255, 0, 0), 1, cv2.LINE_AA)
    cv2.line(frame, (CENTER_X, 0), (CENTER_X, HEIGHT), (255, 0, 0), 1, cv2.LINE_AA)


# -------------------------------
# Hàm viết chữ có viền (trắng, viền đen)
# -------------------------------
def draw_text_outline(frame, text, pos, font_scale=1, color=(255, 255, 255),
                      outline_color=(0, 0, 0), thickness=2, font=cv2.FONT_HERSHEY_SIMPLEX):
    x, y = pos
    # Viền
    cv2.putText(frame, text, (x, y), font, font_scale, outline_color, thickness + 3, cv2.LINE_AA)
    # Chữ chính
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


# -------------------------------
# Luồng video chính
# -------------------------------
def video_loop():
    global auto_tracking
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    # Cho phép cửa sổ được resize và đặt kích thước hiển thị lớn hơn
    window_name = "Drone Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)      # có thể kéo/resize
    cv2.resizeWindow(window_name, 1280, 720)             # <-- chỉnh kích thước cửa sổ hiển thị ở đây

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=0.3,
            device='cuda',     # nếu không có GPU, đổi thành 'cpu'
            classes=[39],      # 39 = "bottle" trong COCO
            verbose=False
        )
        boxes = results[0].boxes

        if boxes:
            # Chọn box lớn nhất
            box = max(
                boxes,
                key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1])
            )
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Vẽ bounding box và tâm vật thể
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)

            # Tracking nếu bật
            if auto_tracking:
                dx = cx - CENTER_X
                dy = cy - CENTER_Y
                direction = ""
                if abs(dx) > DEAD_ZONE:
                    direction += "Right" if dx > 0 else "Left"
                if abs(dy) > DEAD_ZONE:
                    direction += " Down" if dy > 0 else " Up"
                if direction:
                    draw_text_outline(frame, f"Turn: {direction}", (10, 50), 1.2)

        # Luôn hiển thị grid
        draw_crosshair(frame)

        # Trạng thái tracking
        status = "Tracking: ON" if auto_tracking else "Tracking: OFF"
        draw_text_outline(frame, status, (10, HEIGHT - 20), 0.9,
                          color=(0, 255, 0) if auto_tracking else (0, 0, 255))

        # Hiển thị
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# Thread lệnh điều khiển
# -------------------------------
def command_listener():
    global auto_tracking
    while True:
        cmd = input().strip().lower()
        if cmd == "track":
            auto_tracking = True
            print("-> MODE Tracking: BẬT")
        elif cmd == "stop":
            auto_tracking = False
            print("-> MODE Tracking: TẮT")
        elif cmd == "exit":
            print("Thoát chương trình...")
            break
        else:
            print("Lệnh không hợp lệ! (dùng: track / stop / exit)")


# -------------------------------
# Khởi động
# -------------------------------
threading.Thread(target=video_loop, daemon=True).start()
command_listener()
