#Coded by Duc Le

import threading
import cv2
from ultralytics import YOLO
import time

print("RUNNING...")

model = YOLO("yolo_weights/yolov8m.pt")
CAM_ID, WIDTH, HEIGHT = 0, 640, 480
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
DEAD_ZONE = 20
PX_PER_CM = 20  # 20px ~ 1cm

auto_tracking = False  # Ban đầu tắt tracking


def video_loop():
    global auto_tracking
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf=0.3,
            device='cuda',
            classes=[0],  # 0 - người
            verbose=False
        )
        boxes = results[0].boxes

        if boxes:
            box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Vẽ minh họa
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.circle(frame, (CENTER_X, CENTER_Y), 5, (255, 0, 0), -1)

            # Chỉ in khi tracking được bật
            if auto_tracking:
                dx = cx - CENTER_X
                dy = cy - CENTER_Y

                if abs(dx) > DEAD_ZONE:
                    excess_px = abs(dx) - DEAD_ZONE
                    excess_cm = excess_px / PX_PER_CM
                    direction = "phải" if dx > 0 else "trái"
                    print(f"quay {direction} ({excess_px} px ~ {excess_cm:.1f} cm)")

                if abs(dy) > DEAD_ZONE:
                    excess_px = abs(dy) - DEAD_ZONE
                    excess_cm = excess_px / PX_PER_CM
                    direction = "xuống" if dy > 0 else "lên"
                    print(f"quay {direction} ({excess_px} px ~ {excess_cm:.1f} cm)")

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
            break

    cap.release()
    cv2.destroyAllWindows()


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


# Chạy camera song song với listener
threading.Thread(target=video_loop, daemon=True).start()
command_listener()
