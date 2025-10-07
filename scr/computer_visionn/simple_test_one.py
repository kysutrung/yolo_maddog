#Chương trình này dùng để thử nghiệm nhận dạng vật thể đơn giản

from ultralytics import YOLO

import cv2

model = YOLO("yolo_weights/yolov8x.pt")

results = model.predict(source="0", #nguồn hình ảnh
                        conf=0.3, #độ chính xác
                        device="cuda", #phần cứng xử lý
                        classes=[0], #đối tượng quét
                        show=True)