#Chương trình này dùng để thử nghiệm nhận dạng vật thể đơn giản

from ultralytics import YOLO

import cv2

model = YOLO("yolo_weights/yolov8m.pt")

results = model.predict(source="drone_footages/Whiteman.mp4", #nguồn hình ảnh
                        conf=0.3, #độ chính xác
                        device="cuda", #phần cứng xử lý
                        classes=[0], #đối tượng quét
                        save=True, #lưu video  
                        show=True)