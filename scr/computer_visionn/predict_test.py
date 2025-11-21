#Chương trình này dùng để thử nghiệm nhận dạng vật thể đơn giản

from ultralytics import YOLO

import cv2

model = YOLO("yolo_weights/yolov8m.pt")

results = model.predict(source="0", #nguồn hình ảnh
                        conf=0.6, #độ chính xác
                        device="cuda", #phần cứng xử lý 
                        classes=[0], #đối tượng quét
                        save=False, #lưu video  
                        max_det=5,
                        half=False,  
                        show=True)