from ultralytics import YOLO

model = YOLO("yolo_weights/yolov8m.pt")
model.export(format="engine", device=0, half=True)  # sẽ tạo yolov8m.engine phù hợp GPU của bạn
