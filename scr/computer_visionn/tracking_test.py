from ultralytics import YOLO

model = YOLO("yolo_weights/yolov8m.pt")

model.track(
    source=0,                  
    conf=0.3,                  
    device="cuda",            
    classes=[0],               
    show=True,                 
    persist=True,              # giữ ID giữa các frame
    tracker="bytetrack.yaml",  # hoặc "botsort.yaml"
    save=False                 # True nếu muốn lưu video đầu ra
)
