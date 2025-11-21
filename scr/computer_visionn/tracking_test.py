from ultralytics import YOLO

model = YOLO("yolo_weights/yolov8m.engine")

model.track(
    source="0",                  
    conf=0.3,                  
    device="cuda",            
    classes=[0],               
    show=True,                 
    persist=True,              
    tracker="bytetrack.yaml",
    save=False,
    half=True                 
)
