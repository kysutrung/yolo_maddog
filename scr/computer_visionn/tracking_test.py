from ultralytics import YOLO

model = YOLO("yolo_weights/yolov8m.pt")

model.track(
    source="drone_footages/Whiteman01.mp4",                  
    conf=0.3,                  
    device="cuda",            
    classes=[0],               
    show=True,                 
    persist=True,              #bật tắt ID
    tracker="bytetrack.yaml",  #chọn tracker -> "botsort.yaml"
    save=True                 #lưu video đầu ra
)
