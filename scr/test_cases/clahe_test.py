#tăng tương phản hình ảnh

import cv2
import numpy as np
from ultralytics import YOLO

# Configuration
WEIGHTS = "yolo_weights/yolov8m.pt"
CONF_THRES = 0.30
CLASSES = [0, 2, 3]  # person, car, motorcycle
CAM_INDEX = 1

def apply_clahe(frame):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Merge channels and convert back to BGR
    lab_clahe = cv2.merge([l_clahe, a, b])
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return enhanced

def draw_detections(frame, results, names):
    """Draw bounding boxes on frame"""
    if results.boxes is None or len(results.boxes) == 0:
        return frame
    
    xyxy = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy().astype(int)
    
    for box, conf, cls_id in zip(xyxy, confs, clss):
        x1, y1, x2, y2 = map(int, box)
        
        # Get class name
        label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        txt = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y_text = max(0, y1 - 8)
        cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 2), (0, 0, 0), -1)
        cv2.putText(frame, txt, (x1 + 3, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    
    return frame

def main():
    # Load YOLO model
    print(f"Loading model: {WEIGHTS}")
    model = YOLO(WEIGHTS)
    names = model.names
    
    # Open camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Cannot open camera index {CAM_INDEX}")
        return
    
    print("Press 'q' to quit")
    print("Press 'c' to toggle CLAHE enhancement")
    
    use_clahe = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply CLAHE enhancement
        if use_clahe:
            enhanced = apply_clahe(frame)
        else:
            enhanced = frame.copy()
        
        # Run YOLO detection
        results = model.predict(
            enhanced,
            conf=CONF_THRES,
            classes=CLASSES,
            verbose=False
        )[0]
        
        # Draw detections
        display = draw_detections(enhanced.copy(), results, names)
        
        # Add status text
        status = f"CLAHE: {'ON' if use_clahe else 'OFF'} | Detections: {len(results.boxes) if results.boxes else 0}"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show side-by-side comparison
        original_small = cv2.resize(frame, (320, 240))
        enhanced_small = cv2.resize(enhanced, (320, 240))
        
        cv2.putText(original_small, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(enhanced_small, "CLAHE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        comparison = np.hstack([original_small, enhanced_small])
        
        # Display
        cv2.imshow('YOLO Detection with CLAHE', display)
        cv2.imshow('Comparison', comparison)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            use_clahe = not use_clahe
            print(f"CLAHE: {'ON' if use_clahe else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
