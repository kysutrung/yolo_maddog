#ORB - Oriented FAST and Rotated BRIEF
#Brute Force Matcher - tìm điểm chung giữa hai frame liên tiếp
#RANSAC - Hạn chế các điểm match sai
#Affine Transform

import cv2
import numpy as np
from ultralytics import YOLO

# Configuration
WEIGHTS = "yolo_weights/yolov8m.pt"
CONF_THRES = 0.30
CLASSES = [0, 2, 3]  # person, car, motorcycle
CAM_INDEX = 1

# ORB parameters
N_FEATURES = 500
MIN_MATCH_COUNT = 10
RANSAC_THRESHOLD = 5.0

class ORBTracker:
    """Track objects using ORB feature matching with RANSAC"""
    
    def __init__(self, n_features=500):
        self.orb = cv2.ORB_create(n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.reference_kp = None
        self.reference_des = None
        self.reference_box = None
        
    def set_reference(self, frame, bbox):
        """Set reference object to track"""
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        
        # Detect ORB features
        self.reference_kp, self.reference_des = self.orb.detectAndCompute(crop, None)
        
        if self.reference_des is None or len(self.reference_kp) < MIN_MATCH_COUNT:
            return False
        
        self.reference_box = (x1, y1, x2, y2)
        print(f"Reference set: {len(self.reference_kp)} keypoints detected")
        return True
    
    def track(self, frame):
        """Track reference object in new frame"""
        if self.reference_des is None:
            return None, None
        
        # Detect features in current frame
        kp, des = self.orb.detectAndCompute(frame, None)
        
        if des is None or len(kp) < MIN_MATCH_COUNT:
            return None, None
        
        # Match features using BFMatcher
        matches = self.bf.knnMatch(self.reference_des, des, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < MIN_MATCH_COUNT:
            return None, None
        
        # Extract matched keypoint locations
        ref_pts = np.float32([self.reference_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find affine transformation using RANSAC
        M, mask = cv2.estimateAffinePartial2D(ref_pts, curr_pts, method=cv2.RANSAC, 
                                              ransacReprojThreshold=RANSAC_THRESHOLD)
        
        if M is None:
            return None, None
        
        # Transform reference box corners
        x1, y1, x2, y2 = self.reference_box
        ref_corners = np.float32([
            [0, 0],
            [x2 - x1, 0],
            [x2 - x1, y2 - y1],
            [0, y2 - y1]
        ]).reshape(-1, 1, 2)
        
        # Apply affine transformation
        transformed_corners = cv2.transform(ref_corners, M)
        
        # Get bounding box of transformed corners
        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]
        
        new_x1 = int(np.min(x_coords))
        new_y1 = int(np.min(y_coords))
        new_x2 = int(np.max(x_coords))
        new_y2 = int(np.max(y_coords))
        
        # Clip to frame boundaries
        h, w = frame.shape[:2]
        new_x1 = max(0, min(w - 1, new_x1))
        new_y1 = max(0, min(h - 1, new_y1))
        new_x2 = max(0, min(w - 1, new_x2))
        new_y2 = max(0, min(h - 1, new_y2))
        
        inliers = mask.ravel().tolist()
        inlier_count = sum(inliers)
        
        return (new_x1, new_y1, new_x2, new_y2), {
            'matches': len(good_matches),
            'inliers': inlier_count,
            'transform': M
        }
    
    def reset(self):
        """Reset tracker"""
        self.reference_kp = None
        self.reference_des = None
        self.reference_box = None

def draw_detections(frame, results, names, tracker_box=None):
    """Draw bounding boxes on frame"""
    if results.boxes is not None and len(results.boxes) > 0:
        xyxy = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = map(int, box)
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            txt = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            y_text = max(0, y1 - 8)
            cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 2), (0, 0, 0), -1)
            cv2.putText(frame, txt, (x1 + 3, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Draw tracker box
    if tracker_box is not None:
        x1, y1, x2, y2 = tracker_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(frame, "ORB Tracking", (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
    
    return frame

def main():
    # Load YOLO model
    print(f"Loading model: {WEIGHTS}")
    model = YOLO(WEIGHTS)
    names = model.names
    
    # Create ORB tracker
    tracker = ORBTracker(n_features=N_FEATURES)
    
    # Open camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Cannot open camera index {CAM_INDEX}")
        return
    
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Select first detected object for ORB tracking")
    print("  'r' - Reset tracker")
    
    tracking_active = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        results = model.predict(
            frame,
            conf=CONF_THRES,
            classes=CLASSES,
            verbose=False
        )[0]
        
        tracker_box = None
        tracker_info = None
        
        # Track object if active
        if tracking_active:
            tracker_box, tracker_info = tracker.track(frame)
            if tracker_box is None:
                print("Tracking lost!")
                tracking_active = False
        
        # Draw results
        display = draw_detections(frame.copy(), results, names, tracker_box)
        
        # Add status text
        n_det = len(results.boxes) if results.boxes else 0
        status = f"YOLO Detections: {n_det} | ORB Tracking: {'ON' if tracking_active else 'OFF'}"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show tracker info
        if tracker_info is not None:
            info_text = f"Matches: {tracker_info['matches']} | Inliers: {tracker_info['inliers']}"
            cv2.putText(display, info_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('YOLO + ORB Feature Matching (RANSAC)', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Select first detection for tracking
            if results.boxes is not None and len(results.boxes) > 0:
                first_box = results.boxes.xyxy[0].cpu().numpy()
                if tracker.set_reference(frame, first_box):
                    tracking_active = True
                    print("Tracking started!")
                else:
                    print("Failed to extract features from selected object")
            else:
                print("No detections available")
        elif key == ord('r'):
            tracker.reset()
            tracking_active = False
            print("Tracker reset")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
