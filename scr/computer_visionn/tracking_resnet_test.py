import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

# ============================================================
#        FEATURE EXTRACTOR (ResNet50 – No TorchReID)
# ============================================================

class ResNet50Extractor:
    def __init__(self, device="cuda"):
        self.device = device
        base = models.resnet50(weights="IMAGENET1K_V2")
        base.fc = nn.Identity()  # output 2048-D vector
        self.model = base.to(device).eval()

        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def __call__(self, img):
        if img is None or img.size == 0:
            return None
        img = self.tf(img).unsqueeze(0).to(self.device)
        feat = self.model(img)[0]
        feat = feat / (feat.norm() + 1e-8)
        return feat.cpu().numpy()


# ============================================================
#                  MEMORY RE-ID (Signature)
# ============================================================

class TargetMemory:
    def __init__(self, max_samples=20):
        self.embeds = []
        self.max_samples = max_samples

    def add(self, emb):
        if emb is None:
            return
        self.embeds.append(emb)
        if len(self.embeds) > self.max_samples:
            self.embeds.pop(0)

    def signature(self):
        if len(self.embeds) == 0:
            return None
        sig = np.mean(self.embeds, axis=0)
        sig /= (np.linalg.norm(sig) + 1e-8)
        return sig


# ============================================================
#           DRONE SAFE KALMAN TRACKER (No NaN)
# ============================================================

class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox):

        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])

        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        self.kf.P *= 10
        self.kf.R *= 1
        self.kf.Q *= 0.01

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = max((x2-x1)*(y2-y1), 1e-6)
        r = max((x2-x1)/(y2-y1+1e-6), 0.1)

        self.kf.x[:4] = np.array([[cx], [cy], [s], [r]])

        self.time_since_update = 0
        self.hits = 1
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.last_feat = None
        self.predicted_box = bbox
        self.memory = TargetMemory()

    def predict(self):
        self.kf.predict()

        if self.kf.x[2] < 1e-6:
            self.kf.x[2] = 1e-6
        if self.kf.x[3] < 0.1:
            self.kf.x[3] = 1.0

        self.time_since_update += 1
        self.predicted_box = self.state()
        return self.predicted_box

    def update(self, bbox):
        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        s = max((x2-x1)*(y2-y1), 1e-6)
        r = max((x2-x1)/(y2-y1+1e-6), 0.1)

        z = np.array([[cx], [cy], [s], [r]])
        self.kf.update(z)

        self.hits += 1
        self.time_since_update = 0

    def state(self):
        cx, cy, s, r = self.kf.x[:4].reshape(-1)

        s = max(s, 1e-6)
        r = max(r, 0.1)

        w = np.sqrt(s * r)
        h = s / (w + 1e-6)

        if np.isnan(w) or np.isnan(h) or w <= 0 or h <= 0:
            w, h = 50, 100

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return [int(x1), int(y1), int(x2), int(y2)]


# ============================================================
#                     DRONE TRACKER PRO 2.0
# ============================================================

class DroneTrackerPro:
    def __init__(self, device="cuda"):
        self.device = device
        self.extractor = ResNet50Extractor(device)
        self.trackers = []

        self.base_thr = 0.65
        self.adapt_low = 0.55
        self.max_age = 120

        self.target_id = None
        self.target_sig = None

    def iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        w = max(0, x2-x1)
        h = max(0, y2-y1)
        inter = w*h
        areaA = (a[2]-a[0])*(a[3]-a[1])
        areaB = (b[2]-b[0])*(b[3]-b[1])
        return inter / (areaA + areaB - inter + 1e-6)

    def update(self, dets, frame):
        feats = []
        for (x1, y1, x2, y2, conf, cls) in dets:
            feats.append(self.extractor(frame[y1:y2, x1:x2]))

        for t in self.trackers:
            t.predict()

        if len(self.trackers) == 0:
            for i, det in enumerate(dets):
                tr = KalmanBoxTracker(det[:4])
                tr.last_feat = feats[i]
                tr.memory.add(feats[i])
                self.trackers.append(tr)
        else:
            N = len(self.trackers)
            M = len(dets)
            cost = np.ones((N, M), dtype=np.float32)

            for i, trk in enumerate(self.trackers):
                pred = trk.predicted_box

                for j, feat in enumerate(feats):
                    cos = 0
                    if trk.last_feat is not None and feat is not None:
                        cos = np.dot(trk.last_feat, feat)
                        cos = (cos + 1) / 2

                    iou = self.iou(pred, dets[j][:4])

                    # center similarity
                    bx1, by1, bx2, by2 = pred
                    cx = (bx1 + bx2) / 2
                    cy = (by1 + by2) / 2
                    tx = (dets[j][0] + dets[j][2]) / 2
                    ty = (dets[j][1] + dets[j][3]) / 2
                    center_sim = np.exp(-0.0005 * ((tx-cx)**2 + (ty-cy)**2))

                    score = 0.65*cos + 0.25*iou + 0.10*center_sim

                    # nếu là target → dùng signature mạnh hơn
                    if trk.id == self.target_id and self.target_sig is not None:
                        sig_cos = np.dot(self.target_sig, feat)
                        sig_cos = (sig_cos + 1) / 2
                        score = max(score, sig_cos)

                    cost[i, j] = 1 - score

            row, col = linear_sum_assignment(cost)
            matched = set()

            for r, c in zip(row, col):
                thr = self.adapt_low if self.trackers[r].hits > 20 else self.base_thr

                if cost[r, c] < (1 - thr):
                    self.trackers[r].update(dets[c][:4])
                    self.trackers[r].last_feat = feats[c]
                    self.trackers[r].memory.add(feats[c])
                    if self.trackers[r].id == self.target_id:
                        self.target_sig = self.trackers[r].memory.signature()
                    matched.add(c)

            for j in range(len(dets)):
                if j not in matched:
                    tr = KalmanBoxTracker(dets[j][:4])
                    tr.last_feat = feats[j]
                    tr.memory.add(feats[j])
                    self.trackers.append(tr)

        self.trackers = [
            t for t in self.trackers if t.time_since_update < self.max_age
        ]

        out = []
        for t in self.trackers:
            out.append((t.id, *t.state(), t.time_since_update))
        return out


# ============================================================
#                          MAIN
# ============================================================

def main():

    model = YOLO("yolo_weights/yolov8m.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracker = DroneTrackerPro(device)

    cap = cv2.VideoCapture(0)

    # ========== VIDEO WRITER ==========
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("drone_tracking_output.mp4", fourcc, fps, (w, h))
    # ==================================

    print("Nhấn S: lock target")
    print("Nhấn R: reset")
    print("Nhấn Q: thoát")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        cx, cy = w//2, h//2

        cv2.drawMarker(frame, (cx, cy), (255, 255, 255),
                       cv2.MARKER_CROSS, 30, 3)

        res = model(frame, conf=0.3, classes=[0])[0]
        dets = []

        for b in res.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].cpu().numpy())
            dets.append((x1, y1, x2, y2, float(b.conf), int(b.cls)))

        tracks = tracker.update(dets, frame)

        lock_box = None
        for tid, x1, y1, x2, y2, miss in tracks:
            color = (150,150,150) if miss == 0 else (60,60,60)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"ID:{tid}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if tid == tracker.target_id:
                lock_box = (x1,y1,x2,y2)

        if lock_box:
            x1,y1,x2,y2 = lock_box
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.putText(frame, f"TARGET {tracker.target_id}",
                        (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0), 2)

        # ==== GHI VIDEO ====
        out.write(frame)
        # ====================

        cv2.imshow("DRONE TRACKER PRO 2.0 - Recording", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == ord('r'):
            tracker.target_id = None
            tracker.target_sig = None
            print("Target reset.")

        if key == ord('s'):
            best = None
            best_d = 1e12
            for tid, x1, y1, x2, y2, _ in tracks:
                bx = (x1+x2)/2
                by = (y1+y2)/2
                d = (bx-cx)**2 + (by-cy)**2
                if d < best_d:
                    best_d = d
                    best = tid

            if best is not None:
                tracker.target_id = best
                for t in tracker.trackers:
                    if t.id == best:
                        tracker.target_sig = t.memory.signature()
                print("Locked target:", best)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
