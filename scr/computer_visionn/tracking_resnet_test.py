import cv2
import numpy as np
import torch
from torchvision import models
from ultralytics import YOLO

# ===================== CẤU HÌNH ===================== #
YOLO_WEIGHTS = "yolo_weights/yolov8m.engine"  # đường dẫn model YOLO của bạn
CONF_THRES = 0.3                              # ngưỡng confidence của YOLO

EMB_SIM_THRESHOLD = 0.7                       # ngưỡng cosine similarity để coi là cùng mục tiêu
TARGET_ABS_ID = 1                             # ID tuyệt đối cho mục tiêu

# ImageNet mean/std cho normalize
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ===================== MẠNG TRÍCH ĐẶC TRƯNG ===================== #
def build_feature_extractor(device):
    """
    Tạo ResNet18 pre-trained, bỏ lớp FC cuối để lấy vector đặc trưng 512-D.
    """
    backbone = models.resnet18(pretrained=True)
    backbone.fc = torch.nn.Identity()  # output shape: [B, 512]
    backbone.to(device)
    backbone.eval()
    return backbone


def compute_embedding(roi_bgr, backbone, device):
    """
    Tính embedding deep cho ROI (BGR -> RGB -> resize 224 -> normalize -> ResNet).
    Trả về vector 512-D đã L2-normalize (numpy array).
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return None

    # BGR -> RGB
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    # Resize về 224x224
    roi_resized = cv2.resize(roi_rgb, (224, 224))

    # HWC -> CHW, [0,255] -> [0,1]
    tensor = torch.from_numpy(roi_resized).permute(2, 0, 1).float() / 255.0

    # Normalize theo ImageNet
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    tensor = tensor.unsqueeze(0).to(device)  # thêm batch dim

    with torch.no_grad():
        feat = backbone(tensor)  # [1, 512]
        feat = feat[0]           # [512]
        feat = feat / (feat.norm(p=2) + 1e-8)

    return feat.cpu().numpy()


# ===================== HÀM TIỆN ÍCH ===================== #
def safe_crop(frame, box):
    """
    Cắt ROI theo box (x1, y1, x2, y2) nhưng đảm bảo không vượt khỏi frame.
    Trả về None nếu box không hợp lệ.
    """
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def get_detections(results):
    """
    Chuyển output YOLO Results thành list bounding boxes:
    [(x1, y1, x2, y2, conf, cls_id), ...]
    """
    detections = []
    boxes = results.boxes

    if boxes is None:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    clss = boxes.cls.cpu().numpy()

    for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
        detections.append((
            int(x1), int(y1), int(x2), int(y2),
            float(conf), int(cls_id)
        ))

    return detections


# ===================== CHƯƠNG TRÌNH CHÍNH ===================== #
def main():
    # Thiết bị cho ResNet
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dùng device cho feature extractor:", device)

    # Load model YOLO
    model = YOLO(YOLO_WEIGHTS)

    # Tạo backbone trích đặc trưng
    backbone = build_feature_extractor(device)

    # Mở camera
    cap = cv2.VideoCapture(0)  # "0" là webcam mặc định
    if not cap.isOpened():
        print("Không mở được camera")
        return

    # Trạng thái tracking
    target_emb = None          # vector đặc trưng 512-D của mục tiêu
    target_box = None          # box hiện tại của mục tiêu (x1, y1, x2, y2)
    target_lost = True
    target_last_sim = 0.0
    frames_since_seen = 0

    window_name = "YOLOv8 + Deep Embedding Tracking - ABSOLUTE ID"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được frame từ camera")
            break

        frame_display = frame.copy()
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2  # tâm khung hình

        # Vẽ dấu + ở tâm để căn mục tiêu khi nhấn 's'
        cv2.drawMarker(
            frame_display,
            (cx, cy),
            (255, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=25,
            thickness=2
        )

        # 1. Chạy YOLO detect người (class 0)
        results = model(
            frame,
            conf=CONF_THRES,
            classes=[39]   # 0 = 'person' trong COCO
        )[0]
        detections = get_detections(results)

        # Vẽ tất cả detection (xám) để debug
        for x1, y1, x2, y2, conf, cls_id in detections:
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (100, 100, 100), 1)
            cv2.putText(
                frame_display,
                f"class:{cls_id} conf:{conf:.2f}",
                (x1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (100, 100, 100),
                1
            )

        # 2. Nếu đã có target_emb => tìm detection giống nhất bằng cosine similarity
        detect_match = False

        if target_emb is not None and len(detections) > 0:
            best_sim = -1.0
            best_box = None

            for x1, y1, x2, y2, conf, cls_id in detections:
                roi = safe_crop(frame, (x1, y1, x2, y2))
                if roi is None or roi.size == 0:
                    continue

                emb = compute_embedding(roi, backbone, device)
                if emb is None:
                    continue

                # Cosine similarity do target_emb đã L2-norm
                sim = float(np.dot(target_emb, emb))

                if sim > best_sim:
                    best_sim = sim
                    best_box = (x1, y1, x2, y2)

            if best_box is not None and best_sim > EMB_SIM_THRESHOLD:
                detect_match = True
                target_box = best_box
                target_lost = False
                target_last_sim = best_sim
                frames_since_seen = 0
            else:
                target_lost = True
                frames_since_seen += 1

        # 3. VẼ THÔNG TIN MỤC TIÊU
        if target_box is not None and not target_lost:
            x1, y1, x2, y2 = target_box

            # Khung xanh cho target
            cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # LABEL trên khung
            label_pos_y = y1 - 15 if y1 - 15 > 30 else y1 + 25
            cv2.putText(
                frame_display,
                f"TARGET_ID: {TARGET_ABS_ID}",
                (x1, label_pos_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

            # Thông tin phía trên
            info_text = f"TARGET_ID = {TARGET_ABS_ID} | sim = {target_last_sim:.2f}"
            cv2.putText(
                frame_display,
                info_text,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

            mode_text = "Nguon: YOLO + deep embedding"
            cv2.putText(
                frame_display,
                mode_text,
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        elif target_emb is not None:
            # Đã có mục tiêu nhưng mất tạm thời
            info_text = f"MAT MUC TIEU (TARGET_ID = {TARGET_ABS_ID}) - frames mat: {frames_since_seen}"
            cv2.putText(
                frame_display,
                info_text,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2
            )
            cv2.putText(
                frame_display,
                "Khi muc tieu quay lai, YOLO + embedding se nhan lai dung nguoi.",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )
        else:
            # Chưa chọn mục tiêu
            cv2.putText(
                frame_display,
                "Dat muc tieu gan dau '+' roi nhan 's' de KHOA, 'r' reset, 'q' thoat",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        cv2.imshow(window_name, frame_display)

        # 4. PHÍM BẤM
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('s'):
            # Khóa mục tiêu: chọn detection gần tâm khung nhất
            if len(detections) == 0:
                print("Khong co detection nao khi nhan 's' - hay dam bao co nguoi trong khung.")
            else:
                best_det = None
                best_dist = None

                for det in detections:
                    x1, y1, x2, y2, conf, cls_id = det
                    bx = (x1 + x2) / 2.0
                    by = (y1 + y2) / 2.0
                    dist2 = (bx - cx) ** 2 + (by - cy) ** 2

                    if best_dist is None or dist2 < best_dist:
                        best_dist = dist2
                        best_det = det

                if best_det is not None:
                    x1, y1, x2, y2, conf, cls_id = best_det
                    roi = safe_crop(frame, (x1, y1, x2, y2))
                    emb = compute_embedding(roi, backbone, device)
                    if emb is not None:
                        target_emb = emb
                        target_box = (x1, y1, x2, y2)
                        target_lost = False
                        target_last_sim = 1.0
                        frames_since_seen = 0
                        print(f"Da KHOA muc tieu gan tam khung, TARGET_ID = {TARGET_ABS_ID}.")

        if key == ord('r'):
            # Reset mục tiêu
            target_emb = None
            target_box = None
            target_lost = True
            target_last_sim = 0.0
            frames_since_seen = 0

            print("Da reset muc tieu, dat lai muc tieu gan tam va nhan 's'.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
