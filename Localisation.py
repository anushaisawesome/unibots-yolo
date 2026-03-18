import cv2
import numpy as np
from ultralytics import YOLO

class BallDetector:
    def __init__(self, model_path, camera_matrix, dist_coeffs):
        self.model = YOLO(model_path)
        self.camera_matrix = camera_matrix  # From camera calibration
        self.dist_coeffs = dist_coeffs
        self.KNOWN_DIAMETERS = {
            "ping_pong_ball": 0.040,   # 40mm
            "ball_bearing":   0.010,   # ~10mm (adjust to your bearing size)
        }

    def detect(self, frame):
        results = self.model(frame, conf=0.5)[0]
        detections = []
        
        for box in results.boxes:
            cls_id = int(box.cls)
            cls_name = self.model.names[cls_id]
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Estimate distance using apparent size
            pixel_diameter = max(x2 - x1, y2 - y1)
            real_diameter = self.KNOWN_DIAMETERS.get(cls_name, 0.02)
            focal_length = self.camera_matrix[0][0]
            distance = (real_diameter * focal_length) / pixel_diameter
            
            # 3D position relative to camera
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            x_3d = (cx - self.camera_matrix[0][2]) * distance / focal_length
            y_3d = (cy - self.camera_matrix[1][2]) * distance / focal_length
            
            detections.append({
                "class": cls_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "position_3d": (x_3d, y_3d, distance),
                "center_px": (int(cx), int(cy))
            })
        
        return detections