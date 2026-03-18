"""
yolo-demo.py  —  Raspberry Pi main script
==========================================
Runs your Roboflow-trained model locally on the Pi using the
inference package. No Mac or server needed.

State machine:
    SEARCHING -> APPROACHING -> COLLECTING -> RETURNING_HOME -> DROPPING -> SEARCHING

Setup:
  1. Fill in .env (API key, project ID, model version)
  2. pip install -r requirements.txt
  3. python yolo-demo.py
     - First run will download the model from Roboflow (~few seconds)
     - Subsequent runs use the cached model (no internet needed)
"""

import cv2
import time
import lgpio
import numpy as np
from dotenv import load_dotenv
import os
from inference import get_model

import movement_v2
from sensor import beam_broken

# init
# =============================================================
# 1.  MODEL + CAMERA
# =============================================================

load_dotenv()

API_KEY    = os.getenv("ROBOFLOW_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID", "ball-dataset-merged")
VERSION    = os.getenv("MODEL_VERSION", "2")

if not API_KEY:
    raise RuntimeError("ROBOFLOW_API_KEY not set in .env")

print("Loading model (downloads on first run, cached after)...")
model = get_model(model_id=f"{PROJECT_ID}/{VERSION}", api_key=API_KEY)
print("Model ready")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

# Known real-world diameters for distance estimation (metres)
KNOWN_DIAMETERS = {
    "ping_pong_ball": 0.040,
    "ball_bearing":   0.010,
}

# Camera matrix — replace with calibrated values if you have them
CAMERA_MATRIX = np.array([
    [800,   0, 320],
    [  0, 800, 240],
    [  0,   0,   1]
], dtype=float)


# =============================================================
# 2.  ENCODER
# =============================================================

class EncoderTracker:
    LEFT_A  = 17
    LEFT_B  = 18
    RIGHT_A = 27
    RIGHT_B = 22

    def __init__(self):
        self.left_count  = 0
        self.right_count = 0
        self.h = lgpio.gpiochip_open(0)
        for pin in [self.LEFT_A, self.LEFT_B, self.RIGHT_A, self.RIGHT_B]:
            lgpio.gpio_claim_input(self.h, pin)
        lgpio.callback(self.h, self.LEFT_A,  lgpio.RISING_EDGE, self._left_cb)
        lgpio.callback(self.h, self.RIGHT_A, lgpio.RISING_EDGE, self._right_cb)

    def _left_cb(self, chip, gpio, level, tick):
        b = lgpio.gpio_read(self.h, self.LEFT_B)
        self.left_count += 1 if b == 0 else -1

    def _right_cb(self, chip, gpio, level, tick):
        b = lgpio.gpio_read(self.h, self.RIGHT_B)
        self.right_count += 1 if b == 0 else -1

    def reset(self):
        self.left_count  = 0
        self.right_count = 0

    def close(self):
        lgpio.gpiochip_close(self.h)

encoder = EncoderTracker()


# =============================================================
# 3.  ROBOT PARAMETERS  — tune these for your arena
# =============================================================

FRAME_CENTER         = 320    # Half of 640px frame width

BASE_SPEED           = 80     # Drive speed 0-100
SEARCH_SPEED         = 40     # Rotation speed while scanning
Kp                   = 0.12   # Steering proportional gain

COLLECTION_THRESHOLD = 20     # Distance in cm — triggers collecting
CENTER_TOLERANCE     = 60     # Pixels off-centre before steering

BEAM_COLLECT_TIMEOUT = 3.0    # Seconds to wait for beam after stopping
MAX_CAPACITY         = 5      # Balls before returning home
RETURN_HOME_TIME_S   = 3      # Seconds of reverse to reach home base


# =============================================================
# 4.  DETECTION  (runs on-device via inference package)
# =============================================================

_collected_ids = set()
_track_counter = {}   # simple per-class tracking by position since
                      # inference SDK doesn't include a tracker by default


def get_detections():
    """
    Grabs a frame, runs inference locally, returns:
      - frame
      - list of detection dicts, each with:
          class, confidence, bbox, center_px, distance_cm, track_id
    """
    ret, frame = cap.read()
    if not ret:
        return None, []

    # inference SDK expects BGR numpy array (which is what cv2 gives us)
    results = model.infer(frame, confidence=0.5)[0]

    detections = []
    focal_length = CAMERA_MATRIX[0][0]

    for pred in results.predictions:
        # inference SDK prediction fields:
        # pred.x, pred.y  — centre of box
        # pred.width, pred.height — box size
        # pred.class_name, pred.confidence
        # pred.tracker_id — only set if you use a tracker (see below)

        track_id = getattr(pred, "tracker_id", -1) or -1

        if track_id in _collected_ids:
            continue

        cls_name = pred.class_name
        conf     = pred.confidence

        x1 = int(pred.x - pred.width  / 2)
        y1 = int(pred.y - pred.height / 2)
        x2 = int(pred.x + pred.width  / 2)
        y2 = int(pred.y + pred.height / 2)

        pixel_diameter = max(int(pred.width), int(pred.height))
        if pixel_diameter == 0:
            continue

        real_diameter = KNOWN_DIAMETERS.get(cls_name, 0.02)
        distance_cm   = (real_diameter * focal_length) / pixel_diameter * 100

        cx = int(pred.x)
        cy = int(pred.y)

        detections.append({
            "class":       cls_name,
            "confidence":  conf,
            "bbox":        (x1, y1, x2, y2),
            "center_px":   (cx, cy),
            "distance_cm": distance_cm,
            "track_id":    track_id,
        })

    return frame, detections


def best_target(detections):
    """Returns the closest uncollected ball, or None."""
    if not detections:
        return None
    return min(detections, key=lambda d: d["distance_cm"])


# =============================================================
# 5.  STATE MACHINE
# =============================================================

def run():
    global _collected_ids

    state           = "SEARCHING"
    active_track_id = -1
    ball_count      = 0

    print("Robot starting — press Ctrl+C to stop")

    try:
        while True:
            frame, detections = get_detections()
            if frame is None:
                time.sleep(0.05)
                continue

            target        = best_target(detections)
            ball_detected = target is not None

            if ball_detected:
                ball_cx     = target["center_px"][0]
                distance_cm = target["distance_cm"]
                error       = ball_cx - FRAME_CENTER
                turn_adj    = Kp * error
                close       = distance_cm < COLLECTION_THRESHOLD
                track_id    = target["track_id"]

            # -- Draw bounding boxes --
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame,
                            f'{d["class"]} {d["distance_cm"]:.0f}cm',
                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.imshow("Ball Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # --------------------------------------------------
            # STATES
            # --------------------------------------------------

            if state == "SEARCHING":
                movement_v2.turn_left(SEARCH_SPEED)
                print("SEARCHING")

                if ball_detected:
                    print(f"Ball spotted — APPROACHING")
                    active_track_id = track_id
                    state = "APPROACHING"

            elif state == "APPROACHING":
                if not ball_detected:
                    print("Ball lost — back to SEARCHING")
                    movement_v2.stop_drive()
                    state = "SEARCHING"

                elif close:
                    print(f"Close enough ({distance_cm:.1f}cm) — COLLECTING")
                    active_track_id = track_id
                    movement_v2.stop_drive()
                    state = "COLLECTING"

                else:
                    print(f"APPROACHING | dist={distance_cm:.1f}cm | error={error:+d}px")

                    if turn_adj > 10:
                        movement_v2.turn_right(BASE_SPEED)
                    elif turn_adj < -10:
                        movement_v2.turn_left(BASE_SPEED)
                    else:
                        movement_v2.move_forward(BASE_SPEED)
                        movement_v2.start_spinner()

            elif state == "COLLECTING":
                print("COLLECTING — waiting for beam...")
                movement_v2.stop_drive()
                movement_v2.start_spinner()

                deadline  = time.time() + BEAM_COLLECT_TIMEOUT
                collected = False

                while time.time() < deadline:
                    if beam_broken():
                        print("Beam broken — ball collected!")
                        movement_v2.stop_spinner()
                        movement_v2.lift_up(speed=80)
                        time.sleep(1.0)
                        movement_v2.stop_lift()

                        if active_track_id != -1:
                            _collected_ids.add(active_track_id)

                        ball_count += 1
                        collected  = True
                        print(f"Total: {ball_count}/{MAX_CAPACITY}")
                        break
                    time.sleep(0.05)

                if not collected:
                    print("Beam timeout — missed, back to SEARCHING")

                state = "RETURNING_HOME" if ball_count >= MAX_CAPACITY else "SEARCHING"

            elif state == "RETURNING_HOME":
                print("RETURNING HOME")
                encoder.reset()
                movement_v2.move_backward(BASE_SPEED)
                time.sleep(RETURN_HOME_TIME_S)
                movement_v2.stop_drive()
                state = "DROPPING"

            elif state == "DROPPING":
                print("DROPPING balls")
                movement_v2.lift_down(speed=80)
                time.sleep(1.5)
                movement_v2.stop_lift()

                _collected_ids.clear()
                ball_count = 0
                state      = "SEARCHING"

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        movement_v2.shutdown()
        encoder.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Shutdown complete")



# =============================================================
# 6.  ENTRY POINT
# =============================================================

if __name__ == "__main__":
    run()