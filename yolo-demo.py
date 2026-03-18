"""
yolo-demo.py  —  Raspberry Pi main script
==========================================
Sends camera frames to the Roboflow Inference Server running on your Mac,
then drives the robot through a full state machine:

    SEARCHING → APPROACHING → COLLECTING → RETURNING_HOME → DROPPING → SEARCHING

Setup:
  1. Mac:  `inference server start`
  2. Pi:   set MAC_IP and ROBOFLOW_API_KEY in .env
  3. Pi:   `python yolo-demo.py`

Dependencies (Pi):
  pip install opencv-python requests python-dotenv RPi.GPIO lgpio
"""

import cv2
import os
import time
import requests
import lgpio
from dotenv import load_dotenv

import movement_v2
from sensor import beam_broken   # break-beam confirmation

# =============================================================
# 1.  ENVIRONMENT / SERVER CONFIG
# =============================================================

load_dotenv()

API_KEY    = os.getenv("ROBOFLOW_API_KEY")
MAC_IP     = os.getenv("MAC_IP")          # e.g. 10.0.0.42
PROJECT_ID = os.getenv("PROJECT_ID", "ball-dataset-merged")
VERSION    = os.getenv("MODEL_VERSION", "2")

if not API_KEY:
    raise RuntimeError("❌  ROBOFLOW_API_KEY not set in .env")
if not MAC_IP:
    raise RuntimeError("❌  MAC_IP not set in .env  (run `ifconfig` on your Mac to find it)")

MODEL_ID = f"{PROJECT_ID}/{VERSION}"
API_URL  = f"http://{MAC_IP}:9001/{MODEL_ID}"

print(f"🔗  Inference server: {API_URL}")


# =============================================================
# 2.  CAMERA
# =============================================================

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    raise RuntimeError("❌  Could not open camera")


# =============================================================
# 3.  ENCODER  (wheel tick counting for RETURN_HOME)
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
# 4.  ROBOT PARAMETERS  — tune these for your arena
# =============================================================

FRAME_WIDTH          = 640
FRAME_CENTER         = FRAME_WIDTH // 2   # 320 px

BASE_SPEED           = 80     # Drive speed 0-100
SEARCH_SPEED         = 40     # Rotation speed while scanning
Kp                   = 0.12   # Steering proportional gain

COLLECTION_THRESHOLD = 200    # Ball pixel-width that means "close enough to collect"
CENTER_TOLERANCE     = 60     # Pixels off-centre before we bother steering

BEAM_COLLECT_TIMEOUT = 3.0    # Seconds to wait for beam after stopping
MAX_CAPACITY         = 5      # Balls before returning home
RETURN_HOME_TIME_S   = 3      # Seconds of reverse to reach home

FRAME_SKIP           = 5      # Send 1 in every N frames to server (reduce lag)


# =============================================================
# 5.  DETECTION  (calls Mac inference server)
# =============================================================

_predictions    = []   # Latest predictions, persisted between frames
_frame_count    = 0
_collected_ids  = set()   # Track IDs already collected this run (by Roboflow tracker id)

def _query_server(frame):
    """Send frame to Mac, return list of prediction dicts (or [] on error)."""
    _, buffer = cv2.imencode('.jpg', frame)
    try:
        resp = requests.post(
            API_URL,
            params={"api_key": API_KEY, "confidence": 40},
            files={"file": ("image.jpg", buffer.tobytes(), "image/jpeg")},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json().get("predictions", [])
        else:
            print(f"⚠️   Server error {resp.status_code}")
    except requests.exceptions.Timeout:
        print("⚠️   Server timeout — using last frame's predictions")
    except Exception as e:
        print(f"⚠️   Request error: {e}")
    return None   # None = keep old predictions


def get_predictions():
    """
    Grab a camera frame, optionally query the server, return
    (frame, predictions).  Skips the server call on non-FRAME_SKIP frames
    to keep the loop fast.
    """
    global _predictions, _frame_count

    ret, frame = cap.read()
    if not ret:
        return None, []

    _frame_count += 1
    if _frame_count % FRAME_SKIP == 0:
        result = _query_server(frame)
        if result is not None:          # None means keep old list
            _predictions = result

    return frame, _predictions


def best_target(predictions):
    """
    From the server's prediction list, return the single best ball to chase:
    - filters out already-collected IDs (if server gives track_id)
    - picks the largest bounding box (= closest ball)

    Returns the prediction dict, or None if nothing worth chasing.
    """
    candidates = []
    for p in predictions:
        track_id = p.get("tracker_id", -1)
        if track_id in _collected_ids:
            continue
        candidates.append(p)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p["width"] * p["height"])


# =============================================================
# 6.  STATE MACHINE
# =============================================================

def run():
    global _collected_ids

    state           = "SEARCHING"
    active_track_id = -1
    ball_count      = 0
    at_home         = False
    dropped         = False

    print("🤖  Robot starting — press Ctrl+C to stop")

    try:
        while True:
            frame, predictions = get_predictions()
            if frame is None:
                time.sleep(0.05)
                continue

            target = best_target(predictions)

            # ── Helper values ──────────────────────────────────────
            ball_detected = target is not None
            if ball_detected:
                ball_cx   = target["x"]                          # centre x in pixels
                ball_w    = target["width"]                      # apparent width
                error     = ball_cx - FRAME_CENTER               # +ve = ball right of centre
                close     = ball_w >= COLLECTION_THRESHOLD       # large box = close ball
                track_id  = target.get("tracker_id", -1)

            # ── Draw boxes on frame (for debugging over SSH / display) ─
            for p in predictions:
                x = int(p["x"] - p["width"]  / 2)
                y = int(p["y"] - p["height"] / 2)
                w = int(p["width"])
                h = int(p["height"])
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f'{p.get("class","?")} {p.get("confidence",0):.2f}',
                            (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Ball Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # ──────────────────────────────────────────────────────
            #  STATE BEHAVIOURS
            # ──────────────────────────────────────────────────────

            if state == "SEARCHING":
                movement_v2.turn_left(SEARCH_SPEED)
                print("🔍  SEARCHING")

                if ball_detected:
                    print(f"👁️   Ball spotted (ID {track_id}) — APPROACHING")
                    active_track_id = track_id
                    state = "APPROACHING"

            # ───────────────────────────────────────────────────────
            elif state == "APPROACHING":
                if not ball_detected:
                    print("⚠️   Ball lost — back to SEARCHING")
                    movement_v2.stop_drive()
                    state = "SEARCHING"

                elif close:
                    print(f"📍  Close enough — COLLECTING (ID {track_id})")
                    active_track_id = track_id
                    movement_v2.stop_drive()
                    state = "COLLECTING"

                else:
                    # Proportional steering toward ball
                    turn_adj = Kp * error
                    print(f"🎯  APPROACHING | cx={ball_cx}px | w={ball_w}px | "
                          f"error={error:+d} | adj={turn_adj:+.1f}")

                    if turn_adj > 10:
                        movement_v2.turn_right(BASE_SPEED)
                    elif turn_adj < -10:
                        movement_v2.turn_left(BASE_SPEED)
                    else:
                        movement_v2.move_forward(BASE_SPEED)
                        movement_v2.start_spinner()   # spin ready to intake

            # ───────────────────────────────────────────────────────
            elif state == "COLLECTING":
                print("🤖  COLLECTING — waiting for beam...")
                movement_v2.stop_drive()
                movement_v2.start_spinner()

                deadline  = time.time() + BEAM_COLLECT_TIMEOUT
                collected = False

                while time.time() < deadline:
                    if beam_broken():
                        print("  ✅  Beam broken — ball collected!")
                        movement_v2.stop_spinner()
                        movement_v2.lift_up(speed=80)
                        time.sleep(1.0)
                        movement_v2.stop_lift()

                        if active_track_id != -1:
                            _collected_ids.add(active_track_id)

                        ball_count += 1
                        collected  = True
                        print(f"  🏆  Total: {ball_count}/{MAX_CAPACITY}")
                        break
                    time.sleep(0.05)

                if not collected:
                    print("  ⚠️   Beam timeout — missed, back to SEARCHING")

                state = "RETURNING_HOME" if ball_count >= MAX_CAPACITY else "SEARCHING"

            # ───────────────────────────────────────────────────────
            elif state == "RETURNING_HOME":
                print("🏠  RETURNING HOME")
                encoder.reset()
                movement_v2.move_backward(BASE_SPEED)
                time.sleep(RETURN_HOME_TIME_S)
                movement_v2.stop_drive()
                at_home = True
                state   = "DROPPING"

            # ───────────────────────────────────────────────────────
            elif state == "DROPPING":
                print("📦  DROPPING balls")
                movement_v2.lift_down(speed=80)
                time.sleep(1.5)
                movement_v2.stop_lift()

                _collected_ids.clear()
                ball_count = 0
                at_home    = False
                state      = "SEARCHING"

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑  Stopped by user")

    finally:
        movement_v2.shutdown()
        encoder.close()
        cap.release()
        cv2.destroyAllWindows()
        print("✅  Shutdown complete")


# =============================================================
# 7.  ENTRY POINT
# =============================================================

if __name__ == "__main__":
    run()