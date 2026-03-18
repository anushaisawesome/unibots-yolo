"""
main_robot.py
=============
Integrates:
  - Localisation.py   (BallDetector - YOLO vision)
  - movement_v2.py    (Motor control - GPIO motors)
  - behaviour.py      (State logic + PID steering)
  - state_machine.py  (State transitions)
  - sensor.py         (Break-beam collector confirmation)
  - yolo-demo.py      (model.track() with persistent ball IDs)

Run this single file to operate the robot.

⚠️  PIN CONFLICT RESOLVED:
    sensor.py originally used GPIO 17 for the beam.
    EncoderTracker uses GPIO 17 for the left encoder channel A.
    Fix: wire your beam receiver OUT pin to GPIO 25 instead.
    Change BEAM_PIN below if you use a different pin.
"""

import time
import cv2
import numpy as np
import lgpio
import RPi.GPIO as GPIO

import movement_v2
from Localisation import BallDetector


# =============================================================
# 1. CAMERA & MODEL SETUP
#    Uses model.track() from yolo-demo.py so each ball gets
#    a persistent ID — prevents re-targeting already-collected balls
# =============================================================

CAMERA_MATRIX = np.array([
    [800,   0, 320],
    [  0, 800, 240],
    [  0,   0,   1]
], dtype=float)
# ☝️ Replace with your real calibrated values from cv2.calibrateCamera()

DIST_COEFFS = np.zeros(5)

MODEL_PATH = "runs/detect/train/weights/best.pt"  # From yolo-demo.py training output

detector = BallDetector(MODEL_PATH, CAMERA_MATRIX, DIST_COEFFS)

cap = cv2.VideoCapture(0)
cap.set(3, 640)   # Width  — from yolo-demo.py
cap.set(4, 480)   # Height — from yolo-demo.py


# =============================================================
# 2. BREAK-BEAM SENSOR SETUP (from sensor.py)
#    Physically confirms a ball has entered the collector.
#    More reliable than distance alone — ball could be close
#    but not actually collected (e.g. robot stopped short).
#
#    ⚠️  Wire your beam receiver OUT to GPIO 25, NOT GPIO 17
#        (17 is already used by the left wheel encoder)
# =============================================================

BEAM_PIN = 25   # GPIO pin for break-beam receiver OUT

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(BEAM_PIN, GPIO.IN)

def beam_broken():
    """
    Returns True if beam is broken (ball is passing through collector).
    Returns False if beam is intact (nothing in collector).
    Directly from sensor.py logic.
    """
    return GPIO.input(BEAM_PIN) == GPIO.LOW


# =============================================================
# 3. ENCODER SETUP (from behaviour.py)
#    Tracks wheel ticks to measure distance travelled.
#    Used for accurate RETURN_HOME navigation.
# =============================================================

class EncoderTracker:
    LEFT_A  = 17   # <- This is why beam moved off pin 17
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

    def average_ticks(self):
        return (self.left_count + self.right_count) / 2

    def close(self):
        lgpio.gpiochip_close(self.h)

encoder = EncoderTracker()


# =============================================================
# 4. ROBOT PARAMETERS
# =============================================================

FRAME_CENTER         = 320   # Horizontal centre of 640px wide frame
BASE_SPEED           = 80    # Forward drive speed (0-100)
SEARCH_SPEED         = 40    # Rotation speed while scanning

Kp                   = 0.12  # Steering correction strength

COLLECTION_THRESHOLD = 20    # cm — triggers COLLECTING state
BEAM_COLLECT_TIMEOUT = 3.0   # Seconds to wait for beam to trigger before giving up
MAX_CAPACITY         = 5     # Balls before returning home
RETURN_HOME_TIME_S   = 3     # Seconds of reverse to reach home


# =============================================================
# 5. TRACKING STATE
#    Keeps track of ball IDs already collected so the robot
#    doesn't try to pick up the same ball twice.
#    Ball IDs come from model.track() in yolo-demo.py.
# =============================================================

ball_count       = 0
collected_ids    = set()   # Set of YOLO track IDs already collected this run
at_home          = False
dropped          = False


# =============================================================
# 6. VISION — using model.track() (from yolo-demo.py)
#    track() assigns each detected ball a persistent numeric ID
#    that stays the same across frames, even if the ball moves.
#    This replaces the plain model() call from Localisation.py.
# =============================================================

def detect_balls_tracked():
    """
    Runs YOLO tracking on the latest camera frame.
    Returns a list of detection dicts, each including a 'track_id'.
    Filters out any ball IDs that have already been collected.

    Each dict contains:
        class       : "ping_pong_ball" or "ball_bearing"
        confidence  : float 0-1
        bbox        : (x1, y1, x2, y2) pixels
        center_px   : (cx, cy) pixels
        position_3d : (x, y, distance_metres)
        track_id    : int — persistent ID from YOLO tracker
    """
    ret, frame = cap.read()
    if not ret:
        return [], None

    # model.track() instead of model() — key change from yolo-demo.py
    # persist=True tells the tracker to remember IDs between frames
    results = detector.model.track(frame, persist=True, conf=0.5)

    detections = []

    if results[0].boxes is None:
        return [], frame

    for box in results[0].boxes:
        # track_id is None if tracker hasn't assigned one yet
        track_id = int(box.id) if box.id is not None else -1

        # Skip balls we've already collected this run
        if track_id in collected_ids:
            continue

        cls_id   = int(box.cls)
        cls_name = detector.model.names[cls_id]
        conf     = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Distance estimate from apparent size (from Localisation.py)
        pixel_diameter = max(x2 - x1, y2 - y1)
        real_diameter  = detector.KNOWN_DIAMETERS.get(cls_name, 0.02)
        focal_length   = CAMERA_MATRIX[0][0]

        if pixel_diameter == 0:
            continue   # Avoid divide-by-zero for very small/corrupt detections

        distance_m = (real_diameter * focal_length) / pixel_diameter

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        x_3d = (cx - CAMERA_MATRIX[0][2]) * distance_m / focal_length
        y_3d = (cy - CAMERA_MATRIX[1][2]) * distance_m / focal_length

        detections.append({
            "class":       cls_name,
            "confidence":  conf,
            "bbox":        (x1, y1, x2, y2),
            "center_px":   (int(cx), int(cy)),
            "position_3d": (x_3d, y_3d, distance_m),
            "track_id":    track_id,
        })

    return detections, frame


def detect_ball():
    """
    Convenience wrapper — returns the single closest uncollected ball.

    Returns:
        ball_detected (bool)
        ball_x        (int px)
        distance_cm   (float)
        track_id      (int)
    """
    detections, _ = detect_balls_tracked()

    if not detections:
        return False, 0, 0, -1

    closest     = min(detections, key=lambda d: d["position_3d"][2])
    ball_x      = closest["center_px"][0]
    distance_cm = closest["position_3d"][2] * 100
    track_id    = closest["track_id"]

    return True, ball_x, distance_cm, track_id


# =============================================================
# 7. BEHAVIOUR FUNCTIONS
# =============================================================

def search_behavior():
    """
    SEARCH: No ball visible — rotate slowly to scan.
    """
    print("🔍 SEARCHING — rotating to scan")
    movement_v2.turn_left(SEARCH_SPEED)


def approach_behavior():
    """
    APPROACH: Ball visible — steer towards it using PID correction.
    Spinner runs while driving so it's ready to collect on arrival.
    """
    ball_detected, ball_x, distance, track_id = detect_ball()

    if not ball_detected:
        movement_v2.stop_drive()
        return

    error           = ball_x - FRAME_CENTER
    turn_adjustment = Kp * error

    print(f"  🎯 APPROACH | ball_x={ball_x}px | dist={distance:.1f}cm | "
          f"error={error:+d} | track_id={track_id}")

    if turn_adjustment > 10:
        movement_v2.turn_right(BASE_SPEED)

    elif turn_adjustment < -10:
        movement_v2.turn_left(BASE_SPEED)

    else:
        # Ball is roughly centred — drive forward with spinner ready
        movement_v2.move_forward(BASE_SPEED)
        movement_v2.start_spinner()


def collect_behavior(current_track_id):
    """
    COLLECT: Robot is close enough — wait for beam confirmation.

    Uses the break-beam sensor (sensor.py) to physically verify
    the ball has entered the collector, rather than just trusting distance.

    Returns True if beam confirmed collection, False if timed out.
    """
    print("🤖 COLLECTING — waiting for beam confirmation...")

    movement_v2.stop_drive()
    movement_v2.start_spinner()   # Keep spinning to pull ball in

    deadline = time.time() + BEAM_COLLECT_TIMEOUT

    while time.time() < deadline:
        if beam_broken():
            # Ball physically confirmed inside collector
            print("  ✅ Beam broken — ball collected!")
            movement_v2.stop_spinner()
            movement_v2.lift_up(speed=80)
            time.sleep(1.0)
            movement_v2.stop_lift()

            # Mark this ball's ID so we never target it again this run
            if current_track_id != -1:
                collected_ids.add(current_track_id)

            return True

        time.sleep(0.05)

    # Beam never triggered — ball wasn't actually collected
    print("  ⚠️  Beam timeout — ball may have been missed")
    movement_v2.stop_spinner()
    return False


def return_home_behavior():
    """
    RETURN HOME: Capacity full — drive back to deposit zone.
    """
    print("🏠 RETURNING HOME")
    encoder.reset()
    movement_v2.move_backward(BASE_SPEED)
    time.sleep(RETURN_HOME_TIME_S)
    movement_v2.stop_drive()


def drop_behavior():
    """
    DROP: Lower lift to deposit all collected balls at home base.
    Clears collected_ids so robot can collect any remaining field balls.
    """
    print("📦 DROPPING balls at home")
    movement_v2.lift_down(speed=80)
    time.sleep(1.5)
    movement_v2.stop_lift()

    # Clear IDs — robot starts fresh after deposit
    collected_ids.clear()


# =============================================================
# 8. OPTIONAL: LIVE ANNOTATED DISPLAY (from yolo-demo.py)
#    Shows bounding boxes and track IDs on screen.
#    Set SHOW_DISPLAY = False on a headless robot with no monitor.
# =============================================================

SHOW_DISPLAY = True

def show_annotated_frame():
    """
    Grabs a frame, runs tracking, and displays annotated output.
    Mirrors the cv2.imshow loop from yolo-demo.py.
    """
    ret, frame = cap.read()
    if not ret:
        return

    results = detector.model.track(frame, persist=True, conf=0.5)
    annotated = results[0].plot()   # Draws boxes + track IDs on frame
    cv2.imshow("Ball Tracking", annotated)
    cv2.waitKey(1)


# =============================================================
# 9. STATE MACHINE MAIN LOOP
# =============================================================

def run():
    global ball_count, at_home, dropped

    state           = 'SEARCHING'
    active_track_id = -1   # Track ID of the ball currently being targeted

    print("🤖 Robot starting — press Ctrl+C to stop")

    try:
        while True:

            # --- Run current state behaviour ---
            if state == 'SEARCHING':
                search_behavior()

            elif state == 'APPROACHING':
                approach_behavior()

            elif state == 'COLLECTING':
                success = collect_behavior(active_track_id)
                if success:
                    ball_count += 1
                    print(f"  🏆 Total collected: {ball_count}/{MAX_CAPACITY}")
                else:
                    # Missed — go back to search
                    state = 'SEARCHING'
                    continue

            elif state == 'RETURNING_HOME':
                return_home_behavior()
                at_home = True

            elif state == 'DROPPING':
                drop_behavior()
                ball_count = 0
                dropped    = True
                at_home    = False

            # --- Optional live display ---
            if SHOW_DISPLAY:
                show_annotated_frame()

            # --- Sense current situation for transitions ---
            ball_detected, ball_x, distance, track_id = detect_ball()
            close_to_ball = ball_detected and distance < COLLECTION_THRESHOLD

            # --- State transitions ---
            if state == 'SEARCHING' and ball_detected:
                print(f"👁️  Ball spotted (ID {track_id}) — APPROACHING")
                active_track_id = track_id
                state = 'APPROACHING'

            elif state == 'APPROACHING' and not ball_detected:
                print("⚠️  Ball lost — back to SEARCHING")
                state = 'SEARCHING'

            elif state == 'APPROACHING' and close_to_ball:
                print(f"📍 Close enough — COLLECTING (ID {track_id})")
                active_track_id = track_id
                state = 'COLLECTING'

            elif state == 'COLLECTING':
                if ball_count >= MAX_CAPACITY:
                    print(f"🎒 Full ({ball_count}) — RETURNING HOME")
                    state = 'RETURNING_HOME'
                else:
                    state = 'SEARCHING'

            elif state == 'RETURNING_HOME' and at_home:
                state = 'DROPPING'

            elif state == 'DROPPING' and dropped:
                dropped = False
                state   = 'SEARCHING'

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")

    finally:
        movement_v2.shutdown()
        encoder.close()
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        print("✅ Shutdown complete")


# =============================================================
# 10. ENTRY POINT
# =============================================================

if __name__ == "__main__":
    run()