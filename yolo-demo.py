"""
yolo_demo_debug.py
==================
Same logic as yolo-demo.py but with thorough debug print lines
so you can verify every function is being called correctly.

HOW TO RUN:
-----------
1. Create a .env file in the same folder with:
       ROBOFLOW_API_KEY=your_key_here
       PROJECT_ID=ball-dataset-merged
       MODEL_VERSION=2

2. Install dependencies (once):
       pip install inference roboflow opencv-python python-dotenv lgpio

3. Run:
       python yolo_demo_debug.py

4. Watch the terminal output — each block prints what it's doing
   and what values it's working with.

5. Press Q in the camera window, or Ctrl+C in terminal, to stop.

DEBUG LEVELS (set at bottom of this file):
   DEBUG_VISION    = True   — prints every detection result
   DEBUG_STEERING  = True   — prints steering calculations
   DEBUG_BEAM      = True   — prints beam sensor reads
   DEBUG_ENCODER   = True   — prints encoder tick counts
   DEBUG_STATE     = True   — prints every state transition
"""

import cv2
import time
import numpy as np
from dotenv import load_dotenv
import os
from inference import get_model

import movement_v2
from sensor import beam_broken

# Try to import lgpio (only available on Raspberry Pi)
try:
    import lgpio
    LGPIO_AVAILABLE = True
except ImportError:
    print("⚠️  lgpio not available (expected on Mac, install on Pi with: pip install lgpio)")
    LGPIO_AVAILABLE = False
    # Mock lgpio for development
    class MockLgpio:
        def gpiochip_open(self, chip): return None
        def gpio_claim_input(self, h, pin): pass
        def callback(self, h, gpio, edge, func): pass
        def gpio_read(self, h, pin): return 0
        def gpiochip_close(self, h): pass
        RISING_EDGE = 1
    lgpio = MockLgpio()


# =============================================================
# DEBUG FLAGS — set False to silence sections you don't need
# =============================================================

DEBUG_VISION   = True   # Detection results and distance estimates
DEBUG_STEERING = True   # PID error and turn decisions
DEBUG_BEAM     = True   # Beam sensor reads during collection
DEBUG_ENCODER  = True   # Encoder tick counts
DEBUG_STATE    = True   # State transitions


def dbg(section, msg):
    """Consistent debug print with section label."""
    print(f"[{section}] {msg}")


# =============================================================
# 1.  MODEL + CAMERA
# =============================================================

print("=" * 60)
print("STARTUP — loading environment variables")
load_dotenv()

API_KEY    = os.getenv("ROBOFLOW_API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID", "ball-dataset-merged")
VERSION    = os.getenv("MODEL_VERSION", "2")

# Verify env vars loaded correctly
print(f"  PROJECT_ID   : {PROJECT_ID}")
print(f"  MODEL_VERSION: {VERSION}")
print(f"  API_KEY      : {'SET ✓' if API_KEY else 'MISSING ✗ — check your .env file'}")

if not API_KEY:
    raise RuntimeError("ROBOFLOW_API_KEY not set in .env")

print("\nLoading model — downloading on first run, cached after...")
model = get_model(model_id=f"{PROJECT_ID}/{VERSION}", api_key=API_KEY)
print(f"Model loaded: {PROJECT_ID}/{VERSION} ✓")

print("\nOpening camera...")
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

if not cap.isOpened():
    raise RuntimeError("Could not open camera — check it is connected")
else:
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Camera opened ✓ — resolution: {actual_w:.0f}x{actual_h:.0f}")

KNOWN_DIAMETERS = {
    "ping_pong_ball": 0.040,
    "ball_bearing":   0.010,
}

CAMERA_MATRIX = np.array([
    [800,   0, 320],
    [  0, 800, 240],
    [  0,   0,   1]
], dtype=float)

print(f"Known diameters: {KNOWN_DIAMETERS}")
print(f"Focal length (from camera matrix): {CAMERA_MATRIX[0][0]}px")
print("=" * 60)


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
        print(f"\n[ENCODER] Initialising on pins: "
              f"LEFT_A={self.LEFT_A}, LEFT_B={self.LEFT_B}, "
              f"RIGHT_A={self.RIGHT_A}, RIGHT_B={self.RIGHT_B}")

        self.h = lgpio.gpiochip_open(0)
        print("[ENCODER] GPIO chip opened ✓")

        for pin in [self.LEFT_A, self.LEFT_B, self.RIGHT_A, self.RIGHT_B]:
            lgpio.gpio_claim_input(self.h, pin)
            print(f"[ENCODER]   Pin {pin} claimed as input ✓")

        lgpio.callback(self.h, self.LEFT_A,  lgpio.RISING_EDGE, self._left_cb)
        lgpio.callback(self.h, self.RIGHT_A, lgpio.RISING_EDGE, self._right_cb)
        print("[ENCODER] Callbacks attached ✓")

    def _left_cb(self, chip, gpio, level, tick):
        b = lgpio.gpio_read(self.h, self.LEFT_B)
        self.left_count += 1 if b == 0 else -1
        if DEBUG_ENCODER:
            # Only print every 10 ticks to avoid flooding terminal
            if self.left_count % 10 == 0:
                dbg("ENCODER", f"Left ticks: {self.left_count}")

    def _right_cb(self, chip, gpio, level, tick):
        b = lgpio.gpio_read(self.h, self.RIGHT_B)
        self.right_count += 1 if b == 0 else -1
        if DEBUG_ENCODER:
            if self.right_count % 10 == 0:
                dbg("ENCODER", f"Right ticks: {self.right_count}")

    def reset(self):
        self.left_count  = 0
        self.right_count = 0
        if DEBUG_ENCODER:
            dbg("ENCODER", "Tick counts reset to 0")

    def close(self):
        lgpio.gpiochip_close(self.h)
        print("[ENCODER] GPIO chip closed ✓")

encoder = EncoderTracker()


# =============================================================
# 3.  ROBOT PARAMETERS
# =============================================================

FRAME_CENTER         = 320
BASE_SPEED           = 80
SEARCH_SPEED         = 40
Kp                   = 0.12
COLLECTION_THRESHOLD = 20
CENTER_TOLERANCE     = 60
BEAM_COLLECT_TIMEOUT = 3.0
MAX_CAPACITY         = 5
RETURN_HOME_TIME_S   = 3

print("\n[PARAMS] Robot parameters loaded:")
print(f"  FRAME_CENTER        = {FRAME_CENTER}px")
print(f"  BASE_SPEED          = {BASE_SPEED}")
print(f"  SEARCH_SPEED        = {SEARCH_SPEED}")
print(f"  Kp (steering gain)  = {Kp}")
print(f"  COLLECTION_THRESHOLD= {COLLECTION_THRESHOLD}cm")
print(f"  BEAM_COLLECT_TIMEOUT= {BEAM_COLLECT_TIMEOUT}s")
print(f"  MAX_CAPACITY        = {MAX_CAPACITY} balls")
print(f"  RETURN_HOME_TIME_S  = {RETURN_HOME_TIME_S}s")


# =============================================================
# 4.  DETECTION
# =============================================================

_collected_ids = set()

def get_detections():
    """
    Grabs a frame, runs inference, returns (frame, list of detections).
    Each detection dict: class, confidence, bbox, center_px, distance_cm, track_id
    """
    ret, frame = cap.read()

    if not ret:
        dbg("VISION", "❌ cap.read() failed — camera may have disconnected")
        return None, []

    if DEBUG_VISION:
        dbg("VISION", "Frame captured — running inference...")

    # model.infer() sends frame to local inference engine
    # Returns list of response objects, one per image in the batch
    # We only sent one frame so we take [0]
    results = model.infer(frame, confidence=0.5)[0]

    if DEBUG_VISION:
        raw_count = len(results.predictions)
        dbg("VISION", f"Raw predictions from model: {raw_count}")

    detections   = []
    focal_length = CAMERA_MATRIX[0][0]

    for i, pred in enumerate(results.predictions):
        track_id = getattr(pred, "tracker_id", -1) or -1

        if DEBUG_VISION:
            dbg("VISION", f"  Pred [{i}]: class='{pred.class_name}' "
                          f"conf={pred.confidence:.2f} "
                          f"x={pred.x:.0f} y={pred.y:.0f} "
                          f"w={pred.width:.0f} h={pred.height:.0f} "
                          f"track_id={track_id}")

        # Skip already-collected balls
        if track_id in _collected_ids:
            if DEBUG_VISION:
                dbg("VISION", f"    ↳ Skipping ID {track_id} (already collected)")
            continue

        cls_name = pred.class_name
        conf     = pred.confidence

        # Convert centre+size format to corner format
        x1 = int(pred.x - pred.width  / 2)
        y1 = int(pred.y - pred.height / 2)
        x2 = int(pred.x + pred.width  / 2)
        y2 = int(pred.y + pred.height / 2)

        pixel_diameter = max(int(pred.width), int(pred.height))

        if pixel_diameter == 0:
            dbg("VISION", f"    ↳ Skipping {cls_name} — zero pixel diameter")
            continue

        real_diameter = KNOWN_DIAMETERS.get(cls_name, 0.02)

        # Distance formula: dist = (real_size * focal_length) / pixel_size
        distance_cm = (real_diameter * focal_length) / pixel_diameter * 100

        if DEBUG_VISION:
            dbg("VISION", f"    ↳ pixel_diam={pixel_diameter}px | "
                          f"real_diam={real_diameter*100:.0f}mm | "
                          f"distance={distance_cm:.1f}cm")

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

    if DEBUG_VISION:
        dbg("VISION", f"Returning {len(detections)} valid (uncollected) detections")

    return frame, detections


def best_target(detections):
    """Returns the closest uncollected ball, or None."""
    if not detections:
        return None
    target = min(detections, key=lambda d: d["distance_cm"])
    if DEBUG_VISION:
        dbg("VISION", f"Best target: '{target['class']}' "
                      f"at {target['distance_cm']:.1f}cm "
                      f"(track_id={target['track_id']})")
    return target


# =============================================================
# 5.  STATE MACHINE
# =============================================================

def run():
    global _collected_ids

    state           = "SEARCHING"
    active_track_id = -1
    ball_count      = 0

    print("\n" + "=" * 60)
    print("STATE MACHINE STARTING")
    print("  Press Q in camera window or Ctrl+C in terminal to stop")
    print("=" * 60 + "\n")

    loop_count = 0

    try:
        while True:
            loop_count += 1

            # Print a loop heartbeat every 50 iterations (~2.5 seconds)
            # so you can confirm the loop is running even when quiet
            if loop_count % 50 == 0:
                print(f"[LOOP] Iteration {loop_count} | state={state} | "
                      f"balls={ball_count}/{MAX_CAPACITY} | "
                      f"collected_ids={_collected_ids}")

            # --- Get latest camera frame and detections ---
            frame, detections = get_detections()

            if frame is None:
                dbg("LOOP", "No frame — skipping this iteration")
                time.sleep(0.05)
                continue

            target        = best_target(detections)
            ball_detected = target is not None

            # Pre-compute values we'll use across states
            if ball_detected:
                ball_cx     = target["center_px"][0]
                distance_cm = target["distance_cm"]
                error       = ball_cx - FRAME_CENTER
                turn_adj    = Kp * error
                close       = distance_cm < COLLECTION_THRESHOLD
                track_id    = target["track_id"]

            # --- Draw bounding boxes on the display frame ---
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                colour = (0, 255, 0) if d["track_id"] not in _collected_ids else (128, 128, 128)
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                label = f'{d["class"]} {d["distance_cm"]:.0f}cm id={d["track_id"]}'
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

            # Show current state on screen too
            cv2.putText(frame, f"State: {state}  Balls: {ball_count}/{MAX_CAPACITY}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Ball Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[LOOP] Q pressed — exiting")
                break

            # ==============================================
            # SEARCHING
            # ==============================================
            if state == "SEARCHING":
                if DEBUG_STATE:
                    dbg("STATE", "SEARCHING — calling movement_v2.turn_left("
                               f"{SEARCH_SPEED})")
                movement_v2.turn_left(SEARCH_SPEED)

                if ball_detected:
                    if DEBUG_STATE:
                        dbg("STATE", f"Ball spotted! class='{target['class']}' "
                                     f"dist={distance_cm:.1f}cm "
                                     f"track_id={track_id} "
                                     f"→ switching to APPROACHING")
                    active_track_id = track_id
                    state = "APPROACHING"

            # ==============================================
            # APPROACHING
            # ==============================================
            elif state == "APPROACHING":

                if not ball_detected:
                    dbg("STATE", "Ball lost mid-approach → SEARCHING")
                    movement_v2.stop_drive()
                    state = "SEARCHING"

                elif close:
                    dbg("STATE", f"Close enough! dist={distance_cm:.1f}cm "
                                 f"(threshold={COLLECTION_THRESHOLD}cm) → COLLECTING")
                    active_track_id = track_id
                    movement_v2.stop_drive()
                    state = "COLLECTING"

                else:
                    # Steering logic
                    if DEBUG_STEERING:
                        dbg("STEER", f"ball_cx={ball_cx}px | centre={FRAME_CENTER}px | "
                                     f"error={error:+d}px | Kp={Kp} | "
                                     f"turn_adj={turn_adj:+.1f} | dist={distance_cm:.1f}cm")

                    if turn_adj > 10:
                        if DEBUG_STEERING:
                            dbg("STEER", f"turn_adj={turn_adj:+.1f} > 10 → "
                                         f"turn_right({BASE_SPEED})")
                        movement_v2.turn_right(BASE_SPEED)

                    elif turn_adj < -10:
                        if DEBUG_STEERING:
                            dbg("STEER", f"turn_adj={turn_adj:+.1f} < -10 → "
                                         f"turn_left({BASE_SPEED})")
                        movement_v2.turn_left(BASE_SPEED)

                    else:
                        if DEBUG_STEERING:
                            dbg("STEER", f"turn_adj={turn_adj:+.1f} within ±10 → "
                                         f"move_forward({BASE_SPEED}) + start_spinner()")
                        movement_v2.move_forward(BASE_SPEED)
                        movement_v2.start_spinner()

            # ==============================================
            # COLLECTING
            # ==============================================
            elif state == "COLLECTING":
                dbg("STATE", f"COLLECTING — target track_id={active_track_id} | "
                             f"stopping drive, starting spinner")

                movement_v2.stop_drive()
                movement_v2.start_spinner()

                deadline  = time.time() + BEAM_COLLECT_TIMEOUT
                collected = False

                dbg("STATE", f"Waiting up to {BEAM_COLLECT_TIMEOUT}s for beam to break...")

                while time.time() < deadline:
                    beam_status = beam_broken()

                    #if DEBUG_BEAM:
                        #dbg("BEAM", f"beam_broken() returned: {beam_status}")

                    if beam_status:
                        dbg("BEAM", "✅ Beam BROKEN — ball confirmed in collector!")
                        dbg("STATE", f"Calling stop_spinner()")
                        movement_v2.stop_spinner()

                        dbg("STATE", f"Calling lift_up(speed=80)")
                        movement_v2.lift_up(speed=80)
                        time.sleep(1.0)

                        dbg("STATE", "Calling stop_lift()")
                        movement_v2.stop_lift()

                        if active_track_id != -1:
                            _collected_ids.add(active_track_id)
                            dbg("STATE", f"Added track_id={active_track_id} to "
                                         f"collected_ids={_collected_ids}")

                        ball_count += 1
                        collected   = True
                        dbg("STATE", f"ball_count incremented → {ball_count}/{MAX_CAPACITY}")
                        break

                    time.sleep(0.05)

                if not collected:
                    dbg("BEAM", f"⚠️  Beam timeout after {BEAM_COLLECT_TIMEOUT}s "
                                f"— ball not confirmed, returning to SEARCHING")
                    movement_v2.stop_spinner()

                # Decide next state
                if ball_count >= MAX_CAPACITY:
                    dbg("STATE", f"Capacity reached ({ball_count}) → RETURNING_HOME")
                    state = "RETURNING_HOME"
                else:
                    dbg("STATE", f"Capacity not reached ({ball_count}/{MAX_CAPACITY}) "
                                 f"→ SEARCHING")
                    state = "SEARCHING"

            # ==============================================
            # RETURNING HOME
            # ==============================================
            elif state == "RETURNING_HOME":
                dbg("STATE", f"RETURNING_HOME — calling move_backward({BASE_SPEED}) "
                             f"for {RETURN_HOME_TIME_S}s")
                encoder.reset()
                movement_v2.move_backward(BASE_SPEED)
                time.sleep(RETURN_HOME_TIME_S)

                dbg("STATE", f"Encoder ticks after return: "
                             f"left={encoder.left_count}, right={encoder.right_count}")
                movement_v2.stop_drive()
                dbg("STATE", "Arrived home → DROPPING")
                state = "DROPPING"

            # ==============================================
            # DROPPING
            # ==============================================
            elif state == "DROPPING":
                dbg("STATE", "DROPPING — calling lift_down(speed=80)")
                movement_v2.lift_down(speed=80)
                time.sleep(1.5)
                movement_v2.stop_lift()
                dbg("STATE", "Lift lowered — balls deposited")

                _collected_ids.clear()
                ball_count = 0
                dbg("STATE", f"collected_ids cleared, ball_count reset → SEARCHING")
                state = "SEARCHING"

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[LOOP] Ctrl+C — shutting down cleanly")

    finally:
        print("[SHUTDOWN] Stopping motors...")
        movement_v2.shutdown()
        print("[SHUTDOWN] Closing encoder GPIO...")
        encoder.close()
        print("[SHUTDOWN] Releasing camera...")
        cap.release()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Complete ✓")


# =============================================================
# 6.  ENTRY POINT
# =============================================================

if __name__ == "__main__":
    run()
