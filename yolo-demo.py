"""
yolo_demo_debug.py
==================
Same logic as yolo-demo.py but with thorough debug print lines
so you can verify every function is being called correctly.

BEHAVIOUR:
  - Collects exactly 4 balls, then stops looking for more
  - Drives straight home after 4th ball (timed reverse, no encoders)
  - Only raises and lowers lift AFTER arriving home to deposit balls
  - Lift is NOT raised during collection — spinner pulls balls in,
    beam sensor confirms each one

HOW TO RUN:
-----------
1. Create a .env file in the same folder with:
       ROBOFLOW_API_KEY=your_key_here
       PROJECT_ID=ball-dataset-merged
       MODEL_VERSION=2

2. Install dependencies (once):
       pip install inference roboflow opencv-python python-dotenv

3. Run:
       python yolo_demo_debug.py

4. Watch the terminal output — each block prints what it's doing
   and what values it's working with.

5. Press Q in the camera window, or Ctrl+C in terminal, to stop.

DEBUG FLAGS (near top of file):
   DEBUG_VISION    = True   — prints every detection result
   DEBUG_STEERING  = True   — prints steering calculations
   DEBUG_BEAM      = True   — prints beam sensor reads during collection
   DEBUG_STATE     = True   — prints every state and transition
"""

import cv2
import time
import numpy as np
from dotenv import load_dotenv
import os
from inference import get_model

import movement_v2
from sensor import beam_broken


# =============================================================
# DEBUG FLAGS — set False to silence sections you don't need
# =============================================================

DEBUG_VISION   = True
DEBUG_STEERING = True
DEBUG_BEAM     = True
DEBUG_STATE    = True


def dbg(section, msg):
    """Consistent debug print with section label and timestamp."""
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
# 2.  ROBOT PARAMETERS
# =============================================================

FRAME_CENTER         = 320    # Half of 640px frame width
BASE_SPEED           = 80     # Drive speed 0-100
SEARCH_SPEED         = 40     # Rotation speed while scanning
Kp                   = 0.12   # Steering proportional gain
COLLECTION_THRESHOLD = 20     # Distance in cm that triggers COLLECTING state
BEAM_COLLECT_TIMEOUT = 3.0    # Seconds to wait for beam confirmation per ball
MAX_CAPACITY         = 4      # Stop collecting after this many balls
RETURN_HOME_TIME_S   = 3      # Seconds of reverse driving to reach home base
                               # ← Tune this to match your arena size

print("\n[PARAMS] Robot parameters:")
print(f"  FRAME_CENTER         = {FRAME_CENTER}px")
print(f"  BASE_SPEED           = {BASE_SPEED}")
print(f"  SEARCH_SPEED         = {SEARCH_SPEED}")
print(f"  Kp (steering gain)   = {Kp}")
print(f"  COLLECTION_THRESHOLD = {COLLECTION_THRESHOLD}cm")
print(f"  BEAM_COLLECT_TIMEOUT = {BEAM_COLLECT_TIMEOUT}s")
print(f"  MAX_CAPACITY         = {MAX_CAPACITY} balls")
print(f"  RETURN_HOME_TIME_S   = {RETURN_HOME_TIME_S}s")


# =============================================================
# 3.  DETECTION
# =============================================================

_collected_ids = set()   # Track IDs confirmed collected — never targeted again


def get_detections():
    """
    Grabs a camera frame, runs local inference, returns:
      - frame  : the raw BGR image (used for display)
      - list of detection dicts, each containing:
          class, confidence, bbox, center_px, distance_cm, track_id
    Balls whose track_id is in _collected_ids are filtered out.
    """
    ret, frame = cap.read()

    if not ret:
        dbg("VISION", "❌ cap.read() failed — camera may have disconnected")
        return None, []

    if DEBUG_VISION:
        dbg("VISION", "Frame captured — running inference...")

    results = model.infer(frame, confidence=0.5)[0]

    if DEBUG_VISION:
        dbg("VISION", f"Raw predictions from model: {len(results.predictions)}")

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

        if track_id in _collected_ids:
            if DEBUG_VISION:
                dbg("VISION", f"    ↳ Skipping ID {track_id} (already collected)")
            continue

        cls_name = pred.class_name
        conf     = pred.confidence

        x1 = int(pred.x - pred.width  / 2)
        y1 = int(pred.y - pred.height / 2)
        x2 = int(pred.x + pred.width  / 2)
        y2 = int(pred.y + pred.height / 2)

        pixel_diameter = max(int(pred.width), int(pred.height))
        if pixel_diameter == 0:
            dbg("VISION", f"    ↳ Skipping {cls_name} — zero pixel diameter")
            continue

        real_diameter = KNOWN_DIAMETERS.get(cls_name, 0.02)
        distance_cm   = (real_diameter * focal_length) / pixel_diameter * 100

        if DEBUG_VISION:
            dbg("VISION", f"    ↳ pixel_diam={pixel_diameter}px | "
                          f"real_diam={real_diameter*100:.0f}mm | "
                          f"distance={distance_cm:.1f}cm")

        detections.append({
            "class":       cls_name,
            "confidence":  conf,
            "bbox":        (x1, y1, x2, y2),
            "center_px":   (int(pred.x), int(pred.y)),
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
        dbg("VISION", f"Best target: '{target['class']}' at "
                      f"{target['distance_cm']:.1f}cm (track_id={target['track_id']})")
    return target


# =============================================================
# 4.  STATE MACHINE
# =============================================================

def run():
    global _collected_ids

    state           = "SEARCHING"
    active_track_id = -1
    ball_count      = 0     # Confirmed collected balls this run

    print("\n" + "=" * 60)
    print("STATE MACHINE STARTING")
    print("  Collecting 4 balls, then returning home to deposit")
    print("  Press Q in camera window or Ctrl+C to stop")
    print("=" * 60 + "\n")

    loop_count = 0

    try:
        while True:
            loop_count += 1

            # Heartbeat every 50 iterations so you can see the loop is alive
            if loop_count % 50 == 0:
                print(f"[LOOP] iter={loop_count} | state={state} | "
                      f"balls={ball_count}/{MAX_CAPACITY} | "
                      f"collected_ids={_collected_ids}")

            # --- Grab frame and run detection ---
            frame, detections = get_detections()

            if frame is None:
                dbg("LOOP", "No frame — skipping iteration")
                time.sleep(0.05)
                continue

            target        = best_target(detections)
            ball_detected = target is not None

            # Pre-compute steering values if a ball is visible
            if ball_detected:
                ball_cx     = target["center_px"][0]
                distance_cm = target["distance_cm"]
                error       = ball_cx - FRAME_CENTER
                turn_adj    = Kp * error
                close       = distance_cm < COLLECTION_THRESHOLD
                track_id    = target["track_id"]

            # --- Annotate and display frame ---
            for d in detections:
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = (f'{d["class"]} {d["distance_cm"]:.0f}cm '
                         f'id={d["track_id"]}')
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(frame,
                        f"State: {state}  Balls: {ball_count}/{MAX_CAPACITY}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Ball Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[LOOP] Q pressed — exiting")
                break

            # ==============================================
            # SEARCHING
            # Robot rotates slowly until it sees a ball.
            # Once a ball is spotted it switches to APPROACHING.
            # ==============================================
            if state == "SEARCHING":
                if DEBUG_STATE:
                    dbg("STATE", f"SEARCHING — turn_left({SEARCH_SPEED})")
                movement_v2.turn_left(SEARCH_SPEED)

                if ball_detected:
                    if DEBUG_STATE:
                        dbg("STATE", f"✅ Ball spotted! class='{target['class']}' "
                                     f"dist={distance_cm:.1f}cm track_id={track_id} "
                                     f"→ APPROACHING")
                    active_track_id = track_id
                    state = "APPROACHING"

            # ==============================================
            # APPROACHING
            # Steers toward the closest ball using proportional
            # control on the horizontal pixel error.
            # Switches to COLLECTING once close enough.
            # ==============================================
            elif state == "APPROACHING":

                if not ball_detected:
                    dbg("STATE", "⚠️  Ball lost mid-approach → SEARCHING")
                    movement_v2.stop_drive()
                    state = "SEARCHING"

                elif close:
                    dbg("STATE", f"📍 Close enough! dist={distance_cm:.1f}cm "
                                 f"(threshold={COLLECTION_THRESHOLD}cm) → COLLECTING")
                    active_track_id = track_id
                    movement_v2.stop_drive()
                    state = "COLLECTING"

                else:
                    if DEBUG_STEERING:
                        dbg("STEER", f"ball_cx={ball_cx}px | centre={FRAME_CENTER}px | "
                                     f"error={error:+d}px | turn_adj={turn_adj:+.1f} | "
                                     f"dist={distance_cm:.1f}cm")

                    if turn_adj > 10:
                        if DEBUG_STEERING:
                            dbg("STEER", f"turn_adj={turn_adj:+.1f} > +10 "
                                         f"→ turn_right({BASE_SPEED})")
                        movement_v2.turn_right(BASE_SPEED)

                    elif turn_adj < -10:
                        if DEBUG_STEERING:
                            dbg("STEER", f"turn_adj={turn_adj:+.1f} < -10 "
                                         f"→ turn_left({BASE_SPEED})")
                        movement_v2.turn_left(BASE_SPEED)

                    else:
                        if DEBUG_STEERING:
                            dbg("STEER", f"turn_adj within ±10 "
                                         f"→ move_forward({BASE_SPEED}) + start_spinner()")
                        movement_v2.move_forward(BASE_SPEED)
                        movement_v2.start_spinner()

            # ==============================================
            # COLLECTING
            # Robot has stopped in front of the ball.
            # Spinner pulls it in. Beam sensor confirms entry.
            # Lift is NOT raised here — balls stay in hopper.
            # After 4 confirmed balls → go straight home.
            # After fewer → go back to search for more.
            # ==============================================
            elif state == "COLLECTING":
                dbg("STATE", f"COLLECTING — track_id={active_track_id} | "
                             f"stop_drive() + start_spinner()")

                movement_v2.stop_drive()
                movement_v2.start_spinner()

                deadline  = time.time() + BEAM_COLLECT_TIMEOUT
                collected = False

                dbg("STATE", f"Waiting up to {BEAM_COLLECT_TIMEOUT}s "
                             f"for beam to break...")

                while time.time() < deadline:
                    beam_status = beam_broken()

                    if DEBUG_BEAM:
                        dbg("BEAM", f"beam_broken() = {beam_status}")

                    if beam_status:
                        dbg("BEAM", "✅ Beam BROKEN — ball physically confirmed!")

                        # Stop spinner — ball is safely inside
                        dbg("STATE", "stop_spinner()")
                        movement_v2.stop_spinner()

                        # NOTE: No lift_up here — balls are held in the hopper
                        # and will only be raised and deposited at home base.

                        # Record this ball as collected
                        if active_track_id != -1:
                            _collected_ids.add(active_track_id)
                            dbg("STATE", f"track_id={active_track_id} added to "
                                         f"collected_ids={_collected_ids}")

                        ball_count += 1
                        collected   = True
                        dbg("STATE", f"ball_count → {ball_count}/{MAX_CAPACITY}")
                        break

                    time.sleep(0.05)

                if not collected:
                    dbg("BEAM", f"⚠️  Beam timeout after {BEAM_COLLECT_TIMEOUT}s "
                                f"— missed, stop_spinner()")
                    movement_v2.stop_spinner()

                # --- Decide next state ---
                if ball_count >= MAX_CAPACITY:
                    # Hit the target — head home immediately, no more searching
                    dbg("STATE", f"🎯 {MAX_CAPACITY} balls collected — "
                                 f"stop collecting → RETURNING_HOME")
                    state = "RETURNING_HOME"
                else:
                    # Still need more balls — keep searching
                    dbg("STATE", f"Need more balls ({ball_count}/{MAX_CAPACITY}) "
                                 f"→ SEARCHING")
                    state = "SEARCHING"

            # ==============================================
            # RETURNING_HOME
            # Drives in reverse for a fixed time to reach home.
            # No encoders — purely time-based.
            # Tune RETURN_HOME_TIME_S to match your arena.
            # Robot does NOT look for or react to balls here.
            # ==============================================
            elif state == "RETURNING_HOME":
                dbg("STATE", f"🏠 RETURNING_HOME — move_backward({BASE_SPEED}) "
                             f"for {RETURN_HOME_TIME_S}s (no encoders)")

                movement_v2.move_backward(BASE_SPEED)
                time.sleep(RETURN_HOME_TIME_S)
                movement_v2.stop_drive()

                dbg("STATE", "Arrived at home base → DROPPING")
                state = "DROPPING"

            # ==============================================
            # DROPPING
            # Robot is at home. Now raise the lift to bring
            # all collected balls up, then lower to deposit them.
            # This is the ONLY place lift_up is called.
            # After depositing, robot resets for another run.
            # ==============================================
            elif state == "DROPPING":
                dbg("STATE", "📦 DROPPING — raising lift to deposit balls")

                # Raise lift to eject/deposit the collected balls
                dbg("STATE", "lift_up(speed=80)")
                movement_v2.lift_up(speed=80)
                time.sleep(2.0)   # Hold up long enough to fully eject balls
                                  # ← Tune this duration if needed

                dbg("STATE", "stop_lift()")
                movement_v2.stop_lift()

                # Lower lift back to starting position ready for next run
                dbg("STATE", "lift_down(speed=80)")
                movement_v2.lift_down(speed=80)
                time.sleep(1.5)

                dbg("STATE", "stop_lift()")
                movement_v2.stop_lift()

                # Reset all state for next run
                _collected_ids.clear()
                ball_count = 0
                dbg("STATE", "collected_ids cleared, ball_count reset → SEARCHING")
                state = "SEARCHING"

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[LOOP] Ctrl+C — shutting down cleanly")

    finally:
        print("[SHUTDOWN] Stopping all motors...")
        movement_v2.shutdown()
        print("[SHUTDOWN] Releasing camera...")
        cap.release()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Complete ✓")


# =============================================================
# 5.  ENTRY POINT
# =============================================================

if __name__ == "__main__":
    run()