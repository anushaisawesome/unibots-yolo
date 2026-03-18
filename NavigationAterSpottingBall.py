import math
import time

class RobotNavigator:
    def __init__(self, detector, camera_matrix):
        self.detector = detector
        self.camera_matrix = camera_matrix

        # --- Tuning values - adjust these for your robot ---
        self.ALIGNMENT_TOLERANCE_PX  = 20    # Pixels off-centre before we bother turning
        self.ARRIVAL_DISTANCE_M      = 0.10  # Stop when within 10cm of ball
        self.TURN_SPEED_SLOW         = 0.3   # Turning speed for fine corrections (0-1 scale)
        self.TURN_SPEED_FAST         = 0.7   # Turning speed for large corrections
        self.MOVE_SPEED              = 0.4   # Forward movement speed (0-1 scale)
        self.LARGE_ANGLE_THRESHOLD   = 30    # Degrees - above this, stop and turn before moving
        self.LOST_BALL_TIMEOUT_S     = 2.0   # Seconds to scan before giving up on a lost ball
        self.FEEDBACK_INTERVAL_S     = 0.1   # How often to re-check position while moving

    # -------------------------------------------------------
    # HELPER: Get the image centre x pixel
    # -------------------------------------------------------
    def _image_centre_x(self, frame):
        # frame.shape gives (height, width, channels) — we want width
        return frame.shape[1] // 2

    # -------------------------------------------------------
    # HELPER: Find closest detection in a list
    # -------------------------------------------------------
    def _find_closest(self, detections):
        # Sort by the Z component of position_3d, which is distance in metres
        # min() with a key is like saying "find the one with the smallest distance"
        return min(detections, key=lambda d: d["position_3d"][2])

    # -------------------------------------------------------
    # HELPER: Calculate how far off-centre the ball is
    # Returns pixel error (negative = ball is left, positive = ball is right)
    # -------------------------------------------------------
    def _horizontal_error_px(self, detection, frame):
        ball_centre_x  = detection["center_px"][0]   # Ball's x pixel
        image_centre_x = self._image_centre_x(frame) # Middle of camera view
        return ball_centre_x - image_centre_x
        # e.g. if ball is at pixel 400 and image centre is 320, error = +80 (ball is to the right)

    # -------------------------------------------------------
    # HELPER: Convert pixel error to an estimated angle in degrees
    # Uses the camera focal length so the angle is physically meaningful
    # -------------------------------------------------------
    def _pixel_error_to_degrees(self, pixel_error):
        focal_length = self.camera_matrix[0][0]  # Focal length in pixels
        angle_rad    = math.atan2(pixel_error, focal_length)
        return math.degrees(angle_rad)

    # -------------------------------------------------------
    # HELPER: Decide turn direction and speed from pixel error
    # Returns ("left"/"right"/"none", speed)
    # -------------------------------------------------------
    def _compute_turn(self, pixel_error):
        abs_error = abs(pixel_error)

        if abs_error < self.ALIGNMENT_TOLERANCE_PX:
            return "none", 0  # Close enough to centre, no turn needed

        # Use faster turn for large errors, slower for small fine-tuning
        speed = self.TURN_SPEED_FAST if abs_error > 60 else self.TURN_SPEED_SLOW

        direction = "right" if pixel_error > 0 else "left"
        return direction, speed

    # -------------------------------------------------------
    # MAIN: Navigate to a single target ball
    # Continuously re-detects and adjusts course
    # -------------------------------------------------------
    def navigate_to_ball(self, target_detection, cap):
        """
        cap        : your cv2.VideoCapture object (the live camera)
        target     : the detection dict we're heading towards
        """
        print(f"\n🎯 Navigating to {target_detection['class']} "
              f"at {target_detection['position_3d'][2]:.2f}m")

        lost_timer = None  # Will track how long the ball has been missing

        while True:

            # --- 1. GRAB FRESH CAMERA FRAME ---
            ret, frame = cap.read()
            if not ret:
                print("  Camera read failed")
                break

            # --- 2. RUN DETECTION ON LATEST FRAME ---
            detections = self.detector.detect(frame)

            # --- 3. CHECK IF WE CAN STILL SEE ANY BALL ---
            if not detections:
                if lost_timer is None:
                    lost_timer = time.time()  # Start counting how long ball is gone
                    print("  Ball lost — scanning...")
                    move_forward(0)           # Stop moving while lost
                elif time.time() - lost_timer > self.LOST_BALL_TIMEOUT_S:
                    print(" Ball lost for too long — skipping this target")
                    return False              # Give up on this ball
                continue                     # Loop again, try to re-find it

            lost_timer = None  # Ball found again — reset the lost timer

            # --- 4. RE-ACQUIRE THE CLOSEST BALL ---
            # We always re-target the closest visible ball, not the original one
            # This handles cases where another ball is now closer mid-journey
            current_target = self._find_closest(detections)
            distance_m     = current_target["position_3d"][2]

            print(f"  Ball at {distance_m:.2f}m | "
                  f"centre_x={current_target['center_px'][0]}px")

            # --- 5. CHECK IF WE'VE ARRIVED ---
            if distance_m <= self.ARRIVAL_DISTANCE_M:
                move_forward(0)   # Stop
                print(f" Arrived at {current_target['class']}!")
                return True       # Signal success to the caller

            # --- 6. CALCULATE HOW MISALIGNED WE ARE ---
            pixel_error   = self._horizontal_error_px(current_target, frame)
            angle_error   = self._pixel_error_to_degrees(pixel_error)
            turn_dir, turn_speed = self._compute_turn(pixel_error)

            print(f"  ↔️  Pixel error: {pixel_error:+.0f}px | "
                  f"Angle: {angle_error:+.1f}° | Turn: {turn_dir}")

            # --- 7. DECIDE: TURN FIRST OR MOVE + STEER ---
            if abs(angle_error) > self.LARGE_ANGLE_THRESHOLD:
                # Ball is far off to the side — stop and turn to face it first
                # before moving, so we don't drive past it
                move_forward(0)
                if turn_dir == "left":
                    turn(-turn_speed)
                elif turn_dir == "right":
                    turn(turn_speed)

            else:
                # Ball is roughly ahead — move forward while making small corrections
                move_forward(self.MOVE_SPEED)
                if turn_dir == "left":
                    turn(-turn_speed)   # Gentle left correction while moving
                elif turn_dir == "right":
                    turn(turn_speed)    # Gentle right correction while moving
                else:
                    turn(0)             # Dead ahead, no turn needed

            # --- 8. WAIT BRIEFLY BEFORE NEXT FEEDBACK LOOP ---
            # This controls how frequently we re-check position
            time.sleep(self.FEEDBACK_INTERVAL_S)

    # -------------------------------------------------------
    # MAIN LOOP: Collect all balls one by one
    # -------------------------------------------------------
    def collect_all_balls(self, cap, pick_up_fn):
        """
        cap        : cv2.VideoCapture object
        pick_up_fn : the function that runs your gripper/arm to grab the ball
        """
        collected_count = 0

        while True:

            # --- Scan for all visible balls ---
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.detector.detect(frame)

            if not detections:
                print("🔍 No balls visible — done or need to reposition")
                break

            # --- Always go for the closest ball first ---
            target = self._find_closest(detections)
            print(f"\n{'='*50}")
            print(f"Next target: {target['class']} at "
                  f"{target['position_3d'][2]:.2f}m "
                  f"(total visible: {len(detections)})")

            # --- Navigate to it ---
            success = self.navigate_to_ball(target, cap)

            if success:
                # --- Attempt pickup ---
                pick_up_fn(target)
                collected_count += 1
                print(f" Total collected: {collected_count}")
                time.sleep(0.5)  # Brief pause to let things settle after pickup
            else:
                print(f"   Skipping — moving on to next ball")

        print(f"\n Collection complete! Picked up {collected_count} balls.")
