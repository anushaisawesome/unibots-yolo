import time
import movement_v2   # your movement class file
import lgpio

# =========================
# ROBOT PARAMETERS
# =========================

FRAME_CENTER = 320

BASE_SPEED = 80
SEARCH_SPEED = 40

Kp = 0.12

COLLECTION_THRESHOLD = 20


# =========================
# ROBOT STATES
# =========================

SEARCH = 0
APPROACH = 1
RETURN_HOME = 2

state = SEARCH


# =========================
# BALL DETECTION PLACEHOLDER
# =========================

def detect_ball():
    """
    Replace with real vision system.
    Returns:
    ball_detected
    ball_x
    distance
    """

    ball_detected = False
    ball_x = 0
    distance = 0

    return ball_detected, ball_x, distance


# =========================
# BEHAVIOUR LOOP
# =========================

def run():

    global state

    while True:

        ball_detected, ball_x, distance = detect_ball()

        # -------------------------
        # SEARCH MODE
        # -------------------------
        if state == SEARCH:

            if not ball_detected:

                movement_v2.turn_left(SEARCH_SPEED)

            else:
                state = APPROACH


        # -------------------------
        # APPROACH TARGET
        # -------------------------
        elif state == APPROACH:

            if not ball_detected:

                state = SEARCH
                continue

            error = ball_x - FRAME_CENTER

            turn_adjustment = Kp * error

            left_speed = BASE_SPEED - turn_adjustment
            right_speed = BASE_SPEED + turn_adjustment

            # simple steering logic using movement functions
            if turn_adjustment > 10:
                movement_v2.turn_right(BASE_SPEED)

            elif turn_adjustment < -10:
                movement_v2.turn_left(BASE_SPEED)

            else:
                movement_v2.move_forward(BASE_SPEED)
                movement_v2.start_spinner()

            if distance < COLLECTION_THRESHOLD:
                movement_v2.stop_drive()
                movement_v2.stop_spinner()

                state = RETURN_HOME


        # -------------------------
        # RETURN HOME
        # -------------------------
        elif state == RETURN_HOME:

            movement_v2.move_forward(BASE_SPEED)

            time.sleep(3)

            movement_v2.stop_drive()

            state = SEARCH
            
            import time

            # GPIO pin numbers (BCM)
            LEFT_ENCODER_A  = 17
            LEFT_ENCODER_B  = 18
            RIGHT_ENCODER_A = 27
            RIGHT_ENCODER_B  = 22

            # State
            left_count  = 0
            right_count = 0

            h = lgpio.gpiochip_open(0)

            # Set pins as inputs
            for pin in [LEFT_ENCODER_A, LEFT_ENCODER_B, RIGHT_ENCODER_A, RIGHT_ENCODER_B]:
                lgpio.gpio_claim_input(h, pin)

            # Interrupt callbacks
            def left_encoder_callback(chip, gpio, level, tick):
                global left_count
                # Quadrature direction detection
                b = lgpio.gpio_read(h, LEFT_ENCODER_B)
                left_count += 1 if b == 0 else -1

            def right_encoder_callback(chip, gpio, level, tick):
                global right_count
                b = lgpio.gpio_read(h, RIGHT_ENCODER_B)
                right_count += 1 if b == 0 else -1

            # Attach rising-edge interrupts to channel A pins
            lgpio.callback(h, LEFT_ENCODER_A,  lgpio.RISING_EDGE, left_encoder_callback)
            lgpio.callback(h, RIGHT_ENCODER_A, lgpio.RISING_EDGE, right_encoder_callback)

            # Main loop
            try:
                while True:
                    print(f"Left: {left_count} ticks | Right: {right_count} ticks")
                    time.sleep(0.1)
            except KeyboardInterrupt:
                lgpio.gpiochip_close(h)