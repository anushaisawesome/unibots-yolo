try:
    import RPi.GPIO as GPIO
except ImportError:
    print("GPIO library not available (not running on Raspberry Pi)")

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# =========================
# MOTOR CLASS
# =========================

class Motor:
    def __init__(self, in1, in2, enable):
        self.in1 = in1
        self.in2 = in2
        self.enable = enable

        GPIO.setup(in1, GPIO.OUT)
        GPIO.setup(in2, GPIO.OUT)
        GPIO.setup(enable, GPIO.OUT)

        self.pwm = GPIO.PWM(enable, 1000)
        self.pwm.start(0)

    def forward(self, speed=100):
        GPIO.output(self.in1, True)
        GPIO.output(self.in2, False)
        self.pwm.ChangeDutyCycle(speed)

    def reverse(self, speed=100):
        GPIO.output(self.in1, False)
        GPIO.output(self.in2, True)
        self.pwm.ChangeDutyCycle(speed)

    def stop(self):
        GPIO.output(self.in1, False)
        GPIO.output(self.in2, False)
        self.pwm.ChangeDutyCycle(0)

# =========================
# PIN CONFIGURATION
# =========================

# Left drive motor
LEFT_IN1 = 5
LEFT_IN2 = 6
LEFT_EN = 13

# Right drive motor
RIGHT_IN1 = 19
RIGHT_IN2 = 26
RIGHT_EN = 12

# Spinner motor
SPIN_IN1 = 20
SPIN_IN2 = 21
SPIN_EN = 16

# Lift motor
LIFT_IN1 = 23
LIFT_IN2 = 24
LIFT_EN = 18

# =========================
# CREATE MOTOR OBJECTS
# =========================

left_motor = Motor(LEFT_IN1, LEFT_IN2, LEFT_EN)
right_motor = Motor(RIGHT_IN1, RIGHT_IN2, RIGHT_EN)
spinner = Motor(SPIN_IN1, SPIN_IN2, SPIN_EN)
lift = Motor(LIFT_IN1, LIFT_IN2, LIFT_EN)

# =========================
# DRIVE FUNCTIONS
# =========================

def move_forward(speed=80):
    left_motor.forward(speed)
    right_motor.forward(speed)

def move_backward(speed=80):
    left_motor.reverse(speed)
    right_motor.reverse(speed)

def turn_left(speed=80):
    left_motor.reverse(speed)
    right_motor.forward(speed)

def turn_right(speed=80):
    left_motor.forward(speed)
    right_motor.reverse(speed)

def turn(left_speed, right_speed):
    if left_speed >= 0:
        left_motor.forward(left_speed)
    else:
        left_motor.reverse(-left_speed)

    if right_speed >= 0:
        right_motor.forward(right_speed)
    else:
        right_motor.reverse(-right_speed)

def stop_drive():
    left_motor.stop()
    right_motor.stop()

# =========================
# SPINNER FUNCTIONS
# =========================

def start_spinner(speed=100):
    spinner.forward(speed)

def reverse_spinner(speed=100):
    spinner.reverse(speed)

def stop_spinner():
    spinner.stop()

# =========================
# LIFT FUNCTIONS
# =========================

def lift_up(speed=80):
    lift.forward(speed)

def lift_down(speed=80):
    lift.reverse(speed)

def stop_lift():
    lift.stop()

# =========================
# CLEANUP FUNCTION
# =========================

def shutdown():
    stop_drive()
    stop_spinner()
    stop_lift()
    GPIO.cleanup()