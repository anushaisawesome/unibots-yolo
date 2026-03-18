import RPi.GPIO as GPIO
import time

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

BEAM_PIN = 17   # GPIO connected to receiver OUT

GPIO.setup(BEAM_PIN, GPIO.IN)

def beam_broken():
    """
    Returns True if beam is broken
    Returns False if beam is intact
    """

    if GPIO.input(BEAM_PIN) == GPIO.LOW:
        return True
    else:
        return False


try:

    print("Break beam sensor running...")

    while True:

        if beam_broken():
            print("Beam BROKEN")
        else:
            print("Beam intact")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Stopping")

finally:
    GPIO.cleanup()