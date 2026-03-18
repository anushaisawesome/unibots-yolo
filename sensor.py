"""
sensor.py
=========
Break-beam sensor — imported by yolo-demo.py.

Wire your beam receiver OUT pin to GPIO 25
(GPIO 17 is taken by the left wheel encoder).
"""

import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

BEAM_PIN = 25   # ⚠️  Changed from 17 — encoder uses 17

GPIO.setup(BEAM_PIN, GPIO.IN)


def beam_broken():
    """
    Returns True if the beam is broken (ball passing through collector).
    Returns False if the beam is intact.
    """
    return GPIO.input(BEAM_PIN) == GPIO.LOW


# ── Standalone test ──────────────────────────────────────────
# Run `python sensor.py` directly to check wiring before full test

if __name__ == "__main__":
    import time
    print(f"Break-beam sensor test on GPIO {BEAM_PIN} — Ctrl+C to stop")
    try:
        while True:
            print("BROKEN" if beam_broken() else "intact")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Done")
    finally:
        GPIO.cleanup()