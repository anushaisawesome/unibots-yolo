"""
sensor.py
=========
Break-beam sensor — imported by yolo-demo.py.

Wire your beam receiver OUT pin to GPIO 25
(GPIO 17 is taken by the left wheel encoder).
"""

# Try to import RPi.GPIO (only available on Raspberry Pi)
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    print("⚠️  RPi.GPIO not available (expected on Mac, install on Pi with: pip install RPi.GPIO)")
    GPIO_AVAILABLE = False
    # Mock GPIO for development
    class MockGPIO:
        BCM = 'BCM'
        OUT = 'OUT'
        IN = 'IN'
        LOW = 0
        HIGH = 1
        def setmode(self, mode): pass
        def setwarnings(self, state): pass
        def setup(self, pin, mode): pass
        def input(self, pin): return self.HIGH  # Assume beam is intact
        def cleanup(self): pass
    GPIO = MockGPIO()

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