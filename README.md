# Unibots Ball Grabbing Robot

A Raspberry Pi robot that detects and collects balls using a YOLO model trained on Roboflow, running entirely on-device.

---

## Architecture

Everything runs on the Pi. No Mac needed.

```
Raspberry Pi
  camera → inference (Roboflow model, cached on-device) → state machine → motors
```

The model is downloaded from Roboflow on first run, then cached locally.
After that, no internet connection is needed.

State machine: `SEARCHING → APPROACHING → COLLECTING → RETURNING_HOME → DROPPING`

---

## Setup

### 1. Clone the repo
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ `inference` is a large install — give it a few minutes on the Pi.

### 4. Set up environment variables
```bash
cp .env.example .env
```

Open `.env` and fill in:
```
ROBOFLOW_API_KEY=your_api_key_here
PROJECT_ID=ball-dataset-merged
MODEL_VERSION=2
```

Your API key is at **roboflow.com → Settings → API**.

### 5. First run (needs internet — downloads and caches the model)
```bash
python yolo-demo.py
```

After this the Pi doesn't need internet again — the model is cached locally.
Do this before competition somewhere with WiFi.

### 6. Test the break-beam sensor
```bash
python sensor.py
```
Should print `intact` normally and `BROKEN` when something passes through.

---

## Files

| File | Purpose |
|---|---|
| `yolo-demo.py` | Main entry point — full state machine |
| `movement_v2.py` | Motor control |
| `sensor.py` | Break-beam collector confirmation |
| `.env` | API key + project config |

---

## GPIO Pin Reference

| Pin | Purpose |
|---|---|
| 17 | Left encoder A |
| 18 | Left encoder B |
| 27 | Right encoder A |
| 22 | Right encoder B |
| 25 | Break-beam sensor ⚠️ NOT pin 17 |

---

## Tuning Parameters

All in the top section of `yolo-demo.py`:

| Parameter | Default | What it does |
|---|---|---|
| `BASE_SPEED` | 80 | Forward drive speed (0–100) |
| `SEARCH_SPEED` | 40 | Rotation speed while scanning |
| `Kp` | 0.12 | Steering correction strength |
| `COLLECTION_THRESHOLD` | 20 | Distance in cm that triggers collecting |
| `CENTER_TOLERANCE` | 60 | Pixels off-centre before steering kicks in |
| `MAX_CAPACITY` | 5 | Balls collected before returning home |
| `RETURN_HOME_TIME_S` | 3 | Seconds of reverse to reach home base |

---

## Notes

- Never commit `.env` — it's in `.gitignore`
- `venv/` is also gitignored
