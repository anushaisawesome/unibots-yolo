# Unibots Ball Grabbing Robot

A Raspberry Pi robot that detects and collects balls using a YOLO model served from a Mac inference server.

---

## Architecture

```
Mac  ──────────────────────────────────────────────────
  inference server (port 9001)
  runs the YOLO model, receives frames, returns detections

Raspberry Pi  ─────────────────────────────────────────
  captures camera frames
  sends frames to Mac over WiFi
  drives motors based on detections
  state machine: SEARCH → APPROACH → COLLECT → RETURN HOME → DROP
```

Both devices must be on the **same WiFi network**.

---

## Mac Setup (inference server)

### 1. Install the inference server
```bash
pip install inference
```

### 2. Start the server
```bash
inference server start
```

Leave this running whenever you use the robot. The server listens on port 9001.

### 3. Find your Mac's IP address
```bash
ifconfig | grep "inet 10."
```
You'll need this for the Pi's `.env` file.

---

## Raspberry Pi Setup

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

### 4. Set up environment variables
```bash
cp .env.example .env
```

Open `.env` and fill in your values:
```
ROBOFLOW_API_KEY=your_api_key_here
MAC_IP=10.X.X.X
PROJECT_ID=ball-dataset-merged
MODEL_VERSION=2
```

Your Roboflow API key is at **roboflow.com → Settings → API**.

### 5. Test the connection to the Mac
```bash
curl http://10.X.X.X:9001
```
You should get a JSON response. If it times out, the Mac server isn't reachable — check both devices are on the same network.

### 6. Test the break-beam sensor
```bash
python sensor.py
```
Should print `intact` normally and `BROKEN` when something passes through. The beam uses **GPIO 25** — do not wire it to GPIO 17 (that's used by the left encoder).

### 7. Run the robot
```bash
python yolo-demo.py
```

Press `q` to quit.

---

## Files

| File | Runs on | Purpose |
|---|---|---|
| `yolo-demo.py` | Pi | Main entry point — full state machine |
| `movement_v2.py` | Pi | Motor control |
| `sensor.py` | Pi | Break-beam collector confirmation |
| `.env` | Pi | API key, Mac IP, project config |
| `train.py` | Mac | Model training (already done) |

---

## GPIO Pin Reference

| Pin | Purpose |
|---|---|
| 17 | Left encoder A |
| 18 | Left encoder B |
| 27 | Right encoder A |
| 22 | Right encoder B |
| 25 | Break-beam sensor |

---

## Tuning Parameters

All in the top section of `yolo-demo.py`:

| Parameter | Default | What it does |
|---|---|---|
| `BASE_SPEED` | 80 | Forward drive speed (0–100) |
| `SEARCH_SPEED` | 40 | Rotation speed while scanning |
| `Kp` | 0.12 | Steering correction strength |
| `COLLECTION_THRESHOLD` | 200 | Ball pixel-width that triggers collecting |
| `CENTER_TOLERANCE` | 60 | Pixels off-centre before steering kicks in |
| `MAX_CAPACITY` | 5 | Balls collected before returning home |
| `RETURN_HOME_TIME_S` | 3 | Seconds of reverse to reach home base |
| `FRAME_SKIP` | 5 | Send 1 in every N frames to server |

---

## Notes

- Never commit `.env` — it's in `.gitignore`
- `venv/` is also gitignored
- `Localisation.py`, `Main_robot.py`, `behaviour.py`, `NavigationAfterSpottingBall.py` are kept for reference but not used — the inference server approach replaces them