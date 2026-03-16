
## Setup Instructions

### 1. Create a virtual environment (Mac)
```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set up environment variables
- Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
- Open `.env` and add your Roboflow API key:
```
ROBOFLOW_API_KEY=your_actual_api_key_here
```

### 4. Run the demo
```bash
python yolo-demo.py
```

Press `q` to quit the webcam window.

### Notes
- Never commit the `.env` file (it's ignored in `.gitignore`)
- The `venv/` folder is also ignored and won't be pushed to GitHub
- When someone clones this repo, they just need to run steps 1-2 above to get set up
