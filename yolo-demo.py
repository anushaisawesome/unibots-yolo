import cv2
import os
from dotenv import load_dotenv
import requests
import json
import time

# Load environment variables
load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not API_KEY:
    print("❌ Error: ROBOFLOW_API_KEY not set. Please add it to your .env file.")
    exit(1)
# Update these with your actual Roboflow project details
PROJECT_ID = "ball-dataset-merged"  # Replace with your project ID
VERSION = 2  # Replace with your model version
MODEL_ID = f"{PROJECT_ID}/{VERSION}"

# Import robot movement module (from Main_robot.py context)
try:
    import movement_v2
    print("✅ movement_v2 loaded successfully")
except ImportError:
    movement_v2 = None
    print("⚠️  movement_v2 not found. Robot commands will be skipped.")

# Robot Logic
def decide_movement(predictions, frame_width):
    if movement_v2 is None:
        return

    # 1. SEARCH: If no balls seen, spin to find one
    if not predictions:
        print("🔍 Search Mode: Spinning")
        movement_v2.turn_left(40)
        return

    # 2. TARGET: Find the largest (closest) ball
    target = max(predictions, key=lambda p: p['width'] * p['height'])
    
    x_center = target['x']
    obj_width = target['width']
    
    screen_center = frame_width / 2
    alignment_error = x_center - screen_center
    
    # 3. CONTROLS
    CENTER_TOLERANCE = 60    # Pixels
    CLOSE_ENOUGH_WIDTH = 200 # Pixels (Adjust this based on when the ball is reachable)

    if abs(alignment_error) > CENTER_TOLERANCE:
        # ALIGNMENT
        if alignment_error > 0:
            movement_v2.turn_right(35)
        else:
            movement_v2.turn_left(35)
    elif obj_width < CLOSE_ENOUGH_WIDTH:
        # APPROACH
        movement_v2.move_forward(50)
        movement_v2.start_spinner() # Start intake
    else:
        # GRAB
        print("🎯 Target Acquired: Grabbing!")
        movement_v2.stop_drive()
        movement_v2.lift_up(80)
        time.sleep(1.0)
        movement_v2.stop_lift()
        movement_v2.stop_spinner()

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print(f"🚀 Starting ball detection with model: {MODEL_ID}")
print("Press 'q' to quit\n")

frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        frame_count += 1
        
        # Convert to JPEG for API
        _, buffer = cv2.imencode('.jpg', frame)
        
        # Send to Roboflow API
        try:
            response = requests.post(
                f"https://detect.roboflow.com/{MODEL_ID}",
                params={"api_key": API_KEY, "confidence": 40},
                files={"file": ("image.jpg", buffer.tobytes(), "image/jpeg")},
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Drive Robot
                predictions = result.get("predictions", [])
                decide_movement(predictions, frame.shape[1])

                # Draw predictions
                for pred in predictions:
                        x = int(pred["x"] - pred["width"] / 2)
                        y = int(pred["y"] - pred["height"] / 2)
                        w = int(pred["width"])
                        h = int(pred["height"])
                        conf = pred.get("confidence", 0)
                        class_name = pred.get("class", "unknown")
                        
                        # Print to console to clarify 'o' vs '0'
                        print(f"Frame {frame_count}: Detected '{class_name}' ({conf:.2f})")
                        
                        # Draw rectangle
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f"{class_name} ({conf:.2f})"
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                print(f"API Error: Status {response.status_code} - {response.text}")
            
        except Exception as e:
            print(f"API Error: {e}")
        
        # Show frame
        cv2.imshow("Ball Detection", frame)
        
        # Break on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
if movement_v2:
    movement_v2.stop_drive()
    movement_v2.shutdown()
print("\n✅ Detection stopped")
