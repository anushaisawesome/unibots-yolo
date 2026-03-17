import cv2
import os
from dotenv import load_dotenv
import requests
import json

# Load environment variables
load_dotenv()

API_KEY = os.getenv("ROBOFLOW_API_KEY", "qncEDfiUooZ7Q01hojBk")
# Update these with your actual Roboflow project details
PROJECT_ID = "ball-dataset-merged"  # Replace with your project ID
VERSION = 1  # Replace with your model version
MODEL_ID = f"{PROJECT_ID}/{VERSION}"

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
                
                # Draw predictions
                if "predictions" in result:
                    predictions = result["predictions"]
                    
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
print("\n✅ Detection stopped")
