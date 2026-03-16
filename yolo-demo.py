import cv2
import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from inference_sdk.webrtc import WebcamSource, StreamConfig, VideoMetadata

# Load environment variables from .env file
load_dotenv()

# Initialize client
client = InferenceHTTPClient.init(
    api_url="http://localhost:9001",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# Configure video source (webcam)
source = WebcamSource(resolution=(1280, 720))

# Configure streaming options
config = StreamConfig(
    # stream_output=["my_stream_output"], # Uncomment and check your stream output name
    data_output=["predictions"]      # Get prediction data via datachannel,
    processing_timeout=3600              # 60 minutes
)

# Create streaming session
session = client.webrtc.stream(
    source=source,
    workflow="custom-workflow",
    workspace="anushas-workspace-kso2j",
    image_input="image",
    config=config
)

# Handle incoming video frames
@session.on_frame
def show_frame(frame, metadata):
    cv2.imshow("Workflow Output", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        session.close()

# Handle prediction data via datachannel
@session.on_data()
def on_data(data: dict, metadata: VideoMetadata):
    print(f"Frame {metadata.frame_id}: {data}")

# Run the session (blocks until closed)
session.run()
