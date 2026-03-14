
from ultralytics import YOLO


model = YOLO("yolo26n.pt")  # load YOLO26 model
model.train(data="/Users/anusha/Desktop/unibots-yolo-demo1/ball-dataset-merged.v1i.yolo26/data.yaml", epochs=50, batch=16, device="mps") # train on merged dataset for 100 epochs