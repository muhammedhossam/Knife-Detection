from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.yaml")

# Train the model
train_results = model.train(data="config.yaml", epochs=1)