from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m.pt")


# Train the model
train_results = model.train(data="config.yaml", epochs=10)