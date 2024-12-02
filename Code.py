import os
import time

from ultralytics import YOLO
import cv2
import numpy as np


# cap = cv2.VideoCapture(0)
#
# print("we load the model wait . . . !!!")
#
# model_path = os.path.join('.', 'Train', 'runs', 'detect', 'train', 'weights', 'best.pt')
# # load the model
# model = YOLO(model_path)
#
# # Detection threshold
# threshold = 0.5
#
# print("Press 'q' to quit.")
#
# while True:
#
#     ret, frame = cap.read()
#
#     # Run the YOLO model
#     results = model(frame)[0]
#
#     # print("the score is ==== ",results)
#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result
#         if score > threshold:
#             # Draw bounding box and label
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
#
#     cv2.imshow('frame1', frame)
#
#     if cv2.waitKey(1) == ord('b'):
#         break
#
# cap.release()
# cv2.destroyAllWindow()


video_path = os.path.join('.', 'Data', 'IMG_0050.MOV')
video_path_out = '{}_out.mp4'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))


# Load a model
model = YOLO('./best.pt')  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()


#
# # Path to the image file
# image_path = os.path.join('.', 'Data', 'datasets', 'images', 'val', '12b9323540615016.jpg')
#
# # Load the image
# image = cv2.imread(image_path)
#
# # Load the YOLO model
# model = YOLO('./best.pt')  # load a custom model
#
# threshold = 0.5
#
# # Perform inference on the image
# results = model(image)[0]
#
# # Draw bounding boxes and labels
# for result in results:
#     for keypoint in result.keypoints.tolist():
#         print(keypoint)
#
# # Save the output image
# cv2.imwrite('./out.jpg', image)
#
# # Optionally, display the image
# # cv2.imshow('Detected Image', image)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()