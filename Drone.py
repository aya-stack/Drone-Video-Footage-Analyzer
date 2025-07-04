import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Load YOLOv5 model (this will download weights first time)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Classes of interest (modify as you want)
TARGET_CLASSES = ['person', 'car', 'dog', 'cat', 'bird']  

# Video path
video_path = r"C:\Users\Administrator\Downloads\drone.mp4"


# Open video
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Store detections: object_type -> list of (x_center, y_center)
detections = defaultdict(list)

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Run inference
    results = model(frame)

    # Extract detections
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        class_name = model.names[int(cls)]
        if class_name in TARGET_CLASSES:
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            detections[class_name].append((x_center, y_center))

    if frame_count % 50 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
print(f"Processed total {frame_count} frames.")

# Generate summary report
summary = {cls: len(coords) for cls, coords in detections.items()}
print("Summary of detections:")
for cls, count in summary.items():
    print(f"{cls}: {count}")

# Generate heatmap for person detections (example)
if 'person' in detections:
    heatmap_data = np.zeros((frame_height, frame_width))
    for (x, y) in detections['person']:
        x_int, y_int = int(x), int(y)
        # Increase heatmap intensity, with bounds check
        if 0 <= y_int < frame_height and 0 <= x_int < frame_width:
            heatmap_data[y_int, x_int] += 1

    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
    plt.title('Heatmap of Person Detections')
    plt.colorbar()
    plt.show()
