# Drone-Video-Footage-Analyzer
# Drone Video Footage Analyzer

A Python-based tool that processes drone video footage to detect objects such as people, cars, and animals using AI-powered object detection (YOLOv5).  
The project analyzes video frames, generates detection summaries, and visualizes activity heatmaps.  

---

## Features

- Uses state-of-the-art YOLOv5 pretrained model for real-time object detection  
- Supports detecting common objects like persons, cars, dogs, cats, and birds  
- Processes video files frame-by-frame efficiently  
- Generates a summary report of detected objects  
- Creates heatmaps visualizing areas of high activity (e.g., where people are detected most)  
- Easily extensible for additional object classes and features  
 ![image](https://github.com/user-attachments/assets/8d8917c7-9e28-43a4-8c04-bcff81d9bda5)

---

## Requirements

- Python 3.7+  
- PyTorch  
- OpenCV  
- matplotlib  
- pandas  
- numpy  

Install dependencies via:

```bash
pip install torch torchvision opencv-python matplotlib pandas numpy
