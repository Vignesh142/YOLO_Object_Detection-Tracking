# Object Detection and Tracking Project using YOLO and MediaPipe and OpenCV

## Introduction
The Project is the Implementation of Object Detection and Tracking using YOLO, MediaPipe and OpenCV.

Using the YOLO v8 model to detect objects in Real-Time and track the objects using the Deep SORT algorithm.

Used the torch and torchvision libraries with CUDA support for the model to run on the GPU for the faster detection and tracking of objects.

## Steps to Run the Project
1. Clone the Repository
2. Install the Required Libraries
3. Run the files

## Installation
1. Clone the Repository
```bash
git clone https://github.com/Vignesh142/YOLO_Object_Detection-Tracking.git
```
2. Install the Required Libraries
```bash
pip install -r requirements.txt
```
3. Run the files
```bash
python file.py
```
4. Test the GPU Support
```bash
python Running-YOLO/check-gpu.py
```
5. Download the YOLO Weights
```bash
cd Running-YOLO
python yolo_basics.py
```

## Files

1. Running-YOLO
    - check-gpu.py
    - yolo_basics.py
    - yolo.py
    - yolo_utils.py
    - yolo_v3.weights
    - coco.names

2. Object-Detection



## Libraries Used
- torch
- torchvision
- numpy
- opencv-python
- mediapipe
- torch-cudnn
- torch-cuda
- torch-cudatoolkit
- cvzone
- ultralytics
- deep_sort_realtime




