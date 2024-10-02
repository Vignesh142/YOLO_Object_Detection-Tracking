from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('../YOLO-weights/yolov8l.pt')
results = model('../images/bikes.png', show=True)  
cv2.waitKey(0)