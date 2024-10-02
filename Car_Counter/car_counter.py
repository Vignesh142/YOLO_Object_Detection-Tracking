import math
import random
import time
from ultralytics import YOLO
import cv2
import cvzone
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# Determine the device to use (GPU if available, else CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# cap = cv2.VideoCapture(0) # For webcam
# cap.set(3, 1280) # Set the width
# cap.set(4, 720) # Set the height
cap = cv2.VideoCapture('../videos/cars.mp4') # For video file

# Load the model
model = YOLO('../YOLO-weights/yolov8l.pt')

object_tracker = DeepSort(max_age=0, nms_max_overlap=0, max_cosine_distance=0.1)

classNames = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

colors = [(random.randint(0,255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]

limits = [200, 400, 673, 400]
totalCount = []

mask = cv2.imread('mask-car.png')
success, img = cap.read()
ptime = 0
while success:
    success, img = cap.read()
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    imgRegion = cv2.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            w, h = x2 - x1, y2 - y1

            # Confidence score
            conf = math.ceil(box.conf[0] * 100) /100

            # Class name
            classId = int(box.cls[0])
            currClass = classNames[classId]
            req = ['car', 'truck', 'bus', 'motorbike', 'bicycle']
            if currClass in req and conf > 0.3:
                # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                # cvzone.putTextRect(img, f'{classNames[classId]} {conf}', (max(0,x1), max(35, y1)), 2, 2, offset=5)
                arr = ([x1, y1, w, h], conf, currClass)
                detections.append(arr)
    
    tracks = object_tracker.update_tracks(detections, frame=img)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = int(track.track_id)
        track_class = track.det_class
        track_conf = track.det_conf

        bbox = track.to_tlbr()
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), l=9, rt=2, colorR=(255,0, 0))
        cvzone.putTextRect(img, f'#{track_id} {track_class} {track_conf}', (max(0,x1), max(35, y1)), scale=1, thickness=1, offset=8)

        # Check if the object is crossing the line
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[1]+20:
            if totalCount.count(track_id)==0:
                totalCount.append(track_id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
                cv2.circle(img, (cx, cy), 5, (0, 255,0), cv2.FILLED)
                cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), l=9, rt=2, colorR=(0,255,0))
                cvzone.putTextRect(img, f'#{track_id} {track_class} {track_conf}', (max(0,x1), max(35, y1)), scale=1, thickness=1, offset=8, colorR=(0, 255, 0))
            
    cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50))
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image", img)
    # cv2.imshow("Region", imgRegion)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()