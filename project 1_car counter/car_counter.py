from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from book_detector.sort import *

# cap = cv.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)

cap = cv.VideoCapture('./videos/cars.mp4')
model = YOLO('./yolo_weights/yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv.imread('./project 1_car counter/mask.png')

tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

limits = [423, 280, 673, 280]
total_count = []

while True:
    success, img = cap.read()
    imgRegion = cv.bitwise_and(img, mask)
    results = model(imgRegion, stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bbox
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            # cv.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 3)
            cvzone.cornerRect(img, (x1, y1, w, h), l=9)

            # confidence
            conf = math.ceil(box.conf[0]*100)/100

            # class anme
            cls = int(box.cls[0])
            crr_class = classNames[cls]
            if crr_class in ['car', 'truck', 'bus', 'motorbike'] and conf >= 0.5:
                # cvzone.putTextRect(img, f'{crr_class} {conf}', (max(20, x1), max(40, y1)),
                #                    scale=0.6, thickness=1, colorT=(0, 0, 0), colorR=(0, 255, 0), offset=10)
                crr_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, crr_array))

    result_tracker = tracker.update(detections)

    cv.line(img, (limits[0], limits[1]),
            (limits[2], limits[3]), (255, 0, 0), 2)

    for result in result_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        # print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.putTextRect(img, f'{id}', (max(20, x1), max(40, y1)),
                           scale=0.6, thickness=1, colorT=(0, 0, 0), colorR=(0, 255, 0), offset=10)
        cx, cy = (x1+w//2), (y1+h//2)
        cv.circle(img, (cx, cy), 5, (0, 255, 0), cv.FILLED)
        if limits[0] < cx < limits[2] and limits[1] - 10 < cy < limits[1] + 10:
            if total_count.count(id) == 0:
                total_count.append(id)
                cv.line(img, (limits[0], limits[1]),
                        (limits[2], limits[3]), (0, 255, 0), 2)

    cvzone.putTextRect(img, f'Count: {len(total_count)}', (20, 20),
                       scale=2, thickness=2, colorT=(0, 0, 0), colorR=(0, 255, 0), offset=10)

    cv.imshow('image', img)
    # cv.imshow('imgRegion', imgRegion)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
