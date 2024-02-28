from ultralytics import YOLO
import cv2 as cv
import time

model = YOLO('./yolo_weights/yolov8l-seg.pt')


cap = cv.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while cap.isOpened():
    success, frame = cap.read()
    start = time.perf_counter()

    if success:
        results = model(frame)
        end = time.perf_counter()
        total_time = end - start
        fps = 1/total_time

        segmented_frame = results[0].plot()        

        cv.imshow('Webcam', segmented_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
