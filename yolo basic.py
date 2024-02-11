from ultralytics import YOLO
import cv2 as cv

model = YOLO("./yolo_weights/yolov8n.pt")
result = model("images/3.jpg", show=True)
cv.waitKey(0)
