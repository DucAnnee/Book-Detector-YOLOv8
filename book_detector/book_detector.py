from ultralytics import YOLO
from sort import *
import cv2 as cv
import cvzone
import math
import pytesseract
import os
import requests
import urllib.parse
from bs4 import BeautifulSoup


class BookDetector:
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

    subdir = './book_detector/cropped_books'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    model = YOLO('./yolo_weights/yolov8l.pt', verbose=False)
    detected_text = []
    cap = cv.VideoCapture(0)

    def __init__(self, width=1280, height=720, video_path='', model_path=''):
        if not os.path.exists(self.subdir):
            os.makedirs(self.subdir)

        if video_path != '':
            self.cap = cv.VideoCapture(video_path)
            print(f'Capture path = {video_path}')
        else:
            self.cap.set(3, width)
            self.cap.set(4, height)
            print(f'Capture mode: Webcam ({width}x{height})')

        if model_path != '':
            self.model = YOLO(model_path)
            print(f'Model path = {model_path}')
        else:
            print('Model path = ./yolo_weights/yolov8l.pt')

        print('Detector initialized!')

    def detect(self):
        detected_id = []
        tracker = Sort(max_age=1000, min_hits=5, iou_threshold=0.4)
        counter = 0
        while True:
            success, img = self.cap.read()
            results = self.model.predict(img, stream=True, verbose=False)
            detections = np.empty((0, 5))

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    conf = math.ceil(box.conf[0] * 100) / 100
                    cls = int(box.cls[0])
                    if self.classNames[cls] == 'book' and conf >= 0.40:
                        # cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        # cvzone.putTextRect(img, f'Book {conf}', (max(20, x1), max(40, y1)),
                        #                    scale=0.6, thickness=1, colorT=(0, 0, 0), colorR=(0, 255, 0), offset=10)

                        crr_array = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, crr_array))

            result_tracker = tracker.update(detections)

            for result in result_tracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2, id = int(x1), int(
                    y1), int(x2), int(y2), int(id)
                detected_id.append(id)

                cvzone.putTextRect(img, f'{id}', (max(20, x1), max(40, y1)),
                                   scale=0.6, thickness=1, colorT=(0, 0, 0), colorR=(0, 255, 0), offset=10)

                book_img = img[y1:y2, x1:x2]
                book_text = pytesseract.image_to_string(book_img)

                if (book_text != '') and (book_text not in self.detected_text):
                    cv.imwrite(
                        f'{self.subdir}/book_img_{id}.jpg', book_img)
                    counter += 1
                    self.detected_text.append(book_text)

            cv.imshow('image', img)
            # cv.imshow('imgRegion', imgRegion)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # for text in self.detected_text:
        #     print(text)
        self.cap.release()
        cv.destroyAllWindows()

    def show_detected(self):
        image_files = [file for file in os.listdir(
            self.subdir) if file.endswith(".jpg") or file.endswith(".png")]

        for file in image_files:
            image_path = os.path.join(self.subdir, file)
            img = cv.imread(image_path)

            text = pytesseract.image_to_string(img)
            print(f"Text extracted from {file}:")
            print(text)
            print("---")
            cv.imshow('image', img)
            cv.waitKey(0)
            cv.destroyAllWindows()
            cv.waitKey(1)


def search_book(self, title):
    query = urllib.parse.quote(title)
    url = f"https://www.google.com/search?q={query}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    search_results = soup.find_all("a")
    for result in search_results:
        link = result.get("href")
        if link.startswith("/url?q="):
            link = link[7:]
            if link.startswith("https://www.amazon.com/"):
                response = requests.get(link)
                soup = BeautifulSoup(response.text, "html.parser")
                image_element = soup.find("img", class_="s-image")
                if image_element:
                    image_url = image_element.get("src")
                    image_response = requests.get(image_url)
                    with open(f"{self.subdir}/{title}_cover.jpg", "wb") as f:
                        f.write(image_response.content)
                    break
