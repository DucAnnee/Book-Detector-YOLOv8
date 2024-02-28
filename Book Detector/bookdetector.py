from ultralytics import YOLO
from sort import *
import cv2 as cv
import cvzone
import math
import pytesseract
import os


class BookDetector:
    subdir = "./book_detector/cropped_books"
    # pytesseract.pytesseract.tesseract_cmd = (
    #     r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # )
    # detected_text = []

    def __init__(
        self,
        model_path="./yolo_weights/yolov8l.pt",
        detect_threshold=0.5,
        capture_width=1280,
        capture_height=720,
        video_path=None,
    ):
        """
        Initializes the BookDetector object.

        Args:
            model_path (str, optional): Path to the YOLO model weights file. Defaults to "./yolo_weights/yolov8l.pt".
            detect_threshold (float, optional): Detection threshold for object detection. Defaults to 0.5.
            capture_width (int, optional): Width of the video capture. Defaults to 1280.
            capture_height (int, optional): Height of the video capture. Defaults to 720.
            video_path (str, optional): Path to the video file to be processed. Defaults to None.
        """

        if not os.path.exists(self.subdir):
            os.makedirs(self.subdir)

        if video_path != None:
            self.cap = cv.VideoCapture(video_path)
            print(f"Capture path = {video_path}")
        else:
            self.cap.set(3, capture_width)
            self.cap.set(4, capture_height)
            print(f"Capture mode: Webcam ({capture_width}x{capture_height})")

        self.model = YOLO(model_path)
        print(f"Model path = {model_path}")

        self.threshold = detect_threshold
        print(f"Detection threshold = {detect_threshold}")

        print("Detector initialized!")

    def detect(self):
        detected_id = []
        tracker = Sort(max_age=1000, min_hits=5, iou_threshold=0.4)
        counter = 0
        while True:
            _, img = self.cap.read()
            results = self.model.predict(img, stream=True, verbose=False)
            detections = np.empty((0, 5))

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    conf = math.ceil(box.conf[0] * 100) / 100
                    if box.cls[0] == "book" and conf >= self.threshold:
                        # cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        # cvzone.putTextRect(img, f'Book {conf}', (max(20, x1), max(40, y1)),
                        #                    scale=0.6, thickness=1, colorT=(0, 0, 0), colorR=(0, 255, 0), offset=10)

                        crr_array = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, crr_array))

            result_tracker = tracker.update(detections)

            for result in result_tracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
                detected_id.append(id)

                cvzone.putTextRect(
                    img,
                    f"{id}",
                    (max(20, x1), max(40, y1)),
                    scale=0.6,
                    thickness=1,
                    colorT=(0, 0, 0),
                    colorR=(0, 255, 0),
                    offset=10,
                )

                book_img = img[y1:y2, x1:x2]
                book_text = pytesseract.image_to_string(book_img)

                if (book_text != "") and (book_text not in self.detected_text):
                    cv.imwrite(f"{self.subdir}/book_img_{id}.jpg", book_img)
                    counter += 1
                    self.detected_text.append(book_text)

            cv.imshow("image", img)
            # cv.imshow('imgRegion', imgRegion)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        # for text in self.detected_text:
        #     print(text)
        self.cap.release()
        cv.destroyAllWindows()


# def search_book(self, title):
