from book_detector import BookDetector


def main():
    book_detector = BookDetector(
        width=1280, height=720, model_path='./yolo_weights/yolov8n-seg.pt')

    while True:
        print("Options:")
        print("1. Camera mode")
        print("2. Library mode")
        print("3. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            # Camera mode
            book_detector.detect()
        elif choice == "2":
            # Library mode
            book_detector.show_detected()
        elif choice == "3":
            # Exit
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
