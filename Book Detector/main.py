from bookdetector import BookDetector


def main():
    book_detector = BookDetector()
    while True:
        print("Options:\n1. Camera mode\n2. Exit")

        choice = input("Enter your choice: ")

        if choice == "1":
            # Camera mode
            book_detector.detect()
        elif choice == "2":
            # Exit
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
