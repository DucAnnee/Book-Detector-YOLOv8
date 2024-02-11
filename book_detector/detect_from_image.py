# FILEPATH: main.py
import os
import pytesseract
from PIL import Image, ImageDraw
image_dir = "./book_detector/cropped_books"
image_files = [file for file in os.listdir(
    image_dir) if file.endswith(".jpg") or file.endswith(".png")]

# Loop through each image file
for file in image_files:
    # Open the image file
    image_path = os.path.join(image_dir, file)
    image = Image.open(image_path)

    # Extract text from the image using pytesseract
    text = pytesseract.image_to_string(image)

    # Create a drawing object
    draw = ImageDraw.Draw(image)

    # Loop through each character in the extracted text
    for char in text:
        # Get the bounding box of the character
        bbox = pytesseract.image_to_boxes(image).split('\n')
        for box in bbox:
            if box:
                char_info = box.split(' ')
                char_x, char_y, char_width, char_height = int(char_info[1]), int(
                    char_info[2]), int(char_info[3]), int(char_info[4])
                # Draw a rectangle around the character
                draw.rectangle(
                    [(char_x, char_y), (char_width, char_height)], outline="red")

    # Save the image with rectangles
    image.save(f"./book_detector/cropped_books/{file}")

    # Print the extracted text
    print(f"Text extracted from {file}:")
    print(text)
    print("---")
