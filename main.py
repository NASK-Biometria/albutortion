import os
import cv2
import logging
from src.pipeline import apply_and_save, TRANSFORMATIONS


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    image_list = [
        "example_data/celeb-df-2/id0/real/id0_0000_0001.jpg",
    ]

    for image_name in image_list:
        image_path = os.path.abspath(image_name)
        if not os.path.exists(image_path):
            logging.error(f"Image not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to load image: {image_path}")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for name, transformation in TRANSFORMATIONS:
            apply_and_save(image_name, image, name, transformation)


if __name__ == "__main__":
    main()
